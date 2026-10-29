"""
SPDX-FileCopyrightText: Copyright (c) 2026 Kernel.
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import logging

import cupy as cp
import holoscan as hs
import numpy as np
from utils.reconstruction.assets import get_assets

from ..stream import SampleOutput
from .types import BuildRHSOutput, VoxelMetadata

logger = logging.getLogger(__name__)


class BuildRHSOperator(hs.Operator):
    """Convert realtime moments tensors into trimmed Right-Hand Side (RHS)/Jacobian batches."""

    moments = hs.Input("moments", hs.pyobject)
    batch = hs.Output("batch", hs.pyobject)
    data_dir = hs.Param("", description="Directory holding the Jacobian and voxel assets.")

    def setup(self, spec) -> None:
        spec.input(self.moments)
        spec.output(self.batch)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.moments))

    def start(self) -> None:
        assets = get_assets(self.data_dir)
        # Keep CPU copies for disk-loaded assets; GPU copies are created lazily on first compute.
        self._model_optical_properties_cpu = np.concatenate((assets.mua, assets.musp)).astype(
            np.float32, copy=False
        )
        self._mega_jacobians_cpu = assets.mega_jacobian
        self._channel_mapping = assets.channel_mapping
        self._idxs_significant_voxels_cpu = assets.idxs_significant_voxels
        self._voxel_metadata = VoxelMetadata(
            ijk=assets.ijk, xyz=assets.xyz, resolution=assets.resolution
        )
        self._wavelengths = assets.wavelengths

        # GPU caches
        self._mega_jacobians_gpu = None
        self._model_optical_properties_gpu = None
        self._idxs_significant_voxels_gpu = None
        self._jacobian_cache = None
        self._last_frame = None  # previous frame (GPU)

    def _apply_baseline(self, realtime_moments):
        """
        simple diff against previous frame
        """
        if self._last_frame is None:
            self._last_frame = realtime_moments.copy()
            return None

        # diff with last frame and update last frame
        diff = realtime_moments - self._last_frame
        self._last_frame = realtime_moments.copy()
        return diff

    def _zero_out_invalids(self, data_rhs) -> None:
        invalid_samples = ~cp.isfinite(data_rhs)
        if not cp.any(invalid_samples):
            return

        # NOTE: this is in-place on GPU, async on the current CUDA stream.
        cp.nan_to_num(data_rhs, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    def _get_channel_indices(self, optode_order: list[tuple[int, int, int, int]]) -> list[int]:
        """Map optode tuples to jacobian channel indices (CPU-side dict lookups)."""
        indices: list[int] = []
        missing: list[tuple[int, int, int, int]] = []
        for src_module, src, det_module, det in optode_order:
            try:
                srcs = self._channel_mapping[str(src_module)]
                detectors = srcs[str(src)][str(det_module)]
                jacobian_index = detectors[str(det)]
            except KeyError:
                missing.append((int(src_module), int(src), int(det_module), int(det)))
                continue
            indices.append(int(jacobian_index))
        if missing:
            raise ValueError(
                f"{len(missing)} of {len(optode_order)} channels have no Jacobian mapping, "
                f"for example (source module, source, detector module, detector) "
                f"{', '.join(map(str, missing[:5]))}. Prepare the assets for this headset "
                "with prepare_data.py --headset."
            )
        if not indices:
            raise ValueError("Empty channel mapping; no channels resolved to jacobian indices")
        return indices

    def compute(self, ctx) -> None:
        sample = self.moments.receive()
        payload: SampleOutput = sample.data

        # Host->device copy (may be sync if host memory isn't pinned).
        realtime_moments = cp.asarray(payload.data, dtype=cp.float32)
        # take log of moment 0 to convert to optical density
        # shape is (moments, channels, wavelengths)
        cp.log(realtime_moments[0], out=realtime_moments[0])
        realtime_moments = self._apply_baseline(realtime_moments)
        if realtime_moments is None:
            logger.info("Skipping RHS build for first frame (baseline capture)")
            return

        flowaxis_optodes: list[tuple[int, int, int, int]] = [
            (
                payload.channels.source_module[channel_idx],
                payload.channels.source_number[channel_idx],
                payload.channels.detector_module[channel_idx],
                payload.channels.detector_number[channel_idx],
            )
            for channel_idx in range(len(payload.channels))
        ]

        # Validate that jacobian features dimension matches realtime moments
        # 5D jacobian shape: (channels, features, wavelengths, voxels, simulation_types)
        num_features = realtime_moments.shape[0]
        if self._mega_jacobians_cpu.shape[1] != num_features:
            raise ValueError(
                "Jacobian features dimension mismatch: expected "
                f"{self._mega_jacobians_cpu.shape[1]}, got {num_features} from moments data. "
                "Check SNIRF file format."
            )

        # One-time GPU uploads of large static assets.
        if self._mega_jacobians_gpu is None:
            self._mega_jacobians_gpu = cp.asarray(self._mega_jacobians_cpu, dtype=cp.float32)
        if self._model_optical_properties_gpu is None:
            self._model_optical_properties_gpu = cp.asarray(
                self._model_optical_properties_cpu, dtype=cp.float32
            )
        if self._idxs_significant_voxels_gpu is None:
            self._idxs_significant_voxels_gpu = cp.asarray(
                self._idxs_significant_voxels_cpu, dtype=cp.int64
            )

        if self._jacobian_cache is None:
            channel_indices = self._get_channel_indices(flowaxis_optodes)
            channel_indices_gpu = cp.asarray(channel_indices, dtype=cp.int64)
            jacobians = self._mega_jacobians_gpu[channel_indices_gpu, :, :, :, :]  # 5d
            # swap axes so it's features, channels first
            jacobians = jacobians.transpose(1, 0, 2, 3, 4)
            # reshape to 3d and use Fortran-style ordering
            jacobians = cp.reshape(
                jacobians,
                (
                    jacobians.shape[0] * jacobians.shape[1],
                    jacobians.shape[2],
                    jacobians.shape[3] * jacobians.shape[4],
                ),
                order="F",
            )
            self._jacobian_cache = jacobians

        data_jacobians = self._jacobian_cache

        # swap from (moments, channels, wavelengths) to (channels, moments, wavelengths)
        # then reshape to (channels x moments, wavelengths)
        data_rhs = realtime_moments.transpose(1, 0, 2).reshape(-1, realtime_moments.shape[2])
        self._zero_out_invalids(data_rhs)

        self.batch.forward(
            sample,
            BuildRHSOutput(
                data_jacobians=data_jacobians,
                data_rhs=data_rhs,
                model_optical_properties=self._model_optical_properties_gpu,
                idxs_significant_voxels=self._idxs_significant_voxels_gpu,
                num_full_voxels=int(self._voxel_metadata.ijk.shape[0]),
                num_features=int(num_features),
                wavelengths=tuple(self._wavelengths.tolist()),
                voxel_metadata=self._voxel_metadata,
            ),
        )
