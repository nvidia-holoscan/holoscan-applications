"""
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

VoxelStreamToVolume operator: converts streaming voxel data to a dense 3D volume.
"""

from __future__ import annotations

from typing import NamedTuple

import cupy as cp
import cupyx.scipy.ndimage
import holoscan as hs
import nibabel as nib
import numpy as np

from operators.reconstruction.types import HbVolume

# Gray and white matter, the anatomy regions the 4.x renderer showed.
VISIBLE_LABELS = (3, 4)


class RenderVolume(NamedTuple):
    density: cp.ndarray  # float32 (K, J, I) on the anatomy mask grid
    mask: np.ndarray  # uint8 (K, J, I); 255 where the anatomy is drawn
    affine: np.ndarray  # 4x4 map from mask voxel indices (I, J, K) to anatomy space


class VoxelStreamToVolumeOp(hs.Operator):
    """Resample HbO voxel data onto the anatomy mask grid and scale it for display."""

    hb_volume = hs.Input("hb_volume", hs.pyobject)
    volume = hs.Output("volume", hs.pyobject)
    mask_path = hs.Param("", description="NIfTI file with integer anatomy labels.")
    stats_alpha = hs.Param(0.1, description="Running min/max smoothing factor.")
    visualization_scale = hs.Param(10.0, description="Amplifies small activations.")
    density_min = hs.Param(-100.0, description="Lowest displayed density.")
    density_max = hs.Param(100.0, description="Highest displayed density.")

    def setup(self, spec):
        spec.input(self.hb_volume)
        spec.output(self.volume)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.hb_volume))

    def start(self):
        if not self.mask_path:
            raise ValueError("VoxelStreamToVolume: No mask NIfTI path provided")
        try:
            img = nib.load(self.mask_path)
            labels = np.asarray(img.get_fdata(), dtype=np.uint8)
        except Exception as e:
            raise RuntimeError(
                f"VoxelStreamToVolume: Failed to load mask NIfTI '{self.mask_path}': {e}"
            ) from e
        self.mask_affine = np.asarray(img.affine, dtype=np.float64)
        self.mask_shape = labels.shape
        visible = np.isin(labels, VISIBLE_LABELS).astype(np.uint8) * 255
        self.mask = np.ascontiguousarray(visible.transpose(2, 1, 0))
        self.global_min = None
        self.global_max = None
        self.frame_count = 0

    def compute(self, ctx):
        sample = self.hb_volume.receive()
        hb: HbVolume = sample.data
        hb_voxel = hb.data
        if not isinstance(hb_voxel, cp.ndarray):
            raise TypeError(
                f"VoxelStreamToVolume: Invalid voxel data type: {type(hb_voxel)}, "
                "expected cupy array"
            )
        if hb_voxel.ndim != 3:
            raise ValueError(
                f"VoxelStreamToVolume: Invalid voxel data shape: {hb_voxel.shape}, expected 3D"
            )

        self._update_running_statistics(hb_voxel)
        hb_voxel_normalized = self._normalize_and_process_activated_voxels(
            hb_voxel,
            normalize_min_value=self.density_min + 1,
            normalize_max_value=self.density_max - 1,
        )
        volume_gpu = self._cupy_resample(
            hb_voxel_normalized, hb.affine, self.mask_affine, self.mask_shape
        )
        # (I, J, K) -> (K, J, I), so a C-ordered array matches VTK's x-fastest point order.
        volume_gpu = cp.ascontiguousarray(cp.transpose(volume_gpu, (2, 1, 0)), dtype=cp.float32)
        self.volume.forward(
            sample, RenderVolume(density=volume_gpu, mask=self.mask, affine=self.mask_affine)
        )

    def _update_running_statistics(self, hb_voxel: cp.ndarray):
        """
        Update running min/max statistics using exponential moving average.
        Initializes from first frame with valid data.

        Args:
            hb_voxel: Current voxel data (cupy array)
        """
        self.frame_count += 1
        current_min = float(cp.min(hb_voxel))
        current_max = float(cp.max(hb_voxel))

        if self.global_min is None or self.global_max is None:
            if current_min != 0 and current_max != 0:
                self.global_min = current_min
                self.global_max = current_max
            return

        alpha = self.stats_alpha if self.frame_count > 10 else 0.3
        self.global_min = (1 - alpha) * self.global_min + alpha * current_min
        self.global_max = (1 - alpha) * self.global_max + alpha * current_max

    def _normalize_and_process_activated_voxels(
        self, hb_voxel: cp.ndarray, normalize_min_value: float, normalize_max_value: float
    ):
        """
        Normalize the volume to [normalize_min_value, normalize_max_value] while preserving 0 as
        baseline. Applies visualization scale factor to amplify small activations in white/gray
        matter.
        """
        if self.global_min is None or self.global_max is None:
            return cp.zeros_like(hb_voxel, dtype=cp.float32)

        hb = hb_voxel.astype(cp.float32, copy=False)
        hb_voxel_normalized = cp.zeros_like(hb, dtype=cp.float32)

        if self.global_max > 0:
            pos_scale = (
                float(normalize_max_value) / float(self.global_max)
            ) * self.visualization_scale
            pos_mask = hb >= 0
            hb_voxel_normalized[pos_mask] = hb[pos_mask] * pos_scale

        if self.global_min < 0:
            neg_scale = (
                float(abs(normalize_min_value)) / float(abs(self.global_min))
            ) * self.visualization_scale
            neg_mask = hb < 0
            hb_voxel_normalized[neg_mask] = hb[neg_mask] * neg_scale

        cp.clip(
            hb_voxel_normalized, normalize_min_value, normalize_max_value, out=hb_voxel_normalized
        )
        return hb_voxel_normalized

    def _cupy_resample(self, data_gpu, src_affine, target_affine, target_shape):
        inv_src_affine = np.linalg.inv(src_affine)
        mapping_matrix = inv_src_affine @ target_affine
        matrix = mapping_matrix[:3, :3]
        offset = mapping_matrix[:3, 3]

        data_gpu = cp.asarray(data_gpu, dtype=cp.float32)
        matrix_gpu = cp.asarray(matrix)
        offset_gpu = cp.asarray(offset)

        resampled_gpu = cupyx.scipy.ndimage.affine_transform(
            data_gpu,
            matrix=matrix_gpu,
            offset=offset_gpu,
            output_shape=target_shape,
            order=1,
        )
        return resampled_gpu
