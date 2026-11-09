# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write a small synthetic dataset in the layout the application reads from its data directory."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np

WAVELENGTHS = (690, 905)
MOMENTS = 3
SAMPLE_PERIOD = 1 / 4.75  # Kernel Flow's native frame rate
GRID = 10  # reconstruction voxels per axis
SPACING_MM = 4.0
MASK_SHAPE = (24, 24, 24)
MASK_SPACING_MM = 2.0
# 4.x HbO/HbR molar extinction coefficients are only needed to be well conditioned here.
EXTINCTION = {690: (0.95, 4.85), 905: (2.83, 1.79)}


def _channels(modules: int, sources: int, detectors: int) -> list[tuple[int, int, int, int]]:
    return [
        (module, source, module, detector)
        for module in range(1, modules + 1)
        for source in range(1, sources + 1)
        for detector in range(1, detectors + 1)
    ]


def write_synthetic_dataset(directory: Path, samples: int = 12, seed: int = 0) -> Path:
    """Write SNIRF moments, reconstruction assets, and an anatomy mask under directory."""
    rng = np.random.default_rng(seed)
    directory.mkdir(parents=True, exist_ok=True)
    voxel_info = directory / "voxel_info"
    voxel_info.mkdir(exist_ok=True)

    axis = np.arange(GRID, dtype=np.float32)
    ijk = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    xyz = (ijk - (GRID - 1) / 2) * SPACING_MM
    significant = np.flatnonzero(np.linalg.norm(xyz, axis=1) < 0.45 * GRID * SPACING_MM)
    np.save(voxel_info / "ijk.npy", ijk)
    np.save(voxel_info / "xyz.npy", xyz.astype(np.float32))
    np.save(voxel_info / "idxs_significant_voxels.npy", significant)
    np.save(voxel_info / "wavelengths.npy", np.array(WAVELENGTHS))
    np.save(voxel_info / "resolution.npy", np.full(3, SPACING_MM))
    optical = rng.uniform(0.5, 1.5, size=(significant.size, len(WAVELENGTHS)))
    np.save(voxel_info / "mua.npy", (0.01 * optical).astype(np.float32))
    np.save(voxel_info / "musp.npy", (1.0 * optical).astype(np.float32))

    channels = _channels(modules=2, sources=1, detectors=3)
    jacobian = rng.uniform(
        0.1, 1.0, size=(len(channels), MOMENTS, len(WAVELENGTHS), significant.size, 2)
    ).astype(np.float32)
    np.save(directory / "flow_mega_jacobian.npy", jacobian)
    mapping: dict = {}
    for index, (source_module, source, detector_module, detector) in enumerate(channels):
        mapping.setdefault(str(source_module), {}).setdefault(str(source), {}).setdefault(
            str(detector_module), {}
        )[str(detector)] = index
    (directory / "flow_channel_map.json").write_text(json.dumps(mapping))

    with (directory / "extinction_coefficients_mua.csv").open("w") as f:
        f.write("Wavelength,HbO,deoxyHb,Water,Lipids,LuTex,GdTex\n")
        for wavelength, (hbo, hbr) in EXTINCTION.items():
            f.write(f"{wavelength},{hbo},{hbr},0,0,0,0\n")

    _write_snirf(directory / "data.snirf", channels, samples, rng)
    _write_mask(directory / "anatomy_labels_high_res.nii.gz")
    return directory


def _write_snirf(path: Path, channels, samples: int, rng) -> None:
    source_labels = sorted({f"M{m}S{s}" for m, s, _, _ in channels})
    detector_labels = sorted({f"M{m}D{d}" for _, _, m, d in channels})
    phase = np.linspace(0.0, 2.0 * np.pi, samples)
    with h5py.File(path, "w") as snirf:
        probe = snirf.create_group("nirs/probe")
        probe["sourceLabels"] = np.array(source_labels, dtype="S")
        probe["detectorLabels"] = np.array(detector_labels, dtype="S")
        probe["sourcePos3D"] = rng.normal(size=(len(source_labels), 3))
        probe["detectorPos3D"] = rng.normal(size=(len(detector_labels), 3))
        probe["momentOrders"] = np.arange(MOMENTS)
        data = snirf.create_group("nirs/data1")
        series = []
        index = 0
        for moment in range(MOMENTS):
            for wavelength in range(len(WAVELENGTHS)):
                for source_module, source, detector_module, detector in channels:
                    index += 1
                    entry = data.create_group(f"measurementList{index}")
                    entry["sourceIndex"] = source_labels.index(f"M{source_module}S{source}") + 1
                    entry["detectorIndex"] = (
                        detector_labels.index(f"M{detector_module}D{detector}") + 1
                    )
                    entry["dataTypeIndex"] = moment + 1
                    entry["wavelengthIndex"] = wavelength + 1
                    base = (1e5, 1e3, 1e6)[moment]
                    change = 0.02 * np.sin(phase + index) + 0.002 * rng.normal(size=samples)
                    series.append(base * (1.0 + change))
        data["time"] = np.arange(samples) * SAMPLE_PERIOD
        data["dataTimeSeries"] = np.stack(series, axis=1)


def _write_mask(path: Path) -> None:
    axes = [(np.arange(n) - (n - 1) / 2) * MASK_SPACING_MM for n in MASK_SHAPE]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    radius = np.sqrt(x**2 + y**2 + z**2)
    labels = np.zeros(MASK_SHAPE, dtype=np.uint8)
    labels[radius < 20] = 3  # gray matter
    labels[radius < 12] = 4  # white matter
    affine = np.diag([MASK_SPACING_MM] * 3 + [1.0])
    affine[:3, 3] = [axis[0] for axis in axes]
    nib.Nifti1Image(labels, affine).to_filename(path)
