# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check the cortex surface and its depth sampling on a sphere (CPU only)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import ndimage
from vtkmodules.util.numpy_support import vtk_to_numpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from operators.surface import (
    DEPTH_SAMPLES,
    cortex_surface,
    depth_coordinates,
    sample_surface,
    shade_colors,
    surface_shade,
)

SHAPE = (40, 44, 48)  # (K, J, I)
RADIUS = 14.0


class SurfaceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        kji = np.indices(SHAPE, dtype=np.float32)
        cls.center = np.array([(n - 1) / 2 for n in SHAPE], dtype=np.float32)
        cls.radius = np.sqrt(sum((kji[a] - cls.center[a]) ** 2 for a in range(3)))
        cls.mask = (cls.radius < RADIUS).astype(np.uint8) * 255
        cls.surface = cortex_surface(cls.mask)
        cls.points = vtk_to_numpy(cls.surface.GetPoints().GetData())

    def distance(self, ijk: np.ndarray) -> np.ndarray:
        return np.linalg.norm(ijk[:, ::-1] - self.center, axis=1)

    def test_surface_lies_on_the_mask_boundary(self) -> None:
        self.assertGreater(len(self.points), 1000)
        distance = self.distance(self.points)
        self.assertLess(np.abs(distance - RADIUS).max(), 1.5)

    def test_normals_point_outward(self) -> None:
        normals = vtk_to_numpy(self.surface.GetPointData().GetNormals())
        outward = self.points[:, ::-1] - self.center
        self.assertTrue((np.sum(normals[:, ::-1] * outward, axis=1) > 0).all())

    def test_depth_samples_run_inward(self) -> None:
        coordinates = depth_coordinates(self.mask, self.points, depth_voxels=4.0)
        samples = coordinates.reshape(3, DEPTH_SAMPLES, -1)
        first = self.distance(samples[:, 0].T[:, ::-1])
        last = self.distance(samples[:, -1].T[:, ::-1])
        np.testing.assert_allclose(first - last, 4.0, atol=0.5)

    def test_sample_surface_averages_over_depth(self) -> None:
        coordinates = depth_coordinates(self.mask, self.points, depth_voxels=4.0)
        values = sample_surface(self.radius, coordinates, ndimage.map_coordinates)
        self.assertEqual(values.shape, (len(self.points),))
        expected = self.distance(self.points) - 2.0
        np.testing.assert_allclose(values, expected, atol=0.5)

    def test_surface_shade_darkens_folds(self) -> None:
        # A slot cut into the top half of the sphere: its floor sits deep in a fold.
        center_i = SHAPE[2] // 2
        mask = self.mask.copy()
        mask[SHAPE[0] // 2 :, :, center_i - 1 : center_i + 1] = 0
        surface = cortex_surface(mask)
        points = vtk_to_numpy(surface.GetPoints().GetData())
        shade = surface_shade(mask, points, sigma_voxels=3.0)
        i, k = points[:, 0], points[:, 2]
        in_slot = (np.abs(i - (center_i - 0.5)) < 2.0) & (k > SHAPE[0] / 2 + 2)
        bottom = k < self.center[0] - RADIUS + 3
        self.assertGreater(in_slot.sum(), 10)
        self.assertLess(shade[in_slot].mean(), shade[bottom].mean() - 0.2)

    def test_shade_colors_interpolates_and_scales(self) -> None:
        stops = np.array([-1.0, 0.0, 1.0])
        colors = np.array([[0.0, 0.0, 1.0], [0.5, 0.5, 0.5], [1.0, 0.0, 0.0]])
        values = np.array([-2.0, -0.5, 0.0, 1.0])
        shade = np.array([1.0, 1.0, 0.5, 2.0])
        rgb = shade_colors(values, stops, colors, shade)
        np.testing.assert_array_equal(rgb, [[0, 0, 255], [64, 64, 191], [64, 64, 64], [255, 0, 0]])


if __name__ == "__main__":
    unittest.main()
