"""
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import numpy as np
import vtk
from scipy import ndimage
from vtkmodules.util.numpy_support import numpy_to_vtk

DEPTH_SAMPLES = 5
SMOOTHING_ITERATIONS = 30
SULCUS_SIGMA_MM = 4.0
SULCUS_DARKENING = 0.6
# Blurred-mask values at a flat surface (about 0.5) and deep in a fold.
FOLD_RANGE = (0.45, 0.7)


def cortex_surface(mask: np.ndarray) -> vtk.vtkPolyData:
    """Return a smoothed surface around the nonzero voxels of a (K, J, I) mask.

    Points are in (I, J, K) voxel index coordinates and carry outward normals for shading.
    """
    image = vtk.vtkImageData()
    image.SetDimensions(mask.shape[2], mask.shape[1], mask.shape[0])
    inside = np.ascontiguousarray(mask > 0, dtype=np.uint8).ravel()
    image.GetPointData().SetScalars(numpy_to_vtk(inside, deep=1, array_type=vtk.VTK_UNSIGNED_CHAR))
    nets = vtk.vtkSurfaceNets3D()
    nets.SetInputData(image)
    nets.SetLabel(0, 1)
    nets.SetOutputMeshTypeToTriangles()
    nets.SmoothingOn()
    nets.GetSmoother().SetNumberOfIterations(SMOOTHING_ITERATIONS)
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(nets.GetOutputPort())
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.SplittingOff()
    normals.Update()
    surface = vtk.vtkPolyData()
    surface.ShallowCopy(normals.GetOutput())
    return surface


def depth_coordinates(
    mask: np.ndarray, points: np.ndarray, depth_voxels: float, samples: int = DEPTH_SAMPLES
) -> np.ndarray:
    """Return (3, samples * N) (k, j, i) coordinates from each (i, j, k) point into the mask.

    The inward direction is the gradient of the blurred mask, which doesn't depend on how the
    surface normals of separate pieces of anatomy are oriented.
    """
    blurred = ndimage.gaussian_filter((mask > 0).astype(np.float32), 1.0)
    kji = np.asarray(points, dtype=np.float32)[:, ::-1].T
    gradient = np.stack([ndimage.map_coordinates(g, kji, order=1) for g in np.gradient(blurred)])
    length = np.linalg.norm(gradient, axis=0)
    inward = gradient / np.where(length > 0, length, 1.0)
    depths = np.linspace(0.0, depth_voxels, samples, dtype=np.float32)
    coordinates = kji[:, None, :] + inward[:, None, :] * depths[None, :, None]
    return np.ascontiguousarray(coordinates.reshape(3, -1), dtype=np.float32)


def sample_surface(volume, coordinates, map_coordinates, samples: int = DEPTH_SAMPLES):
    """Average a (K, J, I) volume over each surface point's depth samples.

    Pass NumPy arrays with scipy.ndimage.map_coordinates, or CuPy arrays with the cupyx one.
    """
    values = map_coordinates(volume, coordinates, order=1)
    return values.reshape(samples, -1).mean(axis=0)


def surface_shade(
    mask: np.ndarray,
    points: np.ndarray,
    sigma_voxels: float,
    darkening: float = SULCUS_DARKENING,
) -> np.ndarray:
    """Return a fixed brightness per (i, j, k) point, darker in folds.

    Points with more tissue around them, as at the bottom of a sulcus, read higher in the
    blurred mask and are darkened by up to `darkening`; flat and convex areas are not.
    """
    blurred = ndimage.gaussian_filter((mask > 0).astype(np.float32), sigma_voxels)
    enclosed = ndimage.map_coordinates(blurred, np.asarray(points)[:, ::-1].T, order=1)
    low, high = FOLD_RANGE
    fold = np.clip((enclosed - low) / (high - low), 0.0, 1.0)
    return (1.0 - darkening * fold).astype(np.float32)


def shade_colors(values, stops, colors, shade, xp=np):
    """Map values through linear (stops, (S, 3) colors) and scale by shade into (N, 3) uint8.

    Pass NumPy arrays with xp=np or CuPy arrays with xp=cupy.
    """
    rgb = xp.stack([xp.interp(values, stops, colors[:, c]) for c in range(3)], axis=1)
    return (xp.clip(rgb * shade[:, None], 0.0, 1.0) * 255.0 + 0.5).astype(xp.uint8)
