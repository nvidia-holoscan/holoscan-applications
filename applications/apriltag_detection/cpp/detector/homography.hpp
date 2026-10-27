// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// The homography from the ideal tag square to four image corners, shared by the
// host pose estimate and the GPU decoder.

#pragma once

#include <cmath>

#if defined(__CUDACC__)
#define TAG_DETECTION_HOST_DEVICE __host__ __device__
#else
#define TAG_DETECTION_HOST_DEVICE
#endif

namespace tag_detection {

// The homography (row-major 3x3, h[8] = 1) taking the ideal tag corners
// (-1, 1), (1, 1), (1, -1), (-1, -1) to `corners`, in that order (the order of
// Detection::corners). False if the corners are degenerate.
TAG_DETECTION_HOST_DEVICE inline bool homography_from_corners(const double (&corners)[4][2],
                                                              double (&h)[9]) {
  using std::fabs;
  using std::isfinite;
  constexpr double kIdeal[4][2] = {{-1.0, 1.0}, {1.0, 1.0}, {1.0, -1.0}, {-1.0, -1.0}};
  // Direct linear transform: two equations per correspondence, solved by
  // Gaussian elimination with partial pivoting.
  double a[8][9];
  for (int i = 0; i < 4; ++i) {
    const double x = kIdeal[i][0];
    const double y = kIdeal[i][1];
    const double u = corners[i][0];
    const double v = corners[i][1];
    const double row0[9] = {x, y, 1.0, 0.0, 0.0, 0.0, -x * u, -y * u, u};
    const double row1[9] = {0.0, 0.0, 0.0, x, y, 1.0, -x * v, -y * v, v};
    for (int k = 0; k < 9; ++k) {
      a[2 * i][k] = row0[k];
      a[2 * i + 1][k] = row1[k];
    }
  }
  for (int col = 0; col < 8; ++col) {
    int pivot = col;
    for (int row = col + 1; row < 8; ++row) {
      if (fabs(a[row][col]) > fabs(a[pivot][col])) pivot = row;
    }
    if (fabs(a[pivot][col]) < 1e-12) {
      return false;
    }
    for (int k = 0; k < 9; ++k) {
      const double swap = a[col][k];
      a[col][k] = a[pivot][k];
      a[pivot][k] = swap;
    }
    for (int row = 0; row < 8; ++row) {
      if (row == col) continue;
      const double factor = a[row][col] / a[col][col];
      for (int k = col; k < 9; ++k) a[row][k] -= factor * a[col][k];
    }
  }
  for (int i = 0; i < 8; ++i) {
    h[i] = a[i][8] / a[i][i];
    if (!isfinite(h[i])) {
      return false;
    }
  }
  h[8] = 1.0;
  return true;
}

TAG_DETECTION_HOST_DEVICE inline void project(const double (&h)[9], double x, double y,
                                              double (&out)[2]) {
  const double z = h[6] * x + h[7] * y + h[8];
  out[0] = (h[0] * x + h[1] * y + h[2]) / z;
  out[1] = (h[3] * x + h[4] * y + h[5]) / z;
}

}  // namespace tag_detection
