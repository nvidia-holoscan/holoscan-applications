// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "pose.hpp"

// The AprilTag 3 library's C API.
extern "C" {
#include <apriltag.h>
#include <apriltag_pose.h>

#include <common/matd.h>
}

#include <cmath>
#include <memory>
#include <utility>

namespace tag_detection {
namespace {

struct MatdDeleter {
  void operator()(matd_t* m) const noexcept { matd_destroy(m); }
};
using MatdPtr = std::unique_ptr<matd_t, MatdDeleter>;

constexpr std::array<Point2, 4> kIdealCorners{{{-1.0, 1.0}, {1.0, 1.0}, {1.0, -1.0}, {-1.0, -1.0}}};

}  // namespace

std::optional<Homography> homography_from_corners(const std::array<Point2, 4>& corners) {
  // Direct linear transform with h[8] = 1: two equations per correspondence,
  // solved by Gaussian elimination with partial pivoting.
  std::array<std::array<double, 9>, 8> a{};
  for (int i = 0; i < 4; ++i) {
    const double x = kIdealCorners[i][0];
    const double y = kIdealCorners[i][1];
    const double u = corners[i][0];
    const double v = corners[i][1];
    a[2 * i] = {x, y, 1.0, 0.0, 0.0, 0.0, -x * u, -y * u, u};
    a[2 * i + 1] = {0.0, 0.0, 0.0, x, y, 1.0, -x * v, -y * v, v};
  }
  for (int col = 0; col < 8; ++col) {
    int pivot = col;
    for (int row = col + 1; row < 8; ++row) {
      if (std::abs(a[row][col]) > std::abs(a[pivot][col])) pivot = row;
    }
    if (std::abs(a[pivot][col]) < 1e-12) {
      return std::nullopt;
    }
    std::swap(a[col], a[pivot]);
    for (int row = 0; row < 8; ++row) {
      if (row == col) continue;
      const double factor = a[row][col] / a[col][col];
      for (int k = col; k < 9; ++k) a[row][k] -= factor * a[col][k];
    }
  }
  Homography result{};
  for (int i = 0; i < 8; ++i) {
    result[i] = a[i][8] / a[i][i];
    if (!std::isfinite(result[i])) {
      return std::nullopt;
    }
  }
  result[8] = 1.0;
  return result;
}

Point2 project(const Homography& h, double x, double y) noexcept {
  const double xs = h[0] * x + h[1] * y + h[2];
  const double ys = h[3] * x + h[4] * y + h[5];
  const double zs = h[6] * x + h[7] * y + h[8];
  return {xs / zs, ys / zs};
}

std::optional<TagPose> estimate_pose(const Detection& detection, const CameraIntrinsics& intrinsics,
                                     double tag_size) {
  const auto homography = homography_from_corners(detection.corners);
  if (!homography) {
    return std::nullopt;
  }
  MatdPtr h{matd_create_data(3, 3, homography->data())};
  apriltag_detection_t det{};
  det.id = detection.id;
  det.H = h.get();
  det.c[0] = detection.center[0];
  det.c[1] = detection.center[1];
  for (int i = 0; i < 4; ++i) {
    det.p[i][0] = detection.corners[i][0];
    det.p[i][1] = detection.corners[i][1];
  }
  apriltag_detection_info_t info{.det = &det,
                                 .tagsize = tag_size,
                                 .fx = intrinsics.fx,
                                 .fy = intrinsics.fy,
                                 .cx = intrinsics.cx,
                                 .cy = intrinsics.cy};
  apriltag_pose_t pose{};
  const double error = estimate_tag_pose(&info, &pose);
  MatdPtr rotation{pose.R};
  MatdPtr translation{pose.t};
  if (!rotation || !translation || !std::isfinite(error)) {
    return std::nullopt;
  }
  TagPose result;
  for (int i = 0; i < 9; ++i) {
    result.rotation[i] = rotation->data[i];
  }
  for (int i = 0; i < 3; ++i) {
    result.translation[i] = translation->data[i];
  }
  result.error = error;
  return result;
}

}  // namespace tag_detection
