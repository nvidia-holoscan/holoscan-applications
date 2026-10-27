// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// A tag's X, Y and Z axes as arrows (red, green, blue) drawn into an RGBA8
// image. Header-only and free of any SDK, so its test runs anywhere.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>

#include "detector/detection.hpp"

namespace tag_detection {

using Rgba = std::array<std::uint8_t, 4>;
using Pixel = std::array<double, 2>;  // Pixel centers at integer coordinates.

inline constexpr Rgba kRed{255, 0, 0, 255};
inline constexpr Rgba kGreen{0, 255, 0, 255};
inline constexpr Rgba kBlue{0, 0, 255, 255};
inline constexpr std::array<Rgba, 3> kAxisColors{kRed, kGreen, kBlue};
inline constexpr double kArrowheadPx = 6.0;

// An RGBA8 image, rows packed.
struct RgbaCanvas {
  std::uint8_t* pixels;
  int width;
  int height;
};

// Where a tag's origin and its X, Y and Z tips appear; a missing tip is not drawn.
struct TagAxes {
  Pixel origin;
  std::array<std::optional<Pixel>, 3> tips;
};

// With a pose and intrinsics: the origin and points 0.5 * tag_size along the
// tag's x, y and z (z into the tag), projected; nullopt for an origin behind
// the camera. Without: X and Y in the image plane, from the center toward the
// middles of corners 1-2 (+x) and 0-1 (+y).
inline std::optional<TagAxes> tag_axes(const Detection& detection,
                                       const std::optional<CameraIntrinsics>& intrinsics,
                                       double tag_size) {
  const auto middle = [&](int a, int b) {
    return Pixel{0.5 * (detection.corners[a][0] + detection.corners[b][0]),
                 0.5 * (detection.corners[a][1] + detection.corners[b][1])};
  };
  if (!detection.pose || !intrinsics) {
    return TagAxes{detection.center, {middle(1, 2), middle(0, 1), std::nullopt}};
  }
  const auto& r = detection.pose->rotation;
  const auto& t = detection.pose->translation;
  const auto project = [&](double x, double y, double z) -> std::optional<Pixel> {
    const double cx = r[0] * x + r[1] * y + r[2] * z + t[0];
    const double cy = r[3] * x + r[4] * y + r[5] * z + t[1];
    const double cz = r[6] * x + r[7] * y + r[8] * z + t[2];
    if (!(cz > 1e-6)) return std::nullopt;
    return Pixel{intrinsics->fx * cx / cz + intrinsics->cx,
                 intrinsics->fy * cy / cz + intrinsics->cy};
  };
  const auto origin = project(0, 0, 0);
  if (!origin) return std::nullopt;
  const double l = 0.5 * tag_size;
  return TagAxes{*origin, {project(l, 0, 0), project(0, l, 0), project(0, 0, l)}};
}

// A 2 px line from a to b, clipped to the canvas.
inline void draw_line(const RgbaCanvas& canvas, Pixel a, Pixel b, const Rgba& color) {
  const double dx = b[0] - a[0];
  const double dy = b[1] - a[1];
  if (!std::isfinite(dx) || !std::isfinite(dy)) return;
  // Liang-Barsky against [-1, size], which holds every pixel the line can touch.
  double t0 = 0.0;
  double t1 = 1.0;
  const double p[4] = {-dx, dx, -dy, dy};
  const double q[4] = {a[0] + 1.0, canvas.width - a[0], a[1] + 1.0, canvas.height - a[1]};
  for (int i = 0; i < 4; ++i) {
    if (p[i] == 0.0) {
      if (q[i] < 0.0) return;
    } else if (p[i] < 0.0) {
      t0 = std::max(t0, q[i] / p[i]);
    } else {
      t1 = std::min(t1, q[i] / p[i]);
    }
  }
  if (t0 > t1) return;
  // Step at most a pixel at a time, setting the 2x2 pixels around each point.
  const int steps = static_cast<int>(std::ceil(std::max(std::abs(dx), std::abs(dy)) * (t1 - t0)));
  for (int s = 0; s <= steps; ++s) {
    const double t = t0 + (t1 - t0) * (steps == 0 ? 0.0 : static_cast<double>(s) / steps);
    const int x0 = static_cast<int>(std::floor(a[0] + t * dx - 0.5));
    const int y0 = static_cast<int>(std::floor(a[1] + t * dy - 0.5));
    for (int y = std::max(y0, 0); y <= std::min(y0 + 1, canvas.height - 1); ++y) {
      for (int x = std::max(x0, 0); x <= std::min(x0 + 1, canvas.width - 1); ++x) {
        std::copy(color.begin(), color.end(),
                  canvas.pixels + (static_cast<std::size_t>(y) * canvas.width + x) * 4);
      }
    }
  }
}

// Each axis as a line from the origin to its tip with a small arrowhead.
inline void draw_axes(const RgbaCanvas& canvas, const TagAxes& axes) {
  for (std::size_t i = 0; i < axes.tips.size(); ++i) {
    if (!axes.tips[i]) continue;
    const Pixel tip = *axes.tips[i];
    draw_line(canvas, axes.origin, tip, kAxisColors[i]);
    const double length = std::hypot(tip[0] - axes.origin[0], tip[1] - axes.origin[1]);
    if (!(length > 1.0)) continue;
    // Two strokes back from the tip, 25 degrees either side of the shaft.
    const double ux = (axes.origin[0] - tip[0]) / length;
    const double uy = (axes.origin[1] - tip[1]) / length;
    for (const double angle : {0.436, -0.436}) {
      const double c = std::cos(angle) * kArrowheadPx;
      const double s = std::sin(angle) * kArrowheadPx;
      draw_line(canvas, tip, {tip[0] + c * ux - s * uy, tip[1] + s * ux + c * uy}, kAxisColors[i]);
    }
  }
}

}  // namespace tag_detection
