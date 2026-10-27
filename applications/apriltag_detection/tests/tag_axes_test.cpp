// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// The overlay's axis projection and drawing, on known poses and pixels. Needs no
// SDK or GPU, so it runs on hosted CI.

#include "operators/tag_axes.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "test_support.hpp"

namespace tag_detection {
namespace {

constexpr int kWidth = 200;
constexpr int kHeight = 100;
constexpr std::uint8_t kGuard = 0xA5;

bool near(const std::optional<Pixel>& p, double x, double y) {
  return p && std::abs((*p)[0] - x) < 1e-9 && std::abs((*p)[1] - y) < 1e-9;
}

// A transparent canvas with guard bytes on both sides, to catch writes outside it.
struct Canvas {
  std::vector<std::uint8_t> bytes = std::vector<std::uint8_t>(kWidth * kHeight * 4 + 128, 0);
  Canvas() {
    std::fill(bytes.begin(), bytes.begin() + 64, kGuard);
    std::fill(bytes.end() - 64, bytes.end(), kGuard);
  }
  [[nodiscard]] RgbaCanvas view() { return {bytes.data() + 64, kWidth, kHeight}; }
  [[nodiscard]] Rgba at(int x, int y) const {
    const std::uint8_t* p = bytes.data() + 64 + (static_cast<std::size_t>(y) * kWidth + x) * 4;
    return {p[0], p[1], p[2], p[3]};
  }
  [[nodiscard]] bool guards_intact() const {
    return std::all_of(bytes.begin(), bytes.begin() + 64, [](auto b) { return b == kGuard; }) &&
           std::all_of(bytes.end() - 64, bytes.end(), [](auto b) { return b == kGuard; });
  }
};

// Identity rotation, the tag 2 m ahead: each axis tip is 0.1 m along it.
Detection posed_tag() {
  Detection detection;
  detection.pose =
      TagPose{.rotation = {1, 0, 0, 0, 1, 0, 0, 0, 1}, .translation = {0.25, -0.5, 2.0}};
  return detection;
}

void check_projection() {
  const CameraIntrinsics k{400, 400, 50, 150};
  const auto axes = tag_axes(posed_tag(), k, 0.2);
  VERIFY(axes.has_value());
  VERIFY(std::abs(axes->origin[0] - 100.0) < 1e-9 && std::abs(axes->origin[1] - 50.0) < 1e-9);
  VERIFY(near(axes->tips[0], 120.0, 50.0));
  VERIFY(near(axes->tips[1], 100.0, 70.0));
  VERIFY(near(axes->tips[2], 400 * 0.25 / 2.1 + 50, 400 * -0.5 / 2.1 + 150));
  // A tag behind the camera has no axes.
  Detection behind = posed_tag();
  behind.pose->translation[2] = -1.0;
  VERIFY(!tag_axes(behind, k, 0.2).has_value());
}

void check_without_pose() {
  Detection detection;
  detection.center = {50, 50};
  detection.corners = {{{20, 80}, {80, 80}, {80, 20}, {20, 20}}};
  const auto axes = tag_axes(detection, std::nullopt, 0.0);
  VERIFY(axes && axes->origin == (Pixel{50, 50}));
  VERIFY(near(axes->tips[0], 80, 50));  // Middle of corners 1 and 2: +x.
  VERIFY(near(axes->tips[1], 50, 80));  // Middle of corners 0 and 1: +y.
  VERIFY(!axes->tips[2]);
}

void check_colors() {
  Canvas canvas;
  draw_axes(canvas.view(), *tag_axes(posed_tag(), CameraIntrinsics{400, 400, 50, 150}, 0.2));
  VERIFY((canvas.at(110, 50) == kAxisColors[0]));
  VERIFY((canvas.at(100, 62) == kAxisColors[1]));
  VERIFY((canvas.at(97, 54) == kAxisColors[2]));
  VERIFY((canvas.at(10, 10) == Rgba{0, 0, 0, 0}));
  // Arrowheads: pixels beside the shaft just short of the X tip.
  VERIFY((canvas.at(116, 47) == kAxisColors[0] || canvas.at(116, 52) == kAxisColors[0]));
  VERIFY(canvas.guards_intact());
}

void check_clipping() {
  Canvas canvas;
  draw_line(canvas.view(), {-50, 10}, {kWidth + 50, 10}, kAxisColors[0]);
  draw_line(canvas.view(), {-1000, -1000}, {kWidth + 1000, kHeight + 1000}, kAxisColors[1]);
  draw_line(canvas.view(), {-5, -5}, {-50, 40}, kAxisColors[2]);  // Entirely outside.
  draw_line(canvas.view(), {NAN, 0}, {10, 10}, kAxisColors[2]);
  VERIFY((canvas.at(0, 10) == kAxisColors[0] && canvas.at(kWidth - 1, 10) == kAxisColors[0]));
  constexpr Rgba kClear{0, 0, 0, 0};
  VERIFY((canvas.at(kWidth - 1, kHeight - 1) == kClear ||
          canvas.at(kWidth - 1, kHeight - 1) == kAxisColors[1]));
  int blue = 0;
  for (int y = 0; y < kHeight; ++y) {
    for (int x = 0; x < kWidth; ++x) blue += canvas.at(x, y) == kAxisColors[2];
  }
  VERIFY(blue == 0);
  VERIFY(canvas.guards_intact());
}

}  // namespace
}  // namespace tag_detection

int main() {
  return tag_detection::testing::run({
      {"axes projected with a pose", tag_detection::check_projection},
      {"axes in the image plane without a pose", tag_detection::check_without_pose},
      {"axes drawn in their colors", tag_detection::check_colors},
      {"lines clipped to the image", tag_detection::check_clipping},
  });
}
