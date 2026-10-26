// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// A deterministic scene of tags at known poses, rendered without external data,
// so the application and its tests can check detections against ground truth.

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "detection.hpp"
#include "pose.hpp"

namespace tag_detection {

struct SceneTag {
  int id{};
  TagPose pose;  // `error` unused.
};

struct SyntheticScene {
  int width{};
  int height{};
  CameraIntrinsics intrinsics;
  std::string family;
  double tag_size{};  // Meters across the black border.
  std::vector<SceneTag> tags;
};

// Four tags moving smoothly with `frame`: they tilt by up to 35 degrees, sit at
// 0, 90, 180 and 270 degrees in plane (plus up to 30 degrees either way), and
// move between about 0.9 and 1.3 m from the camera.
// @throws std::invalid_argument for an unknown family or a size below 320x240.
[[nodiscard]] SyntheticScene make_scene(const std::string& family, std::uint32_t frame, int width,
                                        int height);

// The scene as gray (1 channel) or RGB (3 channels) pixels, rows packed, over
// a lit, slightly noisy background. Edges are antialiased by supersampling.
[[nodiscard]] std::vector<std::uint8_t> render_scene(const SyntheticScene& scene, int channels);

// Where a tag's corners project, in Detection::corners' order.
[[nodiscard]] std::array<Point2, 4> project_corners(const SceneTag& tag,
                                                    const CameraIntrinsics& intrinsics,
                                                    double tag_size);

}  // namespace tag_detection
