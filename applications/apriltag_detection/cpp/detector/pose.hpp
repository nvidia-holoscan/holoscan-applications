// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Homography and pose from a detection's corners. Every backend reports corners
// only, and poses are estimated here, so all backends report poses the same way.

#pragma once

#include <array>
#include <optional>

#include "detection.hpp"

namespace tag_detection {

using Homography = std::array<double, 9>;  // Row-major 3x3.
using Point2 = std::array<double, 2>;

// The homography taking the ideal tag's corners (-1, 1), (1, 1), (1, -1) and
// (-1, -1) to `corners`, in that order (Detection::corners' order).
[[nodiscard]] std::optional<Homography> homography_from_corners(
    const std::array<Point2, 4>& corners);

[[nodiscard]] Point2 project(const Homography& h, double x, double y) noexcept;

// The pose of a tag `tag_size` meters across its black border, from its corners
// and the camera's intrinsics; nullopt if the corners are degenerate.
[[nodiscard]] std::optional<TagPose> estimate_pose(const Detection& detection,
                                                   const CameraIntrinsics& intrinsics,
                                                   double tag_size);

}  // namespace tag_detection
