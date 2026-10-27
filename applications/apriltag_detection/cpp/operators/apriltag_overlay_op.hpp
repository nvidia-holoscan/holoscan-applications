// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// AprilTagOverlayOp: turns detections into Holoviz layers: each tag's axes as
// red, green and blue arrows on a transparent RGBA8 image, and an "id N" label.

#pragma once

#include <cstdint>
#include <optional>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include "apriltag_detections_schema_traits.hpp"
#include "detector/detection.hpp"

namespace tag_detection {

// Bytes of text per label row after Holoviz's 16-byte run header.
inline constexpr std::uint32_t kOverlayLabelBytes = 16;

struct AprilTagOverlayConfig {
  int width{};  // Largest image, in pixels.
  int height{};
  // With both, tags with a pose get 3D axes 0.5 * tag_size long; others get X and Y.
  std::optional<CameraIntrinsics> intrinsics;
  double tag_size{};
};

class AprilTagOverlayOp final : public holoscan::Operator<> {
 public:
  explicit AprilTagOverlayOp(AprilTagOverlayConfig config) : config_(config) {}

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Input<schema::AprilTagDetectionsT> detections;
  // For a Holoviz RGBA8 image layer: the axes over a transparent background.
  holoscan::Output<holoscan::schema::ImageT> axes;
  // For a Holoviz text layer: one row per tag, empty rows unused.
  holoscan::Output<holoscan::Tensor> labels;

 private:
  AprilTagOverlayConfig config_;
};

}  // namespace tag_detection
