// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// AprilTagOverlayOp: turns detections into Holoviz layers, an outline per tag
// (with a stroke from its center along its x axis) and an "id N" label.

#pragma once

#include <cstdint>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>

#include "apriltag_detections_schema_traits.hpp"

namespace tag_detection {

// Line segments per tag on the outline layer: four edges and the x-axis stroke.
inline constexpr std::uint32_t kOverlayLinesPerTag = 5;
// Bytes of text per label row after Holoviz's 16-byte run header.
inline constexpr std::uint32_t kOverlayLabelBytes = 16;

class AprilTagOverlayOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Input<schema::AprilTagDetectionsT> detections;
  // For a Holoviz kLineList layer: x0, y0, x1, y1 per segment, normalized.
  holoscan::Output<holoscan::Tensor> outlines;
  // For a Holoviz text layer: one row per tag, empty rows unused.
  holoscan::Output<holoscan::Tensor> labels;
};

}  // namespace tag_detection
