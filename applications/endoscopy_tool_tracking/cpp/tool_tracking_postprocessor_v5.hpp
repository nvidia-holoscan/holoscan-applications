/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_contract.hpp>

#include "video_preprocessor.hpp"

namespace endoscopy_tool_tracking {

struct ToolTrackingPostprocessorParams {
  float min_probability{0.5F};
};

HOLOSCAN_PARAM_SCHEMA(
    ToolTrackingPostprocessorParams,
    holoscan::param_schema("com.nvidia.holohub.endoscopy.ToolTrackingPostprocessorParams", 1),
    holoscan::field<&ToolTrackingPostprocessorParams::min_probability>("min_probability")
        .range(0.0F, 1.0F));

// Converts the model's three outputs into HoloViz-native layers:
// - mask: CUDA uint8 [60,107,4], straight-alpha RGBA
// - crosses: host float32 [21], seven packed (x,y,size) cross primitives
class ToolTrackingPostprocessorOp final
    : public holoscan::Operator<ToolTrackingPostprocessorParams> {
 public:
  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Input<holoscan::Tensor> probs;
  holoscan::Input<holoscan::Tensor> scaled_coords;
  holoscan::Input<holoscan::Tensor> binary_masks;
  holoscan::Output<holoscan::schema::ImageT> mask;
  holoscan::Output<holoscan::Tensor> crosses;
};

}  // namespace endoscopy_tool_tracking
