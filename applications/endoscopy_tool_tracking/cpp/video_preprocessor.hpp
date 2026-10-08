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

#include "video_stream_replayer.hpp"

namespace endoscopy_tool_tracking {

struct VideoPreprocessorParams {
  std::uint32_t width{854U};
  std::uint32_t height{480U};
};

HOLOSCAN_PARAM_SCHEMA(VideoPreprocessorParams,
                      holoscan::param_schema("com.nvidia.holohub.endoscopy.VideoPreprocessorParams",
                                             1),
                      holoscan::field<&VideoPreprocessorParams::width>("width").range(1U, 854U),
                      holoscan::field<&VideoPreprocessorParams::height>("height").range(1U, 480U));

inline constexpr DLDataType kFloat32{kDLFloat, 32U, 1U};
// Converts host HWC RGB888 to CUDA NHWC float32. Values remain in [0, 255],
// matching the v4.5 FormatConverter configuration used to train this model.
class VideoPreprocessorOp final : public holoscan::Operator<VideoPreprocessorParams> {
 public:
  VideoPreprocessorOp() = default;
  ~VideoPreprocessorOp() override;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Input<holoscan::schema::ImageT> input;
  holoscan::Output<holoscan::Tensor> output;

 private:
  std::uint8_t* device_rgb_{};
};

}  // namespace endoscopy_tool_tracking
