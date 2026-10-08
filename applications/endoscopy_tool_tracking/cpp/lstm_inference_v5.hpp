/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvInfer.h>

#include <array>
#include <cstdint>
#include <memory>
#include <string>

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

struct LstmInferenceParams {
  std::string model_path;
  std::string engine_cache_dir;
  bool force_engine_update{false};
  // FP16 produces NaN outputs for this recurrent model on GB10 / TensorRT 10.16.
  bool enable_fp16{false};
  std::uint64_t max_workspace_size{2'147'483'648ULL};
};

HOLOSCAN_PARAM_SCHEMA(
    LstmInferenceParams,
    holoscan::param_schema("com.nvidia.holohub.endoscopy.LstmInferenceParams", 1),
    holoscan::field<&LstmInferenceParams::model_path>("model_path"),
    holoscan::field<&LstmInferenceParams::engine_cache_dir>("engine_cache_dir"),
    holoscan::field<&LstmInferenceParams::force_engine_update>("force_engine_update"),
    holoscan::field<&LstmInferenceParams::enable_fp16>("enable_fp16"),
    holoscan::field<&LstmInferenceParams::max_workspace_size>("max_workspace_size"));

// Native Holoscan 5 recurrent TensorRT operator for tool_loc_convlstm.onnx.
// The two state inputs are owned as bounded ping-pong CUDA buffers, initialized
// once to zero, and recycled internally after each inference activation.
class LstmInferenceOp final : public holoscan::Operator<LstmInferenceParams> {
 public:
  LstmInferenceOp();
  ~LstmInferenceOp() override;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Input<holoscan::Tensor> source_video;
  holoscan::Output<holoscan::Tensor> probs;
  holoscan::Output<holoscan::Tensor> scaled_coords;
  holoscan::Output<holoscan::Tensor> binary_masks;

 private:
  class Logger;

  void release() noexcept;
  void load_or_build_engine();
  [[nodiscard]] std::string engine_cache_path() const;

  std::unique_ptr<Logger> logger_;
  nvinfer1::IRuntime* runtime_{};
  nvinfer1::ICudaEngine* engine_{};
  nvinfer1::IExecutionContext* execution_context_{};
  std::array<void*, 2> cell_state_{};
  std::array<void*, 2> hidden_state_{};
  std::uint32_t current_state_{};
  bool state_initialized_{};
};

}  // namespace endoscopy_tool_tracking
