// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <NvInfer.h>

#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <vector>

// Exercise the private engine validator without exposing a test-only public API.
#include "../cpp/lstm_inference_v5.cpp"  // NOLINT(build/include)

namespace {
class TestLogger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kERROR) {
      std::cerr << message << '\n';
    }
  }
};
}  // namespace

int main() {
  using endoscopy_tool_tracking::validate_engine;
  TestLogger logger;
  std::unique_ptr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(logger)};
  if (!builder) {
    throw std::runtime_error("TensorRT builder unavailable");
  }
  std::unique_ptr<nvinfer1::INetworkDefinition> network{builder->createNetworkV2(0U)};
  std::unique_ptr<nvinfer1::IBuilderConfig> configuration{builder->createBuilderConfig()};
  if (!network || !configuration) {
    throw std::runtime_error("TensorRT network/configuration unavailable");
  }
  auto identity_output = [&](const char* input, const char* output, nvinfer1::Dims dimensions) {
    auto* tensor = network->addInput(input, nvinfer1::DataType::kFLOAT, dimensions);
    auto* layer = tensor ? network->addIdentity(*tensor) : nullptr;
    if (!layer) {
      throw std::runtime_error("failed to create identity fixture");
    }
    auto* result = layer->getOutput(0);
    if (std::string_view(output) == "probs:0") {
      auto* flattened = network->addShuffle(*result);
      if (!flattened) {
        throw std::runtime_error("failed to create oversized probability fixture");
      }
      flattened->setReshapeDimensions(nvinfer1::Dims2{1, 480 * 854 * 3});
      result = flattened->getOutput(0);
    }
    result->setName(output);
    network->markOutput(*result);
  };
  // All expected names, dtypes, ranks, and directions are present, but the
  // probability output holds an entire video: it would overflow the 7-float loan.
  identity_output("data_ph:0", "probs:0", nvinfer1::Dims4{1, 480, 854, 3});
  identity_output("cellstate_ph:0", "Model/net_states:0", nvinfer1::Dims4{1, 60, 107, 7});
  identity_output("hiddenstate_ph:0", "Model/net_hidden:0", nvinfer1::Dims4{1, 60, 107, 7});
  const std::vector<float> coordinates(14U, 0.0F);
  const std::vector<float> masks(7U * 60U * 107U, 0.0F);
  auto constant_output = [&](const char* name, nvinfer1::Dims dimensions,
                             const std::vector<float>& values) {
    auto* layer = network->addConstant(dimensions,
                                       nvinfer1::Weights{nvinfer1::DataType::kFLOAT, values.data(),
                                                         static_cast<std::int64_t>(values.size())});
    if (!layer) {
      throw std::runtime_error("failed to create constant fixture");
    }
    layer->getOutput(0)->setName(name);
    network->markOutput(*layer->getOutput(0));
  };
  constant_output("Localize/scaled_coords:0", nvinfer1::Dims3{1, 7, 2}, coordinates);
  constant_output("Localize_1/binary_masks:0", nvinfer1::Dims4{1, 7, 60, 107}, masks);
  std::unique_ptr<nvinfer1::IHostMemory> serialized{
      builder->buildSerializedNetwork(*network, *configuration)};
  if (!serialized) {
    throw std::runtime_error("failed to build malformed engine fixture");
  }

  std::unique_ptr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};
  if (!runtime) {
    throw std::runtime_error("TensorRT runtime unavailable");
  }
  std::unique_ptr<nvinfer1::ICudaEngine> engine{
      runtime->deserializeCudaEngine(serialized->data(), serialized->size())};
  if (!engine) {
    throw std::runtime_error("failed to deserialize malformed engine fixture");
  }
  try {
    validate_engine(*engine);
  } catch (const std::runtime_error& error) {
    if (std::string_view(error.what()).find("shape: probs:0") != std::string_view::npos) {
      std::cout << "oversized probability binding was rejected before inference\n";
      return 0;
    }
    throw;
  }
  throw std::runtime_error("unsafe engine was accepted despite oversized probability output");
}
