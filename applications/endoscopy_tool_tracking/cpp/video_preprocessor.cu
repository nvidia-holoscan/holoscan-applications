/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

#include "video_preprocessor.hpp"

namespace endoscopy_tool_tracking {
namespace {

__global__ void rgb_u8_to_float(const std::uint8_t* input, float* output, std::size_t elements) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    output[index] = static_cast<float>(input[index]);
  }
}

}  // namespace

VideoPreprocessorOp::~VideoPreprocessorOp() { stop(); }

void VideoPreprocessorOp::setup(holoscan::OperatorSpec& spec) {
  const auto& p = params();
  const std::uint64_t bytes = static_cast<std::uint64_t>(p.width) * p.height * 3U * sizeof(float);
  auto input_representation = kRgbU8Frame;
  input_representation.strides = holoscan::TensorStridePolicy::kContiguousOnly;
  spec.input(input, "input")
      .queue_depth(2U)
      .expects_tensor(holoscan::TensorInputSpec{.representation = input_representation});
  spec.output(output, "source_video")
      .produces_tensor(holoscan::TensorOutputSpec{
          .representation =
              holoscan::TensorRepresentation{
                  .memory_kind = holoscan::MemoryKind::kCudaDevice,
                  .dtype = kFloat32,
                  .rank = 4U,
                  .strides = holoscan::TensorStridePolicy::kContiguousOnly,
              },
          .bounds = holoscan::tensor_bounds(bytes),
          .storage = holoscan::TensorOutputStorage::kRuntimePool,
      });
}

holoscan::Contract VideoPreprocessorOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{input});
  return result;
}

void VideoPreprocessorOp::start() {
  stop();
  const auto& p = params();
  const std::size_t bytes = static_cast<std::size_t>(p.width) * p.height * 3U;
  if (cudaError_t error = cudaMalloc(&device_rgb_, bytes); error != cudaSuccess) {
    throw holoscan::RuntimeError{holoscan::ErrorCode::kFailure, cudaGetErrorString(error)};
  }
}

void VideoPreprocessorOp::stop() {
  if (device_rgb_ != nullptr) {
    (void)cudaFree(device_rgb_);
    device_rgb_ = nullptr;
  }
}

holoscan::expected<void, holoscan::Error> VideoPreprocessorOp::compute(
    holoscan::ExecutionContext& context) {
  auto sample = input.receive();
  if (!sample) {
    return holoscan::make_unexpected(std::move(sample).error());
  }
  if (!sample->data.data) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "RGB frame has no tensor"});
  }
  holoscan::Tensor& tensor = *sample->data.data;
  auto synchronized = tensor.synchronize_for_host();
  if (!synchronized) {
    return holoscan::make_unexpected(std::move(synchronized).error());
  }
  const auto shape = tensor.shape_span();
  const auto& p = context.params<VideoPreprocessorParams>();
  if (shape.size() != 3U || shape[0] != static_cast<std::int64_t>(p.height) ||
      shape[1] != static_cast<std::int64_t>(p.width) || shape[2] != 3 || !tensor.is_contiguous()) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kInvalidArgument,
                        "VideoPreprocessorOp expects contiguous configured HWC3 input"});
  }
  const auto* src = tensor.data_as<std::uint8_t>();
  if (src == nullptr) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "invalid RGB input tensor"});
  }

  const std::array<std::int64_t, 4> output_shape{1, static_cast<std::int64_t>(p.height),
                                                 static_cast<std::int64_t>(p.width), 3};
  auto loan =
      output.allocate_tensor(holoscan::TensorLoanRequest{.shape = output_shape, .dtype = kFloat32});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  auto writer = loan->write(context.cuda_stream());
  if (!writer) {
    return holoscan::make_unexpected(std::move(writer).error());
  }

  constexpr std::size_t kThreads = 256U;
  const std::size_t elements = static_cast<std::size_t>(p.width) * p.height * 3U;
  if (cudaError_t error = cudaMemcpyAsync(device_rgb_, src, elements, cudaMemcpyHostToDevice,
                                          context.cuda_stream().get());
      error != cudaSuccess) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kFailure, cudaGetErrorString(error)});
  }
  const std::size_t blocks = (elements + kThreads - 1U) / kThreads;
  rgb_u8_to_float<<<static_cast<unsigned int>(blocks), kThreads, 0, context.cuda_stream().get()>>>(
      device_rgb_, writer->data_as<float>(), elements);
  if (cudaError_t error = cudaPeekAtLastError(); error != cudaSuccess) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kFailure, cudaGetErrorString(error)});
  }

  auto committed = std::move(*writer).commit();
  if (!committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }
  return output.emit(std::move(*loan),
                     holoscan::EmitOptions{.capture_time = context.activation_time()});
}

}  // namespace endoscopy_tool_tracking
