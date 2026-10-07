/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/logger/logger.hpp>
#include <string>
#include <utility>
#include <vector>

#include "tool_tracking_postprocessor_v5.hpp"
#include "video_stream_replayer.hpp"

namespace endoscopy_tool_tracking {
namespace {

constexpr std::size_t kClasses = 7U;
constexpr std::size_t kMaskHeight = 60U;
constexpr std::size_t kMaskWidth = 107U;
constexpr std::size_t kMaskPixels = kMaskHeight * kMaskWidth;
constexpr float kCrossSize = 0.05F;

struct Rgb {
  float r;
  float g;
  float b;
};

__device__ constexpr Rgb kColors[kClasses]{
    {0.12F, 0.47F, 0.71F}, {0.20F, 0.63F, 0.17F}, {0.89F, 0.10F, 0.11F}, {1.00F, 0.50F, 0.00F},
    {0.42F, 0.24F, 0.60F}, {0.69F, 0.35F, 0.16F}, {0.65F, 0.81F, 0.89F},
};

__global__ void compose_masks(const float* probabilities, const float* binary_masks,
                              float threshold, uchar4* output) {
  const std::size_t pixel = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pixel >= kMaskPixels) {
    return;
  }

  float premult_r = 0.0F;
  float premult_g = 0.0F;
  float premult_b = 0.0F;
  float alpha = 0.0F;
  for (std::size_t tool = 0; tool < kClasses; ++tool) {
    if (!(probabilities[tool] > threshold)) {
      continue;
    }
    float value = binary_masks[(tool * kMaskPixels) + pixel];
    value = fminf(fmaxf(value, 0.3F), 0.99F);
    const float layer_alpha = ((value - 0.3F) / 0.69F) * 0.7F;
    premult_r = (kColors[tool].r * layer_alpha) + premult_r * (1.0F - layer_alpha);
    premult_g = (kColors[tool].g * layer_alpha) + premult_g * (1.0F - layer_alpha);
    premult_b = (kColors[tool].b * layer_alpha) + premult_b * (1.0F - layer_alpha);
    alpha = layer_alpha + alpha * (1.0F - layer_alpha);
  }
  const float inv_alpha = alpha > 0.0F ? 1.0F / alpha : 0.0F;
  output[pixel] =
      make_uchar4(static_cast<unsigned char>(255.0F * fminf(premult_r * inv_alpha, 1.0F)),
                  static_cast<unsigned char>(255.0F * fminf(premult_g * inv_alpha, 1.0F)),
                  static_cast<unsigned char>(255.0F * fminf(premult_b * inv_alpha, 1.0F)),
                  static_cast<unsigned char>(255.0F * fminf(alpha, 1.0F)));
}

}  // namespace

void ToolTrackingPostprocessorOp::setup(holoscan::OperatorSpec& spec) {
  const auto device_float = [](std::uint32_t rank) {
    return holoscan::TensorRepresentation{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = kFloat32,
        .rank = rank,
        .strides = holoscan::TensorStridePolicy::kContiguousOnly,
    };
  };
  spec.input(probs, "probs")
      .queue_depth(2U)
      .expects_tensor(holoscan::TensorInputSpec{.representation = device_float(2U)});
  spec.input(scaled_coords, "scaled_coords")
      .queue_depth(2U)
      .expects_tensor(holoscan::TensorInputSpec{.representation = device_float(3U)});
  spec.input(binary_masks, "binary_masks")
      .queue_depth(2U)
      .expects_tensor(holoscan::TensorInputSpec{.representation = device_float(4U)});

  spec.output(mask, "mask")
      .produces_tensor(holoscan::TensorOutputSpec{
          .representation =
              holoscan::TensorRepresentation{
                  .memory_kind = holoscan::MemoryKind::kCudaDevice,
                  .dtype = kRgbU8,
                  .rank = 3U,
                  .strides = holoscan::TensorStridePolicy::kContiguousOnly,
              },
          .bounds = holoscan::tensor_bounds(kMaskPixels * 4U),
          .storage = holoscan::TensorOutputStorage::kRuntimePool,
      })
      .payload_options(holoscan::PayloadOptions{
          .schema = holoscan::schema_identity<holoscan::schema::ImageT>(),
      });
  spec.output(crosses, "crosses")
      .produces_tensor(holoscan::TensorOutputSpec{
          .representation =
              holoscan::TensorRepresentation{
                  .memory_kind = holoscan::MemoryKind::kHost,
                  .dtype = kFloat32,
                  .rank = 1U,
                  .strides = holoscan::TensorStridePolicy::kContiguousOnly,
              },
          .bounds = holoscan::tensor_bounds(kClasses * 3U * sizeof(float)),
          .storage = holoscan::TensorOutputStorage::kRuntimePool,
      });
}

holoscan::Contract ToolTrackingPostprocessorOp::contract() const {
  holoscan::Contract result;
  std::vector<holoscan::PortRef> ports{probs, scaled_coords, binary_masks};
  result.trigger(holoscan::OnAligned{
      .ports = holoscan::PortList{std::move(ports)},
      .window = std::chrono::nanoseconds{0},
  });
  result.input(probs, holoscan::Exact{});
  result.input(scaled_coords, holoscan::Exact{});
  result.input(binary_masks, holoscan::Exact{});
  return result;
}

holoscan::expected<void, holoscan::Error> ToolTrackingPostprocessorOp::compute(
    holoscan::ExecutionContext& context) try {
  auto fail = [](holoscan::ErrorCode code,
                 std::string message) -> holoscan::expected<void, holoscan::Error> {
    HOLOSCAN_LOG_ERROR("ToolTrackingPostprocessorOp: {}", message);
    return holoscan::make_unexpected(holoscan::Error{code, std::move(message)});
  };
  auto probs_sample = probs.receive_data();
  auto coords_sample = scaled_coords.receive_data();
  auto masks_sample = binary_masks.receive_data();
  if (!probs_sample || !coords_sample || !masks_sample) {
    return fail(holoscan::ErrorCode::kReceiveError, "did not receive an aligned tensor set");
  }
  holoscan::Tensor& probs_tensor = *probs_sample;
  holoscan::Tensor& coords_tensor = *coords_sample;
  holoscan::Tensor& masks_tensor = *masks_sample;
  constexpr std::array<std::int64_t, 2> expected_probs{1, 7};
  constexpr std::array<std::int64_t, 3> expected_coords{1, 7, 2};
  constexpr std::array<std::int64_t, 4> expected_masks{1, 7, 60, 107};
  if (!std::ranges::equal(probs_tensor.shape_span(), expected_probs) ||
      !std::ranges::equal(coords_tensor.shape_span(), expected_coords) ||
      !std::ranges::equal(masks_tensor.shape_span(), expected_masks) ||
      !probs_tensor.is_contiguous() || !coords_tensor.is_contiguous() ||
      !masks_tensor.is_contiguous() || probs_tensor.nbytes() != kClasses * sizeof(float) ||
      coords_tensor.nbytes() != kClasses * 2U * sizeof(float) ||
      masks_tensor.nbytes() != kClasses * kMaskPixels * sizeof(float)) {
    return fail(
        holoscan::ErrorCode::kInvalidArgument,
        "unexpected model output byte sizes: probs=" + std::to_string(probs_tensor.nbytes()) +
            " coords=" + std::to_string(coords_tensor.nbytes()) +
            " masks=" + std::to_string(masks_tensor.nbytes()));
  }

  std::array<float, kClasses> host_probs{};
  std::array<float, kClasses * 2U> host_coords{};
  const auto stream = context.cuda_stream().get();
  auto probs_copy = probs_tensor.copy_to_host(host_probs.data(), sizeof(host_probs), stream);
  auto coords_copy = coords_tensor.copy_to_host(host_coords.data(), sizeof(host_coords), stream);
  if (!probs_copy || !coords_copy) {
    return fail(holoscan::ErrorCode::kFailure,
                "failed to copy probabilities/coordinates to host: probs=" +
                    (probs_copy ? std::string{"ok"} : probs_copy.error().message) +
                    " coords=" + (coords_copy ? std::string{"ok"} : coords_copy.error().message));
  }
  HOLOSCAN_LOG_DEBUG("tool probabilities: [{:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}]",
                     host_probs[0], host_probs[1], host_probs[2], host_probs[3], host_probs[4],
                     host_probs[5], host_probs[6]);

  const std::array<std::int64_t, 3> mask_shape{60, 107, 4};
  const std::array<std::int64_t, 1> crosses_shape{21};
  auto mask_loan =
      mask.allocate_tensor(holoscan::TensorLoanRequest{.shape = mask_shape, .dtype = kRgbU8});
  auto crosses_loan = crosses.allocate_tensor(
      holoscan::TensorLoanRequest{.shape = crosses_shape, .dtype = kFloat32});
  if (!mask_loan || !crosses_loan) {
    return fail(holoscan::ErrorCode::kResourceExhausted,
                "failed to allocate mask/cross output tensors");
  }
  auto mask_writer = mask_loan->write(context.cuda_stream());
  if (!mask_writer) {
    return holoscan::make_unexpected(std::move(mask_writer).error());
  }

  constexpr std::size_t kThreads = 256U;
  constexpr std::size_t kBlocks = (kMaskPixels + kThreads - 1U) / kThreads;
  compose_masks<<<kBlocks, kThreads, 0, stream>>>(
      probs_tensor.data_as<float>(), masks_tensor.data_as<float>(),
      context.params<ToolTrackingPostprocessorParams>().min_probability,
      mask_writer->data_as<uchar4>());
  if (cudaError_t error = cudaPeekAtLastError(); error != cudaSuccess) {
    return fail(holoscan::ErrorCode::kFailure,
                std::string("mask kernel launch failed: ") + cudaGetErrorString(error));
  }
  auto mask_committed = std::move(*mask_writer).commit();
  if (!mask_committed) {
    return holoscan::make_unexpected(std::move(mask_committed).error());
  }

  const float threshold = context.params<ToolTrackingPostprocessorParams>().min_probability;
  auto fill_crosses = [&](const holoscan::TensorOutputWriteGuard& writer)
      -> holoscan::expected<void, holoscan::Error> {
    float* output = writer.data_as<float>();
    if (output == nullptr || writer.byte_size() < kClasses * 3U * sizeof(float)) {
      return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kInvalidArgument});
    }
    for (std::size_t tool = 0; tool < kClasses; ++tool) {
      output[(tool * 3U) + 0U] =
          host_probs[tool] > threshold ? host_coords[(tool * 2U) + 0U] : -1.0F;
      output[(tool * 3U) + 1U] =
          host_probs[tool] > threshold ? host_coords[(tool * 2U) + 1U] : -1.0F;
      output[(tool * 3U) + 2U] = kCrossSize;
    }
    return {};
  };
  auto crosses_committed = crosses_loan->write_host_and_commit(fill_crosses);
  if (!crosses_committed) {
    return holoscan::make_unexpected(std::move(crosses_committed).error());
  }

  const holoscan::EmitOptions emit{.capture_time = context.activation_time()};
  holoscan::schema::ImageT image;
  image.width = 107;
  image.height = 60;
  image.encoding = holoscan::schema::ImageEncoding_RGBA8;
  if (auto result = mask.emit_tensor(std::move(*mask_loan), image, emit); !result) {
    return result;
  }
  return crosses.emit(std::move(*crosses_loan), emit);
} catch (const std::exception& exception) {
  HOLOSCAN_LOG_ERROR("ToolTrackingPostprocessorOp exception: {}", exception.what());
  return holoscan::make_unexpected(
      holoscan::Error{holoscan::ErrorCode::kFailure, exception.what()});
}

}  // namespace endoscopy_tool_tracking
