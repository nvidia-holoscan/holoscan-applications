// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "image_sources.hpp"

#include <cuda_runtime_api.h>

#include <array>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

#include "detector/synthetic_scene.hpp"
#include "detector/tag_family.hpp"

namespace tag_detection {
namespace {

constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};

holoscan::TensorOutputSpec image_tensor(holoscan::MemoryKind memory, std::uint64_t bytes) {
  return holoscan::TensorOutputSpec{
      .representation =
          holoscan::TensorRepresentation{
              .memory_kind = memory,
              .dtype = kUInt8,
              .rank = 3U,
              .strides = holoscan::TensorStridePolicy::kContiguousOnly,
          },
      .bounds = holoscan::tensor_bounds(bytes),
      .storage = holoscan::TensorOutputStorage::kRuntimePool,
  };
}

holoscan::Error failure(std::string message) {
  return holoscan::Error{holoscan::ErrorCode::kFailure, std::move(message)};
}

// A consumer still busy with earlier frames makes the emit fail with
// kBackpressured before anything is delivered. The source then sends the same
// frame on its next tick instead of failing the run or skipping the frame.
holoscan::expected<void, holoscan::Error> retry_if_backpressured(
    holoscan::expected<void, holoscan::Error> emitted, std::uint32_t& sent) {
  if (emitted) {
    ++sent;
    return {};
  }
  if (emitted.error().code == holoscan::ErrorCode::kBackpressured) {
    return {};
  }
  return emitted;
}

// Copies `pixels` (rows packed, `channels` per pixel) into a new image and
// emits it, through host memory or, with `device`, CUDA device memory.
holoscan::expected<void, holoscan::Error> emit_pixels(
    holoscan::Output<holoscan::schema::ImageT>& output, holoscan::ExecutionContext& context,
    const std::vector<std::uint8_t>& pixels, int width, int height, int channels, bool device) {
  const std::array<std::int64_t, 3> shape{height, width, channels};
  auto loan = output.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = kUInt8});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  if (device) {
    auto writer = loan->write(context.cuda_stream());
    if (!writer) {
      return holoscan::make_unexpected(std::move(writer).error());
    }
    if (writer->byte_size() != pixels.size()) {
      return holoscan::make_unexpected(failure("image loan has an unexpected size"));
    }
    const cudaError_t copied =
        cudaMemcpyAsync(writer->data_as<std::uint8_t>(), pixels.data(), pixels.size(),
                        cudaMemcpyHostToDevice, context.cuda_stream().get());
    if (copied != cudaSuccess) {
      return holoscan::make_unexpected(failure(cudaGetErrorString(copied)));
    }
    if (auto committed = std::move(*writer).commit(); !committed) {
      return holoscan::make_unexpected(std::move(committed).error());
    }
  } else {
    auto committed = loan->write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& writer)
                                                     -> holoscan::expected<void, holoscan::Error> {
      if (writer.byte_size() != pixels.size()) {
        return holoscan::make_unexpected(failure("image loan has an unexpected size"));
      }
      std::memcpy(writer.data_as<std::uint8_t>(), pixels.data(), pixels.size());
      return {};
    });
    if (!committed) {
      return holoscan::make_unexpected(std::move(committed).error());
    }
  }
  holoscan::schema::ImageT descriptor;
  descriptor.width = width;
  descriptor.height = height;
  descriptor.encoding =
      channels == 1 ? holoscan::schema::ImageEncoding_MONO8 : holoscan::schema::ImageEncoding_RGB8;
  return output.emit_tensor(std::move(*loan), descriptor,
                            holoscan::EmitOptions{.capture_time = context.activation_time()});
}

}  // namespace

SyntheticTagSourceOp::SyntheticTagSourceOp(SyntheticSourceConfig config)
    : config_(std::move(config)) {
  if (!is_known_family(config_.family)) {
    throw std::invalid_argument("unknown AprilTag family '" + config_.family + "'");
  }
  if (config_.width < 320 || config_.height < 240 || config_.width > 4096 ||
      config_.height > 4096) {
    throw std::invalid_argument("synthetic images must be 320x240 to 4096x4096");
  }
  if (config_.count == 0 || config_.period <= std::chrono::nanoseconds::zero()) {
    throw std::invalid_argument("synthetic source needs a positive count and period");
  }
}

void SyntheticTagSourceOp::setup(holoscan::OperatorSpec& spec) {
  const std::uint64_t bytes =
      std::uint64_t{static_cast<std::uint32_t>(config_.width)} * config_.height * 3U;
  spec.output(image, "image")
      .produces_tensor(image_tensor(
          config_.device_output ? holoscan::MemoryKind::kCudaDevice : holoscan::MemoryKind::kHost,
          bytes))
      .payload_options(holoscan::PayloadOptions{
          .schema = holoscan::schema_identity<holoscan::schema::ImageT>(),
      });
}

holoscan::Contract SyntheticTagSourceOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnClock{.period = config_.period});
  return result;
}

void SyntheticTagSourceOp::start() { emitted_ = 0; }

holoscan::expected<void, holoscan::Error> SyntheticTagSourceOp::compute(
    holoscan::ExecutionContext& context) {
  if (emitted_ >= config_.count) {
    return {};  // Done: idle until the consumers request completion.
  }
  const SyntheticScene scene = make_scene(config_.family, emitted_, config_.width, config_.height);
  const std::vector<std::uint8_t> pixels = render_scene(scene, 3);
  return retry_if_backpressured(
      emit_pixels(image, context, pixels, config_.width, config_.height, 3, config_.device_output),
      emitted_);
}

ImageFileSourceOp::ImageFileSourceOp(ImageFileSourceConfig config) : config_(std::move(config)) {
  if (config_.files.empty()) {
    throw std::invalid_argument("image source needs at least one file");
  }
  if (config_.period <= std::chrono::nanoseconds::zero()) {
    throw std::invalid_argument("image source needs a positive period");
  }
  first_ = read_pnm(config_.files.front());
  const auto files = static_cast<std::uint32_t>(config_.files.size());
  total_ = config_.count == 0 ? files : config_.count;
  if (!config_.loop && total_ > files) {
    throw std::invalid_argument("--count exceeds the number of images; add --loop true");
  }
}

void ImageFileSourceOp::setup(holoscan::OperatorSpec& spec) {
  const std::uint64_t bytes =
      std::uint64_t{static_cast<std::uint32_t>(first_.width)} * first_.height * first_.channels;
  spec.output(image, "image")
      .produces_tensor(image_tensor(holoscan::MemoryKind::kHost, bytes))
      .payload_options(holoscan::PayloadOptions{
          .schema = holoscan::schema_identity<holoscan::schema::ImageT>(),
      });
}

holoscan::Contract ImageFileSourceOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnClock{.period = config_.period});
  return result;
}

void ImageFileSourceOp::start() { emitted_ = 0; }

holoscan::expected<void, holoscan::Error> ImageFileSourceOp::compute(
    holoscan::ExecutionContext& context) {
  if (emitted_ >= total_) {
    return {};
  }
  const auto& path = config_.files[emitted_ % config_.files.size()];
  PnmImage frame;
  try {
    frame = read_pnm(path);
  } catch (const std::exception& error) {
    return holoscan::make_unexpected(failure(error.what()));
  }
  if (frame.width != first_.width || frame.height != first_.height ||
      frame.channels != first_.channels) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kInvalidArgument,
        path.string() + ": every image must match the first one's size and channels"});
  }
  return retry_if_backpressured(
      emit_pixels(image, context, frame.pixels, frame.width, frame.height, frame.channels, false),
      emitted_);
}

}  // namespace tag_detection
