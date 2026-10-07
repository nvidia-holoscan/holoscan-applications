/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace endoscopy_tool_tracking {

// Holoscan 5.x replacement for the 4.x GXF VideoStreamReplayerOp.
// Emits HWC uint8 frames from deterministic synthetic content or one contiguous
// `{basename}.rgb` stream. The stream is opened once and read
// sequentially with bounded memory, avoiding tens of thousands of opens/inodes
// during long surgical-video runs.
struct VideoStreamReplayerParams {
  std::string directory{"."};
  std::string basename{"surgical_video"};
  std::uint32_t width{854U};
  std::uint32_t height{480U};
  std::uint32_t channels{3U};
  std::uint32_t count{0U};       // finite playback; 0 means all contiguous-file frames once
  std::uint32_t period_ms{33U};  // Approximate sample-video rate (30 FPS).
  bool synthetic{false};         // true: ignore on-disk frames and synthesize
  bool repeat{false};            // loop the contiguous stream until count is reached
};

HOLOSCAN_PARAM_SCHEMA(
    VideoStreamReplayerParams,
    holoscan::param_schema("com.nvidia.applications.endoscopy.VideoStreamReplayerParams", 1),
    holoscan::field<&VideoStreamReplayerParams::directory>("directory")
        .description("Directory containing raw RGB frames or used as data root."),
    holoscan::field<&VideoStreamReplayerParams::basename>("basename")
        .description("Contiguous RGB filename stem."),
    holoscan::field<&VideoStreamReplayerParams::width>("width")
        .description("Frame width in pixels.")
        .range(1U, 854U),
    holoscan::field<&VideoStreamReplayerParams::height>("height")
        .description("Frame height in pixels.")
        .range(1U, 480U),
    holoscan::field<&VideoStreamReplayerParams::channels>("channels")
        .description("Channel count (RGB=3).")
        .allowed_values({3U}),
    holoscan::field<&VideoStreamReplayerParams::count>("count").description(
        "Frames to emit; zero means all frames in a contiguous file once."),
    holoscan::field<&VideoStreamReplayerParams::period_ms>("period_ms")
        .description("Source clock period in milliseconds.")
        .range(1U, 1000U),
    holoscan::field<&VideoStreamReplayerParams::synthetic>("synthetic")
        .description("When true, synthesize deterministic frames instead of reading files."),
    holoscan::field<&VideoStreamReplayerParams::repeat>("repeat").description(
        "Loop a contiguous input stream until count frames have been emitted."));

static_assert(holoscan::param_schema_availability<VideoStreamReplayerParams>() ==
                  holoscan::ParamSchemaAvailability::kDeclared,
              "VideoStreamReplayerParams schema must be declared in this namespace");

inline constexpr DLDataType kRgbU8{kDLUInt, 8U, 1U};
inline constexpr holoscan::TensorRepresentation kRgbU8Frame{
    .memory_kind = holoscan::MemoryKind::kHost,
    .dtype = kRgbU8,
    .rank = 3U,
    .strides = holoscan::TensorStridePolicy::kContiguousOnly,
};

class VideoStreamReplayerOp final : public holoscan::Operator<VideoStreamReplayerParams> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    // A parameter update cannot change a sealed port's allocation contract.
    // Bound the pool by the declared dimension limits; loans use the selected shape.
    constexpr std::uint64_t bytes = 854U * 480U * 3U;
    spec.output(output, "output")
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation = kRgbU8Frame,
            .bounds = holoscan::tensor_bounds(bytes),
            .storage = holoscan::TensorOutputStorage::kRuntimePool,
        })
        .payload_options(holoscan::PayloadOptions{
            .schema = holoscan::schema_identity<holoscan::schema::ImageT>(),
        });
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    const auto& p = params();
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{
        .period = std::chrono::milliseconds{static_cast<std::int64_t>(p.period_ms)}});
    return result;
  }

  void start() override {
    const auto& p = params();
    emitted_ = 0U;
    contiguous_frames_ = 0U;
    effective_count_ = p.count;
    contiguous_.close();

    if (p.synthetic) {
      if (effective_count_ == 0U) {
        throw holoscan::RuntimeError{holoscan::ErrorCode::kInvalidArgument,
                                     "synthetic playback requires --count > 0"};
      }
      return;
    }

    const std::uint64_t frame_bytes = static_cast<std::uint64_t>(p.width) * p.height * p.channels;
    const std::filesystem::path path = std::filesystem::path(p.directory) / (p.basename + ".rgb");
    std::error_code ec;
    const std::uint64_t file_bytes = std::filesystem::file_size(path, ec);
    if (!ec) {
      if (frame_bytes == 0U || file_bytes == 0U || file_bytes % frame_bytes != 0U) {
        throw holoscan::RuntimeError{
            holoscan::ErrorCode::kInvalidArgument,
            "contiguous RGB stream size is not a positive whole number of frames"};
      }
      contiguous_frames_ = file_bytes / frame_bytes;
      if (contiguous_frames_ > std::numeric_limits<std::uint32_t>::max()) {
        throw holoscan::RuntimeError{holoscan::ErrorCode::kInvalidArgument,
                                     "contiguous RGB stream contains too many frames"};
      }
      if (effective_count_ == 0U) {
        effective_count_ = static_cast<std::uint32_t>(contiguous_frames_);
      } else if (!p.repeat && effective_count_ > contiguous_frames_) {
        throw holoscan::RuntimeError{holoscan::ErrorCode::kInvalidArgument,
                                     "--count exceeds contiguous stream length without --repeat"};
      }
      contiguous_.open(path, std::ios::binary);
      if (!contiguous_) {
        throw holoscan::RuntimeError{holoscan::ErrorCode::kNotFound,
                                     "failed to open contiguous RGB stream"};
      }
    } else {
      throw holoscan::RuntimeError{holoscan::ErrorCode::kInvalidArgument,
                                   "cannot read contiguous RGB stream: " + path.string()};
    }
  }

  void stop() override { contiguous_.close(); }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    const auto& params = context.params<VideoStreamReplayerParams>();
    if (emitted_ >= effective_count_) {
      return {};
    }

    const std::array<std::int64_t, 3> shape{
        static_cast<std::int64_t>(params.height),
        static_cast<std::int64_t>(params.width),
        static_cast<std::int64_t>(params.channels),
    };
    const std::size_t nbytes = static_cast<std::size_t>(shape[0] * shape[1] * shape[2]);

    auto loan =
        output.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = kRgbU8});
    if (!loan) {
      return holoscan::make_unexpected(std::move(loan).error());
    }

    const std::uint32_t frame_id = emitted_;
    auto fill = [&](const holoscan::TensorOutputWriteGuard& writer)
        -> holoscan::expected<void, holoscan::Error> {
      auto* data = writer.data_as<std::uint8_t>();
      if (data == nullptr || writer.byte_size() != nbytes) {
        return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kInvalidArgument});
      }
      if (params.synthetic) {
        fill_synthetic(data, nbytes, frame_id, params);
      } else {
        HOLOSCAN_TRY(read_raw_frame(data, nbytes, frame_id, params));
      }
      return {};
    };

    auto committed = loan->write_host_and_commit(fill);
    if (!committed) {
      return holoscan::make_unexpected(std::move(committed).error());
    }

    ++emitted_;
    holoscan::schema::ImageT image;
    image.width = static_cast<std::int32_t>(params.width);
    image.height = static_cast<std::int32_t>(params.height);
    image.encoding = holoscan::schema::ImageEncoding_RGB8;
    return output.emit_tensor(std::move(*loan), image,
                              holoscan::EmitOptions{.capture_time = context.activation_time()});
  }

  holoscan::Output<holoscan::schema::ImageT> output;

 private:
  static void fill_synthetic(std::uint8_t* data, std::size_t nbytes, std::uint32_t frame_id,
                             const VideoStreamReplayerParams& params) {
    // Deterministic pattern: encode frame_id in the first 4 bytes, fill the rest
    // with a spatially varying RGB ramp so recordings are visually distinct.
    for (std::size_t i = 0; i < nbytes; ++i) {
      data[i] = static_cast<std::uint8_t>((i + frame_id * 17U) & 0xFFU);
    }
    if (nbytes >= 4U) {
      data[0] = static_cast<std::uint8_t>(frame_id & 0xFFU);
      data[1] = static_cast<std::uint8_t>((frame_id >> 8) & 0xFFU);
      data[2] = static_cast<std::uint8_t>((frame_id >> 16) & 0xFFU);
      data[3] = static_cast<std::uint8_t>((frame_id >> 24) & 0xFFU);
    }
    (void)params;
  }

  holoscan::expected<void, holoscan::Error> read_raw_frame(
      std::uint8_t* data, std::size_t nbytes, std::uint32_t frame_id,
      const VideoStreamReplayerParams& params) {
    if (contiguous_.is_open()) {
      const std::uint64_t source_frame = frame_id % contiguous_frames_;
      if (source_frame == 0U && frame_id != 0U) {
        contiguous_.clear();
        contiguous_.seekg(0, std::ios::beg);
      }
      contiguous_.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(nbytes));
      if (static_cast<std::size_t>(contiguous_.gcount()) != nbytes) {
        return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kInvalidArgument});
      }
      return {};
    }

    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kNotFound, "RGB stream is not open"});
  }

  std::uint32_t emitted_{};
  std::uint32_t effective_count_{};
  std::uint64_t contiguous_frames_{};
  std::ifstream contiguous_;
};

}  // namespace endoscopy_tool_tracking
