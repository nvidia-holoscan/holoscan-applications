// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <utility>

#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_contract.hpp>

#include "run_progress.hpp"
#include "video_stream_replayer.hpp"

namespace endoscopy_tool_tracking {

struct VideoStreamRecorderParams {
  std::string directory;  // Empty means consume without recording.
  std::string basename{"input"};
};

HOLOSCAN_PARAM_SCHEMA(VideoStreamRecorderParams,
                      holoscan::param_schema("com.nvidia.applications.endoscopy.RecorderParams", 1),
                      holoscan::field<&VideoStreamRecorderParams::directory>("directory"),
                      holoscan::field<&VideoStreamRecorderParams::basename>("basename"));

// Terminal consumer for replay, optionally recording the unmodified RGB input.
class VideoStreamRecorderOp final : public holoscan::Operator<VideoStreamRecorderParams> {
 public:
  explicit VideoStreamRecorderOp(std::shared_ptr<RunProgress> progress)
      : progress_(std::move(progress)) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(2U)
        .expects_tensor(holoscan::TensorInputSpec{.representation = kRgbU8Frame});
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }

  void start() override {
    const auto& p = params();
    if (p.directory.empty()) {
      return;
    }
    std::filesystem::create_directories(p.directory);
    const auto path = std::filesystem::path(p.directory) / (p.basename + ".rgb");
    // Recordings are explicit new outputs. Refuse to destroy an existing recording.
    if (std::filesystem::exists(path)) {
      throw holoscan::RuntimeError{holoscan::ErrorCode::kInvalidArgument,
                                   "recording already exists: " + path.string()};
    }
    stream_.open(path, std::ios::binary | std::ios::trunc);
    if (!stream_) {
      throw holoscan::RuntimeError{holoscan::ErrorCode::kFailure,
                                   "cannot open recording: " + path.string()};
    }
  }

  void stop() override {
    if (stream_.is_open()) {
      stream_.flush();
      const bool flushed = static_cast<bool>(stream_);
      stream_.close();
      if (!flushed || stream_.fail()) {
        throw holoscan::RuntimeError{holoscan::ErrorCode::kFailure,
                                     "failed to flush or close the RGB recording"};
      }
    }
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto frame = input.receive_data();
    if (!frame) {
      return holoscan::make_unexpected(std::move(frame).error());
    }
    if (!frame->data) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "RGB frame has no tensor"});
    }
    auto ready = frame->data->synchronize_for_host();
    if (!ready) {
      return holoscan::make_unexpected(std::move(ready).error());
    }
    if (stream_.is_open()) {
      stream_.write(reinterpret_cast<const char*>(frame->data->data_as<std::uint8_t>()),
                    static_cast<std::streamsize>(frame->data->nbytes()));
      if (!stream_) {
        return holoscan::make_unexpected(
            holoscan::Error{holoscan::ErrorCode::kFailure, "failed to write RGB recording"});
      }
    }
    progress_->consumed.fetch_add(1U);
    progress_->finish_if_ready(context);
    return {};
  }

  holoscan::Input<holoscan::schema::ImageT> input;

 private:
  std::shared_ptr<RunProgress> progress_;
  std::ofstream stream_;
};

}  // namespace endoscopy_tool_tracking
