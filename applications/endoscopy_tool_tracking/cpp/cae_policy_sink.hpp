// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include "cae_policy.hpp"
#include "run_progress.hpp"

namespace endoscopy_tool_tracking {

// Terminal consumer of policy maps in --cae policy mode.
//
// For every map it checks geometry and frame identity, copies the 6,420 policy bytes to the host,
// rejects values outside 0..4 (including the 255 invalid-input marker), counts the frame toward
// run completion and optionally writes a diagnostic dump. The host copy happens here, at the end
// of the graph, and never in the policy operator.
//
// Dump layout (only when a dump directory is configured):
//   manifest.json      run-level description: geometry, legend, configuration used
//   frames.jsonl       one JSON line per frame: frame_id, capture time, file, per-policy counts
//   policy_NNNNNN.pgm  107x60 8-bit PGM holding the raw policy values of one frame
class CaePolicySinkOp final : public holoscan::Operator<> {
 public:
  CaePolicySinkOp(std::shared_ptr<RunProgress> progress, CaePolicyConfig config,
                  std::string dump_directory, std::uint32_t source_width,
                  std::uint32_t source_height)
      : progress_(std::move(progress)),
        config_(config),
        dump_directory_(std::move(dump_directory)),
        source_width_(source_width),
        source_height_(source_height) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(policy, "policy").queue_depth(2U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{policy});
    return result;
  }

  void start() override {
    next_frame_ = 1U;
    if (dump_directory_.empty()) return;
    // Create the final directory exclusively, so a concurrent run that passed the same startup
    // check cannot share it and overwrite these files.
    std::filesystem::path directory(dump_directory_);
    if (!directory.has_filename()) directory = directory.parent_path();  // trailing separator
    if (directory.has_parent_path()) std::filesystem::create_directories(directory.parent_path());
    if (!std::filesystem::create_directory(directory)) {
      throw holoscan::RuntimeError{holoscan::ErrorCode::kFailure,
                                   "policy dump directory already exists: " + dump_directory_};
    }
    write_manifest();
    frames_.open(std::filesystem::path(dump_directory_) / "frames.jsonl", std::ios::trunc);
    if (!frames_) {
      throw holoscan::RuntimeError{holoscan::ErrorCode::kFailure,
                                   "cannot create " + dump_directory_ + "/frames.jsonl"};
    }
  }

  void stop() override {
    if (frames_.is_open()) {
      frames_.close();
      if (frames_.fail()) {
        throw holoscan::RuntimeError{holoscan::ErrorCode::kFailure,
                                     "failed to close " + dump_directory_ + "/frames.jsonl"};
      }
    }
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto fail = [](std::string message) -> holoscan::expected<void, holoscan::Error> {
      HOLOSCAN_LOG_ERROR("CaePolicySinkOp: {}", message);
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kInvalidArgument, std::move(message)});
    };
    auto view = policy.selected_sample();
    if (!view) return holoscan::make_unexpected(std::move(view).error());
    const auto& sample = view->sample();
    const holoscan::schema::ImageT& image = sample.data;
    const std::uint64_t frame = sample.metadata.frame_id;
    if (frame != next_frame_) {
      return fail("policy frame_id " + std::to_string(frame) + " does not match expected frame " +
                  std::to_string(next_frame_));
    }
    if (image.width != static_cast<std::int32_t>(kPolicyWidth) ||
        image.height != static_cast<std::int32_t>(kPolicyHeight) ||
        image.encoding != holoscan::schema::ImageEncoding_MONO8 || !image.data ||
        image.data->nbytes() != kPolicyCells || !image.header) {
      return fail("malformed policy map at frame " + std::to_string(frame));
    }

    std::array<std::uint8_t, kPolicyCells> host{};
    auto copied = image.data->copy_to_host(host.data(), host.size(), context.cuda_stream().get());
    if (!copied) {
      return fail("failed to read policy map at frame " + std::to_string(frame) + ": " +
                  copied.error().message);
    }
    std::array<std::size_t, kMaxPolicyValue + 1U> counts{};
    for (const std::uint8_t value : host) {
      if (value > kMaxPolicyValue) {
        return fail("invalid inference output at frame " + std::to_string(frame) +
                    " (policy value " + std::to_string(value) + ")");
      }
      ++counts[value];
    }
    if (!dump_directory_.empty()) {
      if (auto dumped = dump(frame, image.header->capture_timestamp_ns, host, counts); !dumped) {
        return dumped;
      }
    }
    ++next_frame_;
    progress_->policies.fetch_add(1U);
    progress_->finish_if_ready(context);
    return {};
  }

  holoscan::Input<holoscan::schema::ImageT> policy;

 private:
  void write_manifest() const {
    std::ofstream out(std::filesystem::path(dump_directory_) / "manifest.json", std::ios::trunc);
    // Enough digits that the thresholds read back as the exact floats the kernel used.
    out << std::setprecision(std::numeric_limits<float>::max_digits10);
    out << "{\n  \"policy_schema_version\": " << kPolicySchemaVersion << ",\n"
        << "  \"map\": {\"width\": " << kPolicyWidth << ", \"height\": " << kPolicyHeight
        << ", \"dtype\": \"uint8\", \"layout\": \"row-major\", \"file_format\": \"pgm\"},\n"
        << "  \"source\": {\"width\": " << source_width_ << ", \"height\": " << source_height_
        << "},\n"
        << "  \"cell_to_source_pixels\": \"column c covers x in [floor(c*W/" << kPolicyWidth
        << "), floor((c+1)*W/" << kPolicyWidth << ")); row r covers y in [floor(r*H/"
        << kPolicyHeight << "), floor((r+1)*H/" << kPolicyHeight << "))\",\n"
        << "  \"frame_id\": \"1-based processed frame number (publication sequence of the LSTM "
           "outputs); equals the source frame number unless the video loops\",\n"
        << "  \"policy_legend\": {";
    for (std::size_t i = 0; i <= kMaxPolicyValue; ++i) {
      out << (i ? ", " : "") << '"' << i << "\": \"" << kPolicyNames[i] << '"';
    }
    out << "},\n  \"invalid_value\": " << static_cast<int>(kInvalidPolicyValue) << ",\n"
        << "  \"config\": {\"class_threshold\": " << config_.class_threshold
        << ", \"mask_threshold\": " << config_.mask_threshold << ", \"background_policy\": \""
        << policy_name(config_.background_policy) << "\", \"classes\": {";
    for (std::size_t i = 0; i < kToolClasses; ++i) {
      out << (i ? ", " : "") << '"' << kToolClassNames[i] << "\": \""
          << policy_name(config_.class_policies[i]) << '"';
    }
    out << "}}\n}\n";
    out.close();  // flush now so a write error is reported below
    if (!out) {
      throw holoscan::RuntimeError{holoscan::ErrorCode::kFailure,
                                   "cannot write " + dump_directory_ + "/manifest.json"};
    }
  }

  holoscan::expected<void, holoscan::Error> dump(
      std::uint64_t frame, std::int64_t capture_ns,
      const std::array<std::uint8_t, kPolicyCells>& values,
      const std::array<std::size_t, kMaxPolicyValue + 1U>& counts) {
    std::array<char, 32> name{};
    std::snprintf(name.data(), name.size(), "policy_%06llu.pgm",
                  static_cast<unsigned long long>(frame));  // NOLINT(runtime/int)
    std::ofstream pgm(std::filesystem::path(dump_directory_) / name.data(),
                      std::ios::binary | std::ios::trunc);
    pgm << "P5\n" << kPolicyWidth << ' ' << kPolicyHeight << "\n255\n";
    pgm.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size()));
    pgm.close();  // closing flushes; a write error surfaces here, not in the destructor
    frames_ << "{\"frame_id\": " << frame << ", \"capture_timestamp_ns\": " << capture_ns
            << ", \"file\": \"" << name.data() << "\", \"counts\": {";
    for (std::size_t i = 0; i <= kMaxPolicyValue; ++i) {
      frames_ << (i ? ", " : "") << '"' << kPolicyNames[i] << "\": " << counts[i];
    }
    frames_ << "}}\n";
    frames_.flush();
    if (!pgm || !frames_) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure,
                          "failed to write policy dump for frame " + std::to_string(frame)});
    }
    return {};
  }

  std::shared_ptr<RunProgress> progress_;
  CaePolicyConfig config_;
  std::string dump_directory_;
  std::uint32_t source_width_;
  std::uint32_t source_height_;
  std::uint64_t next_frame_{1U};
  std::ofstream frames_;
};

}  // namespace endoscopy_tool_tracking
