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
#include <string_view>
#include <system_error>
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
#include "cae_policy_view.hpp"
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
//   README.md                 guide for people: layout, color legend, how to read the raw values
//   manifest.json             run-level description: geometry, legend, palette, configuration used
//   frames.jsonl              one JSON line per frame: frame_id, capture time, files, counts
//   policy/policy_NNNNNN.pgm  107x60 plain-text PGM (P2) holding the policy values of one frame
//   view/view_NNNNNN.png      the same map scaled to the source size, colored by policy
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
    for (const char* subdirectory : {kPolicySubdirectory, kViewSubdirectory}) {
      std::error_code error;
      if (!std::filesystem::create_directory(directory / subdirectory, error)) {
        throw holoscan::RuntimeError{
            holoscan::ErrorCode::kFailure,
            "cannot create " + (directory / subdirectory).string() + ": " + error.message()};
      }
    }
    write_manifest();
    write_readme();
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
        << ", \"dtype\": \"uint8\", \"layout\": \"row-major\", \"file_format\": \"plain-pgm\", "
        << "\"maxval\": " << int{kMaxPolicyValue} << "},\n"
        << "  \"source\": {\"width\": " << source_width_ << ", \"height\": " << source_height_
        << "},\n"
        << "  \"files\": {\"policy\": \"" << kPolicySubdirectory
        << "/policy_NNNNNN.pgm\", \"view\": \"" << kViewSubdirectory << "/view_NNNNNN.png\"},\n"
        << "  \"view\": {\"width\": " << source_width_ << ", \"height\": " << source_height_
        << ", \"file_format\": \"png\", \"color_type\": \"indexed\", \"pixel\": \"policy value of "
           "the covering cell, as in cell_to_source_pixels\", \"palette\": {";
    for (std::size_t i = 0; i <= kMaxPolicyValue; ++i) {
      const Rgb8& color = kPolicyViewPalette[i];
      out << (i ? ", " : "") << '"' << kPolicyNames[i] << "\": [" << int{color.r} << ", "
          << int{color.g} << ", " << int{color.b} << ']';
    }
    out << "}},\n"
        << "  \"cell_to_source_pixels\": \"the map spans the whole frame: source pixel x lies in "
        << "column floor((2x+1)*" << kPolicyWidth << "/(2W)) and pixel y in row floor((2y+1)*"
        << kPolicyHeight << "/(2H)) (pixel centres); "
        << "cells are the model's " << kPolicyHeight << "x" << kPolicyWidth
        << " mask cells, about 8x8 pixels\",\n"
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

  // A guide for people opening the dump: layout, legend, and how to read the raw values.
  void write_readme() const {
    static constexpr std::array<std::string_view, kMaxPolicyValue + 1U> kMeanings{
        "No change", "Keep more quality", "Keep much more quality", "Spend fewer bits",
        "Spend many fewer bits"};
    const std::string cells = std::to_string(kPolicyCells);
    const std::string width = std::to_string(kPolicyWidth);
    std::ofstream out(std::filesystem::path(dump_directory_) / "README.md", std::ios::trunc);
    out << "# CAE policy dump\n\n"
        << "Written by `endoscopy_tool_tracking --cae policy --cae-policy-dump`. It holds one "
           "policy map per processed frame: which encoding policy applies to each cell of the "
           "frame.\n\n"
        << "## Layout\n\n"
        << "| Path | Contents |\n| --- | --- |\n"
        << "| `README.md` | This guide |\n"
        << "| `manifest.json` | Run description for scripts: map geometry, source size, "
           "cell-to-pixel rule, policy legend, view palette and the policy configuration used |\n"
        << "| `frames.jsonl` | One JSON object per line, one line per frame (see below) |\n"
        << "| `" << kPolicySubdirectory << "/policy_NNNNNN.pgm` | The policy map of frame NNNNNN, "
        << "as numbers you can read in a text editor |\n"
        << "| `" << kViewSubdirectory << "/view_NNNNNN.png` | The same map as a colored image |\n\n"
        << "`NNNNNN` is the frame_id with at least six digits: `" << kPolicySubdirectory
        << "/policy_000001.pgm` is frame 1. Sort by frame_id, not by file name, if a run has more "
           "than 999999 frames.\n\n"
        << "## Policy values\n\n"
        << "| Value | Policy | Meaning | View color | RGB |\n| --- | --- | --- | --- | --- |\n";
    for (std::size_t i = 0; i <= kMaxPolicyValue; ++i) {
      const Rgb8& color = kPolicyViewPalette[i];
      out << "| " << i << " | `" << kPolicyNames[i] << "` | " << kMeanings[i] << " | "
          << kPolicyViewColorNames[i] << " | " << int{color.r} << ", " << int{color.g} << ", "
          << int{color.b} << " |\n";
    }
    out << "\nEvery value in this dump is in 0-" << int{kMaxPolicyValue}
        << ": the application checks each map and stops before dumping one with any other "
           "value.\n\n"
        << "## Map geometry\n\n"
        << "A map has " << kPolicyHeight << " rows of " << kPolicyWidth << " cells (" << cells
        << " cells) and covers the whole " << source_width_ << "x" << source_height_
        << " source frame. Source pixel (x, y) belongs to the cell that contains the pixel's "
           "centre: column `floor((2x+1)*"
        << kPolicyWidth << "/(2*" << source_width_ << "))` and row `floor((2y+1)*" << kPolicyHeight
        << "/(2*" << source_height_
        << "))`. The views use the same rule. The cells are the model's mask cells, about 8x8 "
           "pixels.\n\n"
        << "## Raw maps: `" << kPolicySubdirectory << "/*.pgm`\n\n"
        << "Each file is a plain PGM (`P2`), the text form of a standard grayscale image format. "
           "Open it in any text editor to read the policy values:\n\n"
        << "```text\n"
        << "P2          <- format: plain PGM\n"
        << kPolicyWidth << " " << kPolicyHeight
        << "      <- width (cells per row) and height (rows)\n"
        << int{kMaxPolicyValue} << "           <- largest possible value\n"
        << "3 3 3 ... 3 <- map row 0: " << kPolicyWidth << " values, left to right\n"
        << "...         <- " << kPolicyHeight << " rows in total, top to bottom\n"
        << "```\n\n"
        << "Each number is the policy value of one cell. Image viewers open the file too and show "
           "it in gray, from black (0) to white ("
        << int{kMaxPolicyValue}
        << "); the matching view shows it in color. Image programs such as Pillow scale the "
           "values to 0-255 when they load the file (Pillow gives 0, 64, 128, 191 and 255), so "
           "read the policy numbers from the text, not from an image program. Each row is one "
           "line, longer than the 70 characters the PGM format recommends for text files; "
           "readers such as Pillow accept that.\n\n"
        << "To read the values in a script run from this directory:\n\n"
        << "```python\nrow, col = 0, 0  # the cell to read\n"
        << "with open(\"" << kPolicySubdirectory << "/policy_000001.pgm\") as f:\n"
        << "    numbers = f.read().split()\n"
        << "values = [int(n) for n in numbers[4:]]  # skip P2, width, height and maxval\n"
        << "policy = values[row * " << width << " + col]  # cell (row, col)\n```\n\n"
        << "## Views: `" << kViewSubdirectory << "/*.png`\n\n"
        << "Each view is an indexed-color PNG at the source size, " << source_width_ << "x"
        << source_height_
        << ". Every pixel stores the policy value of the cell that covers it, "
           "and the palette gives it the color in the table above. Any image viewer shows the "
           "colors; a script reading the pixel indexes gets the policy values back, for example "
           "`PIL.Image.open(path).getpixel((x, y))`.\n\n"
        << "## frames.jsonl\n\n"
        << "| Field | Meaning |\n| --- | --- |\n"
        << "| `frame_id` | 1-based processed frame number; equals the source frame number unless "
           "the video loops (`--repeat true`) |\n"
        << "| `capture_timestamp_ns` | Capture time of the source frame, in nanoseconds |\n"
        << "| `file` | The raw map, relative to this directory |\n"
        << "| `view` | The view, relative to this directory |\n"
        << "| `counts` | Number of cells with each policy; they add up to " << cells << " |\n";
    out.close();  // flush now so a write error is reported below
    if (!out) {
      throw holoscan::RuntimeError{holoscan::ErrorCode::kFailure,
                                   "cannot write " + dump_directory_ + "/README.md"};
    }
  }

  holoscan::expected<void, holoscan::Error> dump(
      std::uint64_t frame, std::int64_t capture_ns,
      const std::array<std::uint8_t, kPolicyCells>& values,
      const std::array<std::size_t, kMaxPolicyValue + 1U>& counts) {
    auto failed = [frame](const char* file) -> holoscan::expected<void, holoscan::Error> {
      std::string message = "failed to write " + std::string(file) + " for frame " +
                            std::to_string(frame) + " of the policy dump";
      HOLOSCAN_LOG_ERROR("CaePolicySinkOp: {}", message);
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure, std::move(message)});
    };
    std::array<char, 48> name{};
    std::snprintf(name.data(), name.size(), "%s/policy_%06llu.pgm", kPolicySubdirectory,
                  static_cast<unsigned long long>(frame));  // NOLINT(runtime/int)
    std::array<char, 48> view_name{};
    std::snprintf(view_name.data(), view_name.size(), "%s/view_%06llu.png", kViewSubdirectory,
                  static_cast<unsigned long long>(frame));  // NOLINT(runtime/int)
    // Plain PGM: "P2", the size and the maxval on three header lines, then one map row per line,
    // so a text editor shows the policy values. Maxval kMaxPolicyValue makes image viewers show
    // 0 as black and kMaxPolicyValue as white.
    std::ofstream pgm(std::filesystem::path(dump_directory_) / name.data(), std::ios::trunc);
    pgm << "P2\n" << kPolicyWidth << ' ' << kPolicyHeight << '\n' << int{kMaxPolicyValue} << '\n';
    for (std::size_t row = 0; row < kPolicyHeight; ++row) {
      for (std::size_t column = 0; column < kPolicyWidth; ++column) {
        pgm << (column ? " " : "") << int{values[row * kPolicyWidth + column]};
      }
      pgm << '\n';
    }
    pgm.close();  // closing flushes; a write error surfaces here, not in the destructor
    if (!pgm) return failed(name.data());
    if (!write_policy_view_png(std::filesystem::path(dump_directory_) / view_name.data(), values,
                               source_width_, source_height_)) {
      return failed(view_name.data());
    }
    // Listed only once both files are complete, so every line names files that exist.
    frames_ << "{\"frame_id\": " << frame << ", \"capture_timestamp_ns\": " << capture_ns
            << ", \"file\": \"" << name.data() << "\", \"view\": \"" << view_name.data()
            << "\", \"counts\": {";
    for (std::size_t i = 0; i <= kMaxPolicyValue; ++i) {
      frames_ << (i ? ", " : "") << '"' << kPolicyNames[i] << "\": " << counts[i];
    }
    frames_ << "}}\n";
    frames_.flush();
    if (!frames_) return failed("frames.jsonl");
    return {};
  }

  static constexpr const char* kPolicySubdirectory = "policy";
  static constexpr const char* kViewSubdirectory = "view";

  std::shared_ptr<RunProgress> progress_;
  CaePolicyConfig config_;
  std::string dump_directory_;
  std::uint32_t source_width_;
  std::uint32_t source_height_;
  std::uint64_t next_frame_{1U};
  std::ofstream frames_;
};

}  // namespace endoscopy_tool_tracking
