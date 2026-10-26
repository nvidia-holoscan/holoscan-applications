// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Image sources for the application: the synthetic tag scene, and a folder of
// PGM/PPM images. Both emit a fixed number of frames on a clock, then idle.

#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include "detector/pnm_image.hpp"

namespace tag_detection {

struct SyntheticSourceConfig {
  int width{1280};
  int height{720};
  std::string family{"tag36h11"};
  std::uint32_t count{100};
  std::chrono::nanoseconds period{std::chrono::milliseconds{33}};
  // Emit frames in CUDA device memory instead of host memory.
  bool device_output{false};
};

// RGB8 renderings of make_scene(family, frame, width, height), frame = 0, 1, ...
class SyntheticTagSourceOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument for an unknown family or a size out of range.
  explicit SyntheticTagSourceOp(SyntheticSourceConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Output<holoscan::schema::ImageT> image;

 private:
  SyntheticSourceConfig config_;
  std::uint32_t emitted_{};
};

struct ImageFileSourceConfig {
  std::vector<std::filesystem::path> files;
  // Frames to emit; 0 emits every file once.
  std::uint32_t count{0};
  // Start over after the last file until `count` frames are out.
  bool loop{false};
  std::chrono::nanoseconds period{std::chrono::milliseconds{33}};
};

// MONO8 (PGM) or RGB8 (PPM) frames from `files`, in order. Every file must have
// the first one's size and channel count.
class ImageFileSourceOp final : public holoscan::Operator<> {
 public:
  // Reads the first file to size the output.
  // @throws std::invalid_argument or std::runtime_error for unusable files.
  explicit ImageFileSourceOp(ImageFileSourceConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // Frames this source will emit in total.
  [[nodiscard]] std::uint32_t frame_count() const noexcept { return total_; }

  holoscan::Output<holoscan::schema::ImageT> image;

 private:
  ImageFileSourceConfig config_;
  PnmImage first_;
  std::uint32_t total_{};
  std::uint32_t emitted_{};
};

}  // namespace tag_detection
