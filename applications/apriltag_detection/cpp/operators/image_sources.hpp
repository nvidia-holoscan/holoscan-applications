// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Image sources for the application: the synthetic tag scene, a folder of
// PGM/PPM images, and a V4L2 webcam.

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
// A fixed number of frames on a clock, then idle.
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
  std::chrono::nanoseconds period{std::chrono::milliseconds{33}};
};

// MONO8 (PGM) or RGB8 (PPM) frames from `files`, each once, in order. Every
// file must have the first one's size and channel count.
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

  [[nodiscard]] const PnmImage& first() const noexcept { return first_; }

  holoscan::Output<holoscan::schema::ImageT> image;

 private:
  ImageFileSourceConfig config_;
  PnmImage first_;
  std::uint32_t emitted_{};
};

struct CameraSourceConfig {
  std::filesystem::path device{"/dev/video0"};
  int width{640};
  int height{480};
  // Frames to emit; 0 runs until the application is stopped.
  std::uint32_t count{0};
};

// Checks that `config.device` offers YUYV at the configured size.
// @throws std::invalid_argument naming what is wrong, with the YUYV sizes the
//         device offers when the size is the problem.
void check_camera(const CameraSourceConfig& config);

// RGB8 frames from a V4L2 camera that offers YUYV at the requested size,
// stamped with the time the camera captured them. Frames that arrive while the
// detectors are busy are dropped. Linux only.
class CameraSourceOp final : public holoscan::Operator<> {
 public:
  // Opens the device and sets its format.
  // @throws std::invalid_argument as check_camera() does.
  explicit CameraSourceOp(CameraSourceConfig config);
  ~CameraSourceOp() override;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Output<holoscan::schema::ImageT> image;

 private:
  struct Mapping {
    void* data{};
    std::size_t length{};
  };

  void release() noexcept;

  CameraSourceConfig config_;
  int fd_{-1};
  std::size_t pitch_{};
  std::vector<Mapping> buffers_;
  std::uint32_t emitted_{};
};

}  // namespace tag_detection
