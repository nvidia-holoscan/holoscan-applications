// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Image sources for the application: synthetic scenes or PGM/PPM files on a
// clock, and a V4L2 webcam.

#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/memory_kind.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace tag_detection {

// Declares `image` as output `name`: width x height x channels uint8 frames in `memory`.
void image_output(holoscan::OperatorSpec& spec, holoscan::Output<holoscan::schema::ImageT>& image,
                  const char* name, holoscan::MemoryKind memory, int width, int height,
                  int channels);

struct FrameSourceConfig {
  // PGM/PPM files to emit, each once, in order; empty for synthetic scenes.
  std::vector<std::filesystem::path> files;
  // Synthetic scenes: RGB8 renderings of make_scene(family, frame, width, height).
  std::string family{"tag36h11"};
  int width{1280};
  int height{720};
  std::uint32_t count{100};
  std::chrono::nanoseconds period{std::chrono::milliseconds{33}};
  // Emit synthetic frames in CUDA device memory instead of host memory.
  bool device_output{false};
};

// A fixed number of frames on a clock, then idle: synthetic scenes, or MONO8
// (PGM) and RGB8 (PPM) files, which must all match the first's size and channels.
class FrameSourceOp final : public holoscan::Operator<> {
 public:
  // Reads the first file, if any, to size the output.
  // @throws std::invalid_argument or std::runtime_error for an unusable config or file.
  explicit FrameSourceOp(FrameSourceConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  [[nodiscard]] int width() const noexcept { return config_.width; }
  [[nodiscard]] int height() const noexcept { return config_.height; }
  [[nodiscard]] int channels() const noexcept { return channels_; }

  holoscan::Output<holoscan::schema::ImageT> image;

 private:
  FrameSourceConfig config_;
  int channels_{3};
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

// RGB8 frames from a V4L2 camera offering YUYV at the requested size, stamped
// with their capture time; frames arriving while the detectors are busy are dropped.
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

  void unmap() noexcept;
  void release() noexcept;

  CameraSourceConfig config_;
  int fd_{-1};
  std::size_t pitch_{};
  std::vector<Mapping> buffers_;
  std::uint32_t emitted_{};
};

}  // namespace tag_detection
