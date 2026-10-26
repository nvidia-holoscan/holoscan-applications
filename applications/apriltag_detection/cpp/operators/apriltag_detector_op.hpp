// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// AprilTagDetectorOp: an image in, the tags in it out. The detection backend
// (CPU, CUDA or VPI) is chosen by configuration; the operator converts the
// image to gray where that backend wants it, detects, estimates poses when the
// camera's intrinsics are known, and emits one AprilTagDetections per image.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include "apriltag_detections_schema_traits.hpp"
#include "detector/detection.hpp"
#include "detector/detector_backend.hpp"
#include "detector/gray_converter.hpp"

namespace tag_detection {

// Most tags one message carries; further tags are counted in `dropped`.
inline constexpr std::size_t kMaxDetectionsPerFrame = 64;

struct AprilTagDetectorConfig {
  Backend backend{Backend::kCuda};
  DetectorOptions options;
  // With both, each detection carries the tag's pose.
  std::optional<CameraIntrinsics> intrinsics;
  double tag_size{};  // Meters across the tag's black border.
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const AprilTagDetectorConfig& config);

class AprilTagDetectorOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if `config` is not valid.
  // With Backend::kVpiPva, construction asks VPI whether there is a PVA, which
  // starts VPI's worker threads: create any holoscan::ManagedShutdownSignals
  // before building the graph.
  explicit AprilTagDetectorOp(AprilTagDetectorConfig config);
  ~AprilTagDetectorOp() override;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // MONO8, RGB8, BGR8, RGBA8 or BGRA8, in host or CUDA device memory.
  holoscan::Input<holoscan::schema::ImageT> image;
  holoscan::Output<schema::AprilTagDetectionsT> detections;

 private:
  AprilTagDetectorConfig config_;
  // Built on the first image, and again if the image size changes.
  std::unique_ptr<DetectorBackend> backend_;
  int image_width_{};
  int image_height_{};
  GrayConverter converter_;
  // For frames whose activation has no runtime CUDA stream; made on first use.
  cudaStream_t own_stream_{};
  std::uint64_t frames_{};
};

}  // namespace tag_detection
