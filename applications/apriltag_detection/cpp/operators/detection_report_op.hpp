// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// DetectionReportOp: the application's sink. Counts detections and latencies,
// checks synthetic input against ground truth, and requests completion once
// every reporter has seen its expected frames.

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>

#include "apriltag_detections_schema_traits.hpp"

namespace tag_detection {

// One backend's results. Written by its reporter during the run and read by
// main() after the run has stopped.
struct BackendReport {
  std::string backend;
  std::uint32_t frames{};
  std::uint64_t tags{};
  std::uint64_t dropped{};
  std::uint32_t truncated_frames{};  // Frames that ran out of detector capacity.
  std::vector<float> detect_ms;
  // Ground-truth checks; empty when the input is not synthetic.
  bool verified{};
  std::uint64_t expected_tags{};
  std::uint64_t missed_tags{};
  std::uint64_t extra_tags{};
  std::uint64_t misplaced_tags{};  // Found, but a corner or the pose is off.
  double max_corner_error_px{};
  double max_translation_error{};  // Fraction of the tag's distance.

  [[nodiscard]] bool passed() const noexcept {
    return !verified || (missed_tags == 0 && extra_tags == 0 && misplaced_tags == 0);
  }
};

// The synthetic scene the detections should match.
struct GroundTruth {
  std::string family;
  int width{};
  int height{};
};

struct DetectionReportConfig {
  std::shared_ptr<BackendReport> report;
  // Reporters still running, shared by all of a run's; the last requests completion.
  std::shared_ptr<std::atomic<int>> remaining;
  // Frames after which this reporter is done; 0 never finishes.
  std::uint32_t expected_frames{};
  std::optional<GroundTruth> truth;
};

// Largest corner error, in pixels, a verified detection may have.
inline constexpr double kMaxCornerErrorPx = 1.5;
// Largest translation error, as a fraction of the tag's distance.
inline constexpr double kMaxTranslationError = 0.03;

class DetectionReportOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if `report` or `remaining` is missing.
  explicit DetectionReportOp(DetectionReportConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Input<schema::AprilTagDetectionsT> detections;

 private:
  void verify(const schema::AprilTagDetectionsT& message);

  DetectionReportConfig config_;
  bool finished_{};
};

}  // namespace tag_detection
