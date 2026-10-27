// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "detection_report_op.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

#include "detector/synthetic_scene.hpp"

namespace tag_detection {
DetectionReportOp::DetectionReportOp(DetectionReportConfig config) : config_(std::move(config)) {
  if (!config_.report || !config_.completion) {
    throw std::invalid_argument("DetectionReportOp needs a report and a completion counter");
  }
}

void DetectionReportOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(detections, "detections").queue_depth(2U);
}

holoscan::Contract DetectionReportOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{detections});
  return result;
}

void DetectionReportOp::start() { finished_ = false; }

holoscan::expected<void, holoscan::Error> DetectionReportOp::compute(
    holoscan::ExecutionContext& context) {
  auto received = detections.receive_data();
  if (!received) {
    return holoscan::make_unexpected(std::move(received).error());
  }
  const schema::AprilTagDetectionsT& message = *received;
  BackendReport& report = *config_.report;
  report.backend = message.backend;
  ++report.frames;
  report.tags += message.detections.size();
  report.dropped += static_cast<std::uint64_t>(std::max(message.dropped, 0));
  report.detect_ms.push_back(message.detect_time_us / 1000.0F);
  if (config_.truth) {
    verify(message);
  }
  if (!finished_ && config_.expected_frames != 0 && report.frames >= config_.expected_frames) {
    finished_ = true;
    if (config_.completion->remaining.fetch_sub(1) == 1) {
      context.request_completion();
    }
  }
  return {};
}

void DetectionReportOp::verify(const schema::AprilTagDetectionsT& message) {
  BackendReport& report = *config_.report;
  report.verified = true;
  const GroundTruth& truth = *config_.truth;
  const SyntheticScene scene = make_scene(truth.family, static_cast<std::uint32_t>(message.frame),
                                          truth.width, truth.height);
  std::map<int, const SceneTag*> expected;
  for (const SceneTag& tag : scene.tags) expected[tag.id] = &tag;
  report.expected_tags += expected.size();

  for (const schema::TagDetection& tag : message.detections) {
    const auto it = expected.find(tag.id());
    if (it == expected.end()) {
      ++report.extra_tags;
      continue;
    }
    const auto corners = project_corners(*it->second, scene.intrinsics, scene.tag_size);
    double worst = 0.0;
    for (int c = 0; c < 4; ++c) {
      const auto& corner = *tag.corners()->Get(c);
      worst = std::max(worst, std::hypot(corner.x() - corners[c][0], corner.y() - corners[c][1]));
    }
    report.max_corner_error_px = std::max(report.max_corner_error_px, worst);
    bool misplaced = worst > kMaxCornerErrorPx;
    if (tag.has_pose()) {
      const auto& t = it->second->pose.translation;
      const double error =
          std::hypot(tag.translation()->Get(0) - t[0], tag.translation()->Get(1) - t[1],
                     tag.translation()->Get(2) - t[2]) /
          std::hypot(t[0], t[1], t[2]);
      report.max_translation_error = std::max(report.max_translation_error, error);
      misplaced = misplaced || error > kMaxTranslationError;
    }
    if (misplaced) {
      ++report.misplaced_tags;
    }
    expected.erase(it);  // A second report of one tag counts as extra.
  }
  report.missed_tags += expected.size();
}

}  // namespace tag_detection
