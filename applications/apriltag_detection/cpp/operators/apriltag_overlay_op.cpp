// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "apriltag_overlay_op.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <utility>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/sensor_io/provenance.hpp>
#include <holoscan/visualization/holoviz_op.hpp>

#include "apriltag_detector_op.hpp"

namespace tag_detection {
namespace {

namespace viz = holoscan::visualization;

constexpr DLDataType kFloat32{kDLFloat, 32U, 1U};
constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};
constexpr std::uint32_t kFloatsPerTag = 4U * kOverlayLinesPerTag;
constexpr std::uint32_t kLabelRowBytes = viz::kTextRunHeaderBytes + kOverlayLabelBytes;
constexpr std::uint32_t kLabelColor = 0xFF00FFFFU;  // Opaque yellow, 0xAABBGGRR.
constexpr float kLabelSizePx = 26.0F;

holoscan::TensorOutputSpec host_tensor(DLDataType dtype, std::uint32_t rank, std::uint64_t bytes) {
  return holoscan::TensorOutputSpec{
      .representation =
          holoscan::TensorRepresentation{
              .memory_kind = holoscan::MemoryKind::kHost,
              .dtype = dtype,
              .rank = rank,
              .strides = holoscan::TensorStridePolicy::kContiguousOnly,
          },
      .bounds = holoscan::tensor_bounds(bytes),
      .storage = holoscan::TensorOutputStorage::kRuntimePool,
  };
}

}  // namespace

void AprilTagOverlayOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(detections, "detections").queue_depth(2U);
  spec.output(outlines, "outlines")
      .produces_tensor(host_tensor(
          kFloat32, 1U, std::uint64_t{kMaxDetectionsPerFrame} * kFloatsPerTag * sizeof(float)));
  spec.output(labels, "labels")
      .produces_tensor(
          host_tensor(kUInt8, 2U, std::uint64_t{kMaxDetectionsPerFrame} * kLabelRowBytes));
}

holoscan::Contract AprilTagOverlayOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{detections});
  return result;
}

holoscan::expected<void, holoscan::Error> AprilTagOverlayOp::compute(
    holoscan::ExecutionContext& context) {
  auto received = detections.receive();
  if (!received) {
    return holoscan::make_unexpected(std::move(received).error());
  }
  const schema::AprilTagDetectionsT& message = received->data;
  const float sx = message.image_width > 0 ? 1.0F / static_cast<float>(message.image_width) : 0.0F;
  const float sy =
      message.image_height > 0 ? 1.0F / static_cast<float>(message.image_height) : 0.0F;
  const std::size_t tags = std::min(message.detections.size(), kMaxDetectionsPerFrame);
  // The frame's capture time, so Holoviz draws these over the image they came
  // from, or one close to it, and drops them once stale.
  holoscan::EmitOptions emit = holoscan::sensor_io::derive_from(*received);
  if (!received->metadata.source_clock_id.valid()) {
    // Unstamped detections: Holoviz selects layers by capture time, so stamp
    // them now rather than leave them unselectable.
    emit.capture_time = context.activation_time();
  }

  // A frame without tags still needs a primitive: one off-screen point.
  const std::size_t floats = std::max<std::size_t>(tags * kFloatsPerTag, 4U);
  const std::array<std::int64_t, 1> line_shape{static_cast<std::int64_t>(floats)};
  auto line_loan =
      outlines.allocate_tensor(holoscan::TensorLoanRequest{.shape = line_shape, .dtype = kFloat32});
  if (!line_loan) {
    return holoscan::make_unexpected(std::move(line_loan).error());
  }
  auto lines_written =
      line_loan->write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& writer)
                                           -> holoscan::expected<void, holoscan::Error> {
        float* out = writer.data_as<float>();
        std::fill(out, out + floats, -1.0F);
        for (std::size_t t = 0; t < tags; ++t) {
          const schema::TagDetection& tag = message.detections[t];
          float* line = out + t * kFloatsPerTag;
          for (int edge = 0; edge < 4; ++edge) {
            const auto& a = *tag.corners()->Get(edge);
            const auto& b = *tag.corners()->Get((edge + 1) % 4);
            *line++ = a.x() * sx;
            *line++ = a.y() * sy;
            *line++ = b.x() * sx;
            *line++ = b.y() * sy;
          }
          // Corners 1 and 2 bound the tag's +x side.
          const auto& c1 = *tag.corners()->Get(1);
          const auto& c2 = *tag.corners()->Get(2);
          *line++ = tag.center().x() * sx;
          *line++ = tag.center().y() * sy;
          *line++ = 0.5F * (c1.x() + c2.x()) * sx;
          *line++ = 0.5F * (c1.y() + c2.y()) * sy;
        }
        return {};
      });
  if (!lines_written) {
    return holoscan::make_unexpected(std::move(lines_written).error());
  }
  if (auto sent = outlines.emit(std::move(*line_loan), emit); !sent) {
    return sent;
  }

  const std::array<std::int64_t, 2> label_shape{static_cast<std::int64_t>(kMaxDetectionsPerFrame),
                                                kLabelRowBytes};
  auto label_loan =
      labels.allocate_tensor(holoscan::TensorLoanRequest{.shape = label_shape, .dtype = kUInt8});
  if (!label_loan) {
    return holoscan::make_unexpected(std::move(label_loan).error());
  }
  auto labels_written =
      label_loan->write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& writer)
                                            -> holoscan::expected<void, holoscan::Error> {
        auto* out = writer.data_as<std::uint8_t>();
        std::memset(out, 0, kMaxDetectionsPerFrame * kLabelRowBytes);
        for (std::size_t t = 0; t < tags; ++t) {
          const schema::TagDetection& tag = message.detections[t];
          std::uint8_t* row = out + t * kLabelRowBytes;
          // Above the tag's highest corner, so the label does not cover it.
          float top = tag.corners()->Get(0)->y();
          float left = tag.corners()->Get(0)->x();
          for (int c = 1; c < 4; ++c) {
            top = std::min(top, tag.corners()->Get(c)->y());
            left = std::min(left, tag.corners()->Get(c)->x());
          }
          const holoscan::visualization::TextRunHeader header{
              .x = left * sx,
              .y = std::max(0.0F, top - kLabelSizePx - 4.0F) * sy,
              .size_px = kLabelSizePx,
              .color = kLabelColor};
          std::memcpy(row, &header, sizeof(header));
          char text[kOverlayLabelBytes + 1] = {};
          if (tag.has_pose()) {
            // The tag's distance from the camera.
            const auto* t3 = tag.translation();
            std::snprintf(text, sizeof(text), "id %d %.2fm", tag.id(),
                          std::sqrt(t3->Get(0) * t3->Get(0) + t3->Get(1) * t3->Get(1) +
                                    t3->Get(2) * t3->Get(2)));
          } else {
            std::snprintf(text, sizeof(text), "id %d", tag.id());
          }
          std::memcpy(row + holoscan::visualization::kTextRunHeaderBytes, text, std::strlen(text));
        }
        return {};
      });
  if (!labels_written) {
    return holoscan::make_unexpected(std::move(labels_written).error());
  }
  return labels.emit(std::move(*label_loan), emit);
}

}  // namespace tag_detection
