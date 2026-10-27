// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "apriltag_detector_op.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <exception>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/sensor_io/provenance.hpp>

#include "detector/pose.hpp"

namespace tag_detection {
namespace {

// Least time between two warnings about frames over a backend's capacity.
constexpr std::chrono::seconds kOverflowWarningInterval{5};

holoscan::Error invalid(std::string message) {
  return holoscan::Error{holoscan::ErrorCode::kInvalidArgument,
                         "AprilTagDetectorOp: " + std::move(message)};
}

std::optional<PixelFormat> pixel_format(holoscan::schema::ImageEncoding encoding) {
  switch (encoding) {
    case holoscan::schema::ImageEncoding_MONO8:
      return PixelFormat::kMono8;
    case holoscan::schema::ImageEncoding_RGB8:
      return PixelFormat::kRgb8;
    case holoscan::schema::ImageEncoding_BGR8:
      return PixelFormat::kBgr8;
    case holoscan::schema::ImageEncoding_RGBA8:
      return PixelFormat::kRgba8;
    case holoscan::schema::ImageEncoding_BGRA8:
      return PixelFormat::kBgra8;
    default:
      return std::nullopt;
  }
}

// The image's pixels as a view, after checking the tensor matches its encoding.
holoscan::expected<ImageView, holoscan::Error> view_image(const holoscan::schema::ImageT& image,
                                                          const holoscan::Tensor& tensor) {
  const auto format = pixel_format(image.encoding);
  if (!format) {
    return holoscan::make_unexpected(
        invalid(std::string("unsupported image encoding ") +
                holoscan::schema::EnumNameImageEncoding(image.encoding) +
                "; use MONO8, RGB8, BGR8, RGBA8 or BGRA8"));
  }
  const auto dtype = tensor.dtype();
  if (dtype.code != kDLUInt || dtype.bits != 8 || dtype.lanes != 1) {
    return holoscan::make_unexpected(invalid("image tensors must hold uint8 elements"));
  }
  const std::span<const std::int64_t> shape = tensor.shape_span();
  const int pixel_channels = channels(*format);
  const bool hwc = shape.size() == 3 && shape[2] == pixel_channels;
  const bool hw = shape.size() == 2 && pixel_channels == 1;
  if (!hwc && !hw) {
    return holoscan::make_unexpected(
        invalid("image tensor shape must be [height, width, channels] matching its encoding"));
  }
  const std::vector<std::int64_t> strides = tensor.strides();
  if (strides[1] != pixel_channels || (hwc && strides[2] != 1) ||
      strides[0] < shape[1] * pixel_channels) {
    return holoscan::make_unexpected(invalid("image pixels must be packed within each row"));
  }
  if ((image.width != 0 && image.width != shape[1]) ||
      (image.height != 0 && image.height != shape[0])) {
    return holoscan::make_unexpected(invalid("image width and height disagree with its tensor"));
  }
  const DLDevice device = tensor.device();
  const bool on_device = device.device_type == kDLCUDA || device.device_type == kDLCUDAManaged;
  if (!on_device && device.device_type != kDLCPU && device.device_type != kDLCUDAHost) {
    return holoscan::make_unexpected(invalid("image tensors must be in host or CUDA memory"));
  }
  return ImageView{.data = tensor.data_as<std::uint8_t>(),
                   .width = static_cast<int>(shape[1]),
                   .height = static_cast<int>(shape[0]),
                   .pitch = static_cast<std::size_t>(strides[0]),
                   .format = *format,
                   .location = on_device ? MemoryLocation::kDevice : MemoryLocation::kHost};
}

schema::TagDetection to_message(const Detection& detection) {
  std::array<schema::Point2f, 4> corners;
  for (int i = 0; i < 4; ++i) {
    corners[i] = schema::Point2f(static_cast<float>(detection.corners[i][0]),
                                 static_cast<float>(detection.corners[i][1]));
  }
  std::array<float, 9> rotation{};
  std::array<float, 3> translation{};
  float pose_error = 0.0F;
  if (detection.pose) {
    for (int i = 0; i < 9; ++i) rotation[i] = static_cast<float>(detection.pose->rotation[i]);
    for (int i = 0; i < 3; ++i) translation[i] = static_cast<float>(detection.pose->translation[i]);
    pose_error = static_cast<float>(detection.pose->error);
  }
  return schema::TagDetection(detection.id, detection.hamming, detection.decision_margin,
                              schema::Point2f(static_cast<float>(detection.center[0]),
                                              static_cast<float>(detection.center[1])),
                              ::flatbuffers::span<const schema::Point2f, 4>(corners),
                              detection.pose.has_value(),
                              ::flatbuffers::span<const float, 9>(rotation),
                              ::flatbuffers::span<const float, 3>(translation), pose_error);
}

}  // namespace

Detection from_message(const schema::TagDetection& tag) {
  Detection detection{
      .id = tag.id(), .hamming = tag.hamming(), .decision_margin = tag.decision_margin()};
  detection.center = {tag.center().x(), tag.center().y()};
  for (int c = 0; c < 4; ++c) {
    detection.corners[c] = {tag.corners()->Get(c)->x(), tag.corners()->Get(c)->y()};
  }
  if (tag.has_pose()) {
    TagPose& pose = detection.pose.emplace();
    for (int i = 0; i < 9; ++i) pose.rotation[i] = tag.rotation()->Get(i);
    for (int i = 0; i < 3; ++i) pose.translation[i] = tag.translation()->Get(i);
    pose.error = tag.pose_error();
  }
  return detection;
}

void validate_config(const AprilTagDetectorConfig& config) {
  validate_options(config.options);
  check_available(config.backend);
  if (config.intrinsics) {
    const CameraIntrinsics& k = *config.intrinsics;
    if (!(std::isfinite(k.fx) && k.fx > 0 && std::isfinite(k.fy) && k.fy > 0 &&
          std::isfinite(k.cx) && std::isfinite(k.cy))) {
      throw std::invalid_argument("intrinsics need finite, positive fx and fy and finite cx, cy");
    }
    if (!(std::isfinite(config.tag_size) && config.tag_size > 0)) {
      throw std::invalid_argument("tag_size must be positive to estimate poses");
    }
  }
}

AprilTagDetectorOp::AprilTagDetectorOp(AprilTagDetectorConfig config) : config_(std::move(config)) {
  validate_config(config_);
}

void AprilTagDetectorOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(image, "image").queue_depth(2U);
  spec.output(detections, "detections");
}

holoscan::Contract AprilTagDetectorOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{image});
  return result;
}

void AprilTagDetectorOp::start() {
  frames_ = 0;
  overflowed_frames_ = 0;
  last_overflow_warning_.reset();
}

void AprilTagDetectorOp::stop() {
  backend_.reset();
  own_stream_.reset();
}

holoscan::expected<void, holoscan::Error> AprilTagDetectorOp::compute(
    holoscan::ExecutionContext& context) {
  auto sample = image.receive();
  if (!sample) {
    return holoscan::make_unexpected(std::move(sample).error());
  }
  if (!sample->data.data) {
    return holoscan::make_unexpected(invalid("image has no tensor"));
  }
  const holoscan::Tensor& tensor = *sample->data.data;
  auto view = view_image(sample->data, tensor);
  if (!view) {
    return holoscan::make_unexpected(std::move(view).error());
  }
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    // No device memory on our ports, so no runtime stream: use our own, not the
    // legacy default stream, which serializes with every other stream.
    if (!own_stream_) {
      cudaStream_t created = nullptr;
      const cudaError_t status = cudaStreamCreateWithFlags(&created, cudaStreamNonBlocking);
      if (status != cudaSuccess) {
        return holoscan::make_unexpected(holoscan::Error{
            holoscan::ErrorCode::kFailure,
            std::string("AprilTagDetectorOp: cudaStreamCreate: ") + cudaGetErrorString(status)});
      }
      own_stream_.reset(created);
    }
    stream = own_stream_.get();
  }
  // Wait for the producer's writes: on our stream for device memory, on the
  // host otherwise.
  auto ready = view->location == MemoryLocation::kDevice && tensor.has_readiness_stream_contract()
                   ? tensor.synchronize_for_consumer_stream(stream)
                   : tensor.synchronize_for_host();
  if (!ready) {
    return holoscan::make_unexpected(std::move(ready).error());
  }

  FrameDetections found;
  std::chrono::steady_clock::time_point start;
  try {
    if (!backend_ || view->width != image_width_ || view->height != image_height_) {
      backend_ = create_backend(config_.backend, config_.options, view->width, view->height);
      image_width_ = view->width;
      image_height_ = view->height;
    }
    // Detection time leaves out building the backend, which happens once.
    start = std::chrono::steady_clock::now();
    const GrayImageView gray = converter_.convert(*view, backend_->input_location(), stream);
    found = backend_->detect(gray, stream);
    if (config_.intrinsics) {
      for (Detection& detection : found.tags) {
        detection.pose = estimate_pose(detection, *config_.intrinsics, config_.tag_size);
      }
    }
  } catch (const std::exception& error) {
    // Nothing queued on the stream may still read the image after we return.
    cudaStreamSynchronize(stream);
    const bool bad_input = dynamic_cast<const std::invalid_argument*>(&error) != nullptr;
    return holoscan::make_unexpected(holoscan::Error{
        bad_input ? holoscan::ErrorCode::kInvalidArgument : holoscan::ErrorCode::kFailure,
        std::string("AprilTagDetectorOp: ") + error.what()});
  }

  schema::AprilTagDetectionsT message;
  message.frame = frames_++;
  message.device_sequence = sample->data.header ? sample->data.header->device_sequence : 0U;
  message.image_width = view->width;
  message.image_height = view->height;
  message.backend = std::string(backend_->name());
  message.family = config_.options.family;
  message.detections.reserve(found.tags.size());
  for (const Detection& detection : found.tags) {
    message.detections.push_back(to_message(detection));
  }
  const Overflow& overflow = found.overflow;
  message.dropped = static_cast<std::int32_t>(overflow.detections);
  message.truncated = overflow.truncated();
  if (overflow.truncated() || overflow.detections != 0) {
    warn_overflow(overflow);
  }
  message.detect_time_us =
      std::chrono::duration<float, std::micro>(std::chrono::steady_clock::now() - start).count();
  return detections.emit(std::move(message), holoscan::sensor_io::derive_from(*sample));
}

void AprilTagDetectorOp::warn_overflow(const Overflow& overflow) {
  ++overflowed_frames_;
  const auto now = std::chrono::steady_clock::now();
  if (last_overflow_warning_ && now - *last_overflow_warning_ < kOverflowWarningInterval) {
    return;
  }
  HOLOSCAN_LOG_WARN(
      "AprilTagDetectorOp ({} backend): {} frame(s) over capacity since the last warning; the "
      "latest dropped {} boundary points, {} quad candidates, {} quads and {} detections{}",
      backend_->name(),
      overflowed_frames_, overflow.points, overflow.candidates, overflow.quads,
      overflow.detections, overflow.truncated() ? ", so tags may be missing" : "");
  overflowed_frames_ = 0;
  last_overflow_warning_ = now;
}

}  // namespace tag_detection
