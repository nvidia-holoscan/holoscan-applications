// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "image_sources.hpp"

#include <cuda_runtime_api.h>
#include <fcntl.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstring>
#include <ctime>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
// clang-format groups these Linux headers with the SDK's; cpplint wants them first.
#include <linux/videodev2.h>  // NOLINT(build/include_order)
#include <sys/ioctl.h>        // NOLINT(build/include_order)
#include <sys/mman.h>         // NOLINT(build/include_order)

#include "detector/pnm_image.hpp"
#include "detector/synthetic_scene.hpp"
#include "detector/yuyv.hpp"

namespace tag_detection {
namespace {

constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};

}  // namespace

void image_output(holoscan::OperatorSpec& spec, holoscan::Output<holoscan::schema::ImageT>& image,
                  const char* name, holoscan::MemoryKind memory, int width, int height,
                  int channels) {
  const std::uint64_t bytes = std::uint64_t{static_cast<std::uint32_t>(width)} * height *
                              static_cast<std::uint32_t>(channels);
  spec.output(image, name)
      .produces_tensor(holoscan::TensorOutputSpec{
          .representation =
              holoscan::TensorRepresentation{
                  .memory_kind = memory,
                  .dtype = kUInt8,
                  .rank = 3U,
                  .strides = holoscan::TensorStridePolicy::kContiguousOnly,
              },
          .bounds = holoscan::tensor_bounds(bytes),
          .storage = holoscan::TensorOutputStorage::kRuntimePool,
      })
      .payload_options(holoscan::PayloadOptions{
          .schema = holoscan::schema_identity<holoscan::schema::ImageT>(),
      });
}

namespace {

holoscan::Error failure(std::string message) {
  return holoscan::Error{holoscan::ErrorCode::kFailure, std::move(message)};
}

// A busy consumer fails the emit with kBackpressured before anything is
// delivered: send the same frame again on the next tick.
holoscan::expected<void, holoscan::Error> retry_if_backpressured(
    holoscan::expected<void, holoscan::Error> emitted, std::uint32_t& sent) {
  if (emitted) {
    ++sent;
    return {};
  }
  if (emitted.error().code == holoscan::ErrorCode::kBackpressured) {
    return {};
  }
  return emitted;
}

// Emits a `width` x `height` image of `channels` 8-bit channels, rows packed,
// that `fill` writes: into host memory, or with `pixels` into device memory.
holoscan::expected<void, holoscan::Error> emit_image(
    holoscan::Output<holoscan::schema::ImageT>& output, holoscan::ExecutionContext& context,
    int width, int height, int channels, const std::function<void(std::uint8_t*)>& fill,
    const std::vector<std::uint8_t>* device_pixels, holoscan::TimePoint captured) {
  const std::array<std::int64_t, 3> shape{height, width, channels};
  const auto bytes = static_cast<std::size_t>(width) * height * channels;
  auto loan = output.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = kUInt8});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  if (device_pixels != nullptr) {
    auto writer = loan->write(context.cuda_stream());
    if (!writer) {
      return holoscan::make_unexpected(std::move(writer).error());
    }
    if (writer->byte_size() != bytes) {
      return holoscan::make_unexpected(failure("image loan has an unexpected size"));
    }
    const cudaError_t copied =
        cudaMemcpyAsync(writer->data_as<std::uint8_t>(), device_pixels->data(), bytes,
                        cudaMemcpyHostToDevice, context.cuda_stream().get());
    if (copied != cudaSuccess) {
      return holoscan::make_unexpected(failure(cudaGetErrorString(copied)));
    }
    if (auto committed = std::move(*writer).commit(); !committed) {
      return holoscan::make_unexpected(std::move(committed).error());
    }
  } else {
    auto committed = loan->write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& writer)
                                                     -> holoscan::expected<void, holoscan::Error> {
      if (writer.byte_size() != bytes) {
        return holoscan::make_unexpected(failure("image loan has an unexpected size"));
      }
      fill(writer.data_as<std::uint8_t>());
      return {};
    });
    if (!committed) {
      return holoscan::make_unexpected(std::move(committed).error());
    }
  }
  holoscan::schema::ImageT descriptor;
  descriptor.width = width;
  descriptor.height = height;
  descriptor.encoding =
      channels == 1 ? holoscan::schema::ImageEncoding_MONO8 : holoscan::schema::ImageEncoding_RGB8;
  return output.emit_tensor(std::move(*loan), descriptor,
                            holoscan::EmitOptions{.capture_time = captured});
}

}  // namespace

FrameSourceOp::FrameSourceOp(FrameSourceConfig config) : config_(std::move(config)) {
  if (!config_.files.empty()) {
    const PnmImage first = read_pnm(config_.files.front());
    config_.width = first.width;
    config_.height = first.height;
    config_.count = static_cast<std::uint32_t>(config_.files.size());
    channels_ = first.channels;
  } else {
    (void)make_scene(config_.family, 0, config_.width, config_.height);  // Family, minimum size.
    if (config_.width > 4096 || config_.height > 4096) {
      throw std::invalid_argument("synthetic images must be 320x240 to 4096x4096");
    }
  }
  if (config_.count == 0 || config_.period <= std::chrono::nanoseconds::zero()) {
    throw std::invalid_argument("frame source needs a positive count and period");
  }
}

void FrameSourceOp::setup(holoscan::OperatorSpec& spec) {
  image_output(
      spec, image, "image",
      config_.device_output ? holoscan::MemoryKind::kCudaDevice : holoscan::MemoryKind::kHost,
      config_.width, config_.height, channels_);
}

holoscan::Contract FrameSourceOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnClock{.period = config_.period});
  return result;
}

void FrameSourceOp::start() { emitted_ = 0; }

holoscan::expected<void, holoscan::Error> FrameSourceOp::compute(
    holoscan::ExecutionContext& context) {
  if (emitted_ >= config_.count) {
    return {};  // Done: idle until the consumers request completion.
  }
  std::vector<std::uint8_t> pixels;
  if (config_.files.empty()) {
    pixels = render_scene(make_scene(config_.family, emitted_, config_.width, config_.height), 3);
  } else {
    const auto& path = config_.files[emitted_];
    PnmImage frame;
    try {
      frame = read_pnm(path);
    } catch (const std::exception& error) {
      return holoscan::make_unexpected(failure(error.what()));
    }
    if (frame.width != config_.width || frame.height != config_.height ||
        frame.channels != channels_) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kInvalidArgument,
          path.string() + ": every image must match the first one's size and channels"});
    }
    pixels = std::move(frame.pixels);
  }
  return retry_if_backpressured(
      emit_image(
          image, context, config_.width, config_.height, channels_,
          [&pixels](std::uint8_t* out) { std::memcpy(out, pixels.data(), pixels.size()); },
          config_.device_output ? &pixels : nullptr, context.activation_time()),
      emitted_);
}

namespace {

int xioctl(int fd, std::uint64_t request, void* argument) {
  int result = 0;
  do {
    result = ioctl(fd, request, argument);
  } while (result == -1 && errno == EINTR);
  return result;
}

// The YUYV frame sizes `fd` offers, as "640x480, 1280x720".
std::string yuyv_sizes(int fd) {
  std::string sizes;
  v4l2_frmsizeenum size{};
  size.pixel_format = V4L2_PIX_FMT_YUYV;
  for (; xioctl(fd, VIDIOC_ENUM_FRAMESIZES, &size) == 0; ++size.index) {
    if (size.type != V4L2_FRMSIZE_TYPE_DISCRETE) {
      return "any up to " + std::to_string(size.stepwise.max_width) + "x" +
             std::to_string(size.stepwise.max_height);
    }
    sizes += (sizes.empty() ? "" : ", ") + std::to_string(size.discrete.width) + "x" +
             std::to_string(size.discrete.height);
  }
  return sizes.empty() ? "none" : sizes;
}

constexpr std::uint32_t kCameraBuffers = 4;

// Opens `config.device` and sets YUYV at the configured size; returns the
// descriptor and the row pitch.
int open_camera(const CameraSourceConfig& config, std::size_t& pitch) {
  const std::string name = config.device.string();
  const int fd = open(name.c_str(), O_RDWR | O_NONBLOCK);
  if (fd < 0) {
    throw std::invalid_argument("cannot open camera " + name + ": " + std::strerror(errno));
  }
  v4l2_capability capability{};
  if (xioctl(fd, VIDIOC_QUERYCAP, &capability) != 0 ||
      (capability.device_caps & V4L2_CAP_VIDEO_CAPTURE) == 0 ||
      (capability.device_caps & V4L2_CAP_STREAMING) == 0) {
    close(fd);
    throw std::invalid_argument(name + " is not a V4L2 capture device with streaming");
  }
  v4l2_format format{};
  format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  format.fmt.pix.width = static_cast<std::uint32_t>(config.width);
  format.fmt.pix.height = static_cast<std::uint32_t>(config.height);
  format.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
  format.fmt.pix.field = V4L2_FIELD_NONE;
  if (xioctl(fd, VIDIOC_S_FMT, &format) != 0 || format.fmt.pix.pixelformat != V4L2_PIX_FMT_YUYV ||
      format.fmt.pix.width != static_cast<std::uint32_t>(config.width) ||
      format.fmt.pix.height != static_cast<std::uint32_t>(config.height)) {
    const std::string offered = yuyv_sizes(fd);
    close(fd);
    throw std::invalid_argument(name + " does not offer YUYV at " + std::to_string(config.width) +
                                "x" + std::to_string(config.height) +
                                "; its YUYV sizes: " + offered);
  }
  pitch = format.fmt.pix.bytesperline;
  return fd;
}

}  // namespace

void check_camera(const CameraSourceConfig& config) {
  std::size_t pitch = 0;
  close(open_camera(config, pitch));
}

CameraSourceOp::CameraSourceOp(CameraSourceConfig config) : config_(std::move(config)) {
  fd_ = open_camera(config_, pitch_);
}

CameraSourceOp::~CameraSourceOp() { release(); }

void CameraSourceOp::unmap() noexcept {
  for (const Mapping& buffer : buffers_) {
    munmap(buffer.data, buffer.length);
  }
  buffers_.clear();
}

void CameraSourceOp::release() noexcept {
  unmap();
  if (fd_ >= 0) {
    close(fd_);
    fd_ = -1;
  }
}

void CameraSourceOp::setup(holoscan::OperatorSpec& spec) {
  image_output(spec, image, "image", holoscan::MemoryKind::kHost, config_.width, config_.height, 3);
}

holoscan::Contract CameraSourceOp::contract() const {
  // Poll faster than any frame rate; a poll without a new frame does nothing.
  holoscan::Contract result;
  result.trigger(holoscan::OnClock{.period = std::chrono::milliseconds{5}});
  return result;
}

void CameraSourceOp::start() {
  emitted_ = 0;
  v4l2_requestbuffers request{};
  request.count = kCameraBuffers;
  request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  request.memory = V4L2_MEMORY_MMAP;
  if (xioctl(fd_, VIDIOC_REQBUFS, &request) != 0 || request.count == 0) {
    throw holoscan::RuntimeError{holoscan::ErrorCode::kFailure, "VIDIOC_REQBUFS failed"};
  }
  for (std::uint32_t i = 0; i < request.count; ++i) {
    v4l2_buffer buffer{};
    buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buffer.memory = V4L2_MEMORY_MMAP;
    buffer.index = i;
    if (xioctl(fd_, VIDIOC_QUERYBUF, &buffer) != 0) {
      throw holoscan::RuntimeError{holoscan::ErrorCode::kFailure, "VIDIOC_QUERYBUF failed"};
    }
    void* data =
        mmap(nullptr, buffer.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buffer.m.offset);
    if (data == MAP_FAILED) {
      throw holoscan::RuntimeError{holoscan::ErrorCode::kFailure, "mmap of a camera buffer failed"};
    }
    buffers_.push_back({data, buffer.length});
    if (xioctl(fd_, VIDIOC_QBUF, &buffer) != 0) {
      throw holoscan::RuntimeError{holoscan::ErrorCode::kFailure, "VIDIOC_QBUF failed"};
    }
  }
  int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (xioctl(fd_, VIDIOC_STREAMON, &type) != 0) {
    throw holoscan::RuntimeError{holoscan::ErrorCode::kFailure, "VIDIOC_STREAMON failed"};
  }
}

void CameraSourceOp::stop() {
  int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  xioctl(fd_, VIDIOC_STREAMOFF, &type);
  unmap();
  v4l2_requestbuffers request{};
  request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  request.memory = V4L2_MEMORY_MMAP;
  xioctl(fd_, VIDIOC_REQBUFS, &request);
}

holoscan::expected<void, holoscan::Error> CameraSourceOp::compute(
    holoscan::ExecutionContext& context) {
  if (config_.count != 0 && emitted_ >= config_.count) {
    return {};
  }
  v4l2_buffer buffer{};
  buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  buffer.memory = V4L2_MEMORY_MMAP;
  if (xioctl(fd_, VIDIOC_DQBUF, &buffer) != 0) {
    if (errno == EAGAIN) {
      return {};  // No new frame yet.
    }
    return holoscan::make_unexpected(failure("VIDIOC_DQBUF: " + std::string(std::strerror(errno))));
  }
  // Frames that queued up while we were busy are stale: keep only the newest.
  for (v4l2_buffer newer = buffer; xioctl(fd_, VIDIOC_DQBUF, &newer) == 0; buffer = newer) {
    if (xioctl(fd_, VIDIOC_QBUF, &buffer) != 0) {
      return holoscan::make_unexpected(
          failure("VIDIOC_QBUF: " + std::string(std::strerror(errno))));
    }
  }
  // The frame's capture time on the graph's clock: now, less its age on the
  // monotonic clock V4L2 stamps it with.
  holoscan::TimePoint captured = context.activation_time();
  if ((buffer.flags & V4L2_BUF_FLAG_TIMESTAMP_MASK) == V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC) {
    timespec now{};
    clock_gettime(CLOCK_MONOTONIC, &now);
    const std::int64_t age = (now.tv_sec - buffer.timestamp.tv_sec) * 1'000'000'000LL +
                             (now.tv_nsec - buffer.timestamp.tv_usec * 1000LL);
    if (age > 0) {
      captured.timestamp_ns -= age;
    }
  }
  const auto* yuyv = static_cast<const std::uint8_t*>(buffers_[buffer.index].data);
  auto emitted = emit_image(
      image, context, config_.width, config_.height, 3,
      [&](std::uint8_t* rgb) { yuyv_to_rgb(yuyv, pitch_, config_.width, config_.height, rgb); },
      nullptr, captured);
  if (xioctl(fd_, VIDIOC_QBUF, &buffer) != 0) {
    return holoscan::make_unexpected(failure("VIDIOC_QBUF: " + std::string(std::strerror(errno))));
  }
  if (emitted) {
    ++emitted_;
    return {};
  }
  // A live frame is not worth waiting for: drop it while the detectors are busy.
  return emitted.error().code == holoscan::ErrorCode::kBackpressured
             ? holoscan::expected<void, holoscan::Error>{}
             : emitted;
}

}  // namespace tag_detection
