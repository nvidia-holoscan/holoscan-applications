// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "gpu_segmenter.hpp"  // cuda_check
#include "gray_converter.hpp"

namespace tag_detection {
namespace {

// BT.601 luma in 8-bit fixed point.
__host__ __device__ inline std::uint8_t luma(std::uint8_t r, std::uint8_t g, std::uint8_t b) {
  return static_cast<std::uint8_t>((77U * r + 150U * g + 29U * b + 128U) >> 8U);
}

struct ChannelOrder {
  int stride;
  int red;
  int blue;
};

__host__ __device__ inline ChannelOrder order_of(PixelFormat format) {
  switch (format) {
    case PixelFormat::kRgb8:
      return {3, 0, 2};
    case PixelFormat::kBgr8:
      return {3, 2, 0};
    case PixelFormat::kRgba8:
      return {4, 0, 2};
    case PixelFormat::kBgra8:
      return {4, 2, 0};
    default:
      return {1, 0, 0};
  }
}

__global__ void to_gray_kernel(const std::uint8_t* input, std::size_t in_pitch, PixelFormat format,
                               int width, int height, std::uint8_t* output) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  const ChannelOrder order = order_of(format);
  const std::uint8_t* pixel = input + y * in_pitch + x * order.stride;
  output[y * width + x] = luma(pixel[order.red], pixel[1], pixel[order.blue]);
}

void to_gray_host(const ImageView& input, std::uint8_t* output) {
  const ChannelOrder order = order_of(input.format);
  for (int y = 0; y < input.height; ++y) {
    const std::uint8_t* row = input.data + static_cast<std::size_t>(y) * input.pitch;
    std::uint8_t* out = output + static_cast<std::size_t>(y) * input.width;
    for (int x = 0; x < input.width; ++x) {
      const std::uint8_t* pixel = row + x * order.stride;
      out[x] = luma(pixel[order.red], pixel[1], pixel[order.blue]);
    }
  }
}

}  // namespace

int channels(PixelFormat format) noexcept { return order_of(format).stride; }

GrayConverter::~GrayConverter() { release(); }

void GrayConverter::release() noexcept {
  if (host_gray_ != nullptr) cudaFreeHost(host_gray_);
  if (device_gray_ != nullptr) cudaFree(device_gray_);
  host_gray_ = nullptr;
  device_gray_ = nullptr;
  width_ = height_ = 0;
}

void GrayConverter::reserve(int width, int height) {
  if (width == width_ && height == height_) {
    return;
  }
  release();
  const std::size_t bytes = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  cuda_check(cudaMallocHost(reinterpret_cast<void**>(&host_gray_), bytes), "cudaMallocHost(gray)");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&device_gray_), bytes), "cudaMalloc(gray)");
  width_ = width;
  height_ = height;
}

GrayImageView GrayConverter::convert(const ImageView& input, MemoryLocation target,
                                     cudaStream_t stream) {
  if (input.format == PixelFormat::kMono8 && input.location == target) {
    return {input.data, input.width, input.height, input.pitch, target};
  }
  reserve(input.width, input.height);
  const auto width = static_cast<std::size_t>(input.width);
  const auto height = static_cast<std::size_t>(input.height);
  const GrayImageView host{host_gray_, input.width, input.height, width, MemoryLocation::kHost};
  const GrayImageView device{device_gray_, input.width, input.height, width,
                             MemoryLocation::kDevice};

  if (input.location == MemoryLocation::kHost) {
    // Convert on the host, then move a third (or a quarter) of the bytes.
    const std::uint8_t* gray = input.data;
    std::size_t gray_pitch = input.pitch;
    if (input.format != PixelFormat::kMono8) {
      to_gray_host(input, host_gray_);
      gray = host_gray_;
      gray_pitch = width;
    }
    if (target == MemoryLocation::kHost) {
      return host;
    }
    cuda_check(cudaMemcpy2DAsync(device_gray_, width, gray, gray_pitch, width, height,
                                 cudaMemcpyHostToDevice, stream),
               "cudaMemcpy2DAsync(gray to device)");
    return device;
  }

  const std::uint8_t* gray = input.data;
  std::size_t gray_pitch = input.pitch;
  if (input.format != PixelFormat::kMono8) {
    const dim3 block(32, 8);
    const dim3 grid((input.width + block.x - 1) / block.x, (input.height + block.y - 1) / block.y);
    to_gray_kernel<<<grid, block, 0, stream>>>(input.data, input.pitch, input.format, input.width,
                                               input.height, device_gray_);
    cuda_check(cudaGetLastError(), "to_gray_kernel");
    gray = device_gray_;
    gray_pitch = width;
  }
  if (target == MemoryLocation::kDevice) {
    return device;
  }
  cuda_check(cudaMemcpy2DAsync(host_gray_, width, gray, gray_pitch, width, height,
                               cudaMemcpyDeviceToHost, stream),
             "cudaMemcpy2DAsync(gray to host)");
  cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  return host;
}

}  // namespace tag_detection
