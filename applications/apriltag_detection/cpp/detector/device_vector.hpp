// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Device memory for the GPU segmenter's scratch buffers: it grows on request,
// by at least half again, and is released on destruction.

#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace tag_detection {

// Growing discards the contents, and frees device memory, which synchronizes
// the device; it happens only between stream synchronizations. Growing by half
// again keeps a slowly rising requirement from reallocating on every call.
template <typename T>
class DeviceVector {
 public:
  DeviceVector() = default;
  explicit DeviceVector(std::size_t count) { reserve(count); }
  ~DeviceVector() { cudaFree(data_); }
  DeviceVector(const DeviceVector&) = delete;
  DeviceVector& operator=(const DeviceVector&) = delete;

  void reserve(std::size_t count) {
    if (count <= capacity_) {
      return;
    }
    const std::size_t grown = std::max(count, capacity_ + capacity_ / 2);
    cudaFree(data_);
    data_ = nullptr;
    capacity_ = 0;
    const cudaError_t status =
        cudaMalloc(reinterpret_cast<void**>(&data_), std::max<std::size_t>(grown, 1) * sizeof(T));
    if (status != cudaSuccess) {
      data_ = nullptr;
      throw std::runtime_error(std::string("cudaMalloc: ") + cudaGetErrorString(status));
    }
    capacity_ = grown;
  }
  [[nodiscard]] T* get() const noexcept { return data_; }
  [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }

 private:
  T* data_{};
  std::size_t capacity_{};
};

}  // namespace tag_detection
