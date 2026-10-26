// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Converts an input image of any supported packing, on the host or the device,
// into the gray image a backend wants, where it wants it. A gray input already
// in place is passed through without a copy.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

#include "detection.hpp"

namespace tag_detection {

enum class PixelFormat : std::uint8_t { kMono8, kRgb8, kBgr8, kRgba8, kBgra8 };

[[nodiscard]] int channels(PixelFormat format) noexcept;

struct ImageView {
  const std::uint8_t* data{};
  int width{};
  int height{};
  std::size_t pitch{};
  PixelFormat format{PixelFormat::kMono8};
  MemoryLocation location{MemoryLocation::kHost};
};

class GrayConverter {
 public:
  GrayConverter() = default;
  ~GrayConverter();
  GrayConverter(const GrayConverter&) = delete;
  GrayConverter& operator=(const GrayConverter&) = delete;

  // `input` as gray in `target`, valid until the next call. Device work is
  // ordered on `stream`; a host result is complete when this returns.
  // @throws std::runtime_error on a CUDA failure.
  [[nodiscard]] GrayImageView convert(const ImageView& input, MemoryLocation target,
                                      cudaStream_t stream);

 private:
  void reserve(int width, int height);
  void release() noexcept;

  int width_{};
  int height_{};
  std::uint8_t* host_gray_{};    // Pinned, width_ x height_.
  std::uint8_t* device_gray_{};  // width_ x height_.
};

}  // namespace tag_detection
