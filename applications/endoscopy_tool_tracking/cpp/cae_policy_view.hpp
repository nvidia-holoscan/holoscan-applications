// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Human-readable view of a policy map, written next to the raw map in a policy dump.
//
// The view is an indexed-color PNG at the source frame size. Each pixel holds the policy value of
// the cell that covers it, using the same cell-to-pixel rule as manifest.json, and the palette
// gives each policy a color: blues for protection, gray for neutral, orange and red for
// deprioritization. The raw values stay in the file, so a script can read them back exactly.

#include <zlib.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include "cae_policy.hpp"

namespace endoscopy_tool_tracking {

struct Rgb8 {
  std::uint8_t r;
  std::uint8_t g;
  std::uint8_t b;
};

// Indexed by policy value. Darker blue means stronger protection; red means strongest
// deprioritization.
inline constexpr std::array<Rgb8, kMaxPolicyValue + 1U> kPolicyViewPalette{{
    {128, 128, 128},  // neutral: gray
    {100, 170, 255},  // protect: light blue
    {0, 40, 160},     // protect_strong: dark blue
    {255, 140, 0},    // deprioritize: orange
    {220, 0, 0},      // deprioritize_strong: red
}};

// Names of the palette colors, for the dump's README.
inline constexpr std::array<std::string_view, kMaxPolicyValue + 1U> kPolicyViewColorNames{
    "gray", "light blue", "dark blue", "orange", "red"};  // NOLINT(whitespace/indent_namespace)

namespace detail {

// For each output pixel along one axis, the map cell that contains the pixel's centre:
// cell = floor((2 * pixel + 1) * cells / (2 * pixels)).
inline std::vector<std::size_t> cell_of_pixel(std::size_t pixels, std::size_t cells) {
  std::vector<std::size_t> result(pixels);
  for (std::size_t p = 0; p < pixels; ++p) result[p] = (2U * p + 1U) * cells / (2U * pixels);
  return result;
}

// Appends one PNG chunk: big-endian length, type, data, then the CRC-32 of type and data.
inline void append_png_chunk(std::string& png, std::string_view type, const std::uint8_t* data,
                             std::size_t size) {
  const auto size32 = static_cast<std::uint32_t>(size);
  for (int shift = 24; shift >= 0; shift -= 8) {
    png.push_back(static_cast<char>((size32 >> shift) & 0xFFU));
  }
  png.append(type);
  uLong crc =
      crc32(0L, reinterpret_cast<const Bytef*>(type.data()), static_cast<uInt>(type.size()));
  // crc32() with no data would return the initial value, not crc, so skip it for empty chunks.
  if (size != 0U) {
    png.append(reinterpret_cast<const char*>(data), size);
    crc = crc32(crc, data, static_cast<uInt>(size));
  }
  for (int shift = 24; shift >= 0; shift -= 8) {
    png.push_back(static_cast<char>((crc >> shift) & 0xFFU));
  }
}

}  // namespace detail

// Largest view, in pixels, that write_policy_view_png() accepts: 4096x4096, about 16 MiB of raw
// image data. The application's views are 854x480.
inline constexpr std::uint64_t kMaxPolicyViewPixels = 4096U * 4096U;

// Writes the view of one 107x60 policy map, scaled to width x height pixels. Every value must be
// in 0..kMaxPolicyValue; the sink checks that before dumping. Returns false for an empty view, one
// larger than kMaxPolicyViewPixels, or if the file cannot be written.
inline bool write_policy_view_png(const std::filesystem::path& path,
                                  const std::array<std::uint8_t, kPolicyCells>& values,
                                  std::uint32_t width, std::uint32_t height) {
  // Checked before any allocation; the bound also keeps the image data far below PNG's 2^31-1
  // chunk limit and zlib's 32-bit sizes.
  if (width == 0U || height == 0U ||
      static_cast<std::uint64_t>(width) * height > kMaxPolicyViewPixels) {
    return false;
  }
  // Raw image data: each row is a filter-type byte (0, no filter) followed by one palette index
  // per pixel.
  const auto columns = detail::cell_of_pixel(width, kPolicyWidth);
  const auto rows = detail::cell_of_pixel(height, kPolicyHeight);
  std::vector<std::uint8_t> raw;
  raw.reserve(static_cast<std::size_t>(height) * (width + 1U));
  for (std::uint32_t y = 0; y < height; ++y) {
    raw.push_back(0U);
    const std::uint8_t* row = values.data() + rows[y] * kPolicyWidth;
    for (std::uint32_t x = 0; x < width; ++x) raw.push_back(row[columns[x]]);
  }
  uLongf compressed_size = compressBound(static_cast<uLong>(raw.size()));
  std::vector<std::uint8_t> compressed(compressed_size);
  if (compress2(compressed.data(), &compressed_size, raw.data(), static_cast<uLong>(raw.size()),
                Z_DEFAULT_COMPRESSION) != Z_OK) {
    return false;
  }

  std::string png("\x89PNG\r\n\x1a\n", 8);
  // IHDR: width, height, bit depth 8, color type 3 (indexed), default compression, filter and
  // interlace methods.
  std::array<std::uint8_t, 13> header{};
  for (int i = 0; i < 4; ++i) {
    header[i] = static_cast<std::uint8_t>(width >> (24 - 8 * i));
    header[4 + i] = static_cast<std::uint8_t>(height >> (24 - 8 * i));
  }
  header[8] = 8U;
  header[9] = 3U;
  detail::append_png_chunk(png, "IHDR", header.data(), header.size());
  std::array<std::uint8_t, 3U * kPolicyViewPalette.size()> palette{};
  for (std::size_t i = 0; i < kPolicyViewPalette.size(); ++i) {
    palette[3U * i] = kPolicyViewPalette[i].r;
    palette[3U * i + 1U] = kPolicyViewPalette[i].g;
    palette[3U * i + 2U] = kPolicyViewPalette[i].b;
  }
  detail::append_png_chunk(png, "PLTE", palette.data(), palette.size());
  detail::append_png_chunk(png, "IDAT", compressed.data(), compressed_size);
  detail::append_png_chunk(png, "IEND", nullptr, 0U);

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(png.data(), static_cast<std::streamsize>(png.size()));
  out.close();  // closing flushes; a write error surfaces here, not in the destructor
  return static_cast<bool>(out);
}

}  // namespace endoscopy_tool_tracking
