// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Binary PGM (P5) and PPM (P6) images with 8-bit samples: a dependency-free way
// to feed recorded frames to the application. Any image tool converts to them,
// for example `ffmpeg -i video.mp4 frames/%06d.pgm`.

#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

namespace tag_detection {

struct PnmImage {
  int width{};
  int height{};
  int channels{};                    // 1 (PGM) or 3 (PPM).
  std::vector<std::uint8_t> pixels;  // Rows packed.
};

// @throws std::runtime_error naming the file and what is wrong with it.
[[nodiscard]] PnmImage read_pnm(const std::filesystem::path& path);

// @throws std::runtime_error if the file cannot be written.
void write_pnm(const std::filesystem::path& path, const PnmImage& image);

// The .pgm and .ppm files directly in `directory`, sorted by name.
// @throws std::runtime_error if `directory` cannot be listed or holds none.
[[nodiscard]] std::vector<std::filesystem::path> list_pnm_files(
    const std::filesystem::path& directory);

}  // namespace tag_detection
