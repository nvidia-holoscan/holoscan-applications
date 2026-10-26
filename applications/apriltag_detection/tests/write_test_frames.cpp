// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Writes synthetic tag frames as PGM (gray) or PPM (rgb) files, for the tests
// that feed the application a folder of images.
//
// Usage: write_test_frames DIR COUNT gray|rgb [FAMILY]

#include <charconv>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>

#include "detector/pnm_image.hpp"
#include "detector/synthetic_scene.hpp"

int main(int argc, char** argv) try {
  if (argc < 4) {
    std::cerr << "usage: " << argv[0] << " DIR COUNT gray|rgb [FAMILY]\n";
    return 2;
  }
  const std::filesystem::path directory(argv[1]);
  unsigned count = 0;
  const std::string_view count_text(argv[2]);
  std::from_chars(count_text.data(), count_text.data() + count_text.size(), count);
  const bool gray = std::string_view(argv[3]) == "gray";
  const std::string family = argc > 4 ? argv[4] : "tag36h11";
  std::filesystem::create_directories(directory);
  for (unsigned frame = 0; frame < count; ++frame) {
    const auto scene = tag_detection::make_scene(family, frame, 1280, 720);
    const int channels = gray ? 1 : 3;
    tag_detection::PnmImage image{scene.width, scene.height, channels,
                                  tag_detection::render_scene(scene, channels)};
    char name[32];
    std::snprintf(name, sizeof(name), "frame_%04u.%s", frame, gray ? "pgm" : "ppm");
    tag_detection::write_pnm(directory / name, image);
  }
  return 0;
} catch (const std::exception& error) {
  std::cerr << "error: " << error.what() << '\n';
  return 1;
}
