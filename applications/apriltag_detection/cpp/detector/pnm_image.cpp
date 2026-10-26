// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "pnm_image.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace tag_detection {
namespace {

// Reads one header integer, skipping whitespace and comments.
int read_header_value(std::istream& in, const std::filesystem::path& path) {
  int c = in.peek();
  while (c != EOF && (std::isspace(c) || c == '#')) {
    if (c == '#') {
      in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    } else {
      in.get();
    }
    c = in.peek();
  }
  int value = -1;
  if (!(in >> value) || value < 0) {
    throw std::runtime_error(path.string() + ": malformed PNM header");
  }
  return value;
}

}  // namespace

PnmImage read_pnm(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error(path.string() + ": cannot open");
  }
  char magic[2] = {};
  in.read(magic, 2);
  if (!in || magic[0] != 'P' || (magic[1] != '5' && magic[1] != '6')) {
    throw std::runtime_error(path.string() + ": not a binary PGM (P5) or PPM (P6) image");
  }
  PnmImage image;
  image.channels = magic[1] == '5' ? 1 : 3;
  image.width = read_header_value(in, path);
  image.height = read_header_value(in, path);
  const int max_value = read_header_value(in, path);
  if (image.width <= 0 || image.height <= 0 || image.width > 16384 || image.height > 16384) {
    throw std::runtime_error(path.string() + ": image size out of range");
  }
  if (max_value != 255) {
    throw std::runtime_error(path.string() + ": only 8-bit samples (maxval 255) are supported");
  }
  if (!std::isspace(in.get())) {
    throw std::runtime_error(path.string() + ": malformed PNM header");
  }
  image.pixels.resize(static_cast<std::size_t>(image.width) * image.height * image.channels);
  in.read(reinterpret_cast<char*>(image.pixels.data()),
          static_cast<std::streamsize>(image.pixels.size()));
  if (static_cast<std::size_t>(in.gcount()) != image.pixels.size()) {
    throw std::runtime_error(path.string() + ": truncated pixel data");
  }
  return image;
}

void write_pnm(const std::filesystem::path& path, const PnmImage& image) {
  std::ofstream out(path, std::ios::binary);
  out << (image.channels == 1 ? "P5" : "P6") << '\n'
      << image.width << ' ' << image.height << "\n255\n";
  out.write(reinterpret_cast<const char*>(image.pixels.data()),
            static_cast<std::streamsize>(image.pixels.size()));
  if (!out) {
    throw std::runtime_error(path.string() + ": write failed");
  }
}

std::vector<std::filesystem::path> list_pnm_files(const std::filesystem::path& directory) {
  std::error_code error;
  std::vector<std::filesystem::path> files;
  for (const auto& entry : std::filesystem::directory_iterator(directory, error)) {
    const auto extension = entry.path().extension();
    if (entry.is_regular_file() && (extension == ".pgm" || extension == ".ppm")) {
      files.push_back(entry.path());
    }
  }
  if (error) {
    throw std::runtime_error(directory.string() + ": cannot list: " + error.message());
  }
  if (files.empty()) {
    throw std::runtime_error(directory.string() + ": no .pgm or .ppm images");
  }
  std::sort(files.begin(), files.end());
  return files;
}

}  // namespace tag_detection
