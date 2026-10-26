// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Reads a fitted quad's bits and matches them against a tag family, for the
// CUDA backend. Coordinates follow quad_fit.hpp (pixel centers at +0.5).

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "detection.hpp"
#include "quad_fit.hpp"
#include "tag_family.hpp"

namespace tag_detection {

class TagDecoder {
 public:
  // @throws std::invalid_argument if `max_hamming` is not in 0..2.
  TagDecoder(const TagFamily& family, int max_hamming, double decode_sharpening);

  // The tag `quad` frames in `image`, or nullopt if it decodes to no code within
  // `max_hamming` bit errors. Corners and center use quad_fit.hpp's coordinates.
  [[nodiscard]] std::optional<Detection> decode(const Quad& quad, const HostGray& image) const;

  [[nodiscard]] bool normal_border() const noexcept { return !reversed_border_; }
  [[nodiscard]] bool reversed_border() const noexcept { return reversed_border_; }
  [[nodiscard]] int width_at_border() const noexcept { return width_at_border_; }

 private:
  struct Entry {
    std::uint64_t code{};
    std::int32_t id{-1};  // -1 marks an empty slot.
    std::int32_t hamming{};
  };

  void insert(std::uint64_t code, int id, int hamming);
  [[nodiscard]] const Entry* find(std::uint64_t code) const noexcept;
  [[nodiscard]] std::uint64_t rotate(std::uint64_t code) const noexcept;

  int nbits_{};
  int width_at_border_{};
  int total_width_{};
  bool reversed_border_{};
  double decode_sharpening_{};
  std::vector<int> bit_x_;
  std::vector<int> bit_y_;
  // rotation_[i]: the bit a quarter turn moves bit i to.
  std::vector<int> rotation_;
  std::vector<Entry> table_;  // Open addressing; size is a power of two.
};

}  // namespace tag_detection
