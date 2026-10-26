// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "tag_decoder.hpp"

#include <apriltag.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tag_detection {
namespace {

// A plane through gray samples, value = a * x + b * y + c in tag coordinates,
// so the black/white threshold can follow lighting across the tag.
class GrayModel {
 public:
  void add(double x, double y, double value) noexcept {
    const std::array<double, 3> row{x, y, 1.0};
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        normal_[i][j] += row[i] * row[j];
      }
      rhs_[i] += row[i] * value;
    }
  }

  // Least squares; with too few independent samples, the mean only.
  void solve() noexcept {
    std::array<std::array<double, 4>, 3> m{};
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        m[i][j] = normal_[i][j];
      }
      m[i][3] = rhs_[i];
    }
    for (int col = 0; col < 3; ++col) {
      int pivot = col;
      for (int row = col + 1; row < 3; ++row) {
        if (std::abs(m[row][col]) > std::abs(m[pivot][col])) pivot = row;
      }
      if (std::abs(m[pivot][col]) < 1e-9) {
        coefficients_ = {0.0, 0.0, normal_[2][2] > 0.0 ? rhs_[2] / normal_[2][2] : 0.0};
        return;
      }
      std::swap(m[col], m[pivot]);
      for (int row = 0; row < 3; ++row) {
        if (row == col) continue;
        const double factor = m[row][col] / m[col][col];
        for (int k = col; k < 4; ++k) m[row][k] -= factor * m[col][k];
      }
    }
    for (int i = 0; i < 3; ++i) coefficients_[i] = m[i][3] / m[i][i];
  }

  [[nodiscard]] double at(double x, double y) const noexcept {
    return coefficients_[0] * x + coefficients_[1] * y + coefficients_[2];
  }

 private:
  std::array<std::array<double, 3>, 3> normal_{};
  std::array<double, 3> rhs_{};
  std::array<double, 3> coefficients_{};
};

std::uint64_t mix(std::uint64_t code) noexcept {
  code ^= code >> 33;
  code *= 0xff51afd7ed558ccdULL;
  code ^= code >> 33;
  return code;
}

}  // namespace

TagDecoder::TagDecoder(const TagFamily& family, int max_hamming, double decode_sharpening)
    : decode_sharpening_(decode_sharpening) {
  if (max_hamming < 0 || max_hamming > 2) {
    throw std::invalid_argument("max_hamming must be in 0..2");
  }
  const apriltag_family_t* f = family.get();
  nbits_ = static_cast<int>(f->nbits);
  width_at_border_ = f->width_at_border;
  total_width_ = f->total_width;
  reversed_border_ = f->reversed_border;
  bit_x_.assign(f->bit_x, f->bit_x + f->nbits);
  bit_y_.assign(f->bit_y, f->bit_y + f->nbits);

  // A quarter turn about the tag center, (u, v) -> (-v, u) in bit cells.
  rotation_.assign(static_cast<std::size_t>(nbits_), -1);
  const double center = (width_at_border_ - 1) / 2.0;
  for (int i = 0; i < nbits_; ++i) {
    const double u = bit_x_[i] - center;
    const double v = bit_y_[i] - center;
    const double rx = center - v;
    const double ry = center + u;
    for (int j = 0; j < nbits_; ++j) {
      if (std::abs(bit_x_[j] - rx) < 1e-6 && std::abs(bit_y_[j] - ry) < 1e-6) {
        rotation_[i] = j;
      }
    }
    if (rotation_[i] < 0) {
      throw std::runtime_error("tag family layout is not symmetric under rotation");
    }
  }

  std::size_t entries = f->ncodes;
  if (max_hamming >= 1) entries += static_cast<std::size_t>(f->ncodes) * nbits_;
  if (max_hamming >= 2) entries += static_cast<std::size_t>(f->ncodes) * nbits_ * (nbits_ - 1) / 2;
  table_.resize(std::bit_ceil(entries * 2));
  for (std::uint32_t id = 0; id < f->ncodes; ++id) {
    const std::uint64_t code = f->codes[id];
    insert(code, static_cast<int>(id), 0);
    if (max_hamming >= 1) {
      for (int a = 0; a < nbits_; ++a) {
        const std::uint64_t one = code ^ (std::uint64_t{1} << a);
        insert(one, static_cast<int>(id), 1);
        if (max_hamming >= 2) {
          for (int b = 0; b < a; ++b) {
            insert(one ^ (std::uint64_t{1} << b), static_cast<int>(id), 2);
          }
        }
      }
    }
  }
}

void TagDecoder::insert(std::uint64_t code, int id, int hamming) {
  const std::size_t mask = table_.size() - 1;
  for (std::size_t slot = mix(code) & mask;; slot = (slot + 1) & mask) {
    Entry& entry = table_[slot];
    if (entry.id < 0) {
      entry = {code, id, hamming};
      return;
    }
    if (entry.code == code) {
      // Two codes within reach of one word: keep the closer one.
      if (hamming < entry.hamming) entry = {code, id, hamming};
      return;
    }
  }
}

const TagDecoder::Entry* TagDecoder::find(std::uint64_t code) const noexcept {
  const std::size_t mask = table_.size() - 1;
  for (std::size_t slot = mix(code) & mask;; slot = (slot + 1) & mask) {
    const Entry& entry = table_[slot];
    if (entry.id < 0) return nullptr;
    if (entry.code == code) return &entry;
  }
}

std::uint64_t TagDecoder::rotate(std::uint64_t code) const noexcept {
  std::uint64_t rotated = 0;
  for (int i = 0; i < nbits_; ++i) {
    if ((code >> (nbits_ - 1 - i)) & 1U) {
      rotated |= std::uint64_t{1} << (nbits_ - 1 - rotation_[i]);
    }
  }
  return rotated;
}

std::optional<Detection> TagDecoder::decode(const Quad& quad, const HostGray& image) const {
  const auto homography =
      homography_from_corners({quad.corners[3], quad.corners[2], quad.corners[1], quad.corners[0]});
  // homography_from_corners maps (-1, 1), (1, 1), (1, -1), (-1, -1) in order.
  // The quad's corners run clockwise in the image, as tag (-1, -1), (1, -1),
  // (1, 1), (-1, 1) do, so this sends tag (-1, -1) to corner 0 without a mirror.
  if (!homography) {
    return std::nullopt;
  }
  const Homography& h = *homography;
  const double border = width_at_border_;
  const auto to_tag = [border](double cell) { return 2.0 * ((cell + 0.5) / border - 0.5); };

  // Sample the known white ring outside the border and black ring inside it.
  GrayModel white;
  GrayModel black;
  struct Pattern {
    double x0, y0, dx, dy;
    bool white;
  };
  const std::array<Pattern, 8> patterns{{
      {-1.0, 0.0, 0.0, 1.0, true},
      {0.0, 0.0, 0.0, 1.0, false},
      {border, 0.0, 0.0, 1.0, true},
      {border - 1.0, 0.0, 0.0, 1.0, false},
      {0.0, -1.0, 1.0, 0.0, true},
      {0.0, 0.0, 1.0, 0.0, false},
      {0.0, border, 1.0, 0.0, true},
      {0.0, border - 1.0, 1.0, 0.0, false},
  }};
  for (const Pattern& pattern : patterns) {
    for (int i = 0; i < width_at_border_; ++i) {
      const double tx = to_tag(pattern.x0 + i * pattern.dx);
      const double ty = to_tag(pattern.y0 + i * pattern.dy);
      const Point2 p = project(h, tx, ty);
      const double ix = std::floor(p[0]);
      const double iy = std::floor(p[1]);
      if (!(ix >= 0 && iy >= 0 && ix < image.width && iy < image.height)) continue;
      const double value =
          image.data[static_cast<std::size_t>(iy) * image.pitch + static_cast<std::size_t>(ix)];
      (pattern.white ? white : black).add(tx, ty, value);
    }
  }
  white.solve();
  black.solve();
  if ((white.at(0, 0) - black.at(0, 0) < 0) != reversed_border_) {
    return std::nullopt;
  }

  // Bit values relative to the local threshold, on the family's full grid so
  // they can be sharpened with their neighbors.
  const int size = total_width_;
  const int min_coord = (width_at_border_ - total_width_) / 2;
  std::vector<double> values(static_cast<std::size_t>(size * size), 0.0);
  for (int i = 0; i < nbits_; ++i) {
    const double tx = to_tag(bit_x_[i]);
    const double ty = to_tag(bit_y_[i]);
    const Point2 p = project(h, tx, ty);
    double value = 0.0;
    if (!image.sample(p[0], p[1], value)) continue;
    const double threshold = 0.5 * (black.at(tx, ty) + white.at(tx, ty));
    values[static_cast<std::size_t>((bit_y_[i] - min_coord) * size + bit_x_[i] - min_coord)] =
        value - threshold;
  }
  if (decode_sharpening_ != 0.0) {
    std::vector<double> laplacian(values.size(), 0.0);
    for (int y = 0; y < size; ++y) {
      for (int x = 0; x < size; ++x) {
        double sum = 4.0 * values[y * size + x];
        if (x > 0) sum -= values[y * size + x - 1];
        if (x + 1 < size) sum -= values[y * size + x + 1];
        if (y > 0) sum -= values[(y - 1) * size + x];
        if (y + 1 < size) sum -= values[(y + 1) * size + x];
        laplacian[y * size + x] = sum;
      }
    }
    for (std::size_t i = 0; i < values.size(); ++i) values[i] += decode_sharpening_ * laplacian[i];
  }

  std::uint64_t code = 0;
  double white_score = 0.0, black_score = 0.0;
  double white_count = 1.0, black_count = 1.0;
  for (int i = 0; i < nbits_; ++i) {
    const double v =
        values[static_cast<std::size_t>((bit_y_[i] - min_coord) * size + bit_x_[i] - min_coord)];
    code <<= 1U;
    if (v > 0) {
      white_score += v;
      white_count += 1.0;
      code |= 1U;
    } else {
      black_score -= v;
      black_count += 1.0;
    }
  }
  const double margin = std::min(white_score / white_count, black_score / black_count);
  if (margin < 0.0) {
    return std::nullopt;
  }

  for (int turns = 0; turns < 4; ++turns) {
    if (const Entry* entry = find(code)) {
      // The upright tag is the sampled one turned `turns` quarter turns, so its
      // tag coordinates map through the inverse turn first.
      Homography upright = h;
      for (int t = 0; t < turns; ++t) {
        // h * [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]: (u, v) -> (v, -u).
        const Homography prev = upright;
        for (int row = 0; row < 3; ++row) {
          upright[row * 3 + 0] = -prev[row * 3 + 1];
          upright[row * 3 + 1] = prev[row * 3 + 0];
        }
      }
      Detection detection;
      detection.id = entry->id;
      detection.hamming = entry->hamming;
      detection.decision_margin = static_cast<float>(margin);
      constexpr std::array<Point2, 4> kIdeal{{{-1.0, 1.0}, {1.0, 1.0}, {1.0, -1.0}, {-1.0, -1.0}}};
      for (int i = 0; i < 4; ++i) {
        detection.corners[i] = project(upright, kIdeal[i][0], kIdeal[i][1]);
      }
      detection.center = project(upright, 0.0, 0.0);
      return detection;
    }
    code = rotate(code);
  }
  return std::nullopt;
}

}  // namespace tag_detection
