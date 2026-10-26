// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// The host half of the CUDA backend's geometry: fit a quadrilateral to each edge
// cluster, then refine its edges on the full-resolution image.
//
// Coordinates here follow the AprilTag 3 convention, pixel centers at +0.5, so
// sampling matches the reference decoder; the backend converts when it reports.

#pragma once

#include <array>
#include <vector>

#include "gpu_segmenter.hpp"
#include "pose.hpp"

namespace tag_detection {

struct Quad {
  // Clockwise as seen in the image (y down), full-resolution pixels.
  std::array<Point2, 4> corners{};
  // True when the quad's inside is brighter than its outside.
  bool reversed{};
};

struct QuadFitOptions {
  int decimate{1};
  // Which polarities the family's border can have.
  bool normal_border{true};
  bool reversed_border{false};
  // Smallest tag side, in decimated pixels, worth fitting.
  double min_tag_width{3.0};
  // Adjacent edges whose directions' |cos| exceeds this are rejected (10 deg).
  double max_edge_cos{0.984807753};
  // Largest mean squared line-fit error per edge, in doubled decimated pixels.
  double max_line_fit_mse{10.0};
};

// A host gray image, for sampling.
struct HostGray {
  const std::uint8_t* data{};
  int width{};
  int height{};
  std::size_t pitch{};

  // Bilinear sample at (x, y) with pixel centers at +0.5; false outside.
  [[nodiscard]] bool sample(double x, double y, double& value) const noexcept;
};

[[nodiscard]] std::vector<Quad> fit_quads(const QuadCandidates& candidates,
                                          const QuadFitOptions& options);

// Moves each edge of `quad` onto the strongest nearby intensity step of `image`
// (full resolution), searching up to `decimate + 1` pixels to either side.
void refine_edges(Quad& quad, const HostGray& image, int decimate);

}  // namespace tag_detection
