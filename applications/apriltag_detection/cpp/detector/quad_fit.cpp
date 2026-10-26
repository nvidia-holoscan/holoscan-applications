// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Quad fitting follows the method of the AprilTag 3 paper (Krogius, Haggenmiller
// and Olson, IROS 2019). The GPU orders each cluster's points by angle and finds
// the corner candidates, where a sliding line fit is worst; here the four
// candidates whose four edges fit best are chosen, and the edges intersected.

#include "quad_fit.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <vector>

namespace tag_detection {
namespace {

// Weighted line-fit sums: `w` is the total weight and `n` the point count.
struct Moments {
  double mx{}, my{}, mxx{}, mxy{}, myy{}, w{}, n{};

  static Moments of(double x, double y, double weight) noexcept {
    return {weight * x, weight * y, weight * x * x, weight * x * y, weight * y * y, weight, 1.0};
  }
  static Moments of(const LineMoments& m, double count) noexcept {
    return {m.mx, m.my, m.mxx, m.mxy, m.myy, m.w, count};
  }
  Moments& operator+=(const Moments& other) noexcept {
    mx += other.mx;
    my += other.my;
    mxx += other.mxx;
    mxy += other.mxy;
    myy += other.myy;
    w += other.w;
    n += other.n;
    return *this;
  }
  Moments operator-(const Moments& other) const noexcept {
    return {mx - other.mx,   my - other.my, mxx - other.mxx, mxy - other.mxy,
            myy - other.myy, w - other.w,   n - other.n};
  }
};

struct Line {
  Point2 point{};      // A point on the line: the weighted centroid of its samples.
  Point2 direction{};  // Unit direction.
  double error{};      // Mean squared perpendicular distance times the point count.
  double mse{};        // Weighted mean squared perpendicular distance.
};

// The best line through `m`'s points. The direction is the covariance's major
// eigenvector, computed only when the fit is within `max_mse`.
Line fit_line(const Moments& m, double max_mse = HUGE_VAL) {
  Line line;
  const double ex = m.mx / m.w;
  const double ey = m.my / m.w;
  const double cxx = m.mxx / m.w - ex * ex;
  const double cxy = m.mxy / m.w - ex * ey;
  const double cyy = m.myy / m.w - ey * ey;
  const double half_trace = 0.5 * (cxx + cyy);
  const double spread = std::sqrt(0.25 * (cxx - cyy) * (cxx - cyy) + cxy * cxy);
  const double smallest = std::max(0.0, half_trace - spread);
  line.point = {ex, ey};
  line.mse = smallest;
  line.error = smallest * m.n;
  if (smallest > max_mse) {
    return line;
  }
  // (cxy, largest - cxx) and (largest - cyy, cxy) both lie along the major
  // axis; the longer is the better conditioned.
  const double largest = half_trace + spread;
  const double ax = cxy;
  const double ay = largest - cxx;
  const double bx = largest - cyy;
  const double by = cxy;
  const double la = ax * ax + ay * ay;
  const double lb = bx * bx + by * by;
  if (la == 0.0 && lb == 0.0) {
    line.direction = {1.0, 0.0};  // Isotropic: any direction fits equally.
  } else if (la > lb) {
    const double length = std::sqrt(la);
    line.direction = {ax / length, ay / length};
  } else {
    const double length = std::sqrt(lb);
    line.direction = {bx / length, by / length};
  }
  return line;
}

std::optional<Point2> intersect(const Line& a, const Line& b) {
  const double det = a.direction[1] * b.direction[0] - a.direction[0] * b.direction[1];
  if (std::abs(det) < 1e-9) {
    return std::nullopt;
  }
  const double dx = b.point[0] - a.point[0];
  const double dy = b.point[1] - a.point[1];
  const double t = (dy * b.direction[0] - dx * b.direction[1]) / det;
  return Point2{a.point[0] + t * a.direction[0], a.point[1] + t * a.direction[1]};
}

double cross(const Point2& o, const Point2& a, const Point2& b) {
  return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
}

bool corners_acceptable(const std::array<Point2, 4>& corners, const QuadFitOptions& options) {
  double area = 0.0;
  for (int i = 0; i < 4; ++i) {
    const Point2& prev = corners[(i + 3) % 4];
    const Point2& here = corners[i];
    const Point2& next = corners[(i + 1) % 4];
    // Clockwise in y-down image coordinates means every turn is positive.
    if (cross(prev, here, next) <= 0.0) {
      return false;
    }
    const double ax = prev[0] - here[0];
    const double ay = prev[1] - here[1];
    const double bx = next[0] - here[0];
    const double by = next[1] - here[1];
    const double cosine =
        (ax * bx + ay * by) / std::sqrt((ax * ax + ay * ay) * (bx * bx + by * by));
    if (std::abs(cosine) > options.max_edge_cos) {
      return false;
    }
    area += here[0] * next[1] - next[0] * here[1];
  }
  // Corners are in decimated pixels here.
  return 0.5 * std::abs(area) >= 0.95 * options.min_tag_width * options.min_tag_width;
}

// The moments of the points from candidate `from` to candidate `to`, both
// included, wrapping past the cluster's end.
Moments segment(const ClusterSummary& cluster, const CornerCandidate& from,
                const CornerCandidate& to) {
  if (from.index <= to.index) {
    return Moments::of(to.through - from.before, to.index - from.index + 1.0);
  }
  return Moments::of((cluster.total - from.before) + to.through,
                     static_cast<double>(cluster.points) - from.index + to.index + 1.0);
}

std::optional<Quad> fit_cluster(const ClusterSummary& cluster, const CornerCandidate* candidates,
                                const QuadFitOptions& options) {
  const auto& box = cluster.box;
  if ((box[1] - box[0]) < 2.0 * options.min_tag_width ||
      (box[3] - box[2]) < 2.0 * options.min_tag_width) {
    return std::nullopt;
  }
  // Gradients point from dark to bright: outward across a normal border.
  const bool reversed = cluster.polarity < 0.0;
  if (reversed ? !options.reversed_border : !options.normal_border) {
    return std::nullopt;
  }
  const std::size_t m = cluster.candidates;
  if (m < 4) {
    return std::nullopt;
  }

  // The four candidates whose four edges fit best.
  double best_error = HUGE_VAL;
  std::array<std::size_t, 4> best{};
  for (std::size_t a = 0; a + 3 < m; ++a) {
    for (std::size_t b = a + 1; b + 2 < m; ++b) {
      const double max_mse = options.max_line_fit_mse;
      const Line l0 = fit_line(segment(cluster, candidates[a], candidates[b]), max_mse);
      if (l0.mse > max_mse) continue;
      for (std::size_t c = b + 1; c + 1 < m; ++c) {
        const Line l1 = fit_line(segment(cluster, candidates[b], candidates[c]), max_mse);
        if (l1.mse > max_mse) continue;
        const double dot01 = l0.direction[0] * l1.direction[0] + l0.direction[1] * l1.direction[1];
        if (std::abs(dot01) > options.max_edge_cos) continue;
        for (std::size_t d = c + 1; d < m; ++d) {
          // Only the error counts here; the direction is not needed.
          const Line l2 = fit_line(segment(cluster, candidates[c], candidates[d]), 0.0);
          if (l2.mse > max_mse) continue;
          const Line l3 = fit_line(segment(cluster, candidates[d], candidates[a]), 0.0);
          if (l3.mse > max_mse) continue;
          const double total = l0.error + l1.error + l2.error + l3.error;
          if (total < best_error) {
            best_error = total;
            best = {a, b, c, d};
          }
        }
      }
    }
  }
  if (best_error == HUGE_VAL) {
    return std::nullopt;
  }
  std::array<Line, 4> lines;
  for (int i = 0; i < 4; ++i) {
    lines[i] = fit_line(segment(cluster, candidates[best[i]], candidates[best[(i + 1) % 4]]));
  }
  std::array<Point2, 4> corners;
  for (int i = 0; i < 4; ++i) {
    const auto corner = intersect(lines[(i + 3) % 4], lines[i]);
    if (!corner) {
      return std::nullopt;
    }
    corners[i] = *corner;
  }
  if (!corners_acceptable(corners, options)) {
    return std::nullopt;
  }
  Quad quad;
  quad.reversed = reversed;
  for (int i = 0; i < 4; ++i) {
    // A decimated pixel k spans full-resolution [k * d, (k + 1) * d).
    quad.corners[i] = {corners[i][0] * options.decimate, corners[i][1] * options.decimate};
  }
  return quad;
}

}  // namespace

bool HostGray::sample(double x, double y, double& value) const noexcept {
  const double fx = x - 0.5;
  const double fy = y - 0.5;
  const double x0 = std::floor(fx);
  const double y0 = std::floor(fy);
  // Written so that a NaN coordinate fails too.
  if (!(x0 >= 0 && y0 >= 0 && x0 + 1 < width && y0 + 1 < height)) {
    return false;
  }
  const double ax = fx - x0;
  const double ay = fy - y0;
  const std::uint8_t* row0 = data + static_cast<std::size_t>(y0) * pitch;
  const std::uint8_t* row1 = row0 + pitch;
  const auto ix = static_cast<std::size_t>(x0);
  value = (1 - ax) * (1 - ay) * row0[ix] + ax * (1 - ay) * row0[ix + 1] + (1 - ax) * ay * row1[ix] +
          ax * ay * row1[ix + 1];
  return true;
}

std::vector<Quad> fit_quads(const QuadCandidates& candidates, const QuadFitOptions& options) {
  std::vector<Quad> quads;
  for (std::size_t i = 0; i < candidates.clusters.size(); ++i) {
    if (auto quad = fit_cluster(candidates.clusters[i],
                                candidates.corners.data() + i * kMaxCornerCandidates, options)) {
      quads.push_back(*quad);
    }
  }
  return quads;
}

void refine_edges(Quad& quad, const HostGray& image, int decimate) {
  Point2 centroid{0.0, 0.0};
  for (const auto& corner : quad.corners) {
    centroid[0] += 0.25 * corner[0];
    centroid[1] += 0.25 * corner[1];
  }
  std::array<Line, 4> lines;
  for (int edge = 0; edge < 4; ++edge) {
    const Point2& a = quad.corners[edge];
    const Point2& b = quad.corners[(edge + 1) % 4];
    const double length = std::hypot(b[0] - a[0], b[1] - a[1]);
    if (length < 1e-6) {
      return;
    }
    double nx = (b[1] - a[1]) / length;
    double ny = -(b[0] - a[0]) / length;
    // Point the normal at the bright side: outward for a normal border.
    const double mid_x = 0.5 * (a[0] + b[0]) - centroid[0];
    const double mid_y = 0.5 * (a[1] + b[1]) - centroid[1];
    if ((nx * mid_x + ny * mid_y < 0.0) != quad.reversed) {
      nx = -nx;
      ny = -ny;
    }
    const int samples = std::max(16, static_cast<int>(length / 8.0));
    const int range = decimate + 1;
    constexpr int kStepsPerPixel = 4;
    Moments fit;
    for (int s = 0; s < samples; ++s) {
      // Skip the ends: corners are the least reliable part of an edge.
      const double alpha = (1.0 + s) / (samples + 1);
      const double x0 = alpha * a[0] + (1 - alpha) * b[0];
      const double y0 = alpha * a[1] + (1 - alpha) * b[1];
      double weighted = 0.0;
      double total = 0.0;
      for (int step = 0; step <= 2 * kStepsPerPixel * range; ++step) {
        const double offset = -range + static_cast<double>(step) / kStepsPerPixel;
        double bright = 0.0;
        double dark = 0.0;
        if (!image.sample(x0 + (offset + 1.0) * nx, y0 + (offset + 1.0) * ny, bright) ||
            !image.sample(x0 + (offset - 1.0) * nx, y0 + (offset - 1.0) * ny, dark) ||
            bright < dark) {
          continue;
        }
        const double weight = (bright - dark) * (bright - dark);
        weighted += weight * offset;
        total += weight;
      }
      if (total == 0.0) {
        continue;
      }
      const double offset = weighted / total;
      const double x = x0 + offset * nx;
      const double y = y0 + offset * ny;
      fit += Moments::of(x, y, 1.0);
    }
    if (fit.n < 2) {
      return;  // Leave the quad as fitted.
    }
    lines[edge] = fit_line(fit);
  }
  std::array<Point2, 4> refined;
  for (int i = 0; i < 4; ++i) {
    const auto corner = intersect(lines[(i + 3) % 4], lines[i]);
    if (!corner) {
      return;
    }
    refined[i] = *corner;
  }
  quad.corners = refined;
}

}  // namespace tag_detection
