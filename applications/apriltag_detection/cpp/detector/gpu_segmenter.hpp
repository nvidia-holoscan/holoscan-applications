// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// The GPU half of the CUDA backend. It decimates the image, thresholds it
// adaptively, labels connected components, and gathers the boundary points
// between each pair of adjacent black and white components (a cluster, a
// candidate tag border). It orders each cluster's points by angle, accumulates
// their line-fit moments, scores how badly a line fits around each point, and
// keeps each cluster's best corner candidates. The host (quad_fit.hpp) then
// only chooses four corners per cluster and fits the edges between them.

#pragma once

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "detection.hpp"

#if defined(__CUDACC__)
#define TAG_DETECTION_HOST_DEVICE __host__ __device__
#else
#define TAG_DETECTION_HOST_DEVICE
#endif

namespace tag_detection {

// One boundary point between a black and a white pixel of the decimated image,
// at twice the decimated resolution: (x, y) is the midpoint of the two pixel
// centers times two. (gx, gy) points from the dark pixel to the bright one, and
// `weight` is the gray image's gradient magnitude there, plus one.
struct EdgePoint {
  std::int16_t x;
  std::int16_t y;
  std::int8_t gx;
  std::int8_t gy;
  std::uint16_t weight;
};
static_assert(sizeof(EdgePoint) == sizeof(std::uint64_t));

// Weighted sums for fitting a line to points: x and y in decimated pixels with
// centers at +0.5, the units the fit thresholds are given in.
struct LineMoments {
  double mx{}, my{}, mxx{}, mxy{}, myy{}, w{};

  TAG_DETECTION_HOST_DEVICE LineMoments operator+(const LineMoments& o) const {
    return {mx + o.mx, my + o.my, mxx + o.mxx, mxy + o.mxy, myy + o.myy, w + o.w};
  }
  TAG_DETECTION_HOST_DEVICE LineMoments operator-(const LineMoments& o) const {
    return {mx - o.mx, my - o.my, mxx - o.mxx, mxy - o.mxy, myy - o.myy, w - o.w};
  }
};

// Corner candidates kept per cluster, as in AprilTag 3.
inline constexpr int kMaxCornerCandidates = 10;

// A point of a cluster where a line fits its neighborhood worst.
struct CornerCandidate {
  std::uint32_t index{};  // Into the cluster's angle-ordered, duplicate-free points.
  double error{};         // The smoothed line-fit error there.
  LineMoments through{};  // Moments of the cluster's points 0 .. index.
  LineMoments before{};   // Moments of the cluster's points 0 .. index - 1.
};

struct ClusterSummary {
  std::uint32_t raw_points{};  // Boundary points before duplicates were removed.
  std::uint32_t points{};      // Angle-ordered points without duplicates.
  std::uint32_t candidates{};  // Corner candidates kept, at most kMaxCornerCandidates.
  // Bounding box in EdgePoint coordinates: x min, x max, y min, y max.
  std::int32_t box[4]{};
  // Sum over the points of (point - center) . gradient: positive when the
  // inside of the cluster is darker than its outside.
  double polarity{};
  LineMoments total{};
};

// Every cluster's summary, and its candidates in index order at
// corners[c * kMaxCornerCandidates] .. + clusters[c].candidates.
struct QuadCandidates {
  std::vector<ClusterSummary> clusters;
  std::vector<CornerCandidate> corners;
  // The decimated image size.
  int width{};
  int height{};
};

// The point each cluster's points are ordered about: its bounding box's center,
// nudged so that no point lies exactly on it. In EdgePoint coordinates.
inline constexpr float kCenterNudgeX = 0.05118F;
inline constexpr float kCenterNudgeY = -0.028581F;

class GpuSegmenter {
 public:
  // @throws std::invalid_argument for a size or option out of range.
  // @throws std::runtime_error if a device allocation fails.
  GpuSegmenter(int width, int height, int decimate, int min_white_black_diff);
  ~GpuSegmenter();
  GpuSegmenter(const GpuSegmenter&) = delete;
  GpuSegmenter& operator=(const GpuSegmenter&) = delete;

  // Segments `image` (device memory, the configured size) on `stream` and
  // returns each cluster's corner candidates on the host. Synchronizes `stream`.
  // @throws std::runtime_error on a CUDA failure.
  [[nodiscard]] QuadCandidates segment(const GrayImageView& image, cudaStream_t stream);

 private:
  struct Buffers;

  int width_;
  int height_;
  int decimate_;
  int min_white_black_diff_;
  int dec_width_;
  int dec_height_;
  int tiles_x_;
  int tiles_y_;
  std::size_t capacity_;  // Most edge points, and so clusters, a frame can produce.
  std::unique_ptr<Buffers> buffers_;
};

}  // namespace tag_detection
