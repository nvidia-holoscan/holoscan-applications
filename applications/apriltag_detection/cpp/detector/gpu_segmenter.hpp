// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// The first half of the CUDA backend. It decimates the image, thresholds it
// adaptively, labels connected components, and gathers the boundary points
// between each pair of adjacent black and white components (a cluster, a
// candidate tag border). It orders each cluster's points by angle, accumulates
// their line-fit moments, and scores how badly a line fits around each point;
// the local maxima of that score are the cluster's corner candidates.
//
// Every buffer has a fixed capacity and every count stays on the device, so a
// frame needs no host synchronization.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include "detection.hpp"
#include "homography.hpp"  // TAG_DETECTION_HOST_DEVICE

namespace tag_detection {

inline void cuda_check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

// `count` elements of device memory, released on destruction.
template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t count) {
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&data_), (count > 0 ? count : 1) * sizeof(T)),
               "cudaMalloc");
  }
  ~DeviceBuffer() { cudaFree(data_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  [[nodiscard]] T* get() const noexcept { return data_; }

 private:
  T* data_{};
};

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

// Counts the segmenter leaves on the device for each frame.
struct SegmentCounts {
  std::uint32_t points;  // Boundary points found; those past the capacity are dropped.
  std::uint32_t runs;    // Point runs after sorting, including the padding run.
  std::uint32_t unique;  // Points left after repeated points are removed.
  std::uint32_t maxima;  // Corner candidates over all clusters.
};

// The segmenter's results, in device memory, for cluster c < clusters():
// its points offsets[c] .. offsets[c] + point_counts[c] - 1 in angular order, and its
// corner candidates maxima[maxima_offsets[c]] .. + maxima_counts[c] - 1 (point
// indices, in order).
struct ClusterData {
  const SegmentCounts* counts;
  std::uint32_t capacity;
  const std::uint32_t* raw_points;  // Boundary points before repeats were removed.
  const std::uint32_t* offsets;
  const std::uint32_t* point_counts;
  const std::int32_t* box;  // Per cluster in doubled decimated pixels: x min, x max, y min, y max.
  // Sum over the points of (point - center) . gradient: positive when the
  // inside of the cluster is darker than its outside.
  const double* polarity;
  const LineMoments* prefix;  // Inclusive per-cluster prefix sums of the points' moments.
  const double* smoothed;     // The smoothed line-fit error at each point.
  const std::uint32_t* maxima;
  const std::uint32_t* maxima_offsets;
  const std::uint32_t* maxima_counts;

#if defined(__CUDACC__)
  // Clusters in this frame: the runs, less the run of padding when there is one.
  __device__ std::uint32_t clusters() const {
    const std::uint32_t found = counts->points < capacity ? counts->points : capacity;
    return counts->runs - (found < capacity ? 1U : 0U);
  }
#endif
};

class GpuSegmenter {
 public:
  // @throws std::invalid_argument for a size or option out of range.
  // @throws std::runtime_error if a device allocation fails.
  GpuSegmenter(int width, int height, int decimate, int min_white_black_diff);
  ~GpuSegmenter();
  GpuSegmenter(const GpuSegmenter&) = delete;
  GpuSegmenter& operator=(const GpuSegmenter&) = delete;

  // Enqueues the segmentation of `image` (device memory, the configured size)
  // on `stream`; the results are in data() once it has run.
  // @throws std::runtime_error on a CUDA failure.
  void segment(const GrayImageView& image, cudaStream_t stream);

  [[nodiscard]] ClusterData data() const noexcept;
  // Most boundary points, and so clusters, a frame can hold.
  [[nodiscard]] std::uint32_t capacity() const noexcept { return capacity_; }
  // Longest cluster that can still be a tag border inside the image.
  [[nodiscard]] std::uint32_t max_cluster_points() const noexcept {
    return static_cast<std::uint32_t>(3 * (2 * dec_width_ + 2 * dec_height_));
  }

 private:
  struct Buffers;
  // Runs CUB call `stage` over the whole capacity; with `temp` null it only
  // sizes its scratch.
  void run_cub(int stage, void* temp, std::size_t& bytes, cudaStream_t stream);

  int width_;
  int height_;
  int decimate_;
  int min_white_black_diff_;
  int dec_width_;
  int dec_height_;
  int tiles_x_;
  int tiles_y_;
  std::uint32_t capacity_;
  std::unique_ptr<Buffers> buffers_;
};

}  // namespace tag_detection
