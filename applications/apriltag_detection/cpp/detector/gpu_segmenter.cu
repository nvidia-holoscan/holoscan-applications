// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// CUB's headers are C++; cpplint takes any angle-bracketed non-.hpp header for
// a C system header.
#include <cub/device/device_radix_sort.cuh>         // NOLINT(build/include_order)
#include <cub/device/device_run_length_encode.cuh>  // NOLINT(build/include_order)
#include <cub/device/device_scan.cuh>               // NOLINT(build/include_order)
#include <cub/device/device_select.cuh>             // NOLINT(build/include_order)

#include "device_vector.hpp"
#include "gpu_segmenter.hpp"

namespace tag_detection {
namespace {

// Threshold tiles are this many decimated pixels on a side.
constexpr int kTileSize = 4;
// Pixels marked neither black nor white.
constexpr std::uint8_t kUnknown = 127;
// Components smaller than this cannot border a tag and are skipped.
constexpr std::uint32_t kMinComponentSize = 25;
constexpr int kBlock = 256;

void check(cudaError_t status, std::string_view what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

unsigned int blocks_for(std::size_t items) {
  return static_cast<unsigned int>((items + kBlock - 1) / kBlock);
}

// Box-filters `decimate` x `decimate` blocks of the input into one pixel.
__global__ void decimate_kernel(const std::uint8_t* input, std::size_t pitch, int decimate,
                                std::uint8_t* output, int out_width, int out_height) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= out_width * out_height) {
    return;
  }
  const int x = index % out_width;
  const int y = index / out_width;
  unsigned int sum = 0;
  for (int dy = 0; dy < decimate; ++dy) {
    const std::uint8_t* row = input + static_cast<std::size_t>(y * decimate + dy) * pitch;
    for (int dx = 0; dx < decimate; ++dx) {
      sum += row[x * decimate + dx];
    }
  }
  const unsigned int area = static_cast<unsigned int>(decimate * decimate);
  output[index] = static_cast<std::uint8_t>((sum + area / 2) / area);
}

__global__ void tile_minmax_kernel(const std::uint8_t* image, int width, int height, int tiles_x,
                                   int tiles_y, std::uint8_t* tile_min, std::uint8_t* tile_max) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= tiles_x * tiles_y) {
    return;
  }
  const int tx = index % tiles_x;
  const int ty = index / tiles_x;
  std::uint8_t lo = 255;
  std::uint8_t hi = 0;
  for (int dy = 0; dy < kTileSize; ++dy) {
    const int y = ty * kTileSize + dy;
    if (y >= height) {
      break;
    }
    for (int dx = 0; dx < kTileSize; ++dx) {
      const int x = tx * kTileSize + dx;
      if (x >= width) {
        break;
      }
      const std::uint8_t v = image[y * width + x];
      lo = min(lo, v);
      hi = max(hi, v);
    }
  }
  tile_min[index] = lo;
  tile_max[index] = hi;
}

// Widens each tile's extrema to its 3x3 tile neighborhood, so a tag edge on a
// tile boundary sees both of its sides.
__global__ void tile_dilate_kernel(const std::uint8_t* tile_min, const std::uint8_t* tile_max,
                                   int tiles_x, int tiles_y, std::uint8_t* out_min,
                                   std::uint8_t* out_max) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= tiles_x * tiles_y) {
    return;
  }
  const int tx = index % tiles_x;
  const int ty = index / tiles_x;
  std::uint8_t lo = 255;
  std::uint8_t hi = 0;
  for (int dy = -1; dy <= 1; ++dy) {
    const int y = ty + dy;
    if (y < 0 || y >= tiles_y) {
      continue;
    }
    for (int dx = -1; dx <= 1; ++dx) {
      const int x = tx + dx;
      if (x < 0 || x >= tiles_x) {
        continue;
      }
      lo = min(lo, tile_min[y * tiles_x + x]);
      hi = max(hi, tile_max[y * tiles_x + x]);
    }
  }
  out_min[index] = lo;
  out_max[index] = hi;
}

// Marks each pixel black (0), white (255), or unknown where its tile
// neighborhood has too little contrast. Labels every known pixel as its own
// component.
__global__ void threshold_kernel(const std::uint8_t* image, int width, int height,
                                 const std::uint8_t* tile_min, const std::uint8_t* tile_max,
                                 int tiles_x, int min_white_black_diff, std::uint8_t* output,
                                 std::int32_t* labels) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= width * height) {
    return;
  }
  const int x = index % width;
  const int y = index / width;
  const int tile = (y / kTileSize) * tiles_x + x / kTileSize;
  const int lo = tile_min[tile];
  const int hi = tile_max[tile];
  std::uint8_t value = kUnknown;
  if (hi - lo >= min_white_black_diff) {
    value = image[index] > lo + (hi - lo) / 2 ? 255 : 0;
  }
  output[index] = value;
  labels[index] = value == kUnknown ? -1 : index;
}

__device__ int find_root(const std::int32_t* labels, int node) {
  const volatile std::int32_t* parents = labels;
  int parent = parents[node];
  while (parent != node) {
    node = parent;
    parent = parents[node];
  }
  return node;
}

// Lock-free union: the larger root is pointed at the smaller with atomicMin,
// retrying from the new root if another thread moved it first.
__device__ void unite(std::int32_t* labels, int a, int b) {
  while (true) {
    a = find_root(labels, a);
    b = find_root(labels, b);
    if (a == b) {
      return;
    }
    if (a > b) {
      const int swap = a;
      a = b;
      b = swap;
    }
    const int previous = atomicMin(&labels[b], a);
    if (previous == b) {
      return;
    }
    b = previous;
  }
}

// Black pixels join their 4-neighbors and white pixels their 8-neighbors, so a
// thin black border is not split by a diagonal white bridge.
__global__ void merge_kernel(const std::uint8_t* image, int width, int height,
                             std::int32_t* labels) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= width * height) {
    return;
  }
  const std::uint8_t value = image[index];
  if (value == kUnknown) {
    return;
  }
  const int x = index % width;
  const int y = index / width;
  if (x > 0 && image[index - 1] == value) {
    unite(labels, index, index - 1);
  }
  if (y > 0) {
    const int up = index - width;
    if (image[up] == value) {
      unite(labels, index, up);
    }
    if (value == 255) {
      if (x > 0 && image[up - 1] == value) {
        unite(labels, index, up - 1);
      }
      if (x + 1 < width && image[up + 1] == value) {
        unite(labels, index, up + 1);
      }
    }
  }
}

__global__ void flatten_kernel(int count, std::int32_t* labels, std::uint32_t* sizes) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count || labels[index] < 0) {
    return;
  }
  const int root = find_root(labels, index);
  labels[index] = root;
  atomicAdd(&sizes[root], 1U);
}

__device__ std::uint64_t pack_point(int x, int y, int gx, int gy, int weight) {
  const EdgePoint point{static_cast<std::int16_t>(x), static_cast<std::int16_t>(y),
                        static_cast<std::int8_t>(gx), static_cast<std::int8_t>(gy),
                        static_cast<std::uint16_t>(weight)};
  std::uint64_t packed;
  memcpy(&packed, &point, sizeof(packed));
  return packed;
}

// Emits a point between each pixel and those of its right, lower and two lower
// diagonal neighbors with the opposite color, keyed by the pair of components.
__global__ void edge_points_kernel(const std::uint8_t* image, const std::uint8_t* gray, int width,
                                   int height, const std::int32_t* labels,
                                   const std::uint32_t* sizes, std::uint64_t* keys,
                                   std::uint64_t* points, std::uint32_t* counter,
                                   std::uint32_t capacity) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= width * height) {
    return;
  }
  const std::uint8_t v0 = image[index];
  if (v0 == kUnknown) {
    return;
  }
  const int root0 = labels[index];
  if (sizes[root0] < kMinComponentSize) {
    return;
  }
  const int x = index % width;
  const int y = index / width;
  constexpr int kOffsets[4][2] = {{1, 0}, {0, 1}, {-1, 1}, {1, 1}};
  for (const auto& offset : kOffsets) {
    const int x1 = x + offset[0];
    const int y1 = y + offset[1];
    if (x1 < 0 || x1 >= width || y1 >= height) {
      continue;
    }
    const int neighbor = y1 * width + x1;
    const std::uint8_t v1 = image[neighbor];
    if (static_cast<int>(v0) + static_cast<int>(v1) != 255) {
      continue;
    }
    const int root1 = labels[neighbor];
    if (sizes[root1] < kMinComponentSize) {
      continue;
    }
    // Past the buffer's end the point is only counted, so the caller can grow
    // the buffers to the true total and gather again.
    const std::uint32_t slot = atomicAdd(counter, 1U);
    if (slot >= capacity) {
      continue;
    }
    const auto lo = static_cast<std::uint64_t>(min(root0, root1));
    const auto hi = static_cast<std::uint64_t>(max(root0, root1));
    keys[slot] = (lo << 32) | hi;
    const int sign = v1 > v0 ? 1 : -1;
    // Weight by the gray gradient at the pixel the point rounds to, so points
    // on strong edges dominate the line fits.
    const int px = 2 * x + offset[0];
    const int py = 2 * y + offset[1];
    const int ix = (px + 1) / 2;
    const int iy = (py + 1) / 2;
    int weight = 1;
    if (ix > 0 && ix + 1 < width && iy > 0 && iy + 1 < height) {
      const int dx = gray[iy * width + ix + 1] - gray[iy * width + ix - 1];
      const int dy = gray[(iy + 1) * width + ix] - gray[(iy - 1) * width + ix];
      weight = static_cast<int>(sqrtf(static_cast<float>(dx * dx + dy * dy))) + 1;
    }
    points[slot] = pack_point(px, py, offset[0] * sign, offset[1] * sign, weight);
  }
}

__global__ void reset_boxes_kernel(int clusters, std::int32_t* box) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= clusters) {
    return;
  }
  box[4 * index + 0] = INT32_MAX;
  box[4 * index + 1] = INT32_MIN;
  box[4 * index + 2] = INT32_MAX;
  box[4 * index + 3] = INT32_MIN;
}

__device__ EdgePoint unpack_point(std::uint64_t packed) {
  EdgePoint point;
  memcpy(&point, &packed, sizeof(point));
  return point;
}

// Finds each point's cluster (by binary search over the run offsets) and grows
// that cluster's bounding box.
__global__ void cluster_boxes_kernel(const std::uint64_t* points, int count,
                                     const std::uint32_t* offsets, int clusters,
                                     std::uint32_t* point_cluster, std::int32_t* box) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  int lo = 0;
  int hi = clusters - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) / 2;
    if (offsets[mid] <= static_cast<std::uint32_t>(index)) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  point_cluster[index] = static_cast<std::uint32_t>(lo);
  const EdgePoint point = unpack_point(points[index]);
  atomicMin(&box[4 * lo + 0], point.x);
  atomicMax(&box[4 * lo + 1], point.x);
  atomicMin(&box[4 * lo + 2], point.y);
  atomicMax(&box[4 * lo + 3], point.y);
}

// Keys each point by its cluster, then by its angle about the cluster's center,
// so one sort leaves every cluster in angular order.
__global__ void angle_keys_kernel(const std::uint64_t* points, int count,
                                  const std::uint32_t* point_cluster, const std::int32_t* box,
                                  std::uint64_t* keys) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const std::uint32_t cluster = point_cluster[index];
  const EdgePoint point = unpack_point(points[index]);
  const float cx = 0.5F * (box[4 * cluster + 0] + box[4 * cluster + 1]) + kCenterNudgeX;
  const float cy = 0.5F * (box[4 * cluster + 2] + box[4 * cluster + 3]) + kCenterNudgeY;
  const float angle = atan2f(point.y - cy, point.x - cx);
  // Map the float's bits to an unsigned value that sorts in the same order.
  std::uint32_t bits = __float_as_uint(angle);
  bits = (bits & 0x80000000U) != 0U ? ~bits : bits | 0x80000000U;
  keys[index] = (static_cast<std::uint64_t>(cluster) << 32) | bits;
}

// Clusters with fewer raw points cannot be a tag border (AprilTag 3's limit).
constexpr std::uint32_t kMinClusterPoints = 24;

// Marks the first of each run of equal points within a cluster (a boundary
// point two neighbors both reported), and counts each cluster's survivors.
__global__ void unique_points_kernel(const std::uint64_t* keys, const std::uint64_t* points,
                                     int count, std::uint8_t* keep, std::uint32_t* cluster_of,
                                     std::uint32_t* unique_counts) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const auto cluster = static_cast<std::uint32_t>(keys[index] >> 32);
  cluster_of[index] = cluster;
  // The low 32 bits of a packed EdgePoint are its (x, y).
  const bool first =
      index == 0 || static_cast<std::uint32_t>(keys[index - 1] >> 32) != cluster ||
      static_cast<std::uint32_t>(points[index - 1]) != static_cast<std::uint32_t>(points[index]);
  keep[index] = first ? 1U : 0U;
  if (first) {
    atomicAdd(&unique_counts[cluster], 1U);
  }
}

// Each point's line-fit moments, and its share of its cluster's polarity.
__global__ void moments_kernel(const std::uint64_t* points, const std::uint32_t* cluster_of,
                               const std::int32_t* box, int count, LineMoments* moments,
                               double* polarity) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const EdgePoint point = unpack_point(points[index]);
  const double x = 0.5 * point.x + 0.5;
  const double y = 0.5 * point.y + 0.5;
  const double w = point.weight;
  moments[index] = {w * x, w * y, w * x * x, w * x * y, w * y * y, w};
  const std::uint32_t cluster = cluster_of[index];
  const float cx = 0.5F * (box[4 * cluster + 0] + box[4 * cluster + 1]) + kCenterNudgeX;
  const float cy = 0.5F * (box[4 * cluster + 2] + box[4 * cluster + 3]) + kCenterNudgeY;
  atomicAdd(&polarity[cluster],
            static_cast<double>((point.x - cx) * point.gx + (point.y - cy) * point.gy));
}

// The moments of a cluster's points first .. last inclusive, wrapping past the
// end, from the cluster's inclusive prefix sums.
__device__ LineMoments range_moments(const LineMoments* prefix, int first, int last, int count) {
  const LineMoments zero{};
  const LineMoments before = first == 0 ? zero : prefix[first - 1];
  if (first <= last) {
    return prefix[last] - before;
  }
  return (prefix[count - 1] - before) + prefix[last];
}

// The smaller eigenvalue of the weighted covariance: the mean squared distance
// of the points from their best line.
__device__ double line_mse(const LineMoments& m) {
  const double ex = m.mx / m.w;
  const double ey = m.my / m.w;
  const double cxx = m.mxx / m.w - ex * ex;
  const double cxy = m.mxy / m.w - ex * ey;
  const double cyy = m.myy / m.w - ey * ey;
  const double spread = sqrt(0.25 * (cxx - cyy) * (cxx - cyy) + cxy * cxy);
  return fmax(0.0, 0.5 * (cxx + cyy) - spread);
}

// How badly a line fits each point's neighborhood: high at corners.
__global__ void window_error_kernel(const LineMoments* prefix, const std::uint32_t* cluster_of,
                                    const std::uint32_t* offsets, const std::uint32_t* counts,
                                    int count, double* error) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const std::uint32_t cluster = cluster_of[index];
  const int offset = static_cast<int>(offsets[cluster]);
  const int n = static_cast<int>(counts[cluster]);
  const int window = min(20, n / 12);
  if (window < 2) {
    error[index] = 0.0;
    return;
  }
  const int i = index - offset;
  const LineMoments m = range_moments(prefix + offset, (i + n - window) % n, (i + window) % n, n);
  error[index] = (2 * window + 1) * line_mse(m);
}

__global__ void smooth_kernel(const double* error, const std::uint32_t* cluster_of,
                              const std::uint32_t* offsets, const std::uint32_t* counts, int count,
                              double* smoothed) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const std::uint32_t cluster = cluster_of[index];
  const int offset = static_cast<int>(offsets[cluster]);
  const int n = static_cast<int>(counts[cluster]);
  const int i = index - offset;
  // AprilTag 3's unit-sigma Gaussian, seven taps, unnormalized.
  double sum = 0.0;
  for (int k = -3; k <= 3; ++k) {
    sum += exp(-0.5 * k * k) * error[offset + ((i + k) % n + n) % n];
  }
  smoothed[index] = sum;
}

// Flags the local maxima of each fittable cluster's smoothed error, and counts
// them per cluster.
__global__ void maxima_kernel(const double* smoothed, const std::uint32_t* cluster_of,
                              const std::uint32_t* offsets, const std::uint32_t* counts,
                              const std::uint32_t* raw_counts, std::uint32_t max_raw_points,
                              int count, std::uint8_t* is_maximum, std::uint32_t* indices,
                              std::uint32_t* maxima_counts) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  indices[index] = static_cast<std::uint32_t>(index);
  const std::uint32_t cluster = cluster_of[index];
  const int offset = static_cast<int>(offsets[cluster]);
  const int n = static_cast<int>(counts[cluster]);
  const std::uint32_t raw = raw_counts[cluster];
  const int i = index - offset;
  const double here = smoothed[index];
  const bool maximum = raw >= kMinClusterPoints && raw <= max_raw_points && n / 12 >= 2 &&
                       here > smoothed[offset + (i + n - 1) % n] &&
                       here > smoothed[offset + (i + 1) % n];
  is_maximum[index] = maximum ? 1U : 0U;
  if (maximum) {
    atomicAdd(&maxima_counts[cluster], 1U);
  }
}

// One thread per cluster: its summary, and as corner candidates its strongest
// maxima (all of them when there are at most kMaxCornerCandidates, otherwise
// those stronger than the next one, as AprilTag 3 does), in index order.
__global__ void summary_kernel(int clusters, const std::uint32_t* raw_counts,
                               const std::uint32_t* offsets, const std::uint32_t* counts,
                               const std::int32_t* box, const double* polarity,
                               const LineMoments* prefix, const double* smoothed,
                               const std::uint32_t* maxima, const std::uint32_t* maxima_offsets,
                               const std::uint32_t* maxima_counts, ClusterSummary* summaries,
                               CornerCandidate* candidates) {
  const int cluster = blockIdx.x * blockDim.x + threadIdx.x;
  if (cluster >= clusters) {
    return;
  }
  const int offset = static_cast<int>(offsets[cluster]);
  const int n = static_cast<int>(counts[cluster]);
  ClusterSummary summary;
  summary.raw_points = raw_counts[cluster];
  summary.points = static_cast<std::uint32_t>(n);
  for (int k = 0; k < 4; ++k) {
    summary.box[k] = box[4 * cluster + k];
  }
  summary.polarity = polarity[cluster];
  summary.total = prefix[offset + n - 1];

  // The kMaxCornerCandidates + 1 strongest maxima, strongest first.
  double value[kMaxCornerCandidates + 1];
  int at[kMaxCornerCandidates + 1];
  const int found = static_cast<int>(maxima_counts[cluster]);
  const std::uint32_t* mine = maxima + maxima_offsets[cluster];
  int listed = 0;
  for (int m = 0; m < found; ++m) {
    const int index = static_cast<int>(mine[m]);
    const double here = smoothed[index];
    if (listed == kMaxCornerCandidates + 1 && here <= value[kMaxCornerCandidates]) {
      continue;
    }
    int slot = min(listed, kMaxCornerCandidates);
    while (slot > 0 && value[slot - 1] < here) {
      value[slot] = value[slot - 1];
      at[slot] = at[slot - 1];
      --slot;
    }
    value[slot] = here;
    at[slot] = index - offset;
    listed = min(listed + 1, kMaxCornerCandidates + 1);
  }
  const double floor = found > kMaxCornerCandidates ? value[kMaxCornerCandidates] : -1.0;
  int chosen[kMaxCornerCandidates];
  int kept = 0;
  for (int k = 0; k < listed && kept < kMaxCornerCandidates; ++k) {
    if (value[k] > floor) {
      chosen[kept++] = k;
    }
  }
  // Index order, so the four edges between candidates follow the boundary.
  for (int a = 1; a < kept; ++a) {
    const int moving = chosen[a];
    int b = a;
    while (b > 0 && at[chosen[b - 1]] > at[moving]) {
      chosen[b] = chosen[b - 1];
      --b;
    }
    chosen[b] = moving;
  }
  CornerCandidate* out = candidates + static_cast<std::size_t>(cluster) * kMaxCornerCandidates;
  for (int k = 0; k < kept; ++k) {
    const int i = at[chosen[k]];
    out[k].index = static_cast<std::uint32_t>(i);
    out[k].error = value[chosen[k]];
    out[k].through = prefix[offset + i];
    out[k].before = i == 0 ? LineMoments{} : prefix[offset + i - 1];
  }
  summary.candidates = static_cast<std::uint32_t>(kept);
  summaries[cluster] = summary;
}

struct AddMoments {
  __host__ __device__ LineMoments operator()(const LineMoments& a, const LineMoments& b) const {
    return a + b;
  }
};

}  // namespace

struct GpuSegmenter::Buffers {
  Buffers(std::size_t pixels, std::size_t tiles)
      : decimated(pixels),
        tile_min(tiles),
        tile_max(tiles),
        tile_min_dilated(tiles),
        tile_max_dilated(tiles),
        threshold(pixels),
        labels(pixels),
        sizes(pixels),
        keys(pixels),
        points(pixels),
        counters(4) {
    void* pinned = nullptr;
    check(cudaMallocHost(&pinned, 4 * sizeof(std::uint32_t)), "cudaMallocHost(counters)");
    host_counters = static_cast<std::uint32_t*>(pinned);
  }
  ~Buffers() { cudaFreeHost(host_counters); }
  Buffers(const Buffers&) = delete;
  Buffers& operator=(const Buffers&) = delete;

  // Per point.
  void reserve_points(std::size_t count) {
    for (auto* buffer : {&keys, &points, &sorted_keys, &sorted_points, &unique_points}) {
      buffer->reserve(count);
    }
    for (auto* buffer : {&point_cluster, &cluster_of, &unique_cluster_of}) {
      buffer->reserve(count);
    }
    keep.reserve(count);
    is_maximum.reserve(count);
    indices.reserve(count);
    maxima.reserve(count);
    moments.reserve(count);
    prefix.reserve(count);
    error.reserve(count);
    smoothed.reserve(count);
  }
  // The run-length encoding's outputs, before the cluster count is known: at
  // most one cluster per point.
  void reserve_runs(std::size_t points) {
    run_keys.reserve(points);
    run_counts.reserve(points);
  }
  // Per cluster.
  void reserve_clusters(std::size_t count) {
    for (auto* buffer :
         {&run_offsets, &unique_counts, &unique_offsets, &maxima_counts, &maxima_offsets}) {
      buffer->reserve(count);
    }
    polarity.reserve(count);
    box.reserve(4 * count);
    summaries.reserve(count);
    candidates.reserve(count * kMaxCornerCandidates);
  }

  DeviceVector<std::uint8_t> decimated;
  DeviceVector<std::uint8_t> tile_min;
  DeviceVector<std::uint8_t> tile_max;
  DeviceVector<std::uint8_t> tile_min_dilated;
  DeviceVector<std::uint8_t> tile_max_dilated;
  DeviceVector<std::uint8_t> threshold;
  DeviceVector<std::int32_t> labels;
  DeviceVector<std::uint32_t> sizes;
  DeviceVector<std::uint64_t> keys;
  DeviceVector<std::uint64_t> points;
  DeviceVector<std::uint64_t> sorted_keys;
  DeviceVector<std::uint64_t> sorted_points;
  DeviceVector<std::uint32_t> point_cluster;
  DeviceVector<std::uint8_t> keep;
  DeviceVector<std::uint32_t> cluster_of;
  DeviceVector<std::uint64_t> unique_points;
  DeviceVector<std::uint32_t> unique_cluster_of;
  DeviceVector<LineMoments> moments;
  DeviceVector<LineMoments> prefix;
  DeviceVector<double> error;
  DeviceVector<double> smoothed;
  DeviceVector<std::uint8_t> is_maximum;
  DeviceVector<std::uint32_t> indices;
  DeviceVector<std::uint32_t> maxima;
  DeviceVector<std::uint64_t> run_keys;
  DeviceVector<std::uint32_t> run_counts;
  DeviceVector<std::uint32_t> run_offsets;
  DeviceVector<std::uint32_t> unique_counts;
  DeviceVector<std::uint32_t> unique_offsets;
  DeviceVector<std::uint32_t> maxima_counts;
  DeviceVector<std::uint32_t> maxima_offsets;
  DeviceVector<double> polarity;
  DeviceVector<std::int32_t> box;  // Per cluster: x min, x max, y min, y max.
  DeviceVector<ClusterSummary> summaries;
  DeviceVector<CornerCandidate> candidates;
  // [0]: edge points, [1]: clusters, [2]: unique points, [3]: unique cluster
  // ids, then maxima.
  DeviceVector<std::uint32_t> counters;
  std::uint32_t* host_counters{};  // Pinned.
  // CUB scratch, sized per call.
  DeviceVector<std::uint8_t> scratch;
};

namespace {

// Runs a CUB algorithm: once to size the scratch memory, then for real.
template <typename Algorithm>
void run_cub(DeviceVector<std::uint8_t>& scratch, std::string_view what, Algorithm&& algorithm) {
  std::size_t bytes = 0;
  check(algorithm(nullptr, bytes), what);
  scratch.reserve(bytes);
  bytes = scratch.capacity();
  check(algorithm(scratch.get(), bytes), what);
}

}  // namespace

GpuSegmenter::GpuSegmenter(int width, int height, int decimate, int min_white_black_diff)
    : width_(width),
      height_(height),
      decimate_(decimate),
      min_white_black_diff_(min_white_black_diff) {
  if (decimate < 1 || width / decimate < 2 * kTileSize || height / decimate < 2 * kTileSize) {
    throw std::invalid_argument("CUDA backend: the decimated image is too small");
  }
  if (width / decimate > 16383 || height / decimate > 16383) {
    // Edge points store twice the decimated coordinate in 16 bits.
    throw std::invalid_argument("CUDA backend: the decimated image is too large");
  }
  dec_width_ = width / decimate;
  dec_height_ = height / decimate;
  tiles_x_ = (dec_width_ + kTileSize - 1) / kTileSize;
  tiles_y_ = (dec_height_ + kTileSize - 1) / kTileSize;
  const std::size_t pixels = static_cast<std::size_t>(dec_width_) * dec_height_;
  const std::size_t tiles = static_cast<std::size_t>(tiles_x_) * tiles_y_;
  // Up to four boundary points per pixel; the buffers start at one and grow.
  capacity_ = 4U * pixels;
  buffers_ = std::make_unique<Buffers>(pixels, tiles);
}

GpuSegmenter::~GpuSegmenter() = default;

QuadCandidates GpuSegmenter::segment(const GrayImageView& image, cudaStream_t stream) {
  if (image.location != MemoryLocation::kDevice || image.width != width_ ||
      image.height != height_) {
    throw std::invalid_argument("CUDA backend expects a device image of the configured size");
  }
  Buffers& b = *buffers_;
  const std::size_t pixels = static_cast<std::size_t>(dec_width_) * dec_height_;
  const std::size_t tiles = static_cast<std::size_t>(tiles_x_) * tiles_y_;
  const int pixel_count = static_cast<int>(pixels);

  decimate_kernel<<<blocks_for(pixels), kBlock, 0, stream>>>(
      image.data, image.pitch, decimate_, b.decimated.get(), dec_width_, dec_height_);
  tile_minmax_kernel<<<blocks_for(tiles), kBlock, 0, stream>>>(b.decimated.get(), dec_width_,
                                                               dec_height_, tiles_x_, tiles_y_,
                                                               b.tile_min.get(), b.tile_max.get());
  tile_dilate_kernel<<<blocks_for(tiles), kBlock, 0, stream>>>(
      b.tile_min.get(), b.tile_max.get(), tiles_x_, tiles_y_, b.tile_min_dilated.get(),
      b.tile_max_dilated.get());
  threshold_kernel<<<blocks_for(pixels), kBlock, 0, stream>>>(
      b.decimated.get(), dec_width_, dec_height_, b.tile_min_dilated.get(),
      b.tile_max_dilated.get(), tiles_x_, min_white_black_diff_, b.threshold.get(), b.labels.get());
  merge_kernel<<<blocks_for(pixels), kBlock, 0, stream>>>(b.threshold.get(), dec_width_,
                                                          dec_height_, b.labels.get());
  check(cudaMemsetAsync(b.sizes.get(), 0, pixels * sizeof(std::uint32_t), stream),
        "cudaMemsetAsync");
  flatten_kernel<<<blocks_for(pixels), kBlock, 0, stream>>>(pixel_count, b.labels.get(),
                                                            b.sizes.get());
  check(cudaGetLastError(), "segmentation kernels");

  // Gather the boundary points; if there are more than the buffers hold, grow
  // them and gather again.
  std::size_t point_total = 0;
  for (int attempt = 0; attempt < 2; ++attempt) {
    check(cudaMemsetAsync(b.counters.get(), 0, 4 * sizeof(std::uint32_t), stream),
          "cudaMemsetAsync");
    edge_points_kernel<<<blocks_for(pixels), kBlock, 0, stream>>>(
        b.threshold.get(), b.decimated.get(), dec_width_, dec_height_, b.labels.get(),
        b.sizes.get(), b.keys.get(), b.points.get(), b.counters.get(),
        static_cast<std::uint32_t>(b.keys.capacity()));
    check(cudaGetLastError(), "edge_points_kernel");
    check(cudaMemcpyAsync(b.host_counters, b.counters.get(), sizeof(std::uint32_t),
                          cudaMemcpyDeviceToHost, stream),
          "cudaMemcpyAsync(point count)");
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    point_total = std::min<std::size_t>(b.host_counters[0], capacity_);
    if (point_total <= b.keys.capacity()) {
      break;
    }
    b.reserve_points(point_total);
  }

  QuadCandidates result;
  result.width = dec_width_;
  result.height = dec_height_;
  if (point_total == 0) {
    return result;
  }
  b.reserve_points(point_total);
  b.reserve_runs(point_total);
  const auto point_count = static_cast<int>(point_total);

  // Group the points by component pair.
  run_cub(b.scratch, "cub::DeviceRadixSort::SortPairs", [&](void* temp, std::size_t& bytes) {
    return cub::DeviceRadixSort::SortPairs(temp, bytes, b.keys.get(), b.sorted_keys.get(),
                                           b.points.get(), b.sorted_points.get(), point_count, 0,
                                           64, stream);
  });
  run_cub(b.scratch, "cub::DeviceRunLengthEncode::Encode", [&](void* temp, std::size_t& bytes) {
    return cub::DeviceRunLengthEncode::Encode(temp, bytes, b.sorted_keys.get(), b.run_keys.get(),
                                              b.run_counts.get(), b.counters.get() + 1, point_count,
                                              stream);
  });
  check(cudaMemcpyAsync(b.host_counters + 1, b.counters.get() + 1, sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost, stream),
        "cudaMemcpyAsync(cluster count)");
  check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  const auto cluster_count = static_cast<int>(b.host_counters[1]);
  const auto clusters = static_cast<std::size_t>(cluster_count);
  b.reserve_clusters(clusters);

  // Order every cluster's points by angle about its center.
  run_cub(b.scratch, "cub::DeviceScan::ExclusiveSum", [&](void* temp, std::size_t& bytes) {
    return cub::DeviceScan::ExclusiveSum(temp, bytes, b.run_counts.get(), b.run_offsets.get(),
                                         cluster_count, stream);
  });
  reset_boxes_kernel<<<blocks_for(clusters), kBlock, 0, stream>>>(cluster_count, b.box.get());
  cluster_boxes_kernel<<<blocks_for(point_total), kBlock, 0, stream>>>(
      b.sorted_points.get(), point_count, b.run_offsets.get(), cluster_count, b.point_cluster.get(),
      b.box.get());
  angle_keys_kernel<<<blocks_for(point_total), kBlock, 0, stream>>>(
      b.sorted_points.get(), point_count, b.point_cluster.get(), b.box.get(), b.keys.get());
  check(cudaGetLastError(), "angle ordering kernels");
  run_cub(b.scratch, "cub::DeviceRadixSort::SortPairs(angle)", [&](void* temp, std::size_t& bytes) {
    return cub::DeviceRadixSort::SortPairs(temp, bytes, b.keys.get(), b.sorted_keys.get(),
                                           b.sorted_points.get(), b.points.get(), point_count, 0,
                                           64, stream);
  });

  // Drop repeated points, then accumulate each cluster's line-fit moments.
  check(cudaMemsetAsync(b.unique_counts.get(), 0, clusters * sizeof(std::uint32_t), stream),
        "cudaMemsetAsync");
  unique_points_kernel<<<blocks_for(point_total), kBlock, 0, stream>>>(
      b.sorted_keys.get(), b.points.get(), point_count, b.keep.get(), b.cluster_of.get(),
      b.unique_counts.get());
  check(cudaGetLastError(), "unique_points_kernel");
  run_cub(b.scratch, "cub::DeviceSelect::Flagged(points)", [&](void* temp, std::size_t& bytes) {
    return cub::DeviceSelect::Flagged(temp, bytes, b.points.get(), b.keep.get(),
                                      b.unique_points.get(), b.counters.get() + 2, point_count,
                                      stream);
  });
  run_cub(b.scratch, "cub::DeviceSelect::Flagged(clusters)", [&](void* temp, std::size_t& bytes) {
    return cub::DeviceSelect::Flagged(temp, bytes, b.cluster_of.get(), b.keep.get(),
                                      b.unique_cluster_of.get(), b.counters.get() + 3, point_count,
                                      stream);
  });
  run_cub(b.scratch, "cub::DeviceScan::ExclusiveSum(unique)", [&](void* temp, std::size_t& bytes) {
    return cub::DeviceScan::ExclusiveSum(temp, bytes, b.unique_counts.get(), b.unique_offsets.get(),
                                         cluster_count, stream);
  });
  check(cudaMemcpyAsync(b.host_counters + 2, b.counters.get() + 2, sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost, stream),
        "cudaMemcpyAsync(unique count)");
  check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  const auto unique_count = static_cast<int>(b.host_counters[2]);
  const auto unique = static_cast<std::size_t>(unique_count);

  check(cudaMemsetAsync(b.polarity.get(), 0, clusters * sizeof(double), stream), "cudaMemsetAsync");
  moments_kernel<<<blocks_for(unique), kBlock, 0, stream>>>(
      b.unique_points.get(), b.unique_cluster_of.get(), b.box.get(), unique_count, b.moments.get(),
      b.polarity.get());
  check(cudaGetLastError(), "moments_kernel");
  run_cub(b.scratch, "cub::DeviceScan::InclusiveScanByKey", [&](void* temp, std::size_t& bytes) {
    return cub::DeviceScan::InclusiveScanByKey(temp, bytes, b.unique_cluster_of.get(),
                                               b.moments.get(), b.prefix.get(), AddMoments{},
                                               unique_count, ::cuda::std::equal_to<>{}, stream);
  });

  // Score each point, smooth the scores, and keep each cluster's corner candidates.
  window_error_kernel<<<blocks_for(unique), kBlock, 0, stream>>>(
      b.prefix.get(), b.unique_cluster_of.get(), b.unique_offsets.get(), b.unique_counts.get(),
      unique_count, b.error.get());
  smooth_kernel<<<blocks_for(unique), kBlock, 0, stream>>>(
      b.error.get(), b.unique_cluster_of.get(), b.unique_offsets.get(), b.unique_counts.get(),
      unique_count, b.smoothed.get());
  // A cluster longer than this cannot be a tag border inside the image.
  const auto max_raw_points = static_cast<std::uint32_t>(3 * (2 * dec_width_ + 2 * dec_height_));
  check(cudaMemsetAsync(b.maxima_counts.get(), 0, clusters * sizeof(std::uint32_t), stream),
        "cudaMemsetAsync");
  maxima_kernel<<<blocks_for(unique), kBlock, 0, stream>>>(
      b.smoothed.get(), b.unique_cluster_of.get(), b.unique_offsets.get(), b.unique_counts.get(),
      b.run_counts.get(), max_raw_points, unique_count, b.is_maximum.get(), b.indices.get(),
      b.maxima_counts.get());
  check(cudaGetLastError(), "maxima_kernel");
  run_cub(b.scratch, "cub::DeviceSelect::Flagged(maxima)", [&](void* temp, std::size_t& bytes) {
    return cub::DeviceSelect::Flagged(temp, bytes, b.indices.get(), b.is_maximum.get(),
                                      b.maxima.get(), b.counters.get() + 3, unique_count, stream);
  });
  run_cub(b.scratch, "cub::DeviceScan::ExclusiveSum(maxima)", [&](void* temp, std::size_t& bytes) {
    return cub::DeviceScan::ExclusiveSum(temp, bytes, b.maxima_counts.get(), b.maxima_offsets.get(),
                                         cluster_count, stream);
  });
  summary_kernel<<<blocks_for(clusters), kBlock, 0, stream>>>(
      cluster_count, b.run_counts.get(), b.unique_offsets.get(), b.unique_counts.get(), b.box.get(),
      b.polarity.get(), b.prefix.get(), b.smoothed.get(), b.maxima.get(), b.maxima_offsets.get(),
      b.maxima_counts.get(), b.summaries.get(), b.candidates.get());
  check(cudaGetLastError(), "corner candidate kernels");

  result.clusters.resize(clusters);
  result.corners.resize(clusters * kMaxCornerCandidates);
  check(cudaMemcpyAsync(result.clusters.data(), b.summaries.get(),
                        clusters * sizeof(ClusterSummary), cudaMemcpyDeviceToHost, stream),
        "cudaMemcpyAsync(cluster summaries)");
  check(cudaMemcpyAsync(result.corners.data(), b.candidates.get(),
                        result.corners.size() * sizeof(CornerCandidate), cudaMemcpyDeviceToHost,
                        stream),
        "cudaMemcpyAsync(corner candidates)");
  check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  return result;
}

}  // namespace tag_detection
