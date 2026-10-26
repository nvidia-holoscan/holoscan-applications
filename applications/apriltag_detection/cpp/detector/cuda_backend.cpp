// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "cuda_backend.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gpu_segmenter.hpp"
#include "quad_fit.hpp"
#include "tag_decoder.hpp"
#include "tag_family.hpp"

namespace tag_detection {
namespace {

// quad_fit.hpp and tag_decoder.hpp place pixel centers at +0.5; reported
// coordinates put them at integers.
constexpr double kPixelCenterShift = 0.5;

double shortest_side(const Detection& detection) {
  double side = HUGE_VAL;
  for (int i = 0; i < 4; ++i) {
    const auto& a = detection.corners[i];
    const auto& b = detection.corners[(i + 1) % 4];
    side = std::min(side, std::hypot(b[0] - a[0], b[1] - a[1]));
  }
  return side;
}

// Several quads can frame one tag (its outer and inner border edges, both
// polarities); keep the best decode of each tag.
std::vector<Detection> remove_duplicates(std::vector<Detection> detections) {
  std::sort(detections.begin(), detections.end(), [](const Detection& a, const Detection& b) {
    if (a.hamming != b.hamming) return a.hamming < b.hamming;
    return a.decision_margin > b.decision_margin;
  });
  std::vector<Detection> kept;
  for (const Detection& candidate : detections) {
    const bool duplicate =
        std::any_of(kept.begin(), kept.end(), [&candidate](const Detection& other) {
          if (other.id != candidate.id) return false;
          const double distance = std::hypot(other.center[0] - candidate.center[0],
                                             other.center[1] - candidate.center[1]);
          return distance < 0.5 * std::min(shortest_side(other), shortest_side(candidate));
        });
    if (!duplicate) {
      kept.push_back(candidate);
    }
  }
  return kept;
}

class CudaBackend final : public DetectorBackend {
 public:
  CudaBackend(const DetectorOptions& options, int width, int height)
      : family_(options.family),
        decoder_(family_, options.max_hamming, options.decode_sharpening),
        segmenter_(width, height, options.decimate, options.min_white_black_diff),
        width_(width),
        height_(height),
        decimate_(options.decimate),
        refine_edges_(options.refine_edges) {
    fit_options_.decimate = options.decimate;
    fit_options_.normal_border = decoder_.normal_border();
    fit_options_.reversed_border = decoder_.reversed_border();
    fit_options_.min_tag_width =
        std::max(3.0, static_cast<double>(decoder_.width_at_border()) / options.decimate);
    try {
      check(cudaMallocHost(reinterpret_cast<void**>(&host_gray_),
                           static_cast<std::size_t>(width) * static_cast<std::size_t>(height)),
            "cudaMallocHost");
      check(cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking), "cudaStreamCreate");
      check(cudaEventCreateWithFlags(&image_ready_, cudaEventDisableTiming), "cudaEventCreate");
    } catch (...) {
      release();
      throw;
    }
  }

  ~CudaBackend() override { release(); }

  [[nodiscard]] Backend kind() const noexcept override { return Backend::kCuda; }
  [[nodiscard]] MemoryLocation input_location() const noexcept override {
    return MemoryLocation::kDevice;
  }

  [[nodiscard]] std::vector<Detection> detect(const GrayImageView& image,
                                              cudaStream_t stream) override {
    // Copy the full-resolution image to the host for refinement and decoding on
    // a second stream, once it is ready, while the segmentation kernels run.
    check(cudaEventRecord(image_ready_, stream), "cudaEventRecord");
    check(cudaStreamWaitEvent(copy_stream_, image_ready_), "cudaStreamWaitEvent");
    check(cudaMemcpy2DAsync(host_gray_, static_cast<std::size_t>(width_), image.data, image.pitch,
                            static_cast<std::size_t>(width_), static_cast<std::size_t>(height_),
                            cudaMemcpyDeviceToHost, copy_stream_),
          "cudaMemcpy2DAsync");
    // Whatever happens next, the copy must not outlive this call.
    struct CopyGuard {
      cudaStream_t stream;
      ~CopyGuard() { cudaStreamSynchronize(stream); }
    } copy_guard{copy_stream_};
    const QuadCandidates candidates = segmenter_.segment(image, stream);
    check(cudaStreamSynchronize(copy_stream_), "cudaStreamSynchronize(copy)");
    const HostGray gray{host_gray_, width_, height_, static_cast<std::size_t>(width_)};

    std::vector<Detection> detections;
    for (Quad& quad : fit_quads(candidates, fit_options_)) {
      if (refine_edges_) {
        refine_edges(quad, gray, decimate_);
      }
      if (auto detection = decoder_.decode(quad, gray)) {
        detections.push_back(*detection);
      }
    }
    detections = remove_duplicates(std::move(detections));
    for (Detection& detection : detections) {
      detection.center[0] -= kPixelCenterShift;
      detection.center[1] -= kPixelCenterShift;
      for (auto& corner : detection.corners) {
        corner[0] -= kPixelCenterShift;
        corner[1] -= kPixelCenterShift;
      }
    }
    return detections;
  }

 private:
  static void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
      throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
  }

  void release() noexcept {
    if (copy_stream_ != nullptr) {
      cudaStreamSynchronize(copy_stream_);
      cudaStreamDestroy(copy_stream_);
      copy_stream_ = nullptr;
    }
    if (image_ready_ != nullptr) {
      cudaEventDestroy(image_ready_);
      image_ready_ = nullptr;
    }
    if (host_gray_ != nullptr) {
      cudaFreeHost(host_gray_);
      host_gray_ = nullptr;
    }
  }

  TagFamily family_;
  TagDecoder decoder_;
  GpuSegmenter segmenter_;
  QuadFitOptions fit_options_;
  int width_;
  int height_;
  int decimate_;
  bool refine_edges_;
  std::uint8_t* host_gray_{};  // Pinned.
  cudaStream_t copy_stream_{};
  cudaEvent_t image_ready_{};
};

}  // namespace

std::unique_ptr<DetectorBackend> create_cuda_backend(const DetectorOptions& options, int width,
                                                     int height) {
  return std::make_unique<CudaBackend>(options, width, height);
}

}  // namespace tag_detection
