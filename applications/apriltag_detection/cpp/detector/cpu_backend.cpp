// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "cpu_backend.hpp"

// The AprilTag 3 library's C API.
extern "C" {
#include <apriltag.h>

#include <common/image_u8.h>
#include <common/zarray.h>
}

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "tag_family.hpp"

namespace tag_detection {
namespace {

// The AprilTag 3 library places pixel centers at +0.5; reported coordinates put
// them at integers.
constexpr double kPixelCenterShift = 0.5;

class CpuBackend final : public DetectorBackend {
 public:
  CpuBackend(const DetectorOptions& options, int width, int height)
      : family_(options.family), width_(width), height_(height) {
    detector_.reset(apriltag_detector_create());
    if (!detector_) {
      throw std::runtime_error("apriltag_detector_create failed");
    }
    apriltag_detector_add_family_bits(detector_.get(), family_.get(), options.max_hamming);
    // The library leaves the family without a decode table, and so silently
    // never decodes it, when it cannot allocate one.
    if (family_.get()->impl == nullptr) {
      throw std::runtime_error("AprilTag could not allocate the decode table for " +
                               std::string(family_.name()));
    }
    detector_->quad_decimate = static_cast<float>(options.decimate);
    detector_->quad_sigma = 0.0F;
    detector_->refine_edges = options.refine_edges;
    detector_->decode_sharpening = options.decode_sharpening;
    detector_->nthreads = options.cpu_threads;
    detector_->qtp.min_white_black_diff = options.min_white_black_diff;
  }

  [[nodiscard]] Backend kind() const noexcept override { return Backend::kCpu; }
  [[nodiscard]] MemoryLocation input_location() const noexcept override {
    return MemoryLocation::kHost;
  }

  [[nodiscard]] std::vector<Detection> detect(const GrayImageView& image, cudaStream_t) override {
    if (image.location != MemoryLocation::kHost || image.width != width_ ||
        image.height != height_) {
      throw std::invalid_argument("CPU backend expects a host image of the configured size");
    }
    image_u8_t view{.width = image.width,
                    .height = image.height,
                    .stride = static_cast<int32_t>(image.pitch),
                    .buf = const_cast<uint8_t*>(image.data)};
    std::unique_ptr<zarray_t, DetectionsDeleter> found{
        apriltag_detector_detect(detector_.get(), &view)};
    if (!found) {
      throw std::runtime_error("apriltag_detector_detect failed");
    }
    std::vector<Detection> detections;
    detections.reserve(static_cast<std::size_t>(zarray_size(found.get())));
    for (int i = 0; i < zarray_size(found.get()); ++i) {
      apriltag_detection_t* det = nullptr;
      zarray_get(found.get(), i, &det);
      Detection detection;
      detection.id = det->id;
      detection.hamming = det->hamming;
      detection.decision_margin = det->decision_margin;
      detection.center = {det->c[0] - kPixelCenterShift, det->c[1] - kPixelCenterShift};
      for (int corner = 0; corner < 4; ++corner) {
        detection.corners[corner] = {det->p[corner][0] - kPixelCenterShift,
                                     det->p[corner][1] - kPixelCenterShift};
      }
      detections.push_back(detection);
    }
    return detections;
  }

 private:
  struct DetectorDeleter {
    void operator()(apriltag_detector_t* detector) const noexcept {
      apriltag_detector_destroy(detector);
    }
  };
  struct DetectionsDeleter {
    void operator()(zarray_t* detections) const noexcept {
      apriltag_detections_destroy(detections);
    }
  };

  // Declared before the detector, which refers to it, so it outlives it.
  TagFamily family_;
  std::unique_ptr<apriltag_detector_t, DetectorDeleter> detector_;
  int width_;
  int height_;
};

}  // namespace

std::unique_ptr<DetectorBackend> create_cpu_backend(const DetectorOptions& options, int width,
                                                    int height) {
  return std::make_unique<CpuBackend>(options, width, height);
}

}  // namespace tag_detection
