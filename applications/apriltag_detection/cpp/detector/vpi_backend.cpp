// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vpi_backend.hpp"

// VPI's C API.
extern "C" {
#include <vpi/Array.h>
#include <vpi/Context.h>
#include <vpi/Image.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/AprilTags.h>
}

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "tag_family.hpp"

namespace tag_detection {
namespace {

// VPI places pixel centers at +0.5, like the AprilTag 3 library; reported
// coordinates put them at integers.
constexpr double kPixelCenterShift = 0.5;
// Results the detector can return per frame. VPI writes no more than this.
constexpr std::int32_t kMaxDetections = 256;

void check(VPIStatus status, std::string_view what) {
  if (status != VPI_SUCCESS) {
    char detail[VPI_MAX_STATUS_MESSAGE_LENGTH] = {};
    vpiGetLastStatusMessage(detail, sizeof(detail));
    throw std::runtime_error(std::string(what) + " failed: " + vpiStatusGetName(status) + ": " +
                             detail);
  }
}

VPIAprilTagFamily vpi_family(std::string_view name) {
  constexpr std::pair<std::string_view, VPIAprilTagFamily> kMap[] = {
      {"tag16h5", VPI_APRILTAG_16H5},
      {"tag25h9", VPI_APRILTAG_25H9},
      {"tag36h10", VPI_APRILTAG_36H10},
      {"tag36h11", VPI_APRILTAG_36H11},
      {"tagCircle21h7", VPI_APRILTAG_CIRCLE21H7},
      {"tagCircle49h12", VPI_APRILTAG_CIRCLE49H12},
      {"tagCustom48h12", VPI_APRILTAG_CUSTOM48H12},
      {"tagStandard41h12", VPI_APRILTAG_STANDARD41H12},
      {"tagStandard52h13", VPI_APRILTAG_STANDARD52H13},
  };
  for (const auto& [family_name, family] : kMap) {
    if (family_name == name) {
      return family;
    }
  }
  throw std::invalid_argument("VPI does not support AprilTag family '" + std::string(name) + "'");
}

class VpiBackend final : public DetectorBackend {
 public:
  VpiBackend(Backend requested, const DetectorOptions& options, int width, int height)
      : requested_(requested), width_(width), height_(height) {
    hardware_ = resolve_hardware(requested);
    const bool pva_fits = pva_side_usable(width) && pva_side_usable(height);
    if (hardware_ == VPI_BACKEND_PVA && !pva_fits) {
      if (requested == Backend::kVpiPva) {
        throw std::invalid_argument(
            "the PVA AprilTag detector needs each image side in [256, 4096] and not one past a "
            "multiple of 128; got " +
            std::to_string(width) + "x" + std::to_string(height));
      }
      // `vpi` takes the PVA only where it can.
      hardware_ = VPI_BACKEND_CPU;
    }
    VPIAprilTagDecodeParams params{};
    check(vpiInitAprilTagDecodeParams(&params), "vpiInitAprilTagDecodeParams");
    params.family = vpi_family(options.family);
    params.maxBitsCorrected = options.max_hamming;
    try {
      check(vpiCreateAprilTagDetector(hardware_, width, height, &params, &payload_),
            "vpiCreateAprilTagDetector");
      // With the PVA, frames arrive in device memory and are copied in through CUDA.
      const std::uint64_t image_backends =
          hardware_ | VPI_BACKEND_CPU |
          (hardware_ == VPI_BACKEND_PVA ? std::uint64_t{VPI_BACKEND_CUDA} : std::uint64_t{0});
      check(vpiImageCreate(width, height, VPI_IMAGE_FORMAT_U8, image_backends, &gray_),
            "vpiImageCreate");
      check(vpiArrayCreate(kMaxDetections, VPI_ARRAY_TYPE_APRILTAG_DETECTION,
                           hardware_ | VPI_BACKEND_CPU, &detections_),
            "vpiArrayCreate");
      check(vpiStreamCreate(hardware_ | VPI_BACKEND_CPU, &stream_), "vpiStreamCreate");
    } catch (...) {
      release();
      throw;
    }
  }

  ~VpiBackend() override { release(); }

  [[nodiscard]] Backend kind() const noexcept override { return requested_; }
  [[nodiscard]] std::string_view resolved_name() const noexcept override {
    return hardware_ == VPI_BACKEND_PVA ? "vpi-pva" : "vpi-cpu";
  }
  [[nodiscard]] MemoryLocation input_location() const noexcept override {
    // The PVA reads device-visible memory; the CPU detector reads host memory.
    return hardware_ == VPI_BACKEND_PVA ? MemoryLocation::kDevice : MemoryLocation::kHost;
  }

  [[nodiscard]] std::vector<Detection> detect(const GrayImageView& image,
                                              cudaStream_t stream) override {
    if (image.location != input_location() || image.width != width_ || image.height != height_) {
      throw std::invalid_argument("VPI backend received an image of another size or location");
    }
    upload(image, stream);
    check(
        vpiSubmitAprilTagDetector(stream_, hardware_, payload_, kMaxDetections, gray_, detections_),
        "vpiSubmitAprilTagDetector");
    check(vpiStreamSync(stream_), "vpiStreamSync");

    VPIArrayData data{};
    check(vpiArrayLockData(detections_, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &data),
          "vpiArrayLockData");
    // Unlock however the copy below ends.
    struct Unlock {
      VPIArray array;
      ~Unlock() { vpiArrayUnlock(array); }
    } unlock{detections_};
    std::vector<Detection> result;
    const auto count = static_cast<std::size_t>(*data.buffer.aos.sizePointer);
    result.reserve(count);
    const auto* base = static_cast<const std::byte*>(data.buffer.aos.data);
    for (std::size_t i = 0; i < count; ++i) {
      const auto* element = reinterpret_cast<const VPIAprilTagDetection*>(
          base + i * static_cast<std::size_t>(data.buffer.aos.strideBytes));
      Detection detection;
      detection.id = element->id;
      detection.hamming = element->correctedBits;
      detection.decision_margin = element->decisionMargin;
      detection.center = {element->center.x - kPixelCenterShift,
                          element->center.y - kPixelCenterShift};
      for (int corner = 0; corner < 4; ++corner) {
        detection.corners[corner] = {element->corners[corner].x - kPixelCenterShift,
                                     element->corners[corner].y - kPixelCenterShift};
      }
      result.push_back(detection);
    }
    return result;
  }

 private:
  static std::uint64_t resolve_hardware(Backend requested) {
    const bool has_pva = vpi_platform_has_pva();
    switch (requested) {
      case Backend::kVpiPva:
        if (!has_pva) {
          throw std::runtime_error("backend vpi-pva: this platform has no PVA");
        }
        return VPI_BACKEND_PVA;
      case Backend::kVpiCpu:
        return VPI_BACKEND_CPU;
      case Backend::kVpi:
        return has_pva ? VPI_BACKEND_PVA : VPI_BACKEND_CPU;
      default:
        throw std::invalid_argument("not a VPI backend");
    }
  }

  // Copies `image` into the VPI-owned gray image, through the buffer type that
  // matches where the image lives.
  void upload(const GrayImageView& image, cudaStream_t stream) {
    const bool device = image.location == MemoryLocation::kDevice;
    VPIImageData data{};
    check(vpiImageLockData(
              gray_, VPI_LOCK_WRITE,
              device ? VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR : VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR,
              &data),
          "vpiImageLockData");
    const VPIImagePlanePitchLinear& plane = data.buffer.pitch.planes[0];
    const auto pitch = static_cast<std::size_t>(plane.pitchBytes);
    const auto width = static_cast<std::size_t>(width_);
    cudaError_t synced = cudaSuccess;
    if (device) {
      synced =
          cudaMemcpy2DAsync(plane.pBase, pitch, image.data, image.pitch, width,
                            static_cast<std::size_t>(height_), cudaMemcpyDeviceToDevice, stream);
      if (synced == cudaSuccess) {
        synced = cudaStreamSynchronize(stream);
      }
    } else {
      auto* rows = static_cast<std::uint8_t*>(plane.pBase);
      for (int y = 0; y < height_; ++y) {
        std::memcpy(rows + static_cast<std::size_t>(y) * pitch,
                    image.data + static_cast<std::size_t>(y) * image.pitch, width);
      }
    }
    check(vpiImageUnlock(gray_), "vpiImageUnlock");
    if (synced != cudaSuccess) {
      throw std::runtime_error(std::string("copy into the VPI image failed: ") +
                               cudaGetErrorString(synced));
    }
  }

  void release() noexcept {
    if (stream_ != nullptr) {
      vpiStreamSync(stream_);
      vpiStreamDestroy(stream_);
      stream_ = nullptr;
    }
    vpiArrayDestroy(detections_);
    detections_ = nullptr;
    vpiImageDestroy(gray_);
    gray_ = nullptr;
    vpiPayloadDestroy(payload_);
    payload_ = nullptr;
  }

  Backend requested_;
  int width_;
  int height_;
  std::uint64_t hardware_{VPI_BACKEND_CPU};
  VPIPayload payload_{nullptr};
  VPIImage gray_{nullptr};
  VPIArray detections_{nullptr};
  VPIStream stream_{nullptr};
};

}  // namespace

bool vpi_platform_has_pva() {
  VPIContext context = nullptr;
  check(vpiContextGetCurrent(&context), "vpiContextGetCurrent");
  std::uint64_t flags = 0;
  check(vpiContextGetFlags(context, &flags), "vpiContextGetFlags");
  return (flags & VPI_BACKEND_PVA) != 0U;
}

std::unique_ptr<DetectorBackend> create_vpi_backend(Backend backend, const DetectorOptions& options,
                                                    int width, int height) {
  return std::make_unique<VpiBackend>(backend, options, width, height);
}

}  // namespace tag_detection
