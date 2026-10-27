// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Types shared by every detector backend, the operators, and the tests. Nothing
// here depends on the Holoscan SDK, so the detector library builds and tests on
// its own.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace tag_detection {

// Where a backend runs its detection stage.
enum class Backend : std::uint8_t {
  kCpu,     // AprilTag 3 reference library on the CPU.
  kCuda,    // The whole detector on the GPU.
  kVpi,     // VPI: the PVA when the platform has one, the CPU otherwise.
  kVpiPva,  // VPI on the PVA only (Jetson Orin and Thor).
  kVpiCpu,  // VPI on the CPU only.
};

[[nodiscard]] std::string_view backend_name(Backend backend) noexcept;
// Parses "cpu", "cuda", "vpi", "vpi-pva" or "vpi-cpu"; nullopt for anything else.
[[nodiscard]] std::optional<Backend> parse_backend(std::string_view name) noexcept;
// Backends compiled into this build, in a stable order.
[[nodiscard]] std::vector<Backend> available_backends();

// Where a gray image lives.
enum class MemoryLocation : std::uint8_t { kHost, kDevice };

// A borrowed 8-bit single-channel image. `pitch` is the row stride in bytes.
struct GrayImageView {
  const std::uint8_t* data{};
  int width{};
  int height{};
  std::size_t pitch{};
  MemoryLocation location{MemoryLocation::kHost};
};

// Pinhole intrinsics in pixels. Pixel centers are at integer coordinates, the
// convention of OpenCV and ROS `CameraInfo`.
struct CameraIntrinsics {
  double fx{};
  double fy{};
  double cx{};
  double cy{};
};

// Detector settings. Every backend uses `family` and `max_hamming`; VPI's
// detector has no other settings and ignores the rest.
struct DetectorOptions {
  // An AprilTag 3 family name, for example "tag36h11" or "tagStandard41h12".
  std::string family{"tag36h11"};
  // Bit errors corrected when decoding, 0..2. More admits more false positives
  // and needs more memory for the decode table.
  int max_hamming{1};
  // Quad detection runs on the image shrunk by this integer factor, 1..8; edges
  // are then refined on the full-resolution image. CPU and CUDA backends only.
  int decimate{2};
  // Refine each quad's edges on the full-resolution image. CPU and CUDA only.
  bool refine_edges{true};
  // Laplacian sharpening of the sampled bit values before thresholding, 0..1.
  // CPU and CUDA only.
  double decode_sharpening{0.25};
  // Tiles whose brightest and darkest pixels differ by less than this are not
  // segmented, 1..255. CPU and CUDA only.
  int min_white_black_diff{5};
  // Worker threads for the CPU backend, 1..64.
  int cpu_threads{1};
};

// Rejects options no backend accepts, naming the offending field.
// @throws std::invalid_argument
void validate_options(const DetectorOptions& options);

// Rotation (row-major 3x3) and translation (meters) of the tag frame in the
// camera frame. The camera frame is x right, y down, z forward; the tag frame is
// centered on the tag with x right and y down as the tag is printed, and z
// pointing into the tag, as in the AprilTag 3 library.
struct TagPose {
  std::array<double, 9> rotation{};
  std::array<double, 3> translation{};
  // Object-space reprojection error the pose solver minimized.
  double error{};
};

struct Detection {
  int id{};
  int hamming{};
  float decision_margin{};
  // Pixel coordinates; pixel centers are at integer coordinates.
  std::array<double, 2> center{};
  // Starting at the tag's bottom-left corner and going counter-clockwise as the
  // tag is printed, the AprilTag 3 order.
  std::array<std::array<double, 2>, 4> corners{};
  std::optional<TagPose> pose;
};

}  // namespace tag_detection
