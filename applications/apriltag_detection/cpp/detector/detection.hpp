// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Types shared by the detector backends, the operators and the tests; no SDK.

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
  kCpu,   // AprilTag 3 reference library on the CPU.
  kCuda,  // The whole detector on the GPU.
  kVpi,   // VPI: the PVA when the platform has one and the image fits, the CPU otherwise.
};

[[nodiscard]] std::string_view backend_name(Backend backend) noexcept;
// Parses "cpu", "cuda" or "vpi"; nullopt for anything else.
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

// Pinhole intrinsics in pixels, pixel centers at integers (OpenCV, ROS `CameraInfo`).
struct CameraIntrinsics {
  double fx{};
  double fy{};
  double cx{};
  double cy{};
};

// Detector settings. VPI uses only `family` and `max_hamming`.
struct DetectorOptions {
  std::string family{"tag36h11"};  // An AprilTag 3 family name.
  int max_hamming{1};              // Bit errors corrected when decoding, 0..2.
  int decimate{2};                 // Quads are found at 1/decimate resolution, 1..8.
  bool refine_edges{true};         // Refine quad edges at full resolution.
  double decode_sharpening{0.25};  // Laplacian sharpening of the bit values, 0..1.
  int min_white_black_diff{5};     // Tiles with less contrast are not segmented, 1..255.
};

// Rejects options no backend accepts, naming the offending field.
// @throws std::invalid_argument
void validate_options(const DetectorOptions& options);

// The tag frame in the camera frame: row-major rotation, translation in meters.
// Camera x right, y down, z forward; tag x right, y down as printed, z into the tag.
struct TagPose {
  std::array<double, 9> rotation{};
  std::array<double, 3> translation{};
  double error{};  // Object-space error the pose solver minimized.
};

struct Detection {
  int id{};
  int hamming{};
  float decision_margin{};
  // Pixel coordinates; pixel centers are at integer coordinates.
  std::array<double, 2> center{};
  // From the tag's bottom-left corner, counter-clockwise as printed (AprilTag 3).
  std::array<std::array<double, 2>, 4> corners{};
  std::optional<TagPose> pose;
};

}  // namespace tag_detection
