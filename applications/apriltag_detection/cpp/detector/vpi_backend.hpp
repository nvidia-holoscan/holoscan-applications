// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// The VPI backend: NVIDIA VPI's AprilTag detector on the PVA (Jetson Orin and
// Thor) or the CPU. VPI has no CUDA AprilTag detector.

#pragma once

#include <memory>

#include "detector_backend.hpp"

namespace tag_detection {

// The image sides the PVA detector accepts, as measured with VPI 4 on Jetson
// Thor in an earlier integration: below 256 it refuses the payload, above 4096
// the launch, and a side one pixel past a multiple of 128 builds but fails every
// detect.
inline constexpr int kPvaMinSide = 256;
inline constexpr int kPvaMaxSide = 4096;
inline constexpr int kPvaTileStep = 128;

[[nodiscard]] constexpr bool pva_side_usable(int side) noexcept {
  return side >= kPvaMinSide && side <= kPvaMaxSide && side % kPvaTileStep != 1;
}

// Whether VPI reports a PVA on this platform.
// @throws std::runtime_error if VPI cannot be queried.
[[nodiscard]] bool vpi_platform_has_pva();

// `backend` is kVpi, kVpiPva or kVpiCpu.
[[nodiscard]] std::unique_ptr<DetectorBackend> create_vpi_backend(Backend backend,
                                                                  const DetectorOptions& options,
                                                                  int width, int height);

}  // namespace tag_detection
