// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// The CPU backend: the AprilTag 3 reference detector on a host image.

#pragma once

#include <memory>

#include "detector_backend.hpp"

namespace tag_detection {

[[nodiscard]] std::unique_ptr<DetectorBackend> create_cpu_backend(const DetectorOptions& options,
                                                                  int width, int height);

}  // namespace tag_detection
