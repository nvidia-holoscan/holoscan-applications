// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// The CUDA backend: thresholding, connected components and edge clustering on
// the GPU (gpu_segmenter.hpp); quad fitting, edge refinement and decoding on the
// host (quad_fit.hpp, tag_decoder.hpp).

#pragma once

#include <memory>

#include "detector_backend.hpp"

namespace tag_detection {

[[nodiscard]] std::unique_ptr<DetectorBackend> create_cuda_backend(const DetectorOptions& options,
                                                                   int width, int height);

}  // namespace tag_detection
