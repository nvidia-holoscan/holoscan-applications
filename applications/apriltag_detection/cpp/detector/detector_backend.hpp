// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// The interface every detector backend implements, and the factory that builds
// one. A backend finds tags and their corners; poses are estimated afterwards by
// pose.hpp, the same way for every backend.

#pragma once

#include <cuda_runtime_api.h>

#include <memory>
#include <string_view>
#include <vector>

#include "detection.hpp"

namespace tag_detection {

class DetectorBackend {
 public:
  virtual ~DetectorBackend() = default;
  DetectorBackend(const DetectorBackend&) = delete;
  DetectorBackend& operator=(const DetectorBackend&) = delete;

  [[nodiscard]] virtual Backend kind() const noexcept = 0;
  // The hardware actually used, for backends that choose it: "vpi-pva" or
  // "vpi-cpu" for kVpi. Otherwise backend_name(kind()).
  [[nodiscard]] virtual std::string_view resolved_name() const noexcept {
    return backend_name(kind());
  }
  // Where `detect` wants its image; the caller converts and copies to match.
  [[nodiscard]] virtual MemoryLocation input_location() const noexcept = 0;
  // Finds tags in `image`, which must be `width` x `height` as created and in
  // `input_location()`. Device work is ordered on `stream`; results are on the
  // host when this returns. Poses are left empty.
  // @throws std::runtime_error on a backend failure.
  [[nodiscard]] virtual std::vector<Detection> detect(const GrayImageView& image,
                                                      cudaStream_t stream) = 0;

 protected:
  DetectorBackend() = default;
};

// Checks that `backend` is compiled in and its hardware is present, without
// building it.
// @throws std::invalid_argument naming what is missing.
void check_available(Backend backend);

// Builds `backend` for `width` x `height` gray images.
// @throws std::invalid_argument if the options or size are not usable with this
//         backend, or the backend is not compiled into this build.
// @throws std::runtime_error if the backend fails to initialize (for example,
//         VPI reports no PVA on this platform).
[[nodiscard]] std::unique_ptr<DetectorBackend> create_backend(Backend backend,
                                                              const DetectorOptions& options,
                                                              int width, int height);

}  // namespace tag_detection
