// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "detector_backend.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cpu_backend.hpp"
#include "cuda_backend.hpp"
#include "tag_family.hpp"
#if APRILTAG_DETECTION_WITH_VPI
#include "vpi_backend.hpp"
#endif

namespace tag_detection {
namespace {

constexpr std::array<std::pair<Backend, std::string_view>, 5> kBackendNames{{
    {Backend::kCpu, "cpu"},
    {Backend::kCuda, "cuda"},
    {Backend::kVpi, "vpi"},
    {Backend::kVpiPva, "vpi-pva"},
    {Backend::kVpiCpu, "vpi-cpu"},
}};

// Codes, with every variant up to two bits off, that a decode table may hold.
// AprilTag's own advice: correct two bits only in the smaller families.
constexpr std::uint64_t kMaxDecodeEntries = 8'000'000;

bool is_vpi(Backend backend) noexcept {
  return backend == Backend::kVpi || backend == Backend::kVpiPva || backend == Backend::kVpiCpu;
}

}  // namespace

std::string_view backend_name(Backend backend) noexcept {
  for (const auto& [kind, name] : kBackendNames) {
    if (kind == backend) {
      return name;
    }
  }
  return "unknown";
}

std::optional<Backend> parse_backend(std::string_view name) noexcept {
  for (const auto& [kind, backend_name] : kBackendNames) {
    if (backend_name == name) {
      return kind;
    }
  }
  return std::nullopt;
}

std::vector<Backend> available_backends() {
  std::vector<Backend> backends{Backend::kCpu, Backend::kCuda};
#if APRILTAG_DETECTION_WITH_VPI
  backends.push_back(Backend::kVpi);
#endif
  return backends;
}

void validate_options(const DetectorOptions& options) {
  if (!is_known_family(options.family)) {
    throw std::invalid_argument("family: unknown AprilTag family '" + options.family + "'");
  }
  if (options.max_hamming < 0 || options.max_hamming > 2) {
    throw std::invalid_argument("max_hamming must be in 0..2");
  }
  if (options.max_hamming == 2 && decode_table_entries(options.family, 2) > kMaxDecodeEntries) {
    throw std::invalid_argument("max_hamming 2 with " + options.family +
                                " needs decode tables of several gigabytes; use 0 or 1");
  }
  if (options.decimate < 1 || options.decimate > 8) {
    throw std::invalid_argument("decimate must be in 1..8");
  }
  if (!(options.decode_sharpening >= 0.0 && options.decode_sharpening <= 1.0)) {
    throw std::invalid_argument("decode_sharpening must be in 0..1");
  }
  if (options.min_white_black_diff < 1 || options.min_white_black_diff > 255) {
    throw std::invalid_argument("min_white_black_diff must be in 1..255");
  }
  if (options.cpu_threads < 1 || options.cpu_threads > 64) {
    throw std::invalid_argument("cpu_threads must be in 1..64");
  }
}

void check_available(Backend backend) {
  if (!is_vpi(backend)) {
    return;
  }
#if APRILTAG_DETECTION_WITH_VPI
  if (backend == Backend::kVpiPva && !vpi_platform_has_pva()) {
    throw std::invalid_argument("backend vpi-pva: this platform has no PVA; use vpi or vpi-cpu");
  }
#else
  throw std::invalid_argument("backend " + std::string(backend_name(backend)) +
                              ": this build has no VPI; install VPI 4 and rebuild");
#endif
}

std::unique_ptr<DetectorBackend> create_backend(Backend backend, const DetectorOptions& options,
                                                int width, int height) {
  validate_options(options);
  if (width <= 0 || height <= 0) {
    throw std::invalid_argument("image size must be positive");
  }
  switch (backend) {
    case Backend::kCpu:
      return create_cpu_backend(options, width, height);
    case Backend::kCuda:
      return create_cuda_backend(options, width, height);
    default:
      break;
  }
  if (is_vpi(backend)) {
#if APRILTAG_DETECTION_WITH_VPI
    return create_vpi_backend(backend, options, width, height);
#else
    throw std::invalid_argument("backend " + std::string(backend_name(backend)) +
                                ": this build has no VPI; install VPI 4 and rebuild");
#endif
  }
  throw std::invalid_argument("unknown backend");
}

}  // namespace tag_detection
