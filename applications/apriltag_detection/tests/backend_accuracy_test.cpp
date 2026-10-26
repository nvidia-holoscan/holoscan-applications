// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Every compiled backend against the synthetic scene's ground truth: each tag
// found with its id, corners in order and in place, poses close, and nothing
// else reported. Also checks the backends agree with each other, and the CUDA
// backend's buffer growth on images with many more edges than tags.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "detector/detector_backend.hpp"
#include "detector/device_vector.hpp"
#include "detector/gray_converter.hpp"
#include "detector/pose.hpp"
#include "detector/synthetic_scene.hpp"
#include "test_support.hpp"

namespace tag_detection {
namespace {

constexpr int kWidth = 1280;
constexpr int kHeight = 720;
constexpr std::uint32_t kFrames = 24;

struct Errors {
  int frames{};
  int expected{};
  int found{};
  int extra{};
  int wrong_order{};
  double max_corner_px{};
  double sum_corner_px{};
  int corners{};
  double max_translation_rel{};
  double max_rotation_deg{};
  double detect_ms{};
};

double rotation_error_deg(const std::array<double, 9>& a, const std::array<double, 9>& b) {
  // Angle of a^T b.
  double trace = 0.0;
  for (int i = 0; i < 3; ++i) {
    for (int k = 0; k < 3; ++k) {
      trace += a[k * 3 + i] * b[k * 3 + i];
    }
  }
  return std::acos(std::clamp((trace - 1.0) / 2.0, -1.0, 1.0)) * 180.0 / M_PI;
}

std::vector<Detection> run_backend(DetectorBackend& backend, const std::vector<std::uint8_t>& gray,
                                   GrayConverter& converter, cudaStream_t stream,
                                   std::uint8_t* device_copy, bool device_input) {
  ImageView view{gray.data(), kWidth, kHeight, kWidth, PixelFormat::kMono8, MemoryLocation::kHost};
  if (device_input) {
    cudaMemcpy(device_copy, gray.data(), gray.size(), cudaMemcpyHostToDevice);
    view.data = device_copy;
    view.location = MemoryLocation::kDevice;
  }
  return backend.detect(converter.convert(view, backend.input_location(), stream), stream);
}

Errors evaluate(Backend kind, const std::string& family, bool device_input) {
  DetectorOptions options;
  options.family = family;
  auto backend = create_backend(kind, options, kWidth, kHeight);
  GrayConverter converter;
  cudaStream_t stream = nullptr;
  cudaStreamCreate(&stream);
  std::uint8_t* device_copy = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&device_copy), static_cast<std::size_t>(kWidth) * kHeight);

  Errors errors;
  for (std::uint32_t frame = 0; frame < kFrames; ++frame) {
    const SyntheticScene scene = make_scene(family, frame, kWidth, kHeight);
    const auto gray = render_scene(scene, 1);
    const auto start = std::chrono::steady_clock::now();
    const auto detections =
        run_backend(*backend, gray, converter, stream, device_copy, device_input);
    errors.detect_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    ++errors.frames;
    errors.expected += static_cast<int>(scene.tags.size());
    std::map<int, const SceneTag*> truth;
    for (const SceneTag& tag : scene.tags) truth[tag.id] = &tag;
    for (const Detection& detection : detections) {
      const auto it = truth.find(detection.id);
      if (it == truth.end()) {
        ++errors.extra;
        continue;
      }
      ++errors.found;
      const auto expected = project_corners(*it->second, scene.intrinsics, scene.tag_size);
      double worst = 0.0;
      for (int i = 0; i < 4; ++i) {
        const double e = std::hypot(detection.corners[i][0] - expected[i][0],
                                    detection.corners[i][1] - expected[i][1]);
        worst = std::max(worst, e);
        errors.sum_corner_px += e;
        ++errors.corners;
      }
      if (worst > 5.0) ++errors.wrong_order;
      errors.max_corner_px = std::max(errors.max_corner_px, worst);
      const auto pose = estimate_pose(detection, scene.intrinsics, scene.tag_size);
      if (pose) {
        const auto& t = it->second->pose.translation;
        const double distance = std::hypot(t[0], t[1], t[2]);
        const double dt = std::hypot(pose->translation[0] - t[0], pose->translation[1] - t[1],
                                     pose->translation[2] - t[2]);
        errors.max_translation_rel = std::max(errors.max_translation_rel, dt / distance);
        errors.max_rotation_deg = std::max(
            errors.max_rotation_deg, rotation_error_deg(pose->rotation, it->second->pose.rotation));
      } else {
        errors.max_translation_rel = HUGE_VAL;
      }
      truth.erase(it);  // A second report of one tag counts as extra.
    }
  }
  cudaFree(device_copy);
  cudaStreamDestroy(stream);
  std::printf(
      "  %-8s %-17s %-6s found %d/%d extra %d | corner mean %.3f max %.3f px | "
      "translation max %.2f%% | rotation max %.2f deg | %.2f ms/frame\n",
      std::string(backend_name(kind)).c_str(), family.c_str(), device_input ? "device" : "host",
      errors.found, errors.expected, errors.extra,
      errors.corners ? errors.sum_corner_px / errors.corners : 0.0, errors.max_corner_px,
      100.0 * errors.max_translation_rel, errors.max_rotation_deg,
      errors.detect_ms / errors.frames);
  return errors;
}

void check_backend(Backend kind, const std::string& family, bool device_input) {
  const Errors errors = evaluate(kind, family, device_input);
  VERIFY_MSG(errors.found == errors.expected, "every tag is found");
  VERIFY_MSG(errors.extra == 0, "nothing else is reported");
  VERIFY_MSG(errors.wrong_order == 0, "corners are in the AprilTag order");
  VERIFY_MSG(errors.max_corner_px < 1.0, "corners are within a pixel");
  VERIFY_MSG(errors.max_translation_rel < 0.02, "translation is within 2% of the distance");
  VERIFY_MSG(errors.max_rotation_deg < 3.0, "rotation is within 3 degrees");
}

// The largest error of `detection`'s corners against `truth`.
double worst_corner_error(const Detection& detection, const SceneTag& truth,
                          const SyntheticScene& scene) {
  const auto corners = project_corners(truth, scene.intrinsics, scene.tag_size);
  double worst = 0.0;
  for (int i = 0; i < 4; ++i) {
    worst = std::max(worst, std::hypot(detection.corners[i][0] - corners[i][0],
                                       detection.corners[i][1] - corners[i][1]));
  }
  return worst;
}

// Whether every corner of `truth` is at least `margin` pixels inside the image.
bool fully_visible(const SceneTag& truth, const SyntheticScene& scene, double margin) {
  const auto corners = project_corners(truth, scene.intrinsics, scene.tag_size);
  return std::all_of(corners.begin(), corners.end(), [&](const Point2& c) {
    return c[0] >= margin && c[1] >= margin && c[0] <= kWidth - 1 - margin &&
           c[1] <= kHeight - 1 - margin;
  });
}

// Tags pushed toward the image edges until some are cut off. A clipped tag's
// corners are poorly constrained for every detector (the reference CPU
// backend's err by several pixels), so: fully visible tags must be within a
// pixel; no backend may report a tag that is not there; and the CUDA backend
// must find every tag the reference finds, no less accurately.
void check_edge_tags(Backend kind) {
  DetectorOptions options;
  auto reference = create_backend(Backend::kCpu, options, kWidth, kHeight);
  auto backend = create_backend(kind, options, kWidth, kHeight);
  GrayConverter reference_converter;
  GrayConverter converter;
  int reference_tags = 0;
  int found_tags = 0;
  for (std::uint32_t frame = 0; frame < kFrames; ++frame) {
    SyntheticScene scene = make_scene("tag36h11", frame, kWidth, kHeight);
    for (SceneTag& tag : scene.tags) {
      tag.pose.translation[0] *= 2.1;
      tag.pose.translation[1] *= 1.6;
    }
    const auto gray = render_scene(scene, 1);
    const ImageView view{gray.data(),          kWidth, kHeight, kWidth, PixelFormat::kMono8,
                         MemoryLocation::kHost};
    const auto expected = reference->detect(
        reference_converter.convert(view, MemoryLocation::kHost, nullptr), nullptr);
    const auto found =
        backend->detect(converter.convert(view, backend->input_location(), nullptr), nullptr);
    reference_tags += static_cast<int>(expected.size());
    found_tags += static_cast<int>(found.size());
    const std::string where = "frame " + std::to_string(frame) + ": tag ";
    for (const Detection& got : found) {
      const auto truth = std::find_if(scene.tags.begin(), scene.tags.end(),
                                      [&](const SceneTag& t) { return t.id == got.id; });
      VERIFY_MSG(truth != scene.tags.end(),
                 where + std::to_string(got.id) + " is not in the scene");
      if (truth == scene.tags.end()) continue;
      const double error = worst_corner_error(got, *truth, scene);
      if (fully_visible(*truth, scene, 2.0)) {
        VERIFY_MSG(error < 1.0, where + std::to_string(got.id) + " corners off by " +
                                    std::to_string(error) + " px");
      }
      const auto reference_match = std::find_if(expected.begin(), expected.end(),
                                                [&](const Detection& d) { return d.id == got.id; });
      if (kind == Backend::kCuda && reference_match != expected.end()) {
        const double reference_error = worst_corner_error(*reference_match, *truth, scene);
        VERIFY_MSG(error <= reference_error + 1.0,
                   where + std::to_string(got.id) + " corners off by " + std::to_string(error) +
                       " px, the reference by " + std::to_string(reference_error) + " px");
      }
    }
    if (kind == Backend::kCuda) {
      for (const Detection& want : expected) {
        VERIFY_MSG(std::any_of(found.begin(), found.end(),
                               [&](const Detection& d) { return d.id == want.id; }),
                   where + std::to_string(want.id) + " found by cpu is missing");
      }
    }
  }
  std::printf("  %-8s edge scene: cpu found %d, %s found %d of %u\n",
              std::string(backend_name(kind)).c_str(), reference_tags,
              std::string(backend_name(kind)).c_str(), found_tags, 4 * kFrames);
  // The scene must actually cut tags: fewer than all of them are decodable.
  VERIFY_MSG(reference_tags > 0 && reference_tags < static_cast<int>(4 * kFrames),
             "the edge scene clips some tags");
}

// The segmenter's buffers grow by half again, so a slowly rising requirement
// does not free and reallocate (synchronizing the device) on every frame.
void check_device_vector_growth() {
  DeviceVector<std::uint32_t> buffer(100);
  VERIFY(buffer.capacity() == 100 && buffer.get() != nullptr);
  buffer.reserve(101);
  VERIFY_MSG(buffer.capacity() == 150,
             "grows to 150 elements, not " + std::to_string(buffer.capacity()));
  const std::uint32_t* grown = buffer.get();
  buffer.reserve(150);
  VERIFY_MSG(buffer.get() == grown && buffer.capacity() == 150, "reuses room it already has");
  buffer.reserve(1000);
  VERIFY(buffer.capacity() == 1000);
}

// Vertical black and white stripes one or two decimated pixels wide: every
// pixel borders another component, far more boundary points than the CUDA
// backend's first buffers hold, so it must grow them and gather again. Nothing
// is a tag; a scene detected afterwards still is.
void check_dense_edges() {
  DetectorOptions options;
  auto backend = create_backend(Backend::kCuda, options, kWidth, kHeight);
  GrayConverter converter;
  std::mt19937 random(7);
  std::vector<std::uint8_t> stripes(static_cast<std::size_t>(kWidth) * kHeight);
  std::uint8_t value = 0;
  for (int x = 0; x < kWidth;) {
    const int width = options.decimate * (1 + static_cast<int>(random() & 1U));
    for (int end = std::min(kWidth, x + width); x < end; ++x) {
      for (int y = 0; y < kHeight; ++y) {
        stripes[static_cast<std::size_t>(y) * kWidth + x] = value;
      }
    }
    value = 255 - value;
  }
  for (int pass = 0; pass < 2; ++pass) {
    const auto found =
        run_backend(*backend, stripes, converter, nullptr, nullptr, /*device_input=*/false);
    VERIFY_MSG(found.empty(), std::to_string(found.size()) + " tags found in stripes");
  }
  const SyntheticScene scene = make_scene("tag36h11", 0, kWidth, kHeight);
  const auto found =
      run_backend(*backend, render_scene(scene, 1), converter, nullptr, nullptr, false);
  VERIFY_MSG(found.size() == scene.tags.size(),
             std::to_string(found.size()) + " tags found after the stripes");
}

std::vector<Backend> backends_under_test() {
  std::vector<Backend> backends;
  for (Backend backend : available_backends()) {
    // VPI on the CPU everywhere; its PVA path is exercised where present.
    backends.push_back(backend == Backend::kVpi ? Backend::kVpiCpu : backend);
  }
  return backends;
}

}  // namespace
}  // namespace tag_detection

int main() {
  using tag_detection::Backend;
  using tag_detection::testing::TestCase;
  std::vector<TestCase> cases;
  for (Backend backend : tag_detection::backends_under_test()) {
    for (const char* family : {"tag36h11", "tagStandard41h12", "tag25h9"}) {
      for (bool device_input : {false, true}) {
        cases.push_back({std::string(tag_detection::backend_name(backend)) + " " + family +
                             (device_input ? " device input" : " host input"),
                         [=] { tag_detection::check_backend(backend, family, device_input); }});
      }
    }
  }
  for (Backend backend : tag_detection::backends_under_test()) {
    if (backend == Backend::kCpu) continue;
    cases.push_back({std::string(tag_detection::backend_name(backend)) + " tags cut by the edge",
                     [=] { tag_detection::check_edge_tags(backend); }});
  }
  cases.push_back({"device buffers grow by half", tag_detection::check_device_vector_growth});
  cases.push_back({"cuda many more edges than buffers", tag_detection::check_dense_edges});
  return tag_detection::testing::run(cases);
}
