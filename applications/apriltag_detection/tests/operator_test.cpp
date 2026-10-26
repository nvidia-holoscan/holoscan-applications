// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// The operators on their own, through the SDK's operator test harness: images
// of each supported encoding and layout in, detections with the scene's ids,
// corners and poses out, for every compiled backend; a change of image size
// and non-default options; unsupported images rejected; the detector off the
// legacy default CUDA stream; the reporter's JSON lines; and the overlay's
// Holoviz tensors shaped, scaled and stamped as Holoviz expects.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/testing/operator_test_harness.hpp>

#include "detector/synthetic_scene.hpp"
#include "operators/apriltag_detector_op.hpp"
#include "operators/detection_report_op.hpp"
#if APRILTAG_DETECTION_ENABLE_VISUALIZATION
#include <holoscan/visualization/holoviz_op.hpp>

#include "operators/apriltag_overlay_op.hpp"
#endif
#include "test_support.hpp"

namespace tag_detection {
namespace {

constexpr int kWidth = 640;
constexpr int kHeight = 480;

// A host tensor owning `bytes`, with `shape` and, unless empty, `strides` (in
// elements).
std::shared_ptr<holoscan::Tensor> host_tensor(std::vector<std::uint8_t> bytes,
                                              std::vector<std::int64_t> shape,
                                              std::vector<std::int64_t> strides = {}) {
  struct Storage {
    DLManagedTensor managed{};
    std::vector<std::int64_t> shape;
    std::vector<std::int64_t> strides;
    std::vector<std::uint8_t> bytes;
  };
  auto storage = std::make_unique<Storage>();
  storage->shape = std::move(shape);
  storage->strides = std::move(strides);
  storage->bytes = std::move(bytes);
  storage->managed.dl_tensor = {
      .data = storage->bytes.data(),
      .device = {kDLCPU, 0},
      .ndim = static_cast<std::int32_t>(storage->shape.size()),
      .dtype = {kDLUInt, 8, 1},
      .shape = storage->shape.data(),
      .strides = storage->strides.empty() ? nullptr : storage->strides.data(),
      .byte_offset = 0};
  storage->managed.manager_ctx = storage.get();
  storage->managed.deleter = [](DLManagedTensor* value) {
    delete static_cast<Storage*>(value->manager_ctx);
  };
  return std::make_shared<holoscan::Tensor>(&storage.release()->managed);
}

// How an image's pixels are laid out in its tensor. (The harness stages only
// contiguous tensors, so rows cannot be padded here.)
struct Layout {
  bool rank2{};  // MONO8 as [height, width] rather than [height, width, 1].
};

holoscan::schema::ImageT scene_image(const SyntheticScene& scene,
                                     holoscan::schema::ImageEncoding encoding, Layout layout = {}) {
  using holoscan::schema::ImageEncoding_BGR8;
  using holoscan::schema::ImageEncoding_BGRA8;
  using holoscan::schema::ImageEncoding_MONO8;
  using holoscan::schema::ImageEncoding_RGBA8;
  const bool alpha = encoding == ImageEncoding_RGBA8 || encoding == ImageEncoding_BGRA8;
  const bool bgr = encoding == ImageEncoding_BGR8 || encoding == ImageEncoding_BGRA8;
  const int channels = encoding == ImageEncoding_MONO8 ? 1 : alpha ? 4 : 3;
  const auto rgb = render_scene(scene, channels == 1 ? 1 : 3);
  const int source_channels = channels == 1 ? 1 : 3;
  const std::size_t row = static_cast<std::size_t>(scene.width) * channels;
  std::vector<std::uint8_t> pixels(row * scene.height, 0);
  for (int y = 0; y < scene.height; ++y) {
    for (int x = 0; x < scene.width; ++x) {
      const std::uint8_t* in =
          rgb.data() + (static_cast<std::size_t>(y) * scene.width + x) * source_channels;
      std::uint8_t* out = pixels.data() + y * row + static_cast<std::size_t>(x) * channels;
      if (channels == 1) {
        out[0] = in[0];
        continue;
      }
      out[0] = in[bgr ? 2 : 0];
      out[1] = in[1];
      out[2] = in[bgr ? 0 : 2];
      if (alpha) out[3] = 255;
    }
  }
  std::vector<std::int64_t> shape{scene.height, scene.width};
  std::vector<std::int64_t> strides{static_cast<std::int64_t>(row)};
  if (!layout.rank2) {
    shape.push_back(channels);
    strides.push_back(channels);
  }
  strides.push_back(1);
  holoscan::schema::ImageT image;
  image.width = scene.width;
  image.height = scene.height;
  image.encoding = encoding;
  image.data = host_tensor(std::move(pixels), std::move(shape), std::move(strides));
  return image;
}

// Checks one message's detections against `scene`'s tags, corners and poses.
void check_message(const schema::AprilTagDetectionsT& message, const SyntheticScene& scene,
                   bool poses) {
  VERIFY(message.image_width == scene.width && message.image_height == scene.height);
  VERIFY(message.family == "tag36h11");
  VERIFY(message.dropped == 0);
  VERIFY_MSG(message.detections.size() == scene.tags.size(),
             "frame " + std::to_string(message.frame) + ": " +
                 std::to_string(message.detections.size()) + " tags");
  for (const schema::TagDetection& tag : message.detections) {
    const auto truth = std::find_if(scene.tags.begin(), scene.tags.end(),
                                    [&](const SceneTag& t) { return t.id == tag.id(); });
    VERIFY_MSG(truth != scene.tags.end(), "unexpected id " + std::to_string(tag.id()));
    if (truth == scene.tags.end()) continue;
    const auto corners = project_corners(*truth, scene.intrinsics, scene.tag_size);
    for (int c = 0; c < 4; ++c) {
      const auto& corner = *tag.corners()->Get(c);
      VERIFY(std::hypot(corner.x() - corners[c][0], corner.y() - corners[c][1]) < 1.0);
    }
    VERIFY(tag.has_pose() == poses);
    if (!poses) continue;
    const auto& t = truth->pose.translation;
    VERIFY(std::hypot(tag.translation()->Get(0) - t[0], tag.translation()->Get(1) - t[1],
                      tag.translation()->Get(2) - t[2]) < 0.02 * t[2]);
  }
}

void check_detector(Backend backend, holoscan::schema::ImageEncoding encoding, Layout layout = {}) {
  constexpr std::uint32_t kFrames = 3;
  const SyntheticScene first = make_scene("tag36h11", 0, kWidth, kHeight);
  AprilTagDetectorConfig config{
      .backend = backend, .intrinsics = first.intrinsics, .tag_size = first.tag_size};
  auto h = holoscan::testing::create_operator_test<AprilTagDetectorOp>(config);
  for (std::uint32_t frame = 0; frame < kFrames; ++frame) {
    h.push("image", scene_image(make_scene("tag36h11", frame, kWidth, kHeight), encoding, layout));
  }
  h.capture<schema::AprilTagDetectionsT>("detections");
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return;
  const auto messages = result.values<schema::AprilTagDetectionsT>("detections");
  VERIFY(messages.size() == kFrames);
  for (std::size_t i = 0; i < messages.size(); ++i) {
    VERIFY(messages[i].frame == i);
    VERIFY(messages[i].backend == backend_name(backend));
    check_message(messages[i],
                  make_scene("tag36h11", static_cast<std::uint32_t>(i), kWidth, kHeight), true);
  }
}

// The backend is rebuilt when the image size changes, and detects at both.
void check_size_change(Backend backend) {
  auto h = holoscan::testing::create_operator_test<AprilTagDetectorOp>(
      AprilTagDetectorConfig{.backend = backend});
  const std::array<SyntheticScene, 3> scenes{make_scene("tag36h11", 0, kWidth, kHeight),
                                             make_scene("tag36h11", 1, 800, 600),
                                             make_scene("tag36h11", 2, kWidth, kHeight)};
  for (const SyntheticScene& scene : scenes) {
    h.push("image", scene_image(scene, holoscan::schema::ImageEncoding_MONO8));
  }
  h.capture<schema::AprilTagDetectionsT>("detections");
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return;
  const auto messages = result.values<schema::AprilTagDetectionsT>("detections");
  VERIFY(messages.size() == scenes.size());
  for (std::size_t i = 0; i < messages.size() && i < scenes.size(); ++i) {
    check_message(messages[i], scenes[i], false);
  }
}

// Options other than the defaults reach the backend and still find every tag.
void check_options(Backend backend) {
  AprilTagDetectorConfig config{.backend = backend};
  config.options.decimate = 1;
  config.options.refine_edges = false;
  config.options.max_hamming = 0;
  config.options.decode_sharpening = 0.0;
  auto h = holoscan::testing::create_operator_test<AprilTagDetectorOp>(config);
  const SyntheticScene scene = make_scene("tag36h11", 3, kWidth, kHeight);
  h.push("image", scene_image(scene, holoscan::schema::ImageEncoding_MONO8));
  h.capture<schema::AprilTagDetectionsT>("detections");
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return;
  const auto message = result.value<schema::AprilTagDetectionsT>("detections");
  check_message(message, scene, false);
  for (const schema::TagDetection& tag : message.detections) {
    VERIFY(tag.hamming() == 0);
  }
}

// Detections carry their image's capture time and frame identity.
void check_detector_provenance() {
  constexpr std::chrono::nanoseconds kCaptured = std::chrono::milliseconds(250);
  constexpr std::uint64_t kFrameId = 42;
  auto h = holoscan::testing::create_operator_test<AprilTagDetectorOp>(
      AprilTagDetectorConfig{.backend = Backend::kCpu});
  holoscan::testing::TestSample<holoscan::schema::ImageT> sample{
      .data = scene_image(make_scene("tag36h11", 0, kWidth, kHeight),
                          holoscan::schema::ImageEncoding_MONO8),
      .capture_coordinate = kCaptured};
  sample.metadata.frame_id = kFrameId;
  h.push_samples<holoscan::schema::ImageT>("image", {std::move(sample)});
  h.capture<schema::AprilTagDetectionsT>("detections");
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return;
  const auto samples = result.samples<schema::AprilTagDetectionsT>("detections");
  VERIFY(samples.size() == 1);
  if (samples.size() != 1) return;
  const holoscan::SampleMetadata& metadata = samples.front().metadata;
  VERIFY(metadata.source_clock_id.valid() && metadata.capture_timestamp_ns == kCaptured.count());
  VERIFY_MSG(metadata.frame_id == kFrameId,
             "frame id " + std::to_string(metadata.frame_id) + " forwarded");
}

// How long the unrelated stream below is held.
constexpr std::chrono::seconds kHoldFor{5};

// With host input the runtime gives the detector no CUDA stream. Its CUDA work
// must then not go to the legacy default stream, which waits for every other
// blocking stream in the process: here, one held by a host function while the
// detector handles its second frame (the first allocates, which synchronizes
// the device anyway).
void check_detector_stream_with_host_input() {
  using holoscan::testing::frame;
  using holoscan::testing::input;
  using std::chrono::milliseconds;
  auto h = holoscan::testing::create_operator_test<AprilTagDetectorOp>(
      AprilTagDetectorConfig{.backend = Backend::kCuda});
  const SyntheticScene scene = make_scene("tag36h11", 0, kWidth, kHeight);
  for (int i = 0; i < 2; ++i) {
    h.push_frame(frame(input("image", scene_image(scene, holoscan::schema::ImageEncoding_RGB8))));
  }
  h.capture<schema::AprilTagDetectionsT>("detections");
  auto running = h.run_async({.frames = holoscan::testing::FrameRelease::kManual});
  VERIFY(running.release_frame(milliseconds(30000)).has_value());
  VERIFY(running.wait_for_output("detections", 1, milliseconds(30000)));

  struct Hold {
    std::atomic<bool> released{false};
  } hold;
  cudaStream_t other = nullptr;
  VERIFY(cudaStreamCreate(&other) == cudaSuccess);
  VERIFY(cudaLaunchHostFunc(
             other,
             [](void* data) {
               auto* held = static_cast<Hold*>(data);
               const auto until = std::chrono::steady_clock::now() + kHoldFor;
               while (!held->released.load() && std::chrono::steady_clock::now() < until) {
                 std::this_thread::sleep_for(milliseconds(1));
               }
             },
             &hold) == cudaSuccess);
  const auto start = std::chrono::steady_clock::now();
  VERIFY(running.release_frame(milliseconds(30000)).has_value());
  VERIFY(running.wait_for_output("detections", 2, milliseconds(30000)));
  const auto elapsed = std::chrono::steady_clock::now() - start;
  hold.released = true;
  cudaStreamSynchronize(other);
  cudaStreamDestroy(other);

  const auto result = running.finish();
  VERIFY_MSG(result.ok(), result.describe_failure());
  VERIFY_MSG(elapsed < kHoldFor / 2,
             "the detector waited " +
                 std::to_string(std::chrono::duration_cast<milliseconds>(elapsed).count()) +
                 " ms for an unrelated stream");
  if (result.ok()) {
    for (const auto& message : result.values<schema::AprilTagDetectionsT>("detections")) {
      check_message(message, scene, false);
    }
  }
}

void check_rejects_unsupported_encoding() {
  auto h = holoscan::testing::create_operator_test<AprilTagDetectorOp>(
      AprilTagDetectorConfig{.backend = Backend::kCpu});
  holoscan::schema::ImageT image =
      scene_image(make_scene("tag36h11", 0, kWidth, kHeight), holoscan::schema::ImageEncoding_RGB8);
  image.encoding = holoscan::schema::ImageEncoding_YUYV;
  h.push("image", image);
  h.capture<schema::AprilTagDetectionsT>("detections");
  const auto result = h.run();
  VERIFY_MSG(!result.ok(), "a YUYV image is rejected");
  VERIFY(result.describe_failure().find("unsupported image encoding") != std::string::npos);
}

void check_rejects_bad_config() {
  VERIFY(testing::throws<std::invalid_argument>([] {
    AprilTagDetectorConfig config;
    config.options.family = "tag99h99";
    AprilTagDetectorOp op(config);
  }));
  VERIFY(testing::throws<std::invalid_argument>([] {
    AprilTagDetectorOp op(AprilTagDetectorConfig{.intrinsics = CameraIntrinsics{900, 900, 320, 240},
                                                 .tag_size = 0.0});
  }));
  VERIFY(testing::throws<std::invalid_argument>([] {
    AprilTagDetectorConfig config;
    config.options.decimate = 0;
    AprilTagDetectorOp op(config);
  }));
}

// One tag, id 7, in a 200x100 image; with a pose when `pose` is set.
schema::AprilTagDetectionsT one_tag_message(std::uint64_t frame, bool pose) {
  schema::AprilTagDetectionsT message;
  message.frame = frame;
  message.image_width = 200;
  message.image_height = 100;
  message.backend = "cuda";
  message.family = "tag36h11";
  message.detect_time_us = 1500.0F;
  const std::array<schema::Point2f, 4> corners{schema::Point2f(20, 80), schema::Point2f(80, 80),
                                               schema::Point2f(80, 20), schema::Point2f(20, 20)};
  std::array<float, 9> rotation{};
  std::array<float, 3> translation{};
  if (pose) {
    rotation = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    translation = {0.25F, -0.5F, 2.0F};
  }
  message.detections.push_back(schema::TagDetection(
      7, 0, 50.0F, schema::Point2f(50, 50), ::flatbuffers::span<const schema::Point2f, 4>(corners),
      pose, ::flatbuffers::span<const float, 9>(rotation),
      ::flatbuffers::span<const float, 3>(translation), 0.0F));
  return message;
}

// The reporter writes one JSON object per frame and counts what it saw.
void check_report_json_lines() {
  const std::filesystem::path output =
      std::filesystem::current_path() / "apriltag_operator_test_report.jsonl";
  std::filesystem::remove(output);
  auto report = std::make_shared<BackendReport>();
  auto h = holoscan::testing::create_operator_test<DetectionReportOp>(DetectionReportConfig{
      .report = report, .completion = std::make_shared<RunCompletion>(), .output = output});
  h.push("detections", one_tag_message(0, false));
  h.push("detections", one_tag_message(1, true));
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  VERIFY(report->frames == 2 && report->tags == 2 && report->detect_ms.size() == 2);
  VERIFY(!report->verified && report->passed());

  std::ifstream file(output);
  std::vector<std::string> lines;
  for (std::string line; std::getline(file, line);) lines.push_back(line);
  VERIFY_MSG(lines.size() == 2, std::to_string(lines.size()) + " lines");
  if (lines.size() != 2) return;
  const auto has = [](const std::string& line, const std::string& text) {
    return line.find(text) != std::string::npos;
  };
  VERIFY(has(lines[0], R"({"frame":0,"backend":"cuda","family":"tag36h11","width":200,)"));
  VERIFY(has(lines[0], R"("detect_ms":1.5,"dropped":0,"tags":[{"id":7,"hamming":0,)"));
  VERIFY(has(lines[0], R"("corners":[[20,80],[80,80],[80,20],[20,20]]})"));
  VERIFY(!has(lines[0], "rotation"));
  VERIFY(has(lines[1], R"("rotation":[1,0,0,0,1,0,0,0,1],"translation":[0.25,-0.5,2])"));
  VERIFY(lines[0].back() == '}' && lines[1].back() == '}');
  std::filesystem::remove(output);
}

#if APRILTAG_DETECTION_ENABLE_VISUALIZATION
void check_overlay() {
  // The frame's provenance, which the overlay's layers must carry. (The harness
  // activates the overlay at the capture time, so the frame id is what tells
  // forwarded provenance from a fresh stamp.)
  constexpr std::chrono::nanoseconds kCaptured = std::chrono::milliseconds(1234);
  constexpr std::uint64_t kFrameId = 9;
  auto h = holoscan::testing::create_operator_test<AprilTagOverlayOp>();
  holoscan::testing::TestSample<schema::AprilTagDetectionsT> sample{
      .data = one_tag_message(0, false), .capture_coordinate = kCaptured};
  sample.metadata.frame_id = kFrameId;
  h.push_samples<schema::AprilTagDetectionsT>("detections", {std::move(sample)});
  h.capture_snapshot<holoscan::Tensor>("outlines");
  h.capture_snapshot<holoscan::Tensor>("labels");
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return;

  const auto outlines = result.snapshot<holoscan::Tensor>("outlines");
  for (const auto* metadata : {&outlines.sample_metadata(),
                               &result.snapshot<holoscan::Tensor>("labels").sample_metadata()}) {
    VERIFY_MSG(metadata->source_clock_id.valid() &&
                   metadata->capture_timestamp_ns == kCaptured.count() &&
                   metadata->frame_id == kFrameId,
               "overlay layers carry the frame's provenance, not capture " +
                   std::to_string(metadata->capture_timestamp_ns) + " ns, frame " +
                   std::to_string(metadata->frame_id));
  }
  VERIFY(outlines.shape() == std::vector<std::int64_t>{4 * kOverlayLinesPerTag});
  std::vector<float> lines(4 * kOverlayLinesPerTag);
  std::memcpy(lines.data(), outlines.bytes().data(), lines.size() * sizeof(float));
  const auto near = [&lines](std::size_t first, std::array<float, 4> want) {
    for (std::size_t i = 0; i < want.size(); ++i) {
      if (std::abs(lines[first + i] - want[i]) > 1e-6F) return false;
    }
    return true;
  };
  // The first edge runs from corner 0 to corner 1, normalized by the image size.
  VERIFY(near(0, {0.1F, 0.8F, 0.4F, 0.8F}));
  // The x-axis stroke runs from the center to the middle of corners 1 and 2.
  VERIFY(near(16, {0.25F, 0.5F, 0.4F, 0.5F}));

  const auto labels = result.snapshot<holoscan::Tensor>("labels");
  VERIFY(labels.shape() == (std::vector<std::int64_t>{
                               static_cast<std::int64_t>(kMaxDetectionsPerFrame),
                               holoscan::visualization::kTextRunHeaderBytes + kOverlayLabelBytes}));
  const auto* first_text = reinterpret_cast<const char*>(labels.bytes().data()) +
                           holoscan::visualization::kTextRunHeaderBytes;
  VERIFY(std::string(first_text, 4) == "id 7");
}

// Detections without a capture time still reach Holoviz, which selects layers
// by capture time: the overlay stamps them itself.
void check_overlay_unstamped() {
  auto h = holoscan::testing::create_operator_test<AprilTagOverlayOp>();
  h.push("detections", one_tag_message(0, false));
  h.capture_snapshot<holoscan::Tensor>("outlines");
  h.capture_snapshot<holoscan::Tensor>("labels");
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return;
  VERIFY_MSG(
      result.snapshot<holoscan::Tensor>("outlines").sample_metadata().source_clock_id.valid(),
      "unstamped detections give stamped layers");
}
#endif

}  // namespace
}  // namespace tag_detection

int main() {
  using tag_detection::Backend;
  using tag_detection::testing::TestCase;
  std::vector<TestCase> cases;
  for (Backend backend : tag_detection::available_backends()) {
    const Backend run = backend == Backend::kVpi ? Backend::kVpiCpu : backend;
    const std::string name(tag_detection::backend_name(run));
    for (const auto encoding :
         {holoscan::schema::ImageEncoding_MONO8, holoscan::schema::ImageEncoding_RGB8,
          holoscan::schema::ImageEncoding_BGR8, holoscan::schema::ImageEncoding_RGBA8,
          holoscan::schema::ImageEncoding_BGRA8}) {
      cases.push_back({"detector " + name + " " + holoscan::schema::EnumNameImageEncoding(encoding),
                       [=] { tag_detection::check_detector(run, encoding); }});
    }
    cases.push_back({"detector " + name + " MONO8 as [height, width]", [=] {
                       tag_detection::check_detector(run, holoscan::schema::ImageEncoding_MONO8,
                                                     {.rank2 = true});
                     }});
    cases.push_back({"detector " + name + " image size change",
                     [=] { tag_detection::check_size_change(run); }});
  }
  for (Backend backend : {Backend::kCpu, Backend::kCuda}) {
    cases.push_back(
        {"detector " + std::string(tag_detection::backend_name(backend)) + " non-default options",
         [=] { tag_detection::check_options(backend); }});
  }
  cases.push_back({"detector rejects unsupported encodings",
                   tag_detection::check_rejects_unsupported_encoding});
  cases.push_back({"detector rejects bad configuration", tag_detection::check_rejects_bad_config});
  cases.push_back({"detector forwards provenance", tag_detection::check_detector_provenance});
  cases.push_back({"detector keeps off the legacy default stream with host input",
                   tag_detection::check_detector_stream_with_host_input});
  cases.push_back({"reporter JSON lines", tag_detection::check_report_json_lines});
#if APRILTAG_DETECTION_ENABLE_VISUALIZATION
  cases.push_back({"overlay layers", tag_detection::check_overlay});
  cases.push_back({"overlay stamps unstamped detections", tag_detection::check_overlay_unstamped});
#endif
  return tag_detection::testing::run(cases);
}
