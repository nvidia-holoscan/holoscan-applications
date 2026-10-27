// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// AprilTag Detection: images in, tag ids, corners and poses out, on a choice of
// backends. One source (synthetic scenes, image files or a webcam) fans out to
// one AprilTagDetectorOp per selected backend, so backends can be compared on
// identical frames; each detector feeds a reporter, and the first optionally
// draws on a Holoviz display.

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/managed_shutdown_signals.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/time/realtime_clock.hpp>
#if APRILTAG_DETECTION_ENABLE_VISUALIZATION
#include <holoscan/visualization/holoviz_op.hpp>

#include "operators/apriltag_overlay_op.hpp"
#endif

#include "detector/detection.hpp"
#include "detector/pnm_image.hpp"
#include "detector/synthetic_scene.hpp"
#include "operators/apriltag_detector_op.hpp"
#include "operators/detection_report_op.hpp"
#include "operators/image_sources.hpp"

namespace {

using std::chrono_literals::operator""ms;
namespace td = tag_detection;

constexpr std::string_view kAppName = "apriltag_detection";

enum class DisplayMode { kAuto, kOn, kOff };
enum class Source { kSynthetic, kImages, kCamera };

// How long a finite run may take beyond its expected time.
constexpr auto kTimeoutAllowance = std::chrono::seconds{120};

struct Options {
  Source source{Source::kSynthetic};
  std::filesystem::path images;
  std::filesystem::path device{"/dev/video0"};
  std::vector<td::Backend> backends{td::Backend::kCuda};
  td::DetectorOptions detector;
  double tag_size{0.12};
  std::optional<td::CameraIntrinsics> intrinsics;
  std::optional<int> width;
  std::optional<int> height;
  std::optional<std::uint32_t> count;
  std::uint32_t period_ms{33};
  bool device_input{false};
  DisplayMode display{DisplayMode::kAuto};
  bool compile_only{false};
  bool help{false};
};

bool env_set(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && *value != '\0';
}

bool parse_bool(std::string_view value, std::string_view option) {
  if (value == "true") return true;
  if (value == "false") return false;
  throw std::invalid_argument(std::string(option) + " requires true or false");
}

template <typename T>
T parse_number(std::string_view value, std::string_view option) {
  T result{};
  const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
  if (error != std::errc() || end != value.data() + value.size()) {
    throw std::invalid_argument(std::string(option) + " requires a number, got '" +
                                std::string(value) + "'");
  }
  return result;
}

std::vector<td::Backend> parse_backends(std::string_view value) {
  if (value == "all") {
    return td::available_backends();
  }
  std::vector<td::Backend> backends;
  while (!value.empty()) {
    const std::size_t comma = value.find(',');
    const std::string_view name = value.substr(0, comma);
    const auto backend = td::parse_backend(name);
    if (!backend) {
      throw std::invalid_argument("unknown backend '" + std::string(name) +
                                  "'; use cpu, cuda, vpi, vpi-pva, vpi-cpu or all");
    }
    if (std::find(backends.begin(), backends.end(), *backend) != backends.end()) {
      throw std::invalid_argument("backend '" + std::string(name) + "' is listed twice");
    }
    backends.push_back(*backend);
    value = comma == std::string_view::npos ? std::string_view{} : value.substr(comma + 1);
  }
  if (backends.empty()) {
    throw std::invalid_argument("--backend needs at least one backend");
  }
  return backends;
}

Options parse_args(int argc, char** argv) {
  Options opts;
  std::optional<double> fx, fy, cx, cy;
  for (int i = 1; i < argc; ++i) {
    const std::string_view option{argv[i]};
    if (option == "--help" || option == "-h") {
      opts.help = true;
      return opts;
    }
    if (i + 1 >= argc) {
      throw std::invalid_argument("missing value for " + std::string(option));
    }
    const std::string_view value{argv[++i]};
    if (option == "--source") {
      if (value == "synthetic") {
        opts.source = Source::kSynthetic;
      } else if (value == "images") {
        opts.source = Source::kImages;
      } else if (value == "camera") {
        opts.source = Source::kCamera;
      } else {
        throw std::invalid_argument("--source must be synthetic, images or camera");
      }
    } else if (option == "--images") {
      opts.images = value;
    } else if (option == "--device") {
      opts.device = value;
    } else if (option == "--backend") {
      opts.backends = parse_backends(value);
    } else if (option == "--family") {
      opts.detector.family = value;
    } else if (option == "--tag-size") {
      opts.tag_size = parse_number<double>(value, option);
    } else if (option == "--fx") {
      fx = parse_number<double>(value, option);
    } else if (option == "--fy") {
      fy = parse_number<double>(value, option);
    } else if (option == "--cx") {
      cx = parse_number<double>(value, option);
    } else if (option == "--cy") {
      cy = parse_number<double>(value, option);
    } else if (option == "--decimate") {
      opts.detector.decimate = parse_number<int>(value, option);
    } else if (option == "--max-hamming") {
      opts.detector.max_hamming = parse_number<int>(value, option);
    } else if (option == "--refine-edges") {
      opts.detector.refine_edges = parse_bool(value, option);
    } else if (option == "--sharpening") {
      opts.detector.decode_sharpening = parse_number<double>(value, option);
    } else if (option == "--threads") {
      opts.detector.cpu_threads = parse_number<int>(value, option);
    } else if (option == "--width") {
      opts.width = parse_number<int>(value, option);
    } else if (option == "--height") {
      opts.height = parse_number<int>(value, option);
    } else if (option == "--count") {
      opts.count = parse_number<std::uint32_t>(value, option);
    } else if (option == "--period-ms") {
      opts.period_ms = parse_number<std::uint32_t>(value, option);
    } else if (option == "--device-input") {
      opts.device_input = parse_bool(value, option);
    } else if (option == "--display") {
      opts.display = value == "auto"             ? DisplayMode::kAuto
                     : parse_bool(value, option) ? DisplayMode::kOn
                                                 : DisplayMode::kOff;
    } else if (option == "--compile-only") {
      opts.compile_only = parse_bool(value, option);
    } else {
      throw std::invalid_argument("unknown option: " + std::string(option));
    }
  }
  const int given = fx.has_value() + fy.has_value() + cx.has_value() + cy.has_value();
  if (given != 0 && given != 4) {
    throw std::invalid_argument("--fx, --fy, --cx and --cy go together");
  }
  if (given == 4) {
    if (opts.source == Source::kSynthetic) {
      throw std::invalid_argument("--fx..--cy apply to --source images or camera only");
    }
    opts.intrinsics = td::CameraIntrinsics{*fx, *fy, *cx, *cy};
  }
  return opts;
}

void usage(const char* argv0) {
  std::cout
      << "Usage: " << argv0
      << " [options]\n"
         "Input:\n"
         "  --source synthetic|images|camera\n"
         "                             Rendered tags with ground truth, image files, or a\n"
         "                             V4L2 webcam (default: synthetic)\n"
         "  --images DIR               PGM/PPM images, read once in name order\n"
         "  --device PATH              The webcam (default: /dev/video0)\n"
         "  --count N                  Frames (default: 300 synthetic, every image once,\n"
         "                             camera until Ctrl+C)\n"
         "  --period-ms N              Synthetic and image period, 1..1000 ms (default: 33)\n"
         "  --width N --height N       Synthetic size (default: 1280x720), or a YUYV size\n"
         "                             the camera offers (default: 640x480)\n"
         "  --device-input true|false  Synthetic frames in GPU memory (default: false)\n"
         "Detection:\n"
         "  --backend LIST             Comma-separated: cpu, cuda, vpi, vpi-pva, vpi-cpu, or\n"
         "                             all compiled backends (default: cuda)\n"
         "  --family NAME              AprilTag family (default: tag36h11)\n"
         "  --max-hamming N            Bit errors to correct, 0..2 (default: 1)\n"
         "  --decimate N               Detect quads at 1/N resolution, 1..8 (default: 2)\n"
         "  --refine-edges true|false  Refine edges at full resolution (default: true)\n"
         "  --sharpening X             Bit sharpening, 0..1 (default: 0.25)\n"
         "  --threads N                CPU backend worker threads (default: 1)\n"
         "Pose (--source images or camera; synthetic scenes provide their own):\n"
         "  --fx F --fy F --cx C --cy C  Camera intrinsics in pixels\n"
         "  --tag-size M               Black-border width in meters (default: 0.12)\n"
         "Output:\n"
         "  --display auto|true|false  Show the first backend's detections (default: auto,\n"
         "                             on for the camera, else when a display is available)\n"
         "  --compile-only true|false  Build and compile the graph, then exit without running\n"
         "                             it (default: false)\n";
}

void validate(Options& opts) {
  if (opts.period_ms == 0 || opts.period_ms > 1000) {
    throw std::invalid_argument("--period-ms must be 1..1000");
  }
  if (opts.count && *opts.count == 0) {
    throw std::invalid_argument("--count must be positive");
  }
  if (!opts.images.empty() && opts.source != Source::kImages) {
    throw std::invalid_argument("--images needs --source images");
  }
  if (opts.device_input && opts.source != Source::kSynthetic) {
    throw std::invalid_argument("--device-input applies to synthetic frames only");
  }
  if (opts.source == Source::kImages) {
    if (opts.images.empty()) {
      throw std::invalid_argument("--source images needs --images DIR");
    }
    if (opts.count || opts.width || opts.height) {
      throw std::invalid_argument("--source images reads every image once at its own size");
    }
  } else if (opts.source == Source::kSynthetic) {
    opts.width = opts.width.value_or(1280);
    opts.height = opts.height.value_or(720);
    opts.count = opts.count.value_or(300);
    // The synthetic scene's own camera and tags.
    const td::SyntheticScene scene =
        td::make_scene(opts.detector.family, 0, *opts.width, *opts.height);
    opts.intrinsics = scene.intrinsics;
    opts.tag_size = scene.tag_size;
  } else {
    opts.width = opts.width.value_or(640);
    opts.height = opts.height.value_or(480);
    td::check_camera(td::CameraSourceConfig{
        .device = opts.device, .width = *opts.width, .height = *opts.height});
    if (opts.display == DisplayMode::kAuto) {
      opts.display = DisplayMode::kOn;
    }
  }
  if (!(opts.tag_size > 0)) {
    throw std::invalid_argument("--tag-size must be positive");
  }
  td::validate_options(opts.detector);
  for (const td::Backend backend : opts.backends) {
    td::check_available(backend);
  }
}

double percentile(std::vector<float> values, double fraction) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const auto index = static_cast<std::size_t>(fraction * static_cast<double>(values.size() - 1));
  return values[index];
}

// `expected_frames` is 0 for a live camera run, which has no expected count.
void print_report(const td::BackendReport& report, std::uint32_t expected_frames) {
  double mean = 0.0;
  for (const float ms : report.detect_ms) mean += ms;
  mean = report.detect_ms.empty() ? 0.0 : mean / static_cast<double>(report.detect_ms.size());
  std::cout << std::fixed << std::setprecision(2) << kAppName << ": backend=" << report.backend
            << " frames=" << report.frames;
  if (expected_frames != 0) {
    std::cout << '/' << expected_frames;
  }
  std::cout << " tags=" << report.tags << " detect_ms(mean/p50/p95/max)=" << mean << '/'
            << percentile(report.detect_ms, 0.5) << '/' << percentile(report.detect_ms, 0.95) << '/'
            << percentile(report.detect_ms, 1.0);
  if (report.verified) {
    std::cout << " truth: missed=" << report.missed_tags << " extra=" << report.extra_tags
              << " misplaced=" << report.misplaced_tags << '/' << report.expected_tags
              << " max_corner_px=" << std::setprecision(3) << report.max_corner_error_px
              << " max_translation=" << std::setprecision(2) << 100.0 * report.max_translation_error
              << '%';
  }
  if (report.dropped != 0) {
    std::cout << " dropped=" << report.dropped;
  }
  std::cout << '\n';
}

using SignalDomain = holoscan::expected<holoscan::ManagedShutdownSignals, holoscan::Error>;

int execute(const Options& opts, SignalDomain& signals) {
  holoscan::Graph graph{std::string(kAppName)};
  const auto period = std::chrono::milliseconds(opts.period_ms);
  // Frames the run expects; 0 for a camera that runs until stopped.
  std::uint32_t frames = opts.count.value_or(0);
  int image_width = opts.width.value_or(0);
  int image_height = opts.height.value_or(0);
  int image_channels = 3;
  const holoscan::Output<holoscan::schema::ImageT>* source_output = nullptr;
  if (opts.source == Source::kSynthetic) {
    auto source = graph.op<td::SyntheticTagSourceOp>(
        "source", td::SyntheticSourceConfig{.width = image_width,
                                            .height = image_height,
                                            .family = opts.detector.family,
                                            .count = frames,
                                            .period = period,
                                            .device_output = opts.device_input});
    source_output = &source->image;
  } else if (opts.source == Source::kImages) {
    auto files = td::list_pnm_files(opts.images);
    frames = static_cast<std::uint32_t>(files.size());
    auto source = graph.op<td::ImageFileSourceOp>(
        "source", td::ImageFileSourceConfig{.files = std::move(files), .period = period});
    image_width = source->first().width;
    image_height = source->first().height;
    image_channels = source->first().channels;
    source_output = &source->image;
  } else {
    auto source = graph.op<td::CameraSourceOp>(
        "source",
        td::CameraSourceConfig{
            .device = opts.device, .width = image_width, .height = image_height, .count = frames});
    source_output = &source->image;
  }

  const bool display_available =
      env_set("DISPLAY") || (env_set("WAYLAND_DISPLAY") && env_set("XDG_RUNTIME_DIR"));
  const bool display = opts.display == DisplayMode::kOn ||
                       (opts.display == DisplayMode::kAuto && display_available &&
                        APRILTAG_DETECTION_ENABLE_VISUALIZATION);
  if (display && !APRILTAG_DETECTION_ENABLE_VISUALIZATION) {
    throw std::invalid_argument("--display needs a build with visualization enabled");
  }
  if (display && !display_available && !opts.compile_only) {
    throw std::invalid_argument(
        "the display needs a display server; set DISPLAY or add "
        "--display false");
  }

  // Lossless edges: every detector sees every frame, in order, so reporters can
  // match frame numbers to the synthetic ground truth.
  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking,
                                             .queue_depth = 2U};
  auto completion = std::make_shared<td::RunCompletion>();
  completion->remaining = static_cast<int>(opts.backends.size());
  std::vector<std::shared_ptr<td::BackendReport>> reports;
  std::optional<holoscan::OperatorHandle<td::AprilTagDetectorOp>> first_detector;
  for (const td::Backend backend : opts.backends) {
    const std::string name(td::backend_name(backend));
    auto detector = graph.op<td::AprilTagDetectorOp>(
        "detector_" + name, td::AprilTagDetectorConfig{.backend = backend,
                                                       .options = opts.detector,
                                                       .intrinsics = opts.intrinsics,
                                                       .tag_size = opts.tag_size});
    auto report = std::make_shared<td::BackendReport>();
    report->backend = name;
    reports.push_back(report);
    std::optional<td::GroundTruth> truth;
    if (opts.source == Source::kSynthetic) {
      truth = td::GroundTruth{opts.detector.family, image_width, image_height};
    }
    auto reporter = graph.op<td::DetectionReportOp>(
        "report_" + name,
        td::DetectionReportConfig{
            .report = report, .completion = completion, .expected_frames = frames, .truth = truth});
    graph.add_flow(*source_output, detector->image, lossless);
    graph.add_flow(detector->detections, reporter->detections, lossless);
    if (!first_detector) {
      first_detector.emplace(detector);
    }
  }

#if APRILTAG_DETECTION_ENABLE_VISUALIZATION
  if (display) {
    namespace viz = holoscan::visualization;
    viz::LayerStack layers;
    layers.add_image_layer(
        "video", viz::ImageLayerDesc{.format = image_channels == 1 ? viz::ImageFormat::kR8Unorm
                                                                   : viz::ImageFormat::kR8G8B8Unorm,
                                     .max_width = static_cast<std::uint32_t>(image_width),
                                     .max_height = static_cast<std::uint32_t>(image_height)});
    layers.add_geometry_layer(
        "outlines", viz::GeometryLayerDesc{
                        .topologies = {viz::GeometryTopology::kLineList},
                        .max_primitives = static_cast<std::uint32_t>(td::kMaxDetectionsPerFrame) *
                                          td::kOverlayLinesPerTag,
                        .default_line_width = 8.0F});
    layers.add_text_layer(
        "labels", viz::TextLayerDesc{.max_glyphs = static_cast<std::uint32_t>(
                                         td::kMaxDetectionsPerFrame * td::kOverlayLabelBytes),
                                     .normalized_coordinates = true});
    // Outlines carry their frame's capture time, so they reach Holoviz as old as
    // the detection latency; allow for the slowest backend, or they vanish.
    const auto max_input_age = std::max(4 * period, std::chrono::milliseconds{250});
    auto visualizer = graph.op<viz::HolovizOp>(
        "holoviz", std::move(layers),
        viz::WindowSinkDesc{.title = "AprilTag Detection (" +
                                     std::string(td::backend_name(opts.backends.front())) + ")",
                            .width = static_cast<std::uint32_t>(image_width),
                            .height = static_cast<std::uint32_t>(image_height)},
        viz::FrameTrigger{viz::LayerTrigger{0}}, max_input_age);
    // The display is best effort: it keeps the newest sample and never holds
    // back the detectors.
    const holoscan::ConnectionOptions latest{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                             .queue_depth = 1U};
    auto overlay = graph.op<td::AprilTagOverlayOp>("overlay");
    graph.add_flow(*source_output, visualizer->image_layer_input(0), latest);
    graph.add_flow((*first_detector)->detections, overlay->detections, latest);
    graph.add_flow(overlay->outlines, visualizer->tensor_layer_input(1), latest);
    graph.add_flow(overlay->labels, visualizer->tensor_layer_input(2), latest);
  }
#else
  (void)image_channels;
#endif

  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  holoscan::CompileOptions compile_options;
  // Device-memory frames are on GPU 0.
  if (opts.device_input) {
    compile_options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = "source", .output_port = "image", .device = holoscan::DeviceId{0}});
  }
  const auto plan = holoscan::compile(graph, std::move(compile_options));
  // Warnings on an accepted plan, as well as the reasons for a rejected one.
  if (!plan.diagnostics().empty()) {
    std::cerr << holoscan::format_diagnostics(plan.diagnostics(), plan.graph()) << '\n';
  }
  if (!plan.ok()) {
    return 1;
  }
  if (opts.compile_only) {
    std::cout << kAppName << ": plan=OK\n";
    return 0;
  }
  if (!signals) {
    // Profilers, sanitizers and preloaded libraries can start a thread before
    // main(); the SDK then refuses to take over SIGINT and SIGTERM.
    std::cerr << kAppName << ": warning: running without managed SIGINT/SIGTERM handling: "
              << holoscan::runtime_error_message(signals.error());
    if (signals.error().code == holoscan::ErrorCode::kInvalidArgument) {
      std::cerr << " (another thread was already running)";
    }
    std::cerr << "; an interrupt ends the process without a report\n";
  }
  auto session = signals ? holoscan::run_async(plan, *signals) : holoscan::run_async(plan);
  // A live camera run has no deadline; it ends when stopped.
  const auto deadline = std::chrono::steady_clock::now() + kTimeoutAllowance +
                        std::chrono::milliseconds(2LL * frames * opts.period_ms);
  while (session.is_running() && (frames == 0 || std::chrono::steady_clock::now() < deadline)) {
    std::this_thread::sleep_for(10ms);
  }
  const bool timed_out = session.is_running();
  if (timed_out) {
    std::cerr << kAppName << ": timed out\n";
    session.request_stop();
  }
  session.wait();
  const holoscan::RunTermination termination = session.termination();
  // The SDK logs a failed run's diagnostics itself, unless its report or its
  // logger is turned down; show those of any other run, such as the warning a
  // stop request leaves.
  const bool sdk_reported = termination == holoscan::RunTermination::kFailed &&
                            holoscan::automatic_diagnostic_logging_enabled() &&
                            holoscan::log_level() <= holoscan::LogLevel::ERROR;
  if (!session.diagnostics().empty() && !sdk_reported) {
    std::cerr << holoscan::format_diagnostics(session.diagnostics(), plan.graph()) << '\n';
  }
  if (signals) {
    if (auto closed = signals->close(); !closed) {
      std::cerr << kAppName << ": warning: closing the signal domain failed: "
                << holoscan::runtime_error_message(closed.error()) << '\n';
    }
  }

  // A stop request (SIGINT, SIGTERM, a closed window) ends the run early; that
  // is not a detection failure, but not a pass either. For a live camera it is
  // the normal end.
  const bool stopped = !timed_out && termination == holoscan::RunTermination::kStopped;
  bool passed = !timed_out && session.diagnostics().empty() &&
                termination == holoscan::RunTermination::kRequestedCompleted;
  for (const auto& report : reports) {
    print_report(*report, frames);
    passed = passed && report->frames == frames && report->passed();
  }
  std::cout << kAppName << ": verdict=" << (passed ? "PASS" : stopped ? "STOPPED" : "FAIL") << '\n';
  return passed || (stopped && frames == 0) ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) try {
  // Take over SIGINT and SIGTERM while this is the only thread: validating the
  // options can start VPI's worker threads, and the SDK refuses a process that
  // already runs other threads.
  SignalDomain signals = holoscan::ManagedShutdownSignals::create();
  auto opts = parse_args(argc, argv);
  if (opts.help) {
    usage(argv[0]);
    return 0;
  }
  validate(opts);
  return execute(opts, signals);
} catch (const std::invalid_argument& error) {
  std::cerr << "error: " << error.what() << '\n';
  return 2;
} catch (const std::exception& error) {
  std::cerr << "error: " << error.what() << '\n';
  return 1;
}
