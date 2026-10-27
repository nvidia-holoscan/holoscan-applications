// SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights
// reserved. SPDX-License-Identifier: Apache-2.0

#include <array>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
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
#include <holoscan/core/graph.hpp>
#include <holoscan/core/managed_shutdown_signals.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/time/realtime_clock.hpp>
#if ENDOSCOPY_ENABLE_VISUALIZATION
#include <holoscan/visualization/holoviz_op.hpp>
#endif

#include "lstm_inference_v5.hpp"
#include "run_progress.hpp"
#include "tool_tracking_postprocessor_v5.hpp"
#include "video_preprocessor.hpp"
#include "video_stream_recorder.hpp"
#include "video_stream_replayer.hpp"

namespace {
using std::chrono_literals::operator""ms;
using std::chrono_literals::operator""ns;
using std::chrono_literals::operator""s;
namespace endo = endoscopy_tool_tracking;

enum class DisplayMode { Auto, On, Off };
struct Options {
  endo::VideoStreamReplayerParams replay;
  endo::VideoStreamRecorderParams record;
  DisplayMode display{DisplayMode::Auto};
  bool inference{true};
  bool help{false};
  std::string model;
  std::string engine_cache;
  std::uint32_t timeout_seconds{300U};
};

bool env_set(const char* name) {
  const char* value = std::getenv(name);
  return value && *value;
}

bool parse_bool(std::string_view value, std::string_view option) {
  if (value == "true") return true;
  if (value == "false") return false;
  throw std::invalid_argument(std::string(option) + " requires true or false");
}

std::uint32_t parse_number(std::string_view value, std::string_view option) {
  std::uint32_t result{};
  const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
  if (error != std::errc() || end != value.data() + value.size()) {
    throw std::invalid_argument(std::string(option) + " requires an unsigned 32-bit integer");
  }
  return result;
}

Options parse_args(int argc, char** argv) {
  Options opts;
  if (const char* data = std::getenv("HOLOSCAN_INPUT_PATH"); data && *data) {
    opts.replay.directory = data;
  } else {
    opts.replay.directory = "data/endoscopy";
  }
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
    if (option == "--data") {
      opts.replay.directory = value;
    } else if (option == "--model") {
      opts.model = value;
    } else if (option == "--engine-cache") {
      opts.engine_cache = value;
    } else if (option == "--basename") {
      opts.replay.basename = value;
    } else if (option == "--record-dir") {
      opts.record.directory = value;
    } else if (option == "--record-basename") {
      opts.record.basename = value;
    } else if (option == "--count") {
      opts.replay.count = parse_number(value, option);
    } else if (option == "--width") {
      opts.replay.width = parse_number(value, option);
    } else if (option == "--height") {
      opts.replay.height = parse_number(value, option);
    } else if (option == "--period-ms") {
      opts.replay.period_ms = parse_number(value, option);
    } else if (option == "--timeout-seconds") {
      opts.timeout_seconds = parse_number(value, option);
    } else if (option == "--synthetic") {
      opts.replay.synthetic = parse_bool(value, option);
    } else if (option == "--repeat") {
      opts.replay.repeat = parse_bool(value, option);
    } else if (option == "--inference") {
      opts.inference = parse_bool(value, option);
    } else if (option == "--display") {
      opts.display = value == "auto"
                         ? DisplayMode::Auto
                         : (parse_bool(value, option) ? DisplayMode::On : DisplayMode::Off);
    } else {
      throw std::invalid_argument("unknown option: " + std::string(option));
    }
  }
  return opts;
}

void usage(const char* argv0) {
  std::cout
      << "Usage: " << argv0
      << " [options]\n"
         "  --data DIR                 Prepared model and surgical_video.rgb (or "
         "HOLOSCAN_INPUT_PATH)\n"
         "  --count N                  Frames to process; 0 = whole video (default)\n"
         "  --display auto|true|false   Display when a display server is available (default: "
         "auto)\n"
         "  --model FILE               Override tool_loc_convlstm.onnx in the data directory\n"
         "  --engine-cache DIR         TensorRT cache (default: DATA/engines)\n"
         "  --record-dir DIR           Optionally record original RGB input into a new file\n"
         "  --record-basename NAME     Recording stem (default: input)\n"
         "  --basename NAME            Input stream stem (default: surgical_video)\n"
         "  --repeat true|false        Repeat short video until count is reached (default: false)\n"
         "  --period-ms N              Replay period, 1..1000 ms (default: 33)\n"
         "  --timeout-seconds N        Startup/stall allowance beyond twice playback time "
         "(default: 300)\n"
         "  --synthetic true|false     Deterministic test frames; requires a positive count\n"
         "  --inference true|false     Disable only for replay tests (default: true)\n"
         "  --width N --height N       RGB dimensions (default: 854x480; fixed for inference)\n";
}

void validate(Options& opts) {
  auto& p = opts.replay;
  if (!p.width || p.width > 854 || !p.height || p.height > 480) {
    throw std::invalid_argument("--width must be 1..854 and --height must be 1..480");
  }
  if (!p.period_ms || p.period_ms > 1000 || !opts.timeout_seconds) {
    throw std::invalid_argument("--period-ms must be 1..1000 and --timeout-seconds positive");
  }
  for (const auto& name : {p.basename, opts.record.basename}) {
    if (name.empty() || name == "." || name == ".." ||
        std::filesystem::path(name).filename() != name) {
      throw std::invalid_argument(
          "basenames must be nonempty filenames without directory components");
    }
  }
  if (opts.model.empty()) {
    opts.model = (std::filesystem::path(p.directory) / "tool_loc_convlstm.onnx").string();
  }
  if (opts.engine_cache.empty()) {
    opts.engine_cache = (std::filesystem::path(p.directory) / "engines").string();
  }
  if (opts.inference && !std::filesystem::is_regular_file(opts.model)) {
    throw std::invalid_argument("model is missing: " + opts.model + "; run prepare_data.py first");
  }
  if (opts.inference && (p.width != 854 || p.height != 480)) {
    throw std::invalid_argument("the tracking model requires 854x480 RGB input");
  }
  if (p.synthetic) {
    if (!p.count) throw std::invalid_argument("--synthetic true requires --count > 0");
  } else {
    const auto path = std::filesystem::path(p.directory) / (p.basename + ".rgb");
    std::error_code error;
    const auto bytes = std::filesystem::file_size(path, error);
    const auto frame_bytes = std::uint64_t{p.width} * p.height * 3;
    if (error) throw std::invalid_argument("cannot read RGB video: " + path.string());
    if (!bytes || bytes % frame_bytes) {
      throw std::invalid_argument("RGB video must contain a positive whole number of frames");
    }
    const auto frames = bytes / frame_bytes;
    if (frames > std::numeric_limits<std::uint32_t>::max()) {
      throw std::invalid_argument("RGB video exceeds supported frame count");
    }
    if (!p.count) p.count = static_cast<std::uint32_t>(frames);
    if (!p.repeat && p.count > frames) {
      throw std::invalid_argument("--count exceeds video length; use --repeat true to loop");
    }
  }
  if (!opts.record.directory.empty()) {
    const auto path =
        std::filesystem::path(opts.record.directory) / (opts.record.basename + ".rgb");
    if (std::filesystem::exists(path)) {
      throw std::invalid_argument("recording already exists: " + path.string());
    }
  }
}

class TrackingSinkOp final : public holoscan::Operator<> {
 public:
  explicit TrackingSinkOp(std::shared_ptr<endo::RunProgress> progress)
      : progress_(std::move(progress)) {}
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(mask, "mask").queue_depth(2U);
    spec.input(crosses, "crosses").queue_depth(2U);
  }
  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnAligned{
        .ports = holoscan::PortList{std::vector<holoscan::PortRef>{mask, crosses}}, .window = 0ns});
    result.input(mask, holoscan::Exact{});
    result.input(crosses, holoscan::Exact{});
    return result;
  }
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto image = mask.receive_data();
    auto geometry = crosses.receive_data();
    if (!image) return holoscan::make_unexpected(std::move(image).error());
    if (!geometry) return holoscan::make_unexpected(std::move(geometry).error());
    if (!image->data) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "tracking mask has no tensor"});
    }
    auto ready = image->data->synchronize_for_host();
    if (!ready) return holoscan::make_unexpected(std::move(ready).error());
    progress_->tracked.fetch_add(1U);
    progress_->finish_if_ready(context);
    return {};
  }
  holoscan::Input<holoscan::schema::ImageT> mask;
  holoscan::Input<holoscan::Tensor> crosses;

 private:
  std::shared_ptr<endo::RunProgress> progress_;
};

int execute(const Options& opts) {
  // Establish signal ownership before TensorRT/CUDA create participant threads.
  auto signals = holoscan::ManagedShutdownSignals::create();
  if (!signals) throw std::runtime_error(signals.error().message);
  const bool display_available =
      env_set("DISPLAY") || (env_set("WAYLAND_DISPLAY") && env_set("XDG_RUNTIME_DIR"));
  bool display =
      opts.display == DisplayMode::On ||
      (opts.display == DisplayMode::Auto && display_available && ENDOSCOPY_ENABLE_VISUALIZATION);
  if (display && (!ENDOSCOPY_ENABLE_VISUALIZATION || !display_available)) {
    throw std::invalid_argument(
        "--display true requires a visualization-enabled build and a display server");
  }
  auto progress = std::make_shared<endo::RunProgress>();
  progress->expected = opts.replay.count;
  progress->inference = opts.inference;
  holoscan::Graph graph{"endoscopy-tool-tracking"};
  auto source = graph.op<endo::VideoStreamReplayerOp>("replayer");
  source.set_params(opts.replay);
  auto recorder = graph.op<endo::VideoStreamRecorderOp>("input_sink", progress);
  recorder.set_params(opts.record);
  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking,
                                             .queue_depth = 2U};
  graph.add_flow(source->output, recorder->input, lossless);
  std::optional<holoscan::OperatorHandle<endo::ToolTrackingPostprocessorOp>> post;
  if (opts.inference) {
    auto pre = graph.op<endo::VideoPreprocessorOp>("video_preprocessor");
    pre.set_params(
        endo::VideoPreprocessorParams{.width = opts.replay.width, .height = opts.replay.height});
    auto infer = graph.op<endo::LstmInferenceOp>("lstm_inference");
    infer.set_params(
        endo::LstmInferenceParams{.model_path = opts.model, .engine_cache_dir = opts.engine_cache});
    post.emplace(graph.op<endo::ToolTrackingPostprocessorOp>("postprocessor"));
    auto sink = graph.op<TrackingSinkOp>("tracking_sink", progress);
    graph.add_flow(source->output, pre->input, lossless);
    graph.add_flow(pre->output, infer->source_video, lossless);
    graph.add_flow(infer->probs, (*post)->probs, lossless);
    graph.add_flow(infer->scaled_coords, (*post)->scaled_coords, lossless);
    graph.add_flow(infer->binary_masks, (*post)->binary_masks, lossless);
    graph.add_flow((*post)->mask, sink->mask, lossless);
    graph.add_flow((*post)->crosses, sink->crosses, lossless);
  }
#if ENDOSCOPY_ENABLE_VISUALIZATION
  if (display) {
    namespace viz = holoscan::visualization;
    viz::LayerStack layers;
    layers.add_image_layer("video", viz::ImageLayerDesc{.format = viz::ImageFormat::kR8G8B8Unorm,
                                                        .max_width = opts.replay.width,
                                                        .max_height = opts.replay.height});
    if (opts.inference) {
      layers.add_image_layer("mask", viz::ImageLayerDesc{.format = viz::ImageFormat::kR8G8B8A8Unorm,
                                                         .max_width = 107U,
                                                         .max_height = 60U});
      layers.add_geometry_layer(
          "crosses", viz::GeometryLayerDesc{.topologies = {viz::GeometryTopology::kCrossList},
                                            .max_primitives = 7U,
                                            .default_line_width = 4.0F});
    }
    auto visualizer =
        graph.op<viz::HolovizOp>("holoviz", std::move(layers),
                                 viz::WindowSinkDesc{.title = "Endoscopy Tool Tracking",
                                                     .width = opts.replay.width,
                                                     .height = opts.replay.height},
                                 viz::FrameTrigger{viz::LayerTrigger{0}}, 1s);
    const holoscan::ConnectionOptions latest{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                             .queue_depth = 1U};
    graph.add_flow(source->output, visualizer->image_layer_input(0), latest);
    if (post) {
      graph.add_flow((*post)->mask, visualizer->image_layer_input(1), latest);
      graph.add_flow((*post)->crosses, visualizer->tensor_layer_input(2), latest);
    }
  }
#endif
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  holoscan::CompileOptions compile_options;
  if (opts.inference) {
    for (const auto& [op, port] :
         std::array<std::pair<std::string, std::string>, 5>{{{"video_preprocessor", "source_video"},
                                                             {"lstm_inference", "probs"},
                                                             {"lstm_inference", "scaled_coords"},
                                                             {"lstm_inference", "binary_masks"},
                                                             {"postprocessor", "mask"}}}) {
      compile_options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
          .operator_path = op, .output_port = port, .device = holoscan::DeviceId{0}});
    }
  }
  const auto plan = holoscan::compile(graph, std::move(compile_options));
  if (!plan.ok()) {
    for (const auto& diagnostic : plan.diagnostics()) {
      std::cerr << "compile: " << diagnostic.token << " required=" << diagnostic.required
                << " provided=" << diagnostic.provided << '\n';
    }
    if (env_set("ENDOSCOPY_DEBUG_PLAN")) std::cerr << plan.json() << '\n';
    return 1;
  }
  auto session = holoscan::run_async(plan, *signals);
  const auto timeout = std::chrono::seconds(opts.timeout_seconds) +
                       std::chrono::milliseconds(2LL * opts.replay.count * opts.replay.period_ms);
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (session.is_running() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(10ms);
  }
  const bool finished = !session.is_running();
  if (!finished) {
    std::cerr << "pipeline timed out\n";
    session.request_stop();
  }
  session.wait();
  for (const auto& diag : session.diagnostics()) {
    std::cerr << "diagnostic: " << diag.token << " required=" << diag.required
              << " provided=" << diag.provided << '\n';
  }
  const bool passed = finished && session.diagnostics().empty() &&
                      session.termination() == holoscan::RunTermination::kRequestedCompleted &&
                      progress->consumed == opts.replay.count &&
                      (!opts.inference || progress->tracked == opts.replay.count);
  std::cout << "endoscopy_tool_tracking: expected=" << opts.replay.count
            << " consumed=" << progress->consumed.load() << " tracked=" << progress->tracked.load()
            << " inference=" << (opts.inference ? "on" : "off")
            << " verdict=" << (passed ? "PASS" : "FAIL") << '\n';
  return passed ? 0 : 1;
}
}  // namespace

int main(int argc, char** argv) try {
  auto opts = parse_args(argc, argv);
  if (opts.help) {
    usage(argv[0]);
    return 0;
  }
  validate(opts);
  return execute(opts);
} catch (const std::invalid_argument& error) {
  std::cerr << "error: " << error.what() << '\n';
  return 2;
} catch (const std::exception& error) {
  std::cerr << "error: " << error.what() << '\n';
  return 1;
}
