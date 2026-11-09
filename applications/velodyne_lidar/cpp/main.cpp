/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <arpa/inet.h>  // NOLINT(build/include_order)
#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/managed_shutdown_signals.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/time/realtime_clock.hpp>
#if VELODYNE_ENABLE_VISUALIZATION
#include <holoscan/visualization/holoviz_op.hpp>
#endif

#include "packet_batch.hpp"
#include "point_cloud_render.hpp"
#include "velodyne_cloud.hpp"
#include "vlp_pcap_source.hpp"
#include "vlp_synthetic_source.hpp"
#include "vlp_udp_source.hpp"

namespace {
using std::chrono_literals::operator""ms;
using std::chrono_literals::operator""s;
namespace vl = velodyne_lidar;

enum class Source { kUdp, kPcap, kSynthetic };
enum class DisplayMode { kAuto, kOn, kOff };

struct Options {
  Source source{Source::kUdp};
  vl::VlpUdpSourceParams udp;
  vl::VlpPcapSourceParams pcap;
  vl::VlpSyntheticSourceParams synthetic;
  vl::VelodyneCloudParams cloud;
  vl::PointCloudRenderParams render;
  DisplayMode display{DisplayMode::kAuto};
  std::string snapshot;
  std::uint32_t duration_seconds{0U};
  std::uint32_t timeout_seconds{120U};
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

std::uint32_t parse_number(std::string_view value, std::string_view option) {
  std::uint32_t result{};
  const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
  if (error != std::errc() || end != value.data() + value.size()) {
    throw std::invalid_argument(std::string(option) + " requires an unsigned 32-bit integer");
  }
  return result;
}

float parse_float(std::string_view value, std::string_view option) {
  float result{};
  const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
  if (value.empty() || error != std::errc() || end != value.data() + value.size() ||
      !std::isfinite(result)) {
    throw std::invalid_argument(std::string(option) + " requires a finite number");
  }
  return result;
}

void parse_vec3(std::string_view value, std::string_view option, float& x, float& y, float& z) {
  float* out[3] = {&x, &y, &z};
  for (int i = 0; i < 3; ++i) {
    const auto comma = value.find(',');
    if ((i < 2) == (comma == std::string_view::npos)) {
      throw std::invalid_argument(std::string(option) + " requires X,Y,Z in meters");
    }
    *out[i] = parse_float(value.substr(0, comma), option);
    value = i < 2 ? value.substr(comma + 1) : std::string_view{};
  }
}

Options parse_args(int argc, char** argv) {
  Options opts;
  for (int i = 1; i < argc; ++i) {
    const std::string_view option{argv[i]};
    if (option == "--help" || option == "-h") {
      opts.help = true;
      return opts;
    }
    if (i + 1 >= argc) throw std::invalid_argument("missing value for " + std::string(option));
    const std::string_view value{argv[++i]};
    auto& r = opts.render;
    if (option == "--source") {
      if (value == "udp") {
        opts.source = Source::kUdp;
      } else if (value == "pcap") {
        opts.source = Source::kPcap;
      } else if (value == "synthetic") {
        opts.source = Source::kSynthetic;
      } else {
        throw std::invalid_argument("--source requires udp, pcap, or synthetic");
      }
    } else if (option == "--address") {
      opts.udp.address = value;
    } else if (option == "--port") {
      opts.udp.port = opts.pcap.port = parse_number(value, option);
    } else if (option == "--pcap") {
      opts.pcap.path = value;
      opts.source = Source::kPcap;
    } else if (option == "--loop") {
      opts.pcap.loop = parse_bool(value, option);
    } else if (option == "--count") {
      opts.synthetic.count = parse_number(value, option);
    } else if (option == "--source-period-ms") {
      opts.pcap.period_ms = opts.synthetic.period_ms = parse_number(value, option);
    } else if (option == "--packet-buffer-size") {
      opts.cloud.packet_buffer_size = parse_number(value, option);
    } else if (option == "--view") {
      r.view = value;
    } else if (option == "--width") {
      r.width = parse_number(value, option);
    } else if (option == "--height") {
      r.height = parse_number(value, option);
    } else if (option == "--render-period-ms") {
      r.period_ms = parse_number(value, option);
    } else if (option == "--point-size") {
      r.point_size = parse_number(value, option);
    } else if (option == "--range-m") {
      r.range_m = parse_float(value, option);
    } else if (option == "--eye") {
      parse_vec3(value, option, r.eye_x, r.eye_y, r.eye_z);
    } else if (option == "--target") {
      parse_vec3(value, option, r.target_x, r.target_y, r.target_z);
    } else if (option == "--fov-deg") {
      r.fov_deg = parse_float(value, option);
    } else if (option == "--orbit-deg-per-s") {
      r.orbit_deg_per_s = parse_float(value, option);
    } else if (option == "--display") {
      opts.display = value == "auto"
                         ? DisplayMode::kAuto
                         : (parse_bool(value, option) ? DisplayMode::kOn : DisplayMode::kOff);
    } else if (option == "--snapshot") {
      opts.snapshot = value;
    } else if (option == "--duration-seconds") {
      opts.duration_seconds = parse_number(value, option);
    } else if (option == "--timeout-seconds") {
      opts.timeout_seconds = parse_number(value, option);
    } else {
      throw std::invalid_argument("unknown option: " + std::string(option));
    }
  }
  return opts;
}

void usage(const char* argv0) {
  std::cout << "Usage: " << argv0 << " [options]\n"
            << R"(Sources (one per run):
  --source udp|pcap|synthetic  Packet source (default: udp)
  --address IPV4               UDP bind address (default: 0.0.0.0, all interfaces)
  --port N                     UDP port, also the pcap filter port (default: 2368)
  --pcap FILE                  Replay a classic pcap recording (implies --source pcap)
  --loop true|false            Restart the recording at its end (default: false)
  --count N                    Synthetic packets, 0 = endless (default: 1000)
  --source-period-ms N         Replay/synthetic clock period (default: 10)
Point cloud and view:
  --packet-buffer-size N       Packets in the rolling cloud, 1..256 (default: 80)
  --view top|chase             Fixed camera (default: top)
  --range-m M                  Top view: meters from center to edge (default: 30)
  --eye X,Y,Z --target X,Y,Z   Chase camera pose in meters (default: 0,-14,7 and 0,6,0)
  --fov-deg D                  Chase vertical field of view (default: 60)
  --orbit-deg-per-s D          Chase view: orbit the eye around the target (default: 0)
  --width N --height N         Image size (default: 720x720)
  --point-size N               Point size in pixels (default: 2)
  --render-period-ms N         Render period (default: 33)
Run:
  --display auto|true|false    Show HoloViz when a display server is available (default: auto)
  --snapshot FILE.ppm          Headless finite runs: write the final rendered frame
  --duration-seconds N         Stop after N seconds; 0 runs until done or Ctrl-C (default: 0)
  --timeout-seconds N          Headless finite runs: failure deadline (default: 120)
)";
}

bool finite(const Options& opts) {
  return (opts.source == Source::kSynthetic && opts.synthetic.count != 0U) ||
         (opts.source == Source::kPcap && !opts.pcap.loop);
}

void validate(const Options& opts, bool display) {
  if (opts.source == Source::kUdp) {
    in_addr address{};
    if (::inet_pton(AF_INET, opts.udp.address.c_str(), &address) != 1) {
      throw std::invalid_argument("--address requires an IPv4 address");
    }
  }
  if (opts.udp.port == 0U || opts.udp.port > 65535U) {
    throw std::invalid_argument("--port must be 1..65535");
  }
  if (opts.source == Source::kPcap && !std::filesystem::is_regular_file(opts.pcap.path)) {
    throw std::invalid_argument("--pcap file not found: " + opts.pcap.path);
  }
  if (opts.pcap.period_ms == 0U || opts.pcap.period_ms > 1000U) {
    throw std::invalid_argument("--source-period-ms must be 1..1000");
  }
  if (opts.cloud.packet_buffer_size == 0U || opts.cloud.packet_buffer_size > vl::kMaxCloudPackets) {
    throw std::invalid_argument("--packet-buffer-size must be 1..256");
  }
  const auto& r = opts.render;
  if (r.view != "top" && r.view != "chase")
    throw std::invalid_argument("--view requires top or chase");
  if (r.width < 16U || r.width > vl::kMaxImageSide || r.height < 16U ||
      r.height > vl::kMaxImageSide) {
    throw std::invalid_argument("--width and --height must be 16..2048");
  }
  if (r.period_ms == 0U || r.period_ms > 1000U) {
    throw std::invalid_argument("--render-period-ms must be 1..1000");
  }
  if (r.point_size == 0U || r.point_size > 8U)
    throw std::invalid_argument("--point-size must be 1..8");
  if (!(r.range_m >= 1.0F && r.range_m <= 200.0F)) {
    throw std::invalid_argument("--range-m must be 1..200");
  }
  if (!(r.fov_deg >= 10.0F && r.fov_deg <= 150.0F)) {
    throw std::invalid_argument("--fov-deg must be 10..150");
  }
  if (!(r.orbit_deg_per_s >= -360.0F && r.orbit_deg_per_s <= 360.0F)) {
    throw std::invalid_argument("--orbit-deg-per-s must be -360..360");
  }
  if (r.eye_x == r.target_x && r.eye_y == r.target_y) {
    throw std::invalid_argument("--eye and --target must differ horizontally");
  }
  if (!opts.snapshot.empty() && (display || !finite(opts))) {
    throw std::invalid_argument("--snapshot requires a headless finite synthetic or pcap run");
  }
  if (!opts.snapshot.empty() && std::filesystem::exists(opts.snapshot)) {
    throw std::invalid_argument("snapshot already exists: " + opts.snapshot);
  }
}

// Headless sink: completes a finite run with the first frame rendered from the final cloud,
// optionally saving that frame.
class FrameSinkOp final : public holoscan::Operator<> {
 public:
  FrameSinkOp(std::shared_ptr<vl::RunProgress> progress, bool finite, std::string snapshot)
      : progress_(std::move(progress)), finite_(finite), snapshot_(std::move(snapshot)) {}

  void setup(holoscan::OperatorSpec& spec) override { spec.input(frame, "frame").queue_depth(2U); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{frame});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto sample = frame.receive();
    if (!sample) return holoscan::make_unexpected(std::move(sample).error());
    const holoscan::schema::ImageT& image = sample->data;
    if (!image.data) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "frame has no tensor"});
    }
    if (!finite_ || done_ || !progress_->all_integrated()) return {};
    // Cloud frame_id is the integrated batch count, so the final cloud's is emitted_batches.
    if (sample->metadata.frame_id != progress_->emitted_batches.load()) return {};
    done_ = true;
    if (!snapshot_.empty()) {
      std::vector<std::uint8_t> rgba(image.data->nbytes());
      HOLOSCAN_TRY(image.data->copy_to_host(rgba.data(), rgba.size(), context.cuda_stream().get()));
      std::ofstream out(snapshot_, std::ios::binary);
      out << "P6\n" << image.width << ' ' << image.height << "\n255\n";
      for (std::size_t i = 0; i + 3 < rgba.size(); i += 4) {
        out.write(reinterpret_cast<const char*>(rgba.data() + i), 3);
      }
      if (!out) {
        return holoscan::make_unexpected(
            holoscan::Error{holoscan::ErrorCode::kFailure, "failed to write " + snapshot_});
      }
    }
    context.request_completion();
    return {};
  }

  holoscan::Input<holoscan::schema::ImageT> frame;

 private:
  std::shared_ptr<vl::RunProgress> progress_;
  bool finite_;
  std::string snapshot_;
  bool done_{};
};

int execute(const Options& opts, bool display) {
  auto signals = holoscan::ManagedShutdownSignals::create();
  if (!signals) throw std::runtime_error(signals.error().message);
  auto progress = std::make_shared<vl::RunProgress>();
  holoscan::Graph graph{"velodyne-lidar"};
  auto cloud = graph.op<vl::VelodyneCloudOp>("cloud", progress);
  cloud.set_params(opts.cloud);
  // Finite sources stay lossless for deterministic tests; live UDP drops the oldest batch
  // rather than stall the receive path.
  const holoscan::ConnectionOptions packets{
      .queue_policy =
          finite(opts) ? holoscan::QueuePolicy::kBlocking : holoscan::QueuePolicy::kDropOldest,
      .queue_depth = 8U};
  switch (opts.source) {
    case Source::kUdp: {
      auto source = graph.op<vl::VlpUdpSourceOp>("udp_source", progress);
      source.set_params(opts.udp);
      graph.add_flow(source->packets, cloud->packets, packets);
      break;
    }
    case Source::kPcap: {
      auto source = graph.op<vl::VlpPcapSourceOp>("pcap_source", progress);
      source.set_params(opts.pcap);
      graph.add_flow(source->packets, cloud->packets, packets);
      break;
    }
    case Source::kSynthetic: {
      auto source = graph.op<vl::VlpSyntheticSourceOp>("synthetic_source", progress);
      source.set_params(opts.synthetic);
      graph.add_flow(source->packets, cloud->packets, packets);
      break;
    }
  }
  auto render = graph.op<vl::PointCloudRenderOp>("render", progress);
  render.set_params(opts.render);
  const holoscan::ConnectionOptions latest{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                           .queue_depth = 1U};
  // Depth 1 so the renderer never receives a cloud older than one it already rendered.
  graph.add_flow(cloud->cloud, render->cloud, latest);
#if VELODYNE_ENABLE_VISUALIZATION
  if (display) {
    namespace viz = holoscan::visualization;
    viz::LayerStack layers;
    layers.add_image_layer("cloud", viz::ImageLayerDesc{.format = viz::ImageFormat::kR8G8B8A8Unorm,
                                                        .max_width = opts.render.width,
                                                        .max_height = opts.render.height});
    auto visualizer = graph.op<viz::HolovizOp>(
        "holoviz", std::move(layers),
        viz::WindowSinkDesc{.title = "Velodyne VLP-16 (" + opts.render.view + " view)",
                            .width = opts.render.width,
                            .height = opts.render.height},
        viz::FrameTrigger{viz::LayerTrigger{0}}, 1s);
    graph.add_flow(render->image, visualizer->image_layer_input(0), latest);
  }
#endif
  if (!display) {
    auto sink = graph.op<FrameSinkOp>("frame_sink", progress, finite(opts), opts.snapshot);
    graph.add_flow(render->image, sink->frame, latest);
  }
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  holoscan::CompileOptions compile_options;
  for (const auto& [op, port] : {std::pair<std::string, std::string>{"cloud", "cloud"},
                                 std::pair<std::string, std::string>{"render", "image"}}) {
    compile_options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = op, .output_port = port, .device = holoscan::DeviceId{0}});
  }
  const auto plan = holoscan::compile(graph, std::move(compile_options));
  if (!plan.ok()) {
    for (const auto& diagnostic : plan.diagnostics()) {
      std::cerr << "compile: " << diagnostic.token << " required=" << diagnostic.required
                << " provided=" << diagnostic.provided << '\n';
    }
    if (env_set("VELODYNE_DEBUG_PLAN")) std::cerr << plan.json() << '\n';
    return 1;
  }

  auto session = holoscan::run_async(plan, *signals);
  const bool bounded = finite(opts) && !display;
  const auto start = std::chrono::steady_clock::now();
  bool timed_out = false;
  while (session.is_running()) {
    const auto elapsed = std::chrono::steady_clock::now() - start;
    if (opts.duration_seconds != 0U && elapsed >= std::chrono::seconds(opts.duration_seconds)) {
      session.request_stop();
      break;
    }
    if (bounded && elapsed >= std::chrono::seconds(opts.timeout_seconds)) {
      std::cerr << "pipeline timed out\n";
      timed_out = true;
      session.request_stop();
      break;
    }
    std::this_thread::sleep_for(10ms);
  }
  session.wait();
  for (const auto& diag : session.diagnostics()) {
    std::cerr << "diagnostic: " << diag.token << " required=" << diag.required
              << " provided=" << diag.provided << '\n';
  }
  bool passed = !timed_out && session.diagnostics().empty() &&
                session.termination() != holoscan::RunTermination::kFailed;
  if (bounded && opts.duration_seconds == 0U) {
    passed = passed && session.termination() == holoscan::RunTermination::kRequestedCompleted &&
             progress->emitted_packets.load() > 0U && progress->all_integrated();
  }
  std::cout << "velodyne_lidar: packets=" << progress->emitted_packets.load()
            << " batches=" << progress->emitted_batches.load()
            << " integrated=" << progress->integrated_batches.load()
            << " frames=" << progress->frames.load()
            << " dropped=" << progress->dropped_packets.load()
            << " invalid=" << progress->invalid_packets.load()
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
  const bool display_available =
      env_set("DISPLAY") || (env_set("WAYLAND_DISPLAY") && env_set("XDG_RUNTIME_DIR"));
  const bool display =
      opts.display == DisplayMode::kOn ||
      (opts.display == DisplayMode::kAuto && display_available && VELODYNE_ENABLE_VISUALIZATION);
  if (display && (!VELODYNE_ENABLE_VISUALIZATION || !display_available)) {
    throw std::invalid_argument(
        "--display true requires a visualization-enabled build and a display server");
  }
  validate(opts, display);
  return execute(opts, display);
} catch (const std::invalid_argument& error) {
  std::cerr << "error: " << error.what() << "\nRun with --help for usage.\n";
  return 2;
} catch (const std::exception& error) {
  std::cerr << "error: " << error.what() << '\n';
  return 1;
}
