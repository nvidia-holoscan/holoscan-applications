/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// de_id: binary in-body / out-of-body surgical-video classifier.
//
//   de_id --config config.yaml [--ticks N]
//
// Pipeline: VideoFileReaderOp (decode .mp4) -> TensorPreprocessorOp (resize + normalize) ->
// InferenceOp (de_id.onnx) -> ClassificationPrinterOp (stdout). See README.md.

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <numeric>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/managed_shutdown_signals.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/operators/inference/inference.hpp>
#include <holoscan/operators/tensor_preprocessor/tensor_preprocessor.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "synthetic_mp4.hpp"
#include "video_file_reader.hpp"

namespace {

using holoscan::ops::InferenceOp;
using holoscan::ops::InferenceOpParams;
using holoscan::ops::TensorPreprocessorOp;
using holoscan::ops::TensorPreprocessorOpParams;
using holoscan::ops::VideoFileReaderOp;
using holoscan::ops::VideoFileReaderOpParams;

// -----------------------------------------------------------------------------------------------
// RunObservation: records one wall-clock timestamp per printed classification, so main() knows
// when the run is done. Timestamps are held in a bounded ring (`kMaxTickHistory`) so a long
// `--ticks 0` (loop-forever) run does not grow the vector without limit; `count` is the full
// unbounded tick count, used by `wait_for()`.
// -----------------------------------------------------------------------------------------------

constexpr std::size_t kMaxTickHistory = 10000U;

struct RunObservation {
  static void record_tick() {
    const auto now = std::chrono::steady_clock::now();
    {
      const std::lock_guard lock(mutex);
      ++count;
      if (tick_times.size() >= kMaxTickHistory) {
        tick_times.pop_front();
      }
      tick_times.push_back(now);
    }
    ready.notify_all();
  }

  [[nodiscard]] static bool wait_for(std::size_t expected,
                                     std::chrono::steady_clock::duration timeout) {
    std::unique_lock lock(mutex);
    return ready.wait_for(lock, timeout, [expected] { return count >= expected; });
  }

  [[nodiscard]] static std::size_t current_count() {
    const std::lock_guard lock(mutex);
    return count;
  }

  static void print_summary(std::size_t warmup) {
    std::deque<std::chrono::steady_clock::time_point> ticks_copy;
    {
      const std::lock_guard lock(mutex);
      ticks_copy = tick_times;
    }
    if (ticks_copy.size() <= warmup + 1U) {
      std::cerr << "InferenceTiming: not enough ticks (" << ticks_copy.size()
                << ") to report after warmup=" << warmup << '\n';
      return;
    }
    std::vector<double> deltas_ms;
    deltas_ms.reserve(ticks_copy.size() - 1U - warmup);
    for (std::size_t i = warmup + 1U; i < ticks_copy.size(); ++i) {
      const auto delta = ticks_copy[i] - ticks_copy[i - 1U];
      deltas_ms.push_back(std::chrono::duration<double, std::milli>(delta).count());
    }
    std::vector<double> sorted = deltas_ms;
    std::sort(sorted.begin(), sorted.end());
    const double sum = std::accumulate(deltas_ms.begin(), deltas_ms.end(), 0.0);
    const double mean = sum / static_cast<double>(deltas_ms.size());
    const auto percentile = [&sorted](double p) {
      const auto idx = static_cast<std::size_t>(std::floor(p * (sorted.size() - 1U)));
      return sorted[idx];
    };
    std::cout << "\nInferenceTiming (steady-state, warmup=" << warmup << " ticks skipped):\n"
              << "  samples : " << deltas_ms.size() << '\n'
              << "  min ms  : " << sorted.front() << '\n'
              << "  mean ms : " << mean << '\n'
              << "  p50 ms  : " << percentile(0.50) << '\n'
              << "  p99 ms  : " << percentile(0.99) << '\n'
              << "  max ms  : " << sorted.back() << '\n';
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::size_t count{};
  static inline std::deque<std::chrono::steady_clock::time_point> tick_times{};
};

class ClassificationPrinterOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(input, "in")
        .queue_depth(4U)
        .expects_tensor(holoscan::TensorInputSpec{
            .representation =
                holoscan::TensorRepresentation{.memory_kind = holoscan::MemoryKind::kCudaDevice}});
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto sample = input.receive_data();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    const holoscan::Tensor& tensor = *sample;

    const auto element_count = static_cast<std::size_t>(tensor.nbytes()) / sizeof(float);
    if (element_count != 2U) {
      std::cout << "frame " << ++counter_ << ": unexpected output size (" << element_count
                << " elements, expected 2) -- pipeline ran end-to-end but classification "
                   "interpretation needs tuning for this model output shape\n";
      return {};
    }

    float logits[2];
    auto copied = tensor.copy_to_host(logits, sizeof(logits), context.cuda_stream());
    if (!copied) {
      return holoscan::make_unexpected(std::move(copied).error());
    }

    const float max_logit = std::max(logits[0], logits[1]);
    const float exp0 = std::exp(logits[0] - max_logit);
    const float exp1 = std::exp(logits[1] - max_logit);
    const float sum_exp = exp0 + exp1;
    const float in_body_confidence = exp0 / sum_exp;
    const float out_of_body_confidence = exp1 / sum_exp;

    ++counter_;
    if (in_body_confidence > out_of_body_confidence) {
      std::cout << "frame " << counter_ << ": Likely in-body (confidence=" << in_body_confidence
                << ")\n";
    } else {
      std::cout << "frame " << counter_
                << ": Likely out-of-body (confidence=" << out_of_body_confidence << ")\n";
    }

    RunObservation::record_tick();
    return {};
  }

  holoscan::Input<holoscan::Tensor> input;

 private:
  int counter_{};
};

class CiDropSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(input, "in")
        .queue_depth(4U)
        .expects_tensor(holoscan::TensorInputSpec{
            .representation =
                holoscan::TensorRepresentation{.memory_kind = holoscan::MemoryKind::kCudaDevice}});
  }
  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& /*context*/) override {
    auto sample = input.receive_data();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    received.fetch_add(1, std::memory_order_relaxed);
    return {};
  }
  holoscan::Input<holoscan::Tensor> input;
  static inline std::atomic<std::size_t> received{0};
};

struct CliArgs {
  std::string config_path;
  std::size_t ticks = 0U;
  bool self_test = false;
  std::size_t ci_ticks = 0U;
  // Optional explicit timeout for `--ticks` runs. `0` means "derive from ticks + frame_rate".
  std::size_t timeout_seconds = 0U;
};

[[nodiscard]] bool parse_args(int argc, char** argv, CliArgs& out) {
  // Strict non-negative integer parse: reject empty input, leading sign/whitespace and trailing
  // non-digits. std::stoul would silently accept "10abc" and wrap "-1" to a huge unsigned value.
  const auto parse_unsigned = [](const char* arg_name, const char* value,
                                 std::size_t& dst) -> bool {
    if (value == nullptr || *value == '\0') {
      std::cerr << arg_name << " requires a non-negative integer (got empty value)\n";
      return false;
    }
    for (const char* p = value; *p != '\0'; ++p) {
      if (*p < '0' || *p > '9') {
        std::cerr << arg_name << " requires a non-negative integer with digits only (got '" << value
                  << "')\n";
        return false;
      }
    }
    try {
      std::size_t pos = 0;
      const unsigned long parsed = std::stoul(value, &pos);
      if (pos != std::strlen(value)) {
        std::cerr << arg_name << " requires a non-negative integer (got '" << value << "')\n";
        return false;
      }
      dst = static_cast<std::size_t>(parsed);
      return true;
    } catch (const std::exception&) {
      std::cerr << arg_name << " requires a non-negative integer (got '" << value << "')\n";
      return false;
    }
  };
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "--config" && i + 1 < argc) {
      out.config_path = argv[++i];
    } else if (arg == "--ticks" && i + 1 < argc) {
      if (!parse_unsigned("--ticks", argv[++i], out.ticks)) {
        return false;
      }
    } else if (arg == "--self-test") {
      out.self_test = true;
    } else if (arg == "--ci-ticks" && i + 1 < argc) {
      if (!parse_unsigned("--ci-ticks", argv[++i], out.ci_ticks)) {
        return false;
      }
    } else if (arg == "--timeout" && i + 1 < argc) {
      if (!parse_unsigned("--timeout", argv[++i], out.timeout_seconds)) {
        return false;
      }
    } else {
      std::cerr << "unrecognized argument: " << arg << '\n';
      return false;
    }
  }
  return out.self_test || out.ci_ticks > 0U || !out.config_path.empty();
}

// Create a fresh, private temp directory for one run. Collides-free across concurrent
// `--ci-ticks` invocations even when they share `/tmp`.
[[nodiscard]] std::filesystem::path make_unique_tmp_dir(std::string_view tag) {
  namespace fs = std::filesystem;
  std::string tmpl = (fs::temp_directory_path() / (std::string{tag} + "_XXXXXX")).string();
  // mkdtemp mutates its input in place.
  std::vector<char> buf(tmpl.begin(), tmpl.end());
  buf.push_back('\0');
  const char* created = ::mkdtemp(buf.data());
  if (created == nullptr) {
    throw std::system_error(errno, std::generic_category(), "mkdtemp failed");
  }
  return fs::path{created};
}

[[nodiscard]] int run_ci_pipeline(std::size_t ticks) {
  namespace fs = std::filesystem;
  fs::path tmp_dir;
  try {
    tmp_dir = make_unique_tmp_dir("de_id_ci");
  } catch (const std::exception& exception) {
    std::cerr << "de_id --ci-ticks: failed to create temp dir: " << exception.what() << '\n';
    return 1;
  }
  const fs::path tmp_mp4 = tmp_dir / "synthetic.mp4";
  const auto cleanup = [&tmp_dir]() noexcept {
    std::error_code ec;
    fs::remove_all(tmp_dir, ec);
  };
  try {
    surgical_deid::test::encode_synthetic_mp4(tmp_mp4.string(), /*width=*/128, /*height=*/128,
                                              /*frame_count=*/static_cast<int>(ticks) + 10, 25);
  } catch (const std::exception& exception) {
    std::cerr << "de_id --ci-ticks: failed to encode synthetic MP4: " << exception.what() << '\n';
    cleanup();
    return 1;
  }

  VideoFileReaderOpParams reader_params;
  reader_params.path = tmp_mp4;
  reader_params.tensor_port_name = "frame";
  reader_params.output_pix_fmt = "rgb24";
  reader_params.output_layout = "nhwc";
  reader_params.prepend_batch_dim = false;
  reader_params.memory_kind = holoscan::MemoryKind::kCudaDevice;
  reader_params.frame_count = static_cast<int32_t>(ticks);
  reader_params.loop = false;

  CiDropSink::received.store(0, std::memory_order_relaxed);

  try {
    holoscan::Graph graph{"de_id_ci"};
    const auto reader = graph.op<VideoFileReaderOp>("reader", reader_params);
    const auto sink = graph.op<CiDropSink>("sink");
    graph.add_flow(*reader->output_port(), sink->input,
                   holoscan::ConnectionOptions{.queue_depth = 4U});
    graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

    holoscan::CompileOptions options;
    options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = "reader",
        .output_port = reader_params.tensor_port_name,
        .device = holoscan::DeviceId{0},
    });

    const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
    if (!plan.ok()) {
      std::cerr << plan.json() << '\n';
      cleanup();
      return 1;
    }
    holoscan::RunSession session = holoscan::run_async(plan);
    session.wait();
  } catch (const std::exception& exception) {
    std::cerr << "de_id --ci-ticks: pipeline error: " << exception.what() << '\n';
    cleanup();
    return 1;
  }
  cleanup();
  const std::size_t delivered = CiDropSink::received.load(std::memory_order_relaxed);
  if (delivered < ticks) {
    std::cerr << "de_id --ci-ticks: sink received " << delivered << " of " << ticks
              << " expected frames\n";
    return 2;
  }
  std::cout << "de_id ci-pipeline ok (" << delivered << " ticks delivered)\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  CliArgs args;
  if (!parse_args(argc, argv, args)) {
    std::cerr << "usage: de_id --config <path.yaml> [--ticks N] [--timeout S] | --self-test "
                 "| --ci-ticks N\n";
    return 1;
  }

  if (args.self_test) {
    std::cout << "de_id self-test ok\n";
    return 0;
  }

  if (args.ci_ticks > 0U) {
    return run_ci_pipeline(args.ci_ticks);
  }

  // Own SIGINT/SIGTERM before TRT/CUDA spin up participant threads, so Ctrl-C reaches
  // RunSession::wait() and lets the inference-timing summary print instead of killing the process.
  auto signals = holoscan::ManagedShutdownSignals::create();
  if (!signals) {
    std::cerr << "de_id: ManagedShutdownSignals::create failed: " << signals.error().message
              << '\n';
    return 1;
  }

  YAML::Node config_root;
  try {
    config_root = YAML::LoadFile(args.config_path);
  } catch (const YAML::Exception& exception) {
    std::cerr << "failed to load '" << args.config_path << "': " << exception.what() << '\n';
    return 1;
  }
  const std::filesystem::path config_dir = std::filesystem::path(args.config_path).parent_path();

  if (!config_root["inference"] || !config_root["video_source"] || !config_root["preprocessor"]) {
    std::cerr << "config '" << args.config_path
              << "' is missing required 'inference:', 'video_source:', or 'preprocessor:' "
                 "section\n";
    return 1;
  }

  InferenceOpParams inference_params;
  VideoFileReaderOpParams reader_params;
  TensorPreprocessorOpParams preprocessor_params;
  try {
    inference_params = InferenceOpParams::from_yaml_node(config_root["inference"], config_dir);
    reader_params =
        VideoFileReaderOpParams::from_yaml_node(config_root["video_source"], config_dir);
    preprocessor_params =
        TensorPreprocessorOpParams::from_yaml_node(config_root["preprocessor"], config_dir);
  } catch (const std::exception& exception) {
    std::cerr << "failed to load '" << args.config_path << "': " << exception.what() << '\n';
    return 1;
  }

  try {
    holoscan::Graph graph{"de_id"};

    const auto reader = graph.op<VideoFileReaderOp>("reader", reader_params);
    const auto preprocessor = graph.op<TensorPreprocessorOp>("preprocessor", preprocessor_params);
    const auto infer = graph.op<InferenceOp>("infer", inference_params);
    const auto printer = graph.op<ClassificationPrinterOp>("printer");

    auto* preprocessor_in = preprocessor->input_port();
    auto* infer_in = infer->input_port(preprocessor_params.output_port_name);
    auto* infer_out = infer->output_port("de_id_inferred");
    if (infer_in == nullptr) {
      std::cerr << "InferenceOp declared no input port named '"
                << preprocessor_params.output_port_name << "'\n";
      return 1;
    }
    if (infer_out == nullptr) {
      std::cerr << "InferenceOp declared no output port named 'de_id_inferred'\n";
      return 1;
    }

    graph.add_flow(*reader->output_port(), *preprocessor_in,
                   holoscan::ConnectionOptions{.queue_depth = 4U});
    graph.add_flow(*preprocessor->output_port(), *infer_in,
                   holoscan::ConnectionOptions{.queue_depth = 4U});
    graph.add_flow(*infer_out, printer->input, holoscan::ConnectionOptions{.queue_depth = 4U});

    graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

    holoscan::CompileOptions options;
    options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = "reader",
        .output_port = reader_params.tensor_port_name,
        .device = holoscan::DeviceId{0},
    });
    options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = "preprocessor",
        .output_port = preprocessor_params.output_port_name,
        .device = holoscan::DeviceId{0},
    });
    options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = "infer",
        .output_port = "de_id_inferred",
        .device = holoscan::DeviceId{0},
    });

    const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
    if (!plan.ok()) {
      std::cerr << plan.json() << '\n';
      return 1;
    }

    holoscan::RunSession session = holoscan::run_async(plan, *signals);
    bool complete = true;
    if (args.ticks == 0U) {
      session.wait();
    } else {
      // Derive the deadline from (ticks / frame_rate) when frame_rate > 0, so slow playback at
      // high tick counts does not time out before the pipeline could possibly finish. A caller
      // can override with --timeout. Minimum floor of 60 s keeps the original behavior at very
      // small tick counts.
      std::chrono::seconds timeout{60};
      if (args.timeout_seconds > 0U) {
        timeout = std::chrono::seconds{args.timeout_seconds};
      } else if (reader_params.frame_rate > 0.0F) {
        const double need_s =
            (static_cast<double>(args.ticks) / static_cast<double>(reader_params.frame_rate)) * 1.5;
        timeout = std::max(std::chrono::seconds{60},
                           std::chrono::seconds{static_cast<long>(std::ceil(need_s))});
      }
      // Poll both: either the printer saw `ticks` classifications, or the session finished on its
      // own (video ended before `ticks`, operator failed, SIGINT reached ManagedShutdownSignals).
      const auto poll_step = std::chrono::milliseconds{50};
      const auto deadline = std::chrono::steady_clock::now() + timeout;
      while (std::chrono::steady_clock::now() < deadline) {
        if (RunObservation::wait_for(args.ticks, poll_step)) {
          break;
        }
        if (!session.is_running()) {
          break;
        }
      }
      complete = (RunObservation::current_count() >= args.ticks) || !session.is_running();
      if (session.is_running()) {
        session.request_stop();
      }
      session.wait();
    }

    // Inference-cadence summary. Skips the first 10 intervals so the TRT-warm-up cost does
    // not distort the steady-state percentiles. Requires --ticks >= ~50 to yield meaningful
    // percentiles.
    RunObservation::print_summary(/*warmup=*/10U);

    // Fail loudly if an operator reported a diagnostic or the session ended in an error state,
    // even when the printer reached `args.ticks`. Mirrors the pattern in other C++ apps.
    bool had_error = false;
    for (const auto& diag : session.diagnostics()) {
      std::cerr << "  diagnostic: id=" << diag.id.value << " token=" << diag.token
                << " vertex=" << diag.source.vertex.value << " port=" << diag.source.port.value()
                << " required=" << diag.required << " provided=" << diag.provided << '\n';
      had_error = true;
    }
    const auto termination = session.termination();
    const bool terminated_ok = termination == holoscan::RunTermination::kRequestedCompleted ||
                               termination == holoscan::RunTermination::kCompleted ||
                               termination == holoscan::RunTermination::kStopped;
    if (!terminated_ok) {
      std::cerr << "de_id: run terminated with status=" << static_cast<int>(termination) << '\n';
      had_error = true;
    }
    if (had_error) {
      return 1;
    }

    if (!complete) {
      std::cerr << "run did not complete " << args.ticks << " ticks within the timeout\n";
      return 2;
    }

    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "de_id: error running config '" << args.config_path << "': " << exception.what()
              << '\n';
    return 1;
  }
}
