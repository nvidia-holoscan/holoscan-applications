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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/operators/inference/inference.hpp>
#include <holoscan/operators/tensor_preprocessor/tensor_preprocessor.hpp>
#include <holoscan/time/realtime_clock.hpp>
#include <video_file_reader.hpp>
#include <yaml-cpp/yaml.h>

namespace {

using holoscan::ops::InferenceOp;
using holoscan::ops::InferenceOpParams;
using holoscan::ops::TensorPreprocessorOp;
using holoscan::ops::TensorPreprocessorOpParams;
using holoscan::ops::VideoFileReaderOp;
using holoscan::ops::VideoFileReaderOpParams;

// -----------------------------------------------------------------------------------------------
// RunObservation: mirrors holoinfer_example.cpp's helper of the same name -- records one
// wall-clock timestamp per printed classification, so main() knows when the run is done.
// -----------------------------------------------------------------------------------------------

struct RunObservation {
  static void record_tick() {
    const auto now = std::chrono::steady_clock::now();
    {
      const std::lock_guard lock(mutex);
      ++count;
      tick_times.push_back(now);
    }
    ready.notify_all();
  }

  [[nodiscard]] static bool wait_for(std::size_t expected, std::chrono::seconds timeout) {
    std::unique_lock lock(mutex);
    return ready.wait_for(lock, timeout, [expected] { return count >= expected; });
  }

  static void print_summary(std::size_t warmup) {
    std::vector<std::chrono::steady_clock::time_point> ticks_copy;
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
  static inline std::vector<std::chrono::steady_clock::time_point> tick_times{};
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

    const float in_body_confidence = 1.0F / (1.0F + std::exp(-logits[0]));
    const float out_of_body_confidence = 1.0F / (1.0F + std::exp(-logits[1]));

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

struct CliArgs {
  std::string config_path;
  std::size_t ticks = 10U;
};

[[nodiscard]] bool parse_args(int argc, char** argv, CliArgs& out) {
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "--config" && i + 1 < argc) {
      out.config_path = argv[++i];
    } else if (arg == "--ticks" && i + 1 < argc) {
      out.ticks = static_cast<std::size_t>(std::stoul(argv[++i]));
    } else {
      std::cerr << "unrecognized argument: " << arg << '\n';
      return false;
    }
  }
  return !out.config_path.empty();
}

}  // namespace

int main(int argc, char** argv) {
  CliArgs args;
  if (!parse_args(argc, argv, args)) {
    std::cerr << "usage: de_id --config <path.yaml> [--ticks N]\n";
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

    holoscan::RunSession session = holoscan::run_async(plan);
    const bool complete = RunObservation::wait_for(args.ticks, std::chrono::seconds{60});
    session.request_stop();
    session.wait();

    // Inference-cadence summary. Skips the first 10 intervals so the TRT-warm-up cost does
    // not distort the steady-state percentiles. Requires --ticks >= ~50 to yield meaningful
    // percentiles.
    RunObservation::print_summary(/*warmup=*/10U);

    if (!complete) {
      std::cerr << "run did not complete " << args.ticks << " ticks within the timeout\n";
      for (const auto& diag : session.diagnostics()) {
        std::cerr << "  diagnostic: id=" << diag.id.value << " token=" << diag.token
                  << " vertex=" << diag.source.vertex.value << " port=" << diag.source.port.value()
                  << " required=" << diag.required << " provided=" << diag.provided << '\n';
      }
      return 2;
    }

    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "de_id: error running config '" << args.config_path << "': " << exception.what()
              << '\n';
    return 1;
  }
}
