// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Runs InferenceToPolicyOp in a compiled graph between a synthetic model-output source and a
// capture sink. Checks the published payload contract (MONO8 107x60 device map, frame_id, header
// capture time with no device_sequence), per-frame values, device-side poisoning and lossless
// one-in/one-out order.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "../cpp/inference_to_policy.hpp"

namespace endo = endoscopy_tool_tracking;

namespace {

constexpr std::uint32_t kFrames = 6U;
constexpr std::uint32_t kPoisonFrame = 4U;  // 1-based
constexpr DLDataType kFloat{kDLFloat, 32U, 1U};

struct Captured {
  std::uint64_t frame_id{};
  std::int64_t capture_ns{};
  bool has_header{};
  std::uint64_t header_device_sequence{};
  std::int64_t header_capture_ns{};
  std::int32_t width{};
  std::int32_t height{};
  bool mono8{};
  std::array<std::uint8_t, endo::kPolicyCells> values{};
};

struct Observation {
  static inline std::mutex mutex;
  static inline std::vector<Captured> frames;
  static inline std::vector<std::int64_t> source_capture_ns;  // per emitted frame, in order
  static inline std::atomic<bool> done{false};
};

// Frame n (1-based) covers row n with grasper (Protect) at probability 0.9; frame kPoisonFrame
// carries a NaN probability.
void fill_frame(std::uint32_t frame, std::array<float, endo::kToolClasses>& probs,
                std::vector<float>& masks) {
  probs.fill(0.0F);
  std::fill(masks.begin(), masks.end(), 0.0F);
  probs[0] = frame == kPoisonFrame ? std::numeric_limits<float>::quiet_NaN() : 0.9F;
  for (std::size_t column = 0; column < endo::kPolicyWidth; ++column) {
    masks[(frame * endo::kPolicyWidth) + column] = 1.0F;  // class 0, row = frame
  }
}

class FakeInferenceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    const auto device_float = [](std::uint32_t rank) {
      return holoscan::TensorRepresentation{
          .memory_kind = holoscan::MemoryKind::kCudaDevice,
          .dtype = kFloat,
          .rank = rank,
          .strides = holoscan::TensorStridePolicy::kContiguousOnly};
    };
    spec.output(probs, "probs")
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation = device_float(2U),
            .bounds = holoscan::tensor_bounds(endo::kToolClasses * sizeof(float)),
            .storage = holoscan::TensorOutputStorage::kRuntimePool});
    spec.output(binary_masks, "binary_masks")
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation = device_float(4U),
            .bounds =
                holoscan::tensor_bounds(endo::kToolClasses * endo::kPolicyCells * sizeof(float)),
            .storage = holoscan::TensorOutputStorage::kRuntimePool});
  }
  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = std::chrono::milliseconds{2}});
    return result;
  }
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (emitted_ >= kFrames) return {};
    const std::uint32_t frame = emitted_ + 1U;
    fill_frame(frame, host_probs_, host_masks_);
    const auto stream = context.cuda_stream();
    const std::array<std::int64_t, 2> probs_shape{1, endo::kToolClasses};
    const std::array<std::int64_t, 4> masks_shape{1, endo::kToolClasses, endo::kPolicyHeight,
                                                  endo::kPolicyWidth};
    auto probs_loan =
        probs.allocate_tensor(holoscan::TensorLoanRequest{.shape = probs_shape, .dtype = kFloat});
    auto masks_loan = binary_masks.allocate_tensor(
        holoscan::TensorLoanRequest{.shape = masks_shape, .dtype = kFloat});
    if (!probs_loan || !masks_loan) {
      return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kResourceExhausted});
    }
    auto probs_writer = probs_loan->write(stream);
    auto masks_writer = masks_loan->write(stream);
    if (!probs_writer || !masks_writer) {
      return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kFailure});
    }
    // Pageable host sources: the copies are staged before cudaMemcpyAsync returns.
    cudaMemcpyAsync(probs_writer->data(), host_probs_.data(), sizeof(host_probs_),
                    cudaMemcpyHostToDevice, stream.get());
    cudaMemcpyAsync(masks_writer->data(), host_masks_.data(), host_masks_.size() * sizeof(float),
                    cudaMemcpyHostToDevice, stream.get());
    if (cudaPeekAtLastError() != cudaSuccess) {
      return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kFailure});
    }
    if (auto ok = std::move(*probs_writer).commit(); !ok) {
      return holoscan::make_unexpected(ok.error());
    }
    if (auto ok = std::move(*masks_writer).commit(); !ok) {
      return holoscan::make_unexpected(ok.error());
    }
    const holoscan::EmitOptions emit{.capture_time = context.activation_time()};
    {
      const std::lock_guard lock(Observation::mutex);
      Observation::source_capture_ns.push_back(emit.capture_time->timestamp_ns);
    }
    if (auto ok = probs.emit(std::move(*probs_loan), emit); !ok) {
      return ok;
    }
    if (auto ok = binary_masks.emit(std::move(*masks_loan), emit); !ok) {
      return ok;
    }
    ++emitted_;
    return {};
  }

  holoscan::Output<holoscan::Tensor> probs;
  holoscan::Output<holoscan::Tensor> binary_masks;

 private:
  std::uint32_t emitted_{};
  std::array<float, endo::kToolClasses> host_probs_{};
  std::vector<float> host_masks_ = std::vector<float>(endo::kToolClasses * endo::kPolicyCells);
};

class CaptureOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(policy, "policy").queue_depth(2U);
  }
  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{policy});
    return result;
  }
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto view = policy.selected_sample();
    if (!view) return holoscan::make_unexpected(std::move(view).error());
    const auto& sample = view->sample();
    Captured captured;
    captured.frame_id = sample.metadata.frame_id;
    captured.capture_ns = sample.metadata.capture_timestamp_ns;
    captured.has_header = static_cast<bool>(sample.data.header);
    if (captured.has_header) {
      captured.header_device_sequence = sample.data.header->device_sequence;
      captured.header_capture_ns = sample.data.header->capture_timestamp_ns;
    }
    captured.width = sample.data.width;
    captured.height = sample.data.height;
    captured.mono8 = sample.data.encoding == holoscan::schema::ImageEncoding_MONO8;
    if (!sample.data.data || sample.data.data->nbytes() != endo::kPolicyCells) {
      return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kInvalidArgument});
    }
    if (auto ok = sample.data.data->copy_to_host(captured.values.data(), captured.values.size(),
                                                 context.cuda_stream().get());
        !ok) {
      return holoscan::make_unexpected(ok.error());
    }
    const std::lock_guard lock(Observation::mutex);
    Observation::frames.push_back(captured);
    if (Observation::frames.size() == kFrames) {
      Observation::done = true;
      context.request_completion();
    }
    return {};
  }

  holoscan::Input<holoscan::schema::ImageT> policy;
};

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

}  // namespace

int main() try {
  holoscan::Graph graph{"inference-to-policy-test"};
  auto source = graph.op<FakeInferenceOp>("fake_inference");
  auto policy_op = graph.op<endo::InferenceToPolicyOp>("inference_to_policy");
  endo::CaePolicyConfig config;
  config.class_policies[0] = 1U;  // grasper -> protect
  policy_op.set_params(endo::make_inference_to_policy_params(config));
  auto capture = graph.op<CaptureOp>("capture");
  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking,
                                             .queue_depth = 2U};
  graph.add_flow(source->probs, policy_op->probs, lossless);
  graph.add_flow(source->binary_masks, policy_op->binary_masks, lossless);
  graph.add_flow(policy_op->policy, capture->policy, lossless);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  holoscan::CompileOptions options;
  for (const auto& [op, port] :
       {std::pair{"fake_inference", "probs"}, std::pair{"fake_inference", "binary_masks"},
        std::pair{"inference_to_policy", "policy"}}) {
    options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = op, .output_port = port, .device = holoscan::DeviceId{0}});
  }
  const auto plan = holoscan::compile(graph, std::move(options));
  if (!plan.ok()) {
    for (const auto& diagnostic : plan.diagnostics()) {
      std::cerr << "compile: " << diagnostic.token << " required=" << diagnostic.required
                << " provided=" << diagnostic.provided << '\n';
    }
    return 1;
  }
  auto session = holoscan::run_async(plan);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
  while (session.is_running() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  if (session.is_running()) session.request_stop();
  session.wait();
  for (const auto& diagnostic : session.diagnostics()) {
    std::cerr << "diagnostic: " << diagnostic.token << " required=" << diagnostic.required
              << " provided=" << diagnostic.provided << '\n';
  }
  require(Observation::done, "graph did not deliver every policy map");
  require(session.diagnostics().empty(), "graph reported diagnostics");

  const std::lock_guard lock(Observation::mutex);
  require(Observation::frames.size() == kFrames, "one policy map per frame");
  for (std::uint32_t i = 0; i < kFrames; ++i) {
    const Captured& got = Observation::frames[i];
    const std::uint32_t frame = i + 1U;
    require(got.frame_id == frame, "frame_id must be the 1-based frame number");
    require(got.has_header && got.header_device_sequence == 0U,
            "the header must not claim a device sequence");
    require(got.capture_ns == Observation::source_capture_ns.at(i),
            "the policy sample must forward the source frame's capture time");
    require(got.header_capture_ns == got.capture_ns,
            "the header must carry the same capture time as the policy sample");
    require(i == 0U || got.capture_ns > Observation::frames[i - 1U].capture_ns,
            "capture times must increase frame by frame");
    require(got.width == 107 && got.height == 60 && got.mono8, "payload must be MONO8 107x60");
    for (std::size_t row = 0; row < endo::kPolicyHeight; ++row) {
      for (std::size_t column = 0; column < endo::kPolicyWidth; ++column) {
        const std::uint8_t value = got.values[(row * endo::kPolicyWidth) + column];
        const std::uint8_t expected =
            frame == kPoisonFrame ? endo::kInvalidPolicyValue : (row == frame ? 1U : 3U);
        require(value == expected, "frame " + std::to_string(frame) + " row " +
                                       std::to_string(row) + " has value " + std::to_string(value) +
                                       ", expected " + std::to_string(expected));
      }
    }
  }
  std::cout << "InferenceToPolicy graph test passed\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << "FAIL: " << error.what() << '\n';
  return 1;
}
