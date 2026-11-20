/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/logger/logger.hpp>

#include "inference_to_policy.hpp"

namespace endoscopy_tool_tracking {
namespace {

constexpr DLDataType kPolicyFloat32{kDLFloat, 32U, 1U};
constexpr DLDataType kPolicyU8{kDLUInt, 8U, 1U};

// Kernel arguments; passed by value so the operator needs no device-side parameter storage.
struct PolicyRule {
  float class_threshold;
  float mask_threshold;
  std::uint8_t background_policy;
  std::uint8_t class_policies[kToolClasses];
};

PolicyRule make_rule(const InferenceToPolicyParams& params) {
  PolicyRule rule{.class_threshold = params.class_threshold,
                  .mask_threshold = params.mask_threshold,
                  .background_policy = params.background_policy,
                  .class_policies = {}};
  std::copy_n(params.class_policies.begin(), kToolClasses, rule.class_policies);
  return rule;
}

// One thread per policy cell. probabilities: [7]; masks: [7][60][107], both contiguous FP32.
__global__ void compute_policy(const float* probabilities, const float* masks, PolicyRule rule,
                               std::uint8_t* policy) {
  const std::size_t cell = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (cell >= kPolicyCells) {
    return;
  }

  std::uint8_t selected = rule.background_policy;
  int selected_rank = -1;  // background loses to any covering class
  for (std::size_t tool = 0; tool < kToolClasses; ++tool) {
    const float probability = probabilities[tool];
    const float mask = masks[(tool * kPolicyCells) + cell];
    // Written as negated range tests so NaN fails them.
    if (!(probability >= 0.0F && probability <= 1.0F) || !isfinite(mask)) {
      policy[cell] = kInvalidPolicyValue;
      return;
    }
    if (probability >= rule.class_threshold && mask >= rule.mask_threshold) {
      const std::uint8_t candidate = rule.class_policies[tool];
      if (policy_rank(candidate) > selected_rank) {
        selected = candidate;
        selected_rank = policy_rank(candidate);
      }
    }
  }
  policy[cell] = selected;
}

void launch_compute_policy(const float* probabilities, const float* masks, const PolicyRule& rule,
                           std::uint8_t* policy, cudaStream_t stream) {
  constexpr unsigned int kThreads = 256U;
  constexpr unsigned int kBlocks = (kPolicyCells + kThreads - 1U) / kThreads;
  compute_policy<<<kBlocks, kThreads, 0, stream>>>(probabilities, masks, rule, policy);
}

}  // namespace

void InferenceToPolicyOp::setup(holoscan::OperatorSpec& spec) {
  const auto device_float = [](std::uint32_t rank) {
    return holoscan::TensorRepresentation{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = kPolicyFloat32,
        .rank = rank,
        .strides = holoscan::TensorStridePolicy::kContiguousOnly,
    };
  };
  spec.input(probs, "probs")
      .queue_depth(2U)
      .expects_tensor(holoscan::TensorInputSpec{.representation = device_float(2U)});
  spec.input(binary_masks, "binary_masks")
      .queue_depth(2U)
      .expects_tensor(holoscan::TensorInputSpec{.representation = device_float(4U)});

  spec.output(policy, "policy")
      .produces_tensor(holoscan::TensorOutputSpec{
          .representation =
              holoscan::TensorRepresentation{
                  .memory_kind = holoscan::MemoryKind::kCudaDevice,
                  .dtype = kPolicyU8,
                  .rank = 2U,
                  .strides = holoscan::TensorStridePolicy::kContiguousOnly,
              },
          .bounds = holoscan::tensor_bounds(kPolicyCells),
          .storage = holoscan::TensorOutputStorage::kRuntimePool,
      })
      .payload_options(holoscan::PayloadOptions{
          .schema = holoscan::schema_identity<holoscan::schema::ImageT>(),
      });
}

holoscan::Contract InferenceToPolicyOp::contract() const {
  holoscan::Contract result;
  std::vector<holoscan::PortRef> ports{probs, binary_masks};
  result.trigger(holoscan::OnAligned{
      .ports = holoscan::PortList{std::move(ports)},
      .window = std::chrono::nanoseconds{0},
  });
  result.input(probs, holoscan::Exact{});
  result.input(binary_masks, holoscan::Exact{});
  return result;
}

void InferenceToPolicyOp::start() { last_sequence_ = 0U; }

holoscan::expected<void, holoscan::Error> InferenceToPolicyOp::compute(
    holoscan::ExecutionContext& context) try {
  auto fail = [](holoscan::ErrorCode code,
                 std::string message) -> holoscan::expected<void, holoscan::Error> {
    HOLOSCAN_LOG_ERROR("InferenceToPolicyOp: {}", message);
    return holoscan::make_unexpected(holoscan::Error{code, std::move(message)});
  };

  // Borrowed views: no copy, no read claim. Valid until this activation returns.
  auto probs_view = probs.selected_sample();
  auto masks_view = binary_masks.selected_sample();
  if (!probs_view || !masks_view) {
    return fail(holoscan::ErrorCode::kReceiveError, "did not receive an aligned tensor set");
  }
  const auto& probs_sample = probs_view->sample();
  const auto& masks_sample = masks_view->sample();
  const std::uint64_t sequence = probs_sample.metadata.sequence;

  // Identity: one frame's outputs share a publication sequence that advances by exactly one.
  if (sequence == 0U || masks_sample.metadata.sequence == 0U) {
    return fail(holoscan::ErrorCode::kInvalidArgument,
                "model outputs carry no publication sequence; frame identity is unavailable");
  }
  if (masks_sample.metadata.sequence != sequence) {
    return fail(holoscan::ErrorCode::kInvalidArgument,
                "probabilities and masks belong to different frames: probs sequence=" +
                    std::to_string(sequence) +
                    " masks sequence=" + std::to_string(masks_sample.metadata.sequence));
  }
  if (last_sequence_ != 0U && sequence != last_sequence_ + 1U) {
    return fail(holoscan::ErrorCode::kInvalidArgument,
                "frame sequence is not contiguous: previous=" + std::to_string(last_sequence_) +
                    " current=" + std::to_string(sequence));
  }

  // Layout checks read only tensor metadata; no device data is touched on the host.
  const holoscan::Tensor& probs_tensor = probs_sample.data;
  const holoscan::Tensor& masks_tensor = masks_sample.data;
  constexpr std::array<std::int64_t, 2> expected_probs{1, kToolClasses};
  constexpr std::array<std::int64_t, 4> expected_masks{1, kToolClasses, kPolicyHeight,
                                                       kPolicyWidth};
  if (!std::ranges::equal(probs_tensor.shape_span(), expected_probs) ||
      !std::ranges::equal(masks_tensor.shape_span(), expected_masks) ||
      !probs_tensor.is_contiguous() || !masks_tensor.is_contiguous() ||
      probs_tensor.nbytes() != kToolClasses * sizeof(float) ||
      masks_tensor.nbytes() != kToolClasses * kPolicyCells * sizeof(float)) {
    return fail(holoscan::ErrorCode::kInvalidArgument,
                "unexpected model output layout at frame " + std::to_string(sequence) +
                    ": probs bytes=" + std::to_string(probs_tensor.nbytes()) +
                    " masks bytes=" + std::to_string(masks_tensor.nbytes()));
  }

  const std::array<std::int64_t, 2> policy_shape{kPolicyHeight, kPolicyWidth};
  auto loan = policy.allocate_tensor(
      holoscan::TensorLoanRequest{.shape = policy_shape, .dtype = kPolicyU8});
  if (!loan) {
    return fail(holoscan::ErrorCode::kResourceExhausted, "policy output pool exhausted at frame " +
                                                             std::to_string(sequence) + ": " +
                                                             loan.error().message);
  }
  // Inputs are already ordered on this stream; the kernel queues behind their producers.
  auto writer = loan->write(context.cuda_stream());
  if (!writer) {
    return holoscan::make_unexpected(std::move(writer).error());
  }
  launch_compute_policy(probs_tensor.data_as<float>(), masks_tensor.data_as<float>(),
                        make_rule(context.params<InferenceToPolicyParams>()),
                        writer->data_as<std::uint8_t>(), context.cuda_stream().get());
  if (cudaError_t error = cudaPeekAtLastError(); error != cudaSuccess) {
    return fail(holoscan::ErrorCode::kFailure, "policy kernel launch failed at frame " +
                                                   std::to_string(sequence) + ": " +
                                                   cudaGetErrorString(error));
  }
  // Records readiness on this stream; never blocks.
  auto committed = std::move(*writer).commit();
  if (!committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }

  const holoscan::TimePoint capture_time = probs_sample.metadata.capture_time();
  holoscan::schema::ImageT image;
  image.width = static_cast<std::int32_t>(kPolicyWidth);
  image.height = static_cast<std::int32_t>(kPolicyHeight);
  image.encoding = holoscan::schema::ImageEncoding_MONO8;
  image.header = std::make_shared<holoscan::schema::HeaderT>();
  image.header->capture_timestamp_ns = capture_time.timestamp_ns;
  holoscan::EmitOptions emit{.frame_id = sequence};
  if (capture_time.valid()) {
    emit.capture_time = capture_time;  // forward; never substitute activation time
  }
  if (auto result = policy.emit_tensor(std::move(*loan), image, emit); !result) {
    return result;
  }
  last_sequence_ = sequence;
  return {};
} catch (const std::exception& exception) {
  HOLOSCAN_LOG_ERROR("InferenceToPolicyOp exception: {}", exception.what());
  return holoscan::make_unexpected(
      holoscan::Error{holoscan::ErrorCode::kFailure, exception.what()});
}

}  // namespace endoscopy_tool_tracking
