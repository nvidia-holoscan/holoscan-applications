/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include "cae_policy.hpp"

namespace endoscopy_tool_tracking {

[[nodiscard]] inline bool valid_class_policies(const std::vector<std::uint8_t>& values) noexcept {
  if (values.size() != kToolClasses) return false;
  for (const std::uint8_t value : values) {
    if (value > kMaxPolicyValue) return false;
  }
  return true;
}

struct InferenceToPolicyParams {
  std::vector<std::uint8_t> class_policies =
      std::vector<std::uint8_t>(kToolClasses, static_cast<std::uint8_t>(CaePolicy::kNeutral));
  std::uint8_t background_policy{static_cast<std::uint8_t>(CaePolicy::kDeprioritize)};
  float class_threshold{0.5F};
  float mask_threshold{0.5F};
};

HOLOSCAN_PARAM_SCHEMA(
    InferenceToPolicyParams,
    holoscan::param_schema("com.nvidia.holohub.endoscopy.InferenceToPolicyParams", 1),
    holoscan::field<&InferenceToPolicyParams::class_policies>("class_policies")
        .max_elements(kToolClasses)
        .validator<&valid_class_policies>(
            "com.nvidia.holohub.endoscopy.validator.class-policies.v1")
        .description("Policy value 0..4 for each of the seven model classes, in model order."),
    holoscan::field<&InferenceToPolicyParams::background_policy>("background_policy")
        .range(std::uint8_t{0}, std::uint8_t{kMaxPolicyValue})
        .description("Policy value for cells that no detected class covers."),
    holoscan::field<&InferenceToPolicyParams::class_threshold>("class_threshold")
        .range(0.0F, 1.0F)
        .description("A class participates when its probability is >= this value."),
    holoscan::field<&InferenceToPolicyParams::mask_threshold>("mask_threshold")
        .range(0.0F, 1.0F)
        .description("A participating class covers a cell when its mask value is >= this value."));

[[nodiscard]] inline InferenceToPolicyParams make_inference_to_policy_params(
    const CaePolicyConfig& config) {
  return InferenceToPolicyParams{
      .class_policies =
          std::vector<std::uint8_t>(config.class_policies.begin(), config.class_policies.end()),
      .background_policy = config.background_policy,
      .class_threshold = config.class_threshold,
      .mask_threshold = config.mask_threshold,
  };
}

/// @brief Convert raw tool-tracking inference into a codec-independent CAE policy map.
///
/// Inputs are the LSTM outputs of one source frame on the same CUDA device: FP32 class
/// probabilities [1,7] and spatial masks [1,7,60,107]. The display overlay is never used.
///
/// The output is a U8 MONO8 schema::ImageT of 107x60 policy values, one per mask-grid cell
/// (0 Neutral, 1 Protect, 2 ProtectStrong, 3 Deprioritize, 4 DeprioritizeStrong). A class
/// covers a cell when probability >= class_threshold and mask >= mask_threshold; the covered
/// cell takes that class's configured policy. When several classes cover a cell, the policy
/// with the highest protection priority wins: ProtectStrong > Protect > Neutral > Deprioritize >
/// DeprioritizeStrong. Uncovered cells take background_policy. A frame without detections
/// therefore yields a complete background map.
///
/// The operator never waits on the host. Input values are validated on the device: a non-finite
/// probability or one outside [0,1] marks every cell, and a non-finite mask value marks its cell,
/// with kInvalidPolicyValue (255). Every consumer must reject values above 4 and fail the run.
///
/// Frame identity: both inputs must carry the same Holoscan publication sequence, which must
/// increase by exactly one per frame. Because every stage from the replayer to this operator is
/// lossless and one-in/one-out, that sequence is the 1-based number of the processed frame (the
/// source frame number unless the replayer loops). It is published only as the output sample's
/// frame_id; the image header's device_sequence stays absent (zero) because it is reserved for a
/// sensor's own counter. The input capture time is forwarded unchanged.
class InferenceToPolicyOp final : public holoscan::Operator<InferenceToPolicyParams> {
 public:
  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Input<holoscan::Tensor> probs;
  holoscan::Input<holoscan::Tensor> binary_masks;
  holoscan::Output<holoscan::schema::ImageT> policy;

 private:
  std::uint64_t last_sequence_{};
};

}  // namespace endoscopy_tool_tracking
