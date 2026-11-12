// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

// Compile this translation unit instead of inference_to_policy.cu so the regression exercises the
// exact application kernel without a test-only API.
#include "../cpp/inference_to_policy.cu"  // NOLINT(build/include)

namespace endo = endoscopy_tool_tracking;

namespace {

constexpr std::size_t kCells = endo::kPolicyCells;
constexpr std::uint8_t kNeutral = 0U;
constexpr std::uint8_t kProtect = 1U;
constexpr std::uint8_t kProtectStrong = 2U;
constexpr std::uint8_t kDeprioritize = 3U;
constexpr std::uint8_t kDeprioritizeStrong = 4U;

void check(cudaError_t error) {
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
}

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

std::size_t cell(std::size_t row, std::size_t column) {
  return (row * endo::kPolicyWidth) + column;
}

struct Fixture {
  float* probabilities{};
  float* masks{};
  std::uint8_t* policy{};

  Fixture() {
    check(cudaMallocManaged(&probabilities, endo::kToolClasses * sizeof(float)));
    check(cudaMallocManaged(&masks, endo::kToolClasses * kCells * sizeof(float)));
    check(cudaMallocManaged(&policy, kCells));
    reset();
  }
  ~Fixture() {
    cudaFree(probabilities);
    cudaFree(masks);
    cudaFree(policy);
  }
  // Owns device allocations: copying would free them twice.
  Fixture(const Fixture&) = delete;
  Fixture& operator=(const Fixture&) = delete;
  void reset() {
    std::fill_n(probabilities, endo::kToolClasses, 0.0F);
    std::fill_n(masks, endo::kToolClasses * kCells, 0.0F);
    clear_output();
  }
  // Marker value the kernel never writes, so a skipped cell is detected.
  void clear_output() { std::fill_n(policy, kCells, std::uint8_t{0xAB}); }
  float& mask(std::size_t tool, std::size_t index) { return masks[(tool * kCells) + index]; }
  void run(const endo::InferenceToPolicyParams& params) {
    endo::launch_compute_policy(probabilities, masks, endo::make_rule(params), policy, nullptr);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());
  }
  std::size_t count(std::uint8_t value) const { return std::count(policy, policy + kCells, value); }
};

endo::InferenceToPolicyParams params_with(std::array<std::uint8_t, endo::kToolClasses> classes,
                                          std::uint8_t background) {
  endo::InferenceToPolicyParams params;
  params.class_policies.assign(classes.begin(), classes.end());
  params.background_policy = background;
  params.class_threshold = 0.5F;
  params.mask_threshold = 0.5F;
  return params;
}

}  // namespace

int main() try {
  Fixture f;
  const auto defaults = endo::make_inference_to_policy_params(endo::CaePolicyConfig{});
  const auto mixed = params_with(
      {kProtect, kProtectStrong, kDeprioritizeStrong, kNeutral, kDeprioritize, kProtect, kProtect},
      kDeprioritize);

  // Defaults: every tool Neutral, background Deprioritize.
  require(defaults.background_policy == kDeprioritize, "default background must be Deprioritize");
  require(std::all_of(defaults.class_policies.begin(), defaults.class_policies.end(),
                      [](std::uint8_t v) { return v == kNeutral; }),
          "default class policies must all be Neutral");

  // No detection: the whole map is background.
  f.run(mixed);
  require(f.count(kDeprioritize) == kCells, "a frame without detections must be all background");

  // One covered cell takes its class policy; neighbours stay background.
  f.probabilities[0] = 0.9F;
  f.mask(0, cell(10, 20)) = 0.9F;
  f.run(mixed);
  require(f.policy[cell(10, 20)] == kProtect, "covered cell must take the class policy");
  require(f.policy[cell(10, 21)] == kDeprioritize && f.policy[cell(11, 20)] == kDeprioritize,
          "neighbouring cells must stay background");
  require(f.count(kProtect) == 1U, "exactly one cell must be protected");

  // Threshold equality is inclusive (>=) for both probability and mask.
  f.reset();
  f.probabilities[0] = 0.5F;
  f.mask(0, 0) = 0.5F;
  f.mask(0, 1) = std::nextafter(0.5F, 0.0F);
  f.run(mixed);
  require(f.policy[0] == kProtect, "probability and mask equal to their thresholds must cover");
  require(f.policy[1] == kDeprioritize, "a mask just below the threshold must not cover");
  f.probabilities[0] = std::nextafter(0.5F, 0.0F);
  f.run(mixed);
  require(f.policy[0] == kDeprioritize, "a probability just below the threshold must not cover");

  // Distinct thresholds, set through the configuration path: probability is compared with
  // class_threshold and mask values with mask_threshold, never the other way round.
  {
    endo::CaePolicyConfig config;
    config.class_threshold = 0.3F;
    config.mask_threshold = 0.7F;
    config.class_policies[0] = kProtect;
    const auto split = endo::make_inference_to_policy_params(config);
    require(split.class_threshold == 0.3F && split.mask_threshold == 0.7F,
            "configuration thresholds must reach the operator parameters unchanged");
    f.reset();
    f.probabilities[0] = 0.4F;  // passes 0.3 only
    f.mask(0, 0) = 0.8F;        // passes 0.7
    f.mask(0, 1) = 0.5F;        // passes 0.3 only
    f.run(split);
    require(f.policy[0] == kProtect, "probability >= class_threshold and mask >= mask_threshold");
    require(f.policy[1] == kDeprioritize, "a mask below mask_threshold must not cover");
    f.probabilities[0] = 0.2F;  // below 0.3
    f.run(split);
    require(f.policy[0] == kDeprioritize, "a probability below class_threshold must not cover");
  }

  // A class mapped to Neutral claims its cells, overriding a Deprioritize background.
  f.reset();
  f.probabilities[3] = 1.0F;
  f.mask(3, 5) = 1.0F;
  f.run(mixed);
  require(f.policy[5] == kNeutral, "a Neutral class must override a Deprioritize background");

  // Overlap priority: ProtectStrong > Protect > Neutral > Deprioritize > DeprioritizeStrong.
  f.reset();
  for (std::size_t tool = 0; tool < endo::kToolClasses; ++tool) f.probabilities[tool] = 1.0F;
  // cell 0: Protect(0) + ProtectStrong(1) + DeprioritizeStrong(2)
  f.mask(0, 0) = f.mask(1, 0) = f.mask(2, 0) = 1.0F;
  // cell 1: Protect(0) + Neutral(3) + Deprioritize(4)
  f.mask(0, 1) = f.mask(3, 1) = f.mask(4, 1) = 1.0F;
  // cell 2: Neutral(3) + Deprioritize(4) + DeprioritizeStrong(2)
  f.mask(3, 2) = f.mask(4, 2) = f.mask(2, 2) = 1.0F;
  // cell 3: Deprioritize(4) + DeprioritizeStrong(2)
  f.mask(4, 3) = f.mask(2, 3) = 1.0F;
  // cell 4: DeprioritizeStrong(2) alone, against a Deprioritize background
  f.mask(2, 4) = 1.0F;
  f.run(mixed);
  require(f.policy[0] == kProtectStrong, "ProtectStrong must win every overlap");
  require(f.policy[1] == kProtect, "Protect must beat Neutral and Deprioritize");
  require(f.policy[2] == kNeutral, "Neutral must beat both Deprioritize intents");
  require(f.policy[3] == kDeprioritize, "Deprioritize must beat DeprioritizeStrong");
  require(f.policy[4] == kDeprioritizeStrong, "a covering class must replace the background");

  // The five cells above produced all five policy values; every other cell stays background.
  require(f.count(kDeprioritize) == kCells - 4U, "uncovered cells must stay background");

  // Corner cells map to the first/last row and column without out-of-bounds writes.
  f.reset();
  f.probabilities[0] = 1.0F;
  const std::array<std::size_t, 4> corners{cell(0, 0), cell(0, 106), cell(59, 0), cell(59, 106)};
  for (const std::size_t index : corners) f.mask(0, index) = 1.0F;
  f.run(mixed);
  for (const std::size_t index : corners) {
    require(f.policy[index] == kProtect, "corner cell " + std::to_string(index) + " not covered");
  }
  require(f.count(kProtect) == 4U, "only the four corners must be covered");

  // Invalid probabilities (NaN, Inf, outside [0,1]) poison the whole map, even if not selected.
  for (const float invalid : {std::numeric_limits<float>::quiet_NaN(),
                              std::numeric_limits<float>::infinity(), -0.01F, 1.01F}) {
    f.reset();
    f.probabilities[6] = invalid;
    f.run(mixed);
    require(f.count(endo::kInvalidPolicyValue) == kCells,
            "an invalid probability must mark every cell invalid");
  }

  // A non-finite mask value poisons only its own cell.
  f.reset();
  f.mask(5, cell(30, 50)) = std::numeric_limits<float>::quiet_NaN();
  f.mask(1, cell(30, 51)) = -std::numeric_limits<float>::infinity();
  f.mask(2, cell(30, 52)) = std::numeric_limits<float>::infinity();
  f.run(mixed);
  require(f.policy[cell(30, 50)] == endo::kInvalidPolicyValue &&
              f.policy[cell(30, 51)] == endo::kInvalidPolicyValue &&
              f.policy[cell(30, 52)] == endo::kInvalidPolicyValue,
          "a non-finite mask value must mark its cell invalid");
  require(f.count(endo::kInvalidPolicyValue) == 3U, "other cells must stay valid");

  // Determinism: identical input gives identical output.
  f.reset();
  for (std::size_t i = 0; i < endo::kToolClasses * kCells; ++i) {
    f.masks[i] = static_cast<float>((i * 2654435761U) % 1000U) / 999.0F;
  }
  for (std::size_t tool = 0; tool < endo::kToolClasses; ++tool) {
    f.probabilities[tool] = tool % 2U ? 0.8F : 0.2F;
  }
  f.run(mixed);
  std::array<std::uint8_t, kCells> first{};
  std::copy_n(f.policy, kCells, first.begin());
  for (int repeat = 0; repeat < 20; ++repeat) {
    f.clear_output();
    f.run(mixed);
    require(std::equal(first.begin(), first.end(), f.policy), "policy output is not deterministic");
  }
  require(std::all_of(first.begin(), first.end(), [](std::uint8_t v) { return v <= 4U; }),
          "valid input must produce only values 0..4");

  std::cout << "InferenceToPolicy kernel tests passed\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << "FAIL: " << error.what() << '\n';
  return 1;
}
