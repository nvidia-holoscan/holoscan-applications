/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <vector>

#include "classification_output.hpp"

namespace surgical_deid {
namespace {

// The Tensor borrows host storage and shape until validation has finished. No device allocation
// or inference model is needed to exercise the classifier's output contract.
holoscan::expected<void, holoscan::Error> validate_output(std::vector<int64_t> shape,
                                                          DLDataType dtype) {
  std::array<std::uint8_t, 32> storage{};
  DLManagedTensor managed{};
  managed.dl_tensor.data = storage.data();
  managed.dl_tensor.device = DLDevice{kDLCPU, 0};
  managed.dl_tensor.ndim = static_cast<int32_t>(shape.size());
  managed.dl_tensor.dtype = dtype;
  managed.dl_tensor.shape = shape.data();
  const holoscan::Tensor tensor{&managed};
  return validate_classification_output(tensor);
}

TEST(ClassificationOutput, AcceptsExactlyTwoFloat32ValuesWithOptionalSingletonDimensions) {
  constexpr DLDataType kFloat32{kDLFloat, 32U, 1U};
  for (const auto& shape : std::vector<std::vector<int64_t>>{{2}, {1, 2}, {1, 2, 1}}) {
    const auto valid = validate_output(shape, kFloat32);
    EXPECT_TRUE(valid) << (valid ? "" : valid.error().message);
  }
}

TEST(ClassificationOutput, RejectsWrongNumberOfLogits) {
  constexpr DLDataType kFloat32{kDLFloat, 32U, 1U};
  for (const auto& shape : std::vector<std::vector<int64_t>>{{0}, {1}, {3}, {2, 2}}) {
    SCOPED_TRACE(::testing::PrintToString(shape));
    const auto valid = validate_output(shape, kFloat32);
    ASSERT_FALSE(valid);
    EXPECT_EQ(valid.error().code, holoscan::ErrorCode::kInvalidArgument);
  }
}

TEST(ClassificationOutput, RejectsNonFloat32OutputsEvenWhenTheyOccupyEightBytes) {
  struct Case {
    const char* name;
    int64_t elements;
    DLDataType dtype;
  };
  const Case cases[] = {
      {"int32", 2, {kDLInt, 32U, 1U}},
      {"uint32", 2, {kDLUInt, 32U, 1U}},
      {"float16", 4, {kDLFloat, 16U, 1U}},
      {"float64", 1, {kDLFloat, 64U, 1U}},
      {"vector float32", 1, {kDLFloat, 32U, 2U}},
      {"two float16 values", 2, {kDLFloat, 16U, 1U}},
      {"two float64 values", 2, {kDLFloat, 64U, 1U}},
      {"two vector float32 values", 2, {kDLFloat, 32U, 2U}},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    const auto valid = validate_output({test_case.elements}, test_case.dtype);
    ASSERT_FALSE(valid);
    EXPECT_EQ(valid.error().code, holoscan::ErrorCode::kInvalidArgument);
  }
}

}  // namespace
}  // namespace surgical_deid
