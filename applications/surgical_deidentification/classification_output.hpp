/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>

namespace surgical_deid {

[[nodiscard]] inline holoscan::expected<void, holoscan::Error> validate_classification_output(
    const holoscan::Tensor& tensor) {
  const auto dtype = tensor.dtype();
  if (dtype.code != kDLFloat || dtype.bits != 32U || dtype.lanes != 1U) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kInvalidArgument,
                        "ClassificationPrinterOp: expected float32 logits with one lane"});
  }
  const auto element_count = tensor.size();
  if (element_count != 2) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kInvalidArgument,
                        "ClassificationPrinterOp: expected exactly 2 logits, got " +
                            std::to_string(element_count)});
  }
  return {};
}

}  // namespace surgical_deid
