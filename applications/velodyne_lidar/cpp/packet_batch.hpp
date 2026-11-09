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

// Shared packet-batch port contract and run bookkeeping for the three swappable sources.

#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <utility>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

#include "vlp16_packet.hpp"

namespace velodyne_lidar {

inline constexpr std::uint32_t kMaxBatchPackets = 64U;
inline constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};
inline constexpr DLDataType kFloat32{kDLFloat, 32U, 1U};

/// Host uint8 [N, 1206] tensor of validated VLP-16 packets, 1 <= N <= kMaxBatchPackets.
inline constexpr holoscan::TensorRepresentation kPacketBatch{
    .memory_kind = holoscan::MemoryKind::kHost,
    .dtype = kUInt8,
    .rank = 2U,
    .strides = holoscan::TensorStridePolicy::kContiguousOnly,
};

/// Counters shared by sources, the cloud operator, and headless sinks.
struct RunProgress {
  std::atomic<std::uint64_t> emitted_batches{};
  std::atomic<std::uint64_t> emitted_packets{};
  std::atomic<std::uint64_t> integrated_batches{};
  std::atomic<std::uint64_t> frames{};
  std::atomic<std::uint64_t> dropped_packets{};
  std::atomic<std::uint64_t> invalid_packets{};
  std::atomic<bool> source_done{};

  /// Return true once a finite source finished and every batch reached the cloud.
  [[nodiscard]] bool all_integrated() const noexcept {
    return source_done.load(std::memory_order_acquire) &&
           integrated_batches.load() == emitted_batches.load();
  }
};

inline void declare_packet_output(holoscan::OperatorSpec& spec,
                                  holoscan::Output<holoscan::Tensor>& output) {
  spec.output(output, "packets")
      .produces_tensor(holoscan::TensorOutputSpec{
          .representation = kPacketBatch,
          .bounds = holoscan::tensor_bounds(kMaxBatchPackets * kPacketBytes),
          .storage = holoscan::TensorOutputStorage::kRuntimePool,
      });
}

/// Copy \p count contiguous packets into a pooled tensor and emit it.
[[nodiscard]] inline holoscan::expected<void, holoscan::Error> emit_packets(
    holoscan::Output<holoscan::Tensor>& output, const std::uint8_t* packets, std::uint32_t count,
    holoscan::ExecutionContext& context, RunProgress& progress) {
  const std::array<std::int64_t, 2> shape{static_cast<std::int64_t>(count),
                                          static_cast<std::int64_t>(kPacketBytes)};
  const std::size_t nbytes = std::size_t{count} * kPacketBytes;
  auto loan = output.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = kUInt8});
  if (!loan) return holoscan::make_unexpected(std::move(loan).error());
  auto committed = loan->write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& writer)
                                                   -> holoscan::expected<void, holoscan::Error> {
    if (writer.data() == nullptr || writer.byte_size() != nbytes) {
      return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kInvalidArgument});
    }
    std::memcpy(writer.data(), packets, nbytes);
    return {};
  });
  if (!committed) return holoscan::make_unexpected(std::move(committed).error());
  auto emitted = output.emit(std::move(*loan),
                             holoscan::EmitOptions{.capture_time = context.activation_time()});
  if (emitted) {
    progress.emitted_packets.fetch_add(count);
    progress.emitted_batches.fetch_add(1U);
  }
  return emitted;
}

}  // namespace velodyne_lidar
