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

#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/temporal_contract.hpp>

#include "packet_batch.hpp"
#include "vlp16_synthetic.hpp"

namespace velodyne_lidar {

struct VlpSyntheticSourceParams {
  std::uint32_t count{1000U};          // total packets, 0 = endless; a revolution is ~76
  std::uint32_t packets_per_tick{8U};  // 8 packets per 10 ms matches the sensor's ~754 Hz
  std::uint32_t period_ms{10U};
};

HOLOSCAN_PARAM_SCHEMA(
    VlpSyntheticSourceParams,
    holoscan::param_schema("holoscan.applications.velodyne_lidar.VlpSyntheticSourceParams", 1),
    holoscan::field<&VlpSyntheticSourceParams::count>("count").description(
        "Total packets to generate before finishing; 0 generates endlessly."),
    holoscan::field<&VlpSyntheticSourceParams::packets_per_tick>("packets_per_tick")
        .description("Packets emitted per clock tick.")
        .range(1U, kMaxBatchPackets),
    holoscan::field<&VlpSyntheticSourceParams::period_ms>("period_ms")
        .description("Clock period in milliseconds.")
        .range(1U, 1000U));

/// Generate deterministic VLP-16 packets of a simulated room for tests and CI.
class VlpSyntheticSourceOp final : public holoscan::Operator<VlpSyntheticSourceParams> {
 public:
  explicit VlpSyntheticSourceOp(std::shared_ptr<RunProgress> progress)
      : progress_(std::move(progress)) {}

  void setup(holoscan::OperatorSpec& spec) override { declare_packet_output(spec, packets); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = std::chrono::milliseconds{params().period_ms}});
    return result;
  }

  void start() override {
    emitted_ = 0U;
    staging_.assign(std::size_t{kMaxBatchPackets} * kPacketBytes, 0U);
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    const auto& p = context.params<VlpSyntheticSourceParams>();
    const bool endless = p.count == 0U;
    if (!endless && emitted_ >= p.count) return {};
    const auto count = endless ? p.packets_per_tick
                               : static_cast<std::uint32_t>(std::min<std::uint64_t>(
                                     p.packets_per_tick, p.count - emitted_));
    for (std::uint32_t i = 0; i < count; ++i) {
      make_synthetic_packet(emitted_ + i, staging_.data() + std::size_t{i} * kPacketBytes);
    }
    HOLOSCAN_TRY(emit_packets(packets, staging_.data(), count, context, *progress_));
    emitted_ += count;
    if (!endless && emitted_ >= p.count)
      progress_->source_done.store(true, std::memory_order_release);
    return {};
  }

  holoscan::Output<holoscan::Tensor> packets;

 private:
  std::shared_ptr<RunProgress> progress_;
  std::vector<std::uint8_t> staging_;
  std::uint64_t emitted_{};
};

}  // namespace velodyne_lidar
