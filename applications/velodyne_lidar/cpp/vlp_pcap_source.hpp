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
#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/logger/logger.hpp>

#include "packet_batch.hpp"
#include "vlp16_pcap.hpp"

namespace velodyne_lidar {

struct VlpPcapSourceParams {
  std::string path;
  std::uint32_t port{kDefaultPort};
  std::uint32_t period_ms{10U};
  bool loop{false};
};

HOLOSCAN_PARAM_SCHEMA(
    VlpPcapSourceParams,
    holoscan::param_schema("holoscan.applications.velodyne_lidar.VlpPcapSourceParams", 1),
    holoscan::field<&VlpPcapSourceParams::path>("path").description("Classic pcap recording."),
    holoscan::field<&VlpPcapSourceParams::port>("port")
        .description("UDP destination port of the lidar packets.")
        .range(1U, 65535U),
    holoscan::field<&VlpPcapSourceParams::period_ms>("period_ms")
        .description("Clock period in milliseconds.")
        .range(1U, 1000U),
    holoscan::field<&VlpPcapSourceParams::loop>("loop").description(
        "Restart at the end of the recording."));

/// Replay VLP-16 packets from a pcap recording at their recorded pace.
class VlpPcapSourceOp final : public holoscan::Operator<VlpPcapSourceParams> {
 public:
  explicit VlpPcapSourceOp(std::shared_ptr<RunProgress> progress)
      : progress_(std::move(progress)) {}

  void setup(holoscan::OperatorSpec& spec) override { declare_packet_output(spec, packets); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = std::chrono::milliseconds{params().period_ms}});
    return result;
  }

  void start() override {
    const auto& p = params();
    try {
      reader_.emplace(p.path, static_cast<std::uint16_t>(p.port));
    } catch (const std::exception& error) {
      throw holoscan::RuntimeError{holoscan::ErrorCode::kInvalidArgument, error.what()};
    }
    staging_.assign(std::size_t{kMaxBatchPackets} * kPacketBytes, 0U);
    have_pending_ = false;
    finished_ = false;
    alternate_model_logged_ = false;
    first_record_ns_.reset();
  }

  void stop() override { reader_.reset(); }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override try {
    if (finished_) return {};
    const auto now = std::chrono::steady_clock::now();
    std::uint32_t count = 0;
    while (count < kMaxBatchPackets) {
      if (!have_pending_) {
        std::uint64_t timestamp_ns = 0;
        if (!reader_->next(pending_.data(), timestamp_ns)) {
          if (!context.params<VlpPcapSourceParams>().loop || !first_record_ns_) {
            finished_ = true;
            break;
          }
          reader_->rewind();
          first_record_ns_.reset();
          continue;
        }
        if (!alternate_model_logged_ && pending_[kModelOffset] == kModelVLP16Alternate) {
          alternate_model_logged_ = true;
          HOLOSCAN_LOG_INFO(
              "VlpPcapSourceOp: packets report model 0x21 (HDL-32E); decoding them as VLP-16");
        }
        if (!first_record_ns_) {
          first_record_ns_ = timestamp_ns;
          replay_start_ = now;
        }
        pending_offset_ = std::chrono::nanoseconds{
            static_cast<std::int64_t>(timestamp_ns - std::min(timestamp_ns, *first_record_ns_))};
        have_pending_ = true;
      }
      if (replay_start_ + pending_offset_ > now) break;  // not due yet
      std::copy(pending_.begin(), pending_.end(),
                staging_.begin() + static_cast<std::ptrdiff_t>(count * kPacketBytes));
      have_pending_ = false;
      ++count;
    }
    if (count > 0U)
      HOLOSCAN_TRY(emit_packets(packets, staging_.data(), count, context, *progress_));
    if (finished_) {
      progress_->invalid_packets.store(reader_->skipped());
      if (progress_->emitted_packets.load() == 0U) {
        return holoscan::make_unexpected(
            holoscan::Error{holoscan::ErrorCode::kInvalidArgument,
                            "no VLP-16 data packets to UDP port " +
                                std::to_string(context.params<VlpPcapSourceParams>().port) +
                                " in " + context.params<VlpPcapSourceParams>().path +
                                "; skipped records: " + reader_->skip_summary()});
      }
      HOLOSCAN_LOG_INFO("VlpPcapSourceOp: end of recording; skipped records: {}",
                        reader_->skip_summary());
      progress_->source_done.store(true, std::memory_order_release);
    }
    return {};
  } catch (const std::exception& error) {
    return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kFailure, error.what()});
  }

  holoscan::Output<holoscan::Tensor> packets;

 private:
  std::shared_ptr<RunProgress> progress_;
  std::optional<PcapPacketReader> reader_;
  std::vector<std::uint8_t> staging_;
  std::array<std::uint8_t, kPacketBytes> pending_{};
  std::optional<std::uint64_t> first_record_ns_;
  std::chrono::steady_clock::time_point replay_start_{};
  std::chrono::nanoseconds pending_offset_{};
  bool have_pending_{};
  bool finished_{};
  bool alternate_model_logged_{};
};

}  // namespace velodyne_lidar
