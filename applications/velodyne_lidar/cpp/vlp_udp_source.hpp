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

#include <poll.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stop_token>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <arpa/inet.h>  // NOLINT(build/include_order)
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/lifecycle.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/readiness_source.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/logger/logger.hpp>
#include <netinet/in.h>  // NOLINT(build/include_order)
#include <sys/socket.h>  // NOLINT(build/include_order)

#include "packet_batch.hpp"

namespace velodyne_lidar {

struct VlpUdpSourceParams {
  std::string address{"0.0.0.0"};
  std::uint32_t port{kDefaultPort};
  std::uint32_t queue_packets{4096U};
  std::uint32_t receive_buffer_bytes{8U * 1024U * 1024U};
};

HOLOSCAN_PARAM_SCHEMA(
    VlpUdpSourceParams,
    holoscan::param_schema("holoscan.applications.velodyne_lidar.VlpUdpSourceParams", 1),
    holoscan::field<&VlpUdpSourceParams::address>("address").description(
        "Local IPv4 address to bind; 0.0.0.0 listens on every interface."),
    holoscan::field<&VlpUdpSourceParams::port>("port")
        .description("Local UDP port; VLP-16 sensors send data packets to 2368.")
        .range(1U, 65535U),
    holoscan::field<&VlpUdpSourceParams::queue_packets>("queue_packets")
        .description("Packets buffered between the receive thread and compute; oldest drop first.")
        .range(kMaxBatchPackets, 1048576U),
    holoscan::field<&VlpUdpSourceParams::receive_buffer_bytes>("receive_buffer_bytes")
        .description("Requested SO_RCVBUF size; the kernel may clamp it.")
        .range(65536U, 1073741824U));

/// Receive VLP-16 data packets from a UDP socket.
///
/// SDK 5.0 has no public file-descriptor adapter for OnReadable, so one operator-owned receive
/// thread polls the socket, queues packets, and wakes compute through a NotificationSender, as in
/// the SDK's notification_bridge example.
class VlpUdpSourceOp final : public holoscan::Operator<VlpUdpSourceParams> {
 public:
  explicit VlpUdpSourceOp(std::shared_ptr<RunProgress> progress) : progress_(std::move(progress)) {}

  void setup(holoscan::OperatorSpec& spec) override {
    declare_packet_output(spec, packets);
    spec.notification_source(notification_, "udp-packets")
        .capacity(1U)
        .sender_reference_capacity(1U);
    spec.lifecycle()
        .stage(holoscan::LifecycleStage::kArm, &VlpUdpSourceOp::on_arm)
        .stage(holoscan::LifecycleStage::kStart, &VlpUdpSourceOp::on_start)
        .stage(holoscan::LifecycleStage::kStop, &VlpUdpSourceOp::on_stop)
        .stage(holoscan::LifecycleStage::kRelease, &VlpUdpSourceOp::on_release);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnNotified{.event = notification_});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (const auto code = worker_error_.load(std::memory_order_acquire);
        code != holoscan::ErrorCode::kSuccess) {
      return holoscan::make_unexpected(holoscan::Error{code, "UDP receive thread failed"});
    }
    std::uint32_t count = 0;
    {
      const std::lock_guard lock(mutex_);
      count = static_cast<std::uint32_t>(std::min<std::size_t>(queued_, kMaxBatchPackets));
      for (std::uint32_t i = 0; i < count; ++i) {
        std::memcpy(staging_.data() + std::size_t{i} * kPacketBytes,
                    queue_.data() + ((head_ + i) % capacity_) * kPacketBytes, kPacketBytes);
      }
      head_ = (head_ + count) % capacity_;
      queued_ -= count;
    }
    report_losses();
    if (count == 0U) return {};
    return emit_packets(packets, staging_.data(), count, context, *progress_);
  }

  holoscan::Output<holoscan::Tensor> packets;

 private:
  static constexpr std::chrono::milliseconds kIdlePoll{100};
  static constexpr std::chrono::milliseconds kBacklogPoll{1};

  holoscan::LifecycleStatus on_arm(holoscan::LifecycleContext& context) noexcept {
    const auto& p = params();
    sockaddr_in local{};
    local.sin_family = AF_INET;
    local.sin_port = htons(static_cast<std::uint16_t>(p.port));
    if (::inet_pton(AF_INET, p.address.c_str(), &local.sin_addr) != 1) {
      HOLOSCAN_LOG_ERROR("VlpUdpSourceOp: invalid IPv4 address '{}'", p.address);
      return holoscan::LifecycleStatus::kFatalFailure;
    }
    fd_ = ::socket(AF_INET, SOCK_DGRAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (fd_ < 0) {
      HOLOSCAN_LOG_ERROR("VlpUdpSourceOp: socket() failed: {}", std::strerror(errno));
      return holoscan::LifecycleStatus::kFatalFailure;
    }
    const int reuse = 1;
    static_cast<void>(::setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)));
    const int rcvbuf = static_cast<int>(p.receive_buffer_bytes);
    static_cast<void>(::setsockopt(fd_, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf)));
    if (::bind(fd_, reinterpret_cast<const sockaddr*>(&local), sizeof(local)) != 0) {
      HOLOSCAN_LOG_ERROR("VlpUdpSourceOp: cannot bind {}:{}: {}", p.address, p.port,
                         std::strerror(errno));
      return holoscan::LifecycleStatus::kFatalFailure;
    }
    auto sender = context.notification_sender(notification_);
    if (!sender) {
      HOLOSCAN_LOG_ERROR("VlpUdpSourceOp: no notification sender: {}", sender.error().message);
      return holoscan::LifecycleStatus::kFatalFailure;
    }
    sender_ = std::move(*sender);
    try {
      capacity_ = p.queue_packets;
      queue_.assign(capacity_ * kPacketBytes, 0U);
      staging_.assign(std::size_t{kMaxBatchPackets} * kPacketBytes, 0U);
    } catch (...) {
      return holoscan::LifecycleStatus::kFatalFailure;
    }
    head_ = queued_ = 0U;
    alternate_model_logged_ = false;
    HOLOSCAN_LOG_INFO("VlpUdpSourceOp: listening on {}:{}", p.address, p.port);
    return holoscan::LifecycleStatus::kOk;
  }

  holoscan::LifecycleStatus on_start(holoscan::LifecycleContext&) noexcept {
    try {
      worker_ =
          std::jthread([this](const std::stop_token& token) noexcept { receive_loop(token); });
    } catch (...) {
      return holoscan::LifecycleStatus::kFatalFailure;
    }
    return holoscan::LifecycleStatus::kOk;
  }

  holoscan::LifecycleStatus on_stop(holoscan::LifecycleContext&) noexcept {
    if (worker_.joinable()) {
      worker_.request_stop();
      try {
        worker_.join();  // the bounded poll() timeout bounds join latency
      } catch (...) {
        return holoscan::LifecycleStatus::kFatalFailure;
      }
    }
    return holoscan::LifecycleStatus::kOk;
  }

  holoscan::LifecycleStatus on_release(holoscan::LifecycleContext&) noexcept {
    sender_ = holoscan::NotificationSender{};
    if (fd_ >= 0) {
      static_cast<void>(::close(fd_));
      fd_ = -1;
    }
    return holoscan::LifecycleStatus::kOk;
  }

  void receive_loop(const std::stop_token& token) noexcept {
    std::array<std::uint8_t, 2048> datagram{};
    while (!token.stop_requested()) {
      bool backlog = false;
      {
        const std::lock_guard lock(mutex_);
        backlog = queued_ > 0U;
      }
      pollfd waitable{.fd = fd_, .events = POLLIN, .revents = 0};
      const auto timeout = backlog ? kBacklogPoll : kIdlePoll;
      const int ready = ::poll(&waitable, 1U, static_cast<int>(timeout.count()));
      if (ready < 0 && errno != EINTR) return fail(holoscan::ErrorCode::kFailure, token);
      if (ready > 0 && (waitable.revents & (POLLERR | POLLNVAL)) != 0) {
        return fail(holoscan::ErrorCode::kFailure, token);
      }
      while (ready > 0) {
        const ssize_t size = ::recv(fd_, datagram.data(), datagram.size(), MSG_TRUNC);
        if (size < 0) {
          if (errno == EAGAIN || errno == EWOULDBLOCK) break;
          if (errno == EINTR) continue;
          return fail(holoscan::ErrorCode::kFailure, token);
        }
        if (validate_packet(datagram.data(), static_cast<std::size_t>(size)) != PacketStatus::kOk) {
          progress_->invalid_packets.fetch_add(1U);
          continue;
        }
        if (!alternate_model_logged_ && datagram[kModelOffset] == kModelVLP16Alternate) {
          alternate_model_logged_ = true;
          HOLOSCAN_LOG_INFO(
              "VlpUdpSourceOp: packets report model 0x21 (HDL-32E); decoding them as VLP-16");
        }
        enqueue(datagram.data());
      }
      bool pending = false;
      {
        const std::lock_guard lock(mutex_);
        pending = queued_ > 0U;
      }
      if (!pending) continue;
      // kNotReady (before start) and kBackpressured (activation in flight) retry on the next
      // backlog poll; compute drains every queued packet in batches.
      const auto posted = sender_.post();
      if (!posted && posted.error().code != holoscan::ErrorCode::kNotReady &&
          posted.error().code != holoscan::ErrorCode::kBackpressured) {
        if (token.stop_requested()) return;
        return fail(posted.error().code, token);
      }
    }
  }

  void enqueue(const std::uint8_t* packet) noexcept {
    const std::lock_guard lock(mutex_);
    if (queued_ == capacity_) {
      head_ = (head_ + 1U) % capacity_;
      --queued_;
      progress_->dropped_packets.fetch_add(1U);
    }
    std::memcpy(queue_.data() + ((head_ + queued_) % capacity_) * kPacketBytes, packet,
                kPacketBytes);
    ++queued_;
  }

  void fail(holoscan::ErrorCode code, const std::stop_token& token) noexcept {
    worker_error_.store(code, std::memory_order_release);
    while (!token.stop_requested()) {
      const auto posted = sender_.post();
      if (posted || (posted.error().code != holoscan::ErrorCode::kNotReady &&
                     posted.error().code != holoscan::ErrorCode::kBackpressured)) {
        return;
      }
      std::this_thread::sleep_for(kBacklogPoll);
    }
  }

  void report_losses() {
    const auto now = std::chrono::steady_clock::now();
    if (now - last_report_ < std::chrono::seconds{5}) return;
    last_report_ = now;
    const std::uint64_t dropped = progress_->dropped_packets.load();
    const std::uint64_t invalid = progress_->invalid_packets.load();
    if (dropped != reported_dropped_ || invalid != reported_invalid_) {
      HOLOSCAN_LOG_WARN("VlpUdpSourceOp: {} packets dropped (queue full), {} invalid datagrams",
                        dropped - reported_dropped_, invalid - reported_invalid_);
      reported_dropped_ = dropped;
      reported_invalid_ = invalid;
    }
  }

  std::shared_ptr<RunProgress> progress_;
  holoscan::NotificationSource notification_;
  holoscan::NotificationSender sender_;
  std::jthread worker_;
  std::atomic<holoscan::ErrorCode> worker_error_{holoscan::ErrorCode::kSuccess};
  bool alternate_model_logged_{};  // receive thread only
  std::mutex mutex_;
  std::vector<std::uint8_t> queue_;
  std::vector<std::uint8_t> staging_;
  std::size_t capacity_{};
  std::size_t head_{};
  std::size_t queued_{};
  std::chrono::steady_clock::time_point last_report_{};
  std::uint64_t reported_dropped_{};
  std::uint64_t reported_invalid_{};
  int fd_{-1};
};

}  // namespace velodyne_lidar
