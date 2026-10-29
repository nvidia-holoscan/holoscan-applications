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

// Source-operator and host decoding checks. The graphs use only host tensors.

#include <unistd.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <arpa/inet.h>  // NOLINT(build/include_order)
#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/time/realtime_clock.hpp>
#include <netinet/in.h>  // NOLINT(build/include_order)
#include <sys/socket.h>  // NOLINT(build/include_order)

#include "vlp16_packet.hpp"
#include "vlp16_pcap.hpp"
#include "vlp16_synthetic.hpp"
#include "vlp_pcap_source.hpp"
#include "vlp_synthetic_source.hpp"
#include "vlp_udp_source.hpp"

namespace vl = velodyne_lidar;
using std::chrono_literals::operator""ms;
using std::chrono_literals::operator""s;

namespace {

int failures = 0;

void check(bool condition, const std::string& what) {
  if (!condition) {
    ++failures;
    std::cerr << "FAIL: " << what << '\n';
  }
}

struct Observed {
  std::mutex mutex;
  std::vector<std::uint8_t> first_packet;
  std::atomic<std::uint64_t> packets{};
  std::atomic<std::uint64_t> invalid{};
};

class PacketSinkOp final : public holoscan::Operator<> {
 public:
  PacketSinkOp(std::shared_ptr<Observed> observed, std::uint64_t expected)
      : observed_(std::move(observed)), expected_(expected) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(packets, "packets")
        .queue_depth(8U)
        .expects_tensor(holoscan::TensorInputSpec{.representation = vl::kPacketBatch});
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{packets});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto batch = packets.receive_data();
    if (!batch) return holoscan::make_unexpected(std::move(batch).error());
    const auto shape = batch->shape_span();
    const auto* data = batch->data_as<std::uint8_t>();
    for (std::int64_t i = 0; i < shape[0]; ++i) {
      const std::uint8_t* packet = data + i * vl::kPacketBytes;
      if (vl::validate_packet(packet, vl::kPacketBytes) != vl::PacketStatus::kOk) {
        observed_->invalid.fetch_add(1U);
      }
      const std::lock_guard lock(observed_->mutex);
      if (observed_->first_packet.empty()) {
        observed_->first_packet.assign(packet, packet + vl::kPacketBytes);
      }
    }
    if (observed_->packets.fetch_add(static_cast<std::uint64_t>(shape[0])) +
            static_cast<std::uint64_t>(shape[0]) >=
        expected_) {
      context.request_completion();
    }
    return {};
  }

  holoscan::Input<holoscan::Tensor> packets;

 private:
  std::shared_ptr<Observed> observed_;
  std::uint64_t expected_;
};

// Compile and run a source -> sink graph; \p while_running is polled until completion.
template <typename Source, typename Params>
bool run_source(const char* name, const Params& params, std::shared_ptr<vl::RunProgress> progress,
                std::shared_ptr<Observed> observed, std::uint64_t expected,
                const std::function<void()>& while_running = {}) {
  holoscan::Graph graph{name};
  auto source = graph.op<Source>("source", progress);
  source.set_params(params);
  auto sink = graph.op<PacketSinkOp>("sink", observed, expected);
  graph.add_flow(source->packets, sink->packets,
                 holoscan::ConnectionOptions{.queue_policy = holoscan::QueuePolicy::kBlocking,
                                             .queue_depth = 8U});
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  const auto plan = holoscan::compile(graph);
  if (!plan.ok()) {
    for (const auto& diagnostic : plan.diagnostics()) {
      std::cerr << name << " compile: " << diagnostic.token << '\n';
    }
    return false;
  }
  auto session = holoscan::run_async(plan);
  const auto deadline = std::chrono::steady_clock::now() + 20s;
  while (session.is_running() && std::chrono::steady_clock::now() < deadline) {
    if (while_running) while_running();
    std::this_thread::sleep_for(1ms);
  }
  const bool finished = !session.is_running();
  if (!finished) session.request_stop();
  session.wait();
  for (const auto& diagnostic : session.diagnostics()) {
    std::cerr << name << " diagnostic: " << diagnostic.token << '\n';
  }
  return finished && session.diagnostics().empty() &&
         session.termination() == holoscan::RunTermination::kRequestedCompleted;
}

void test_decode() {
  std::array<std::uint8_t, vl::kPacketBytes> packet{};
  vl::make_synthetic_packet(3, packet.data());
  check(vl::validate_packet(packet.data(), packet.size()) == vl::PacketStatus::kOk,
        "synthetic packet validates");
  check(vl::validate_packet(packet.data(), 1205) == vl::PacketStatus::kWrongSize, "size check");
  auto bad = packet;
  bad[700] = 0;
  check(vl::validate_packet(bad.data(), bad.size()) == vl::PacketStatus::kBadBlockFlag,
        "block flag check");
  bad = packet;
  bad[vl::kModelOffset] = 0x24;
  check(vl::validate_packet(bad.data(), bad.size()) == vl::PacketStatus::kUnsupportedModel,
        "model check");
  bad[vl::kModelOffset] = vl::kModelVLP16Alternate;
  check(vl::validate_packet(bad.data(), bad.size()) == vl::PacketStatus::kOk,
        "model 0x21 decodes as VLP-16");
  for (std::uint32_t block = 0; block < vl::kBlocksPerPacket; ++block) {
    for (std::uint32_t record = 0; record < vl::kRecordsPerBlock; ++record) {
      float p[4];
      vl::decode_point(packet.data(), block, record, p);
      const float horizontal = std::hypot(p[0], p[1]);
      check(p[2] >= -1.81F && p[2] <= 3.8F, "point height inside the room");
      check(horizontal > 0.4F && horizontal < 14.5F, "point inside the room");
      if (vl::laser_pitch_degrees(record % vl::kLasers) < 0.0F && horizontal < 6.0F) {
        check(std::abs(p[2] + 1.8F) < 0.01F, "near downward returns hit the floor");
      }
    }
  }
  // A record at 10 m, laser 0 (-15 deg), block azimuth 90 deg points along +x and down.
  std::array<std::uint8_t, vl::kPacketBytes> manual{};
  for (std::uint32_t block = 0; block < vl::kBlocksPerPacket; ++block) {
    manual[block * 100] = 0xFF;
    manual[block * 100 + 1] = 0xEE;
    manual[block * 100 + 2] = 9000 & 0xFF;
    manual[block * 100 + 3] = 9000 >> 8;
  }
  manual[4] = 5000 & 0xFF;
  manual[5] = 5000 >> 8;
  float p[4];
  vl::decode_point(manual.data(), 0, 0, p);
  check(std::abs(p[0] - 10.0F * std::cos(15.0F * 3.14159265F / 180.0F)) < 1e-3F &&
            std::abs(p[1]) < 1e-3F &&
            std::abs(p[2] + 10.0F * std::sin(15.0F * 3.14159265F / 180.0F)) < 1e-3F,
        "manual record decodes to (9.66, 0, -2.59)");

  // Dual return: block pairs share an azimuth (10, 10, 10.4, 10.4, ..., 12, 12 deg), so the
  // second firing of every block sits halfway to the next pair, the last pair extrapolated.
  std::array<std::uint8_t, vl::kPacketBytes> dual{};
  for (std::uint32_t block = 0; block < vl::kBlocksPerPacket; ++block) {
    const std::uint32_t azimuth = 1000U + 40U * (block / 2U);
    dual[block * 100] = 0xFF;
    dual[block * 100 + 1] = 0xEE;
    dual[block * 100 + 2] = azimuth & 0xFF;
    dual[block * 100 + 3] = azimuth >> 8;
    // Record 17 (laser 1, +1 deg) at 10 m.
    dual[block * 100 + 4 + 17 * 3] = 5000 & 0xFF;
    dual[block * 100 + 4 + 17 * 3 + 1] = 5000 >> 8;
  }
  dual[vl::kReturnModeOffset] = vl::kReturnModeDual;
  dual[vl::kModelOffset] = vl::kModelVLP16;
  check(vl::validate_packet(dual.data(), dual.size()) == vl::PacketStatus::kOk,
        "dual-return packet validates");
  for (std::uint32_t block = 0; block < vl::kBlocksPerPacket; ++block) {
    vl::decode_point(dual.data(), block, 17, p);
    const float expected = 10.2F + 0.4F * static_cast<float>(block / 2U);
    const float decoded = std::atan2(p[0], p[1]) * 180.0F / 3.14159265F;
    check(std::abs(decoded - expected) < 1e-3F,
          "dual-return block " + std::to_string(block) + " azimuth " + std::to_string(decoded));
  }
}

void write_le32(std::ofstream& out, std::uint32_t value) {
  for (int i = 0; i < 4; ++i) out.put(static_cast<char>((value >> (8 * i)) & 0xFFU));
}

// Write \p packets VLP-16 frames plus one ARP frame and one UDP datagram to another port.
void write_fixture(const std::filesystem::path& path, std::uint32_t packets) {
  std::ofstream out(path, std::ios::binary);
  write_le32(out, 0xA1B2C3D4U);
  out.put(2).put(0).put(4).put(0);
  write_le32(out, 0);
  write_le32(out, 0);
  write_le32(out, 65535);
  write_le32(out, 1);
  auto frame = [](std::uint16_t port, const std::uint8_t* payload, std::size_t size) {
    std::vector<std::uint8_t> bytes(14, 0xFF);
    bytes[12] = 0x08;
    bytes[13] = 0x00;
    const std::size_t udp_size = 8 + size;
    const std::uint8_t ip[20] = {0x45,
                                 0,
                                 static_cast<std::uint8_t>((20 + udp_size) >> 8),
                                 static_cast<std::uint8_t>(20 + udp_size),
                                 0,
                                 0,
                                 0x40,
                                 0,
                                 64,
                                 17,
                                 0,
                                 0,
                                 192,
                                 168,
                                 1,
                                 201,
                                 255,
                                 255,
                                 255,
                                 255};
    bytes.insert(bytes.end(), ip, ip + 20);
    const std::uint8_t udp[8] = {0x09,
                                 0x40,
                                 static_cast<std::uint8_t>(port >> 8),
                                 static_cast<std::uint8_t>(port),
                                 static_cast<std::uint8_t>(udp_size >> 8),
                                 static_cast<std::uint8_t>(udp_size),
                                 0,
                                 0};
    bytes.insert(bytes.end(), udp, udp + 8);
    bytes.insert(bytes.end(), payload, payload + size);
    return bytes;
  };
  auto record = [&](std::uint32_t micros, const std::vector<std::uint8_t>& bytes) {
    write_le32(out, micros / 1000000U);
    write_le32(out, micros % 1000000U);
    write_le32(out, static_cast<std::uint32_t>(bytes.size()));
    write_le32(out, static_cast<std::uint32_t>(bytes.size()));
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
  };
  std::array<std::uint8_t, vl::kPacketBytes> packet{};
  for (std::uint32_t i = 0; i < packets; ++i) {
    vl::make_synthetic_packet(i, packet.data());
    record(i * 1327U, frame(vl::kDefaultPort, packet.data(), packet.size()));
    if (i == 0) {
      std::vector<std::uint8_t> arp(42, 0);
      arp[12] = 0x08;
      arp[13] = 0x06;
      record(1U, arp);
      record(2U, frame(8308, packet.data(), packet.size()));
    }
  }
}

void test_pcap_reader(const std::filesystem::path& path) {
  vl::PcapPacketReader reader(path.string(), vl::kDefaultPort);
  std::array<std::uint8_t, vl::kPacketBytes> packet{};
  std::array<std::uint8_t, vl::kPacketBytes> expected{};
  std::uint64_t timestamp = 0;
  std::uint32_t count = 0;
  while (reader.next(packet.data(), timestamp)) {
    vl::make_synthetic_packet(count, expected.data());
    check(packet == expected, "pcap packet bytes match");
    check(timestamp == count * 1327000ULL, "pcap timestamp");
    ++count;
  }
  check(count == 20U, "pcap reader yields every lidar packet");
  check(reader.skipped() == 2U, "pcap reader skips ARP and other ports");
  reader.rewind();
  check(reader.next(packet.data(), timestamp) && timestamp == 0U, "pcap rewind");

  vl::PcapPacketReader other_port(path.string(), vl::kDefaultPort + 1U);
  check(!other_port.next(packet.data(), timestamp), "pcap reader filters by port");
  check(other_port.skip_counts().other_traffic == 22U &&
            other_port.skip_summary() == "22 other traffic or port",
        "pcap reader reports skip reasons");
}

void test_sources(const std::filesystem::path& pcap) {
  {
    auto progress = std::make_shared<vl::RunProgress>();
    auto observed = std::make_shared<Observed>();
    const bool ok = run_source<vl::VlpSyntheticSourceOp>(
        "synthetic",
        vl::VlpSyntheticSourceParams{.count = 50U, .packets_per_tick = 8U, .period_ms = 1U},
        progress, observed, 50U);
    check(ok, "synthetic graph completes");
    check(observed->packets == 50U && observed->invalid == 0U, "synthetic packets arrive");
    check(progress->source_done.load(), "synthetic source finishes");
  }
  {
    auto progress = std::make_shared<vl::RunProgress>();
    auto observed = std::make_shared<Observed>();
    const bool ok = run_source<vl::VlpSyntheticSourceOp>(
        "synthetic-endless",
        vl::VlpSyntheticSourceParams{.count = 0U, .packets_per_tick = 64U, .period_ms = 1U},
        progress, observed, 2000U);
    check(ok, "endless synthetic graph runs until the sink completes");
    check(observed->packets >= 2000U && !progress->source_done.load(),
          "endless synthetic source keeps emitting");
  }
  {
    auto progress = std::make_shared<vl::RunProgress>();
    auto observed = std::make_shared<Observed>();
    const bool ok = run_source<vl::VlpPcapSourceOp>(
        "pcap", vl::VlpPcapSourceParams{.path = pcap.string(), .period_ms = 2U}, progress, observed,
        20U);
    check(ok, "pcap graph completes");
    check(observed->packets == 20U && observed->invalid == 0U, "pcap packets arrive");
    std::array<std::uint8_t, vl::kPacketBytes> expected{};
    vl::make_synthetic_packet(0, expected.data());
    check(std::equal(expected.begin(), expected.end(), observed->first_packet.begin()),
          "pcap source preserves packet bytes");
  }
  {
    auto progress = std::make_shared<vl::RunProgress>();
    auto observed = std::make_shared<Observed>();
    const bool ok = run_source<vl::VlpPcapSourceOp>(
        "pcap-empty",
        vl::VlpPcapSourceParams{
            .path = pcap.string(), .port = vl::kDefaultPort + 1U, .period_ms = 2U},
        progress, observed, 1U);
    check(!ok && observed->packets == 0U, "pcap source fails fast without VLP-16 packets");
  }

  {
    auto progress = std::make_shared<vl::RunProgress>();
    auto observed = std::make_shared<Observed>();
    const std::uint16_t port = static_cast<std::uint16_t>(40000U + (::getpid() % 20000U));
    const int sender = ::socket(AF_INET, SOCK_DGRAM, 0);
    sockaddr_in to{};
    to.sin_family = AF_INET;
    to.sin_port = htons(port);
    ::inet_pton(AF_INET, "127.0.0.1", &to.sin_addr);
    std::array<std::uint8_t, vl::kPacketBytes> packet{};
    std::uint32_t sent = 0;
    // Send until the sink has 40 packets; datagrams sent before the bind are lost.
    auto send_some = [&] {
      if (observed->packets >= 40U) return;
      vl::make_synthetic_packet(sent++, packet.data());
      ::sendto(sender, packet.data(), packet.size(), 0, reinterpret_cast<sockaddr*>(&to),
               sizeof(to));
      ::sendto(sender, packet.data(), 100, 0, reinterpret_cast<sockaddr*>(&to), sizeof(to));
    };
    const bool ok = run_source<vl::VlpUdpSourceOp>(
        "udp", vl::VlpUdpSourceParams{.address = "127.0.0.1", .port = port}, progress, observed,
        40U, send_some);
    ::close(sender);
    check(ok, "udp graph completes");
    check(observed->packets >= 40U && observed->invalid == 0U, "udp packets arrive");
    check(progress->invalid_packets > 0U, "udp source rejects short datagrams");
  }
}

}  // namespace

int main() {
  test_decode();
  const auto pcap = std::filesystem::temp_directory_path() /
                    ("velodyne_sources_test_" + std::to_string(::getpid()) + ".pcap");
  write_fixture(pcap, 20U);
  test_pcap_reader(pcap);
  test_sources(pcap);
  std::filesystem::remove(pcap);
  std::cout << (failures == 0 ? "velodyne_sources_test: PASS\n" : "velodyne_sources_test: FAIL\n");
  return failures == 0 ? 0 : 1;
}
