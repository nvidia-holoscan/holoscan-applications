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

// Minimal classic-pcap reader that yields VLP-16 UDP payloads; no libpcap dependency.

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "vlp16_packet.hpp"

namespace velodyne_lidar {

class PcapPacketReader {
 public:
  static constexpr std::uint32_t kMaxRecordBytes = 262144U;

  PcapPacketReader(const std::string& path, std::uint16_t port) : port_(port) {
    file_.open(path, std::ios::binary);
    if (!file_) throw std::runtime_error("cannot open pcap file: " + path);
    std::array<std::uint8_t, 24> header{};
    if (!file_.read(reinterpret_cast<char*>(header.data()), header.size())) {
      throw std::runtime_error("pcap file is shorter than its global header: " + path);
    }
    const std::uint32_t magic = read32(header.data(), false);
    if (magic == 0xA1B2C3D4U || magic == 0xA1B23C4DU) {
      swapped_ = false;
    } else if (magic == 0xD4C3B2A1U || magic == 0x4D3CB2A1U) {
      swapped_ = true;
    } else {
      throw std::runtime_error("not a classic pcap file (pcapng is not supported): " + path);
    }
    nanosecond_ = magic == 0xA1B23C4DU || magic == 0x4D3CB2A1U;
    link_type_ = read32(header.data() + 20, swapped_);
    if (link_type_ != kLinkEthernet && link_type_ != kLinkRawIP && link_type_ != kLinkLinuxSLL) {
      throw std::runtime_error("unsupported pcap link type " + std::to_string(link_type_));
    }
    data_start_ = file_.tellg();
  }

  /// Read the next VLP-16 payload sent to the configured UDP port.
  /// @return false at end of file; skipped records are counted in \c skipped().
  bool next(std::uint8_t* packet, std::uint64_t& timestamp_ns) {
    std::array<std::uint8_t, 16> record{};
    while (file_.read(reinterpret_cast<char*>(record.data()), record.size())) {
      const std::uint64_t seconds = read32(record.data(), swapped_);
      const std::uint64_t fraction = read32(record.data() + 4, swapped_);
      const std::uint32_t captured = read32(record.data() + 8, swapped_);
      const std::uint32_t original = read32(record.data() + 12, swapped_);
      if (captured > kMaxRecordBytes) throw std::runtime_error("corrupt pcap record length");
      buffer_.resize(captured);
      if (!file_.read(reinterpret_cast<char*>(buffer_.data()), captured)) {
        throw std::runtime_error("truncated pcap record");
      }
      timestamp_ns = seconds * 1000000000ULL + fraction * (nanosecond_ ? 1ULL : 1000ULL);
      if (captured != original) {
        ++skips_.truncated;
        continue;
      }
      std::size_t size = 0;
      const std::uint8_t* payload = udp_payload(size);
      if (payload == nullptr) {
        ++skips_.other_traffic;
        continue;
      }
      switch (validate_packet(payload, size)) {
        case PacketStatus::kOk:
          std::copy(payload, payload + kPacketBytes, packet);
          return true;
        case PacketStatus::kWrongSize:
          ++skips_.wrong_size;
          break;
        case PacketStatus::kBadBlockFlag:
          ++skips_.bad_block_flag;
          break;
        case PacketStatus::kUnsupportedModel:
          ++skips_.unsupported_model;
          skips_.last_model = payload[kModelOffset];
          break;
      }
    }
    if (!file_.eof()) throw std::runtime_error("failed to read pcap record");
    return false;
  }

  /// Restart from the first record.
  void rewind() {
    file_.clear();
    file_.seekg(data_start_);
  }

  /// Why records were skipped; position packets and other traffic count as other_traffic.
  struct SkipCounts {
    std::uint64_t truncated{};
    std::uint64_t other_traffic{};
    std::uint64_t wrong_size{};
    std::uint64_t bad_block_flag{};
    std::uint64_t unsupported_model{};
    std::uint8_t last_model{};
  };

  [[nodiscard]] std::uint64_t skipped() const noexcept {
    return skips_.truncated + skips_.other_traffic + skips_.wrong_size + skips_.bad_block_flag +
           skips_.unsupported_model;
  }
  [[nodiscard]] const SkipCounts& skip_counts() const noexcept { return skips_; }

  /// Describe skipped records, e.g. "12 unsupported model (last 0x24), 3 other traffic".
  [[nodiscard]] std::string skip_summary() const {
    std::string text;
    const auto add = [&text](std::uint64_t count, const std::string& what) {
      if (count == 0U) return;
      if (!text.empty()) text += ", ";
      text += std::to_string(count) + " " + what;
    };
    add(skips_.truncated, "truncated");
    add(skips_.other_traffic, "other traffic or port");
    add(skips_.wrong_size, "wrong payload size");
    add(skips_.bad_block_flag, "bad block flag");
    if (skips_.unsupported_model != 0U) {
      char model[8];
      std::snprintf(model, sizeof(model), "0x%02X", skips_.last_model);
      add(skips_.unsupported_model,
          std::string("unsupported model (last ") + model + ", VLP-16 is 0x22 or 0x21)");
    }
    return text.empty() ? "none" : text;
  }

 private:
  static constexpr std::uint32_t kLinkEthernet = 1U;
  static constexpr std::uint32_t kLinkRawIP = 101U;
  static constexpr std::uint32_t kLinkLinuxSLL = 113U;

  static std::uint32_t read32(const std::uint8_t* b, bool big_endian) {
    return big_endian ? (std::uint32_t{b[0]} << 24U) | (std::uint32_t{b[1]} << 16U) |
                            (std::uint32_t{b[2]} << 8U) | b[3]
                      : (std::uint32_t{b[3]} << 24U) | (std::uint32_t{b[2]} << 16U) |
                            (std::uint32_t{b[1]} << 8U) | b[0];
  }
  static std::uint16_t read16be(const std::uint8_t* b) {
    return static_cast<std::uint16_t>((b[0] << 8U) | b[1]);
  }

  // Return the UDP payload of an unfragmented IPv4 datagram to port_ and set its size.
  [[nodiscard]] const std::uint8_t* udp_payload(std::size_t& size) const {
    std::size_t offset = 0;
    std::uint16_t ether_type = 0x0800U;
    if (link_type_ == kLinkEthernet) {
      offset = 14;
      if (buffer_.size() < offset) return nullptr;
      ether_type = read16be(buffer_.data() + 12);
      while ((ether_type == 0x8100U || ether_type == 0x88A8U) && buffer_.size() >= offset + 4) {
        ether_type = read16be(buffer_.data() + offset + 2);
        offset += 4;
      }
    } else if (link_type_ == kLinkLinuxSLL) {
      offset = 16;
      if (buffer_.size() < offset) return nullptr;
      ether_type = read16be(buffer_.data() + 14);
    }
    if (ether_type != 0x0800U || buffer_.size() < offset + 20) return nullptr;
    const std::uint8_t* ip = buffer_.data() + offset;
    const std::size_t ihl = (ip[0] & 0x0FU) * 4U;
    if ((ip[0] >> 4U) != 4U || ihl < 20 || ip[9] != 17U) return nullptr;
    if ((read16be(ip + 6) & 0x3FFFU) != 0U) return nullptr;  // fragment
    if (buffer_.size() < offset + ihl + 8) return nullptr;
    const std::uint8_t* udp = ip + ihl;
    const std::size_t udp_length = read16be(udp + 4);
    if (read16be(udp + 2) != port_ || udp_length < 8U) return nullptr;
    if (buffer_.size() < offset + ihl + udp_length) return nullptr;
    size = udp_length - 8U;
    return udp + 8;
  }

  std::ifstream file_;
  std::streampos data_start_{};
  std::vector<std::uint8_t> buffer_;
  SkipCounts skips_{};
  std::uint32_t link_type_{};
  std::uint16_t port_{};
  bool swapped_{};
  bool nanosecond_{};
};

}  // namespace velodyne_lidar
