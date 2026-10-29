/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

// VLP-16 data packet layout and decoding, ported from the NVIDIA DeepMap SDK.
// SDK-free and shared by host code, CUDA kernels, and tests.

#pragma once

#include <math.h>

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define VLP16_HOST_DEVICE __host__ __device__
#else
#define VLP16_HOST_DEVICE
#endif

namespace velodyne_lidar {

inline constexpr std::size_t kPacketBytes = 1206U;
inline constexpr std::size_t kBlocksPerPacket = 12U;
inline constexpr std::size_t kBlockBytes = 100U;
inline constexpr std::size_t kRecordsPerBlock = 32U;  // two 16-laser firing sequences
inline constexpr std::size_t kRecordBytes = 3U;
inline constexpr std::size_t kLasers = 16U;
inline constexpr std::size_t kPointsPerPacket = kBlocksPerPacket * kRecordsPerBlock;  // 384
inline constexpr std::size_t kPointComponents = 4U;   // x, y, z (meters), intensity
inline constexpr std::uint16_t kBlockFlag = 0xEEFFU;  // bytes 0xFF 0xEE, little-endian
inline constexpr std::size_t kReturnModeOffset = 1204U;
inline constexpr std::size_t kModelOffset = 1205U;
// In dual-return mode, block pairs (2k, 2k+1) carry two returns of the same firing sequences.
inline constexpr std::uint8_t kReturnModeDual = 0x39U;
inline constexpr std::uint8_t kModelVLP16 = 0x22U;
// Velodyne assigns 0x21 to the HDL-32E, but some VLP-16 recordings (including the VeloView
// samples) report it. HDL-32E is not supported, so 0x21 packets are decoded as VLP-16.
inline constexpr std::uint8_t kModelVLP16Alternate = 0x21U;
inline constexpr float kDistanceUnitMeters = 0.002F;
inline constexpr std::uint16_t kDefaultPort = 2368U;

enum class PacketStatus : std::uint8_t { kOk, kWrongSize, kBadBlockFlag, kUnsupportedModel };

[[nodiscard]] VLP16_HOST_DEVICE inline std::uint16_t read_le16(const std::uint8_t* bytes) {
  return static_cast<std::uint16_t>(bytes[0] | (bytes[1] << 8U));
}

/// Validate the fixed VLP-16 single/dual-return data packet framing.
[[nodiscard]] inline PacketStatus validate_packet(const std::uint8_t* packet, std::size_t size) {
  if (size != kPacketBytes) return PacketStatus::kWrongSize;
  for (std::size_t block = 0; block < kBlocksPerPacket; ++block) {
    if (read_le16(packet + block * kBlockBytes) != kBlockFlag) return PacketStatus::kBadBlockFlag;
  }
  const std::uint8_t model = packet[kModelOffset];
  if (model != kModelVLP16 && model != kModelVLP16Alternate) return PacketStatus::kUnsupportedModel;
  return PacketStatus::kOk;
}

[[nodiscard]] inline const char* packet_status_name(PacketStatus status) {
  switch (status) {
    case PacketStatus::kOk:
      return "ok";
    case PacketStatus::kWrongSize:
      return "wrong size";
    case PacketStatus::kBadBlockFlag:
      return "bad block flag";
    case PacketStatus::kUnsupportedModel:
      return "unsupported model";
  }
  return "unknown";
}

/// Return the elevation of VLP-16 laser 0..15 in degrees (-15, 1, -13, 3, ..., -1, 15).
[[nodiscard]] VLP16_HOST_DEVICE inline float laser_pitch_degrees(std::uint32_t laser) {
  return (laser % 2U == 0U) ? -15.0F + static_cast<float>(laser) : static_cast<float>(laser);
}

/// Decode one record of a validated packet into x, y, z (meters) and intensity.
///
/// Records 16..31 belong to the second firing sequence, whose azimuth is interpolated halfway
/// to the next firing (the next block, or the next block pair in dual-return mode). A zero
/// distance (no return) decodes to the origin.
VLP16_HOST_DEVICE inline void decode_point(const std::uint8_t* packet, std::uint32_t block,
                                           std::uint32_t record, float* out) {
  const std::uint8_t* data = packet + block * kBlockBytes;
  float azimuth = static_cast<float>(read_le16(data + 2U)) * 0.01F;
  if (record >= kLasers) {
    const std::uint32_t stride = packet[kReturnModeOffset] == kReturnModeDual ? 2U : 1U;
    const std::uint32_t first = block - block % stride;
    const bool has_next = first + stride < kBlocksPerPacket;
    const std::uint32_t other = has_next ? first + stride : first - stride;
    const float neighbor = static_cast<float>(read_le16(packet + other * kBlockBytes + 2U)) * 0.01F;
    float gap = has_next ? neighbor - azimuth : azimuth - neighbor;
    if (gap < 0.0F) gap += 360.0F;
    azimuth += 0.5F * gap;
  }
  const std::uint8_t* rec = data + 4U + record * kRecordBytes;
  const float range = static_cast<float>(read_le16(rec)) * kDistanceUnitMeters;
  constexpr float kDegToRad = 3.14159265358979F / 180.0F;
  const float pitch = laser_pitch_degrees(record % kLasers) * kDegToRad;
  const float yaw = azimuth * kDegToRad;
  const float horizontal = range * cosf(pitch);
  out[0] = horizontal * sinf(yaw);
  out[1] = horizontal * cosf(yaw);
  out[2] = range * sinf(pitch);
  out[3] = range > 0.0F ? static_cast<float>(rec[2]) : 0.0F;
}

}  // namespace velodyne_lidar
