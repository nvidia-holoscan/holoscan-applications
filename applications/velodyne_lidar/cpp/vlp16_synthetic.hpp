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
#include <cmath>
#include <cstdint>

#include "vlp16_packet.hpp"

namespace velodyne_lidar {

/// Fill a deterministic VLP-16 packet: a 10 Hz sweep of a 24 m x 16 m room with a pillar,
/// seen from 1.8 m above a flat floor.
inline void make_synthetic_packet(std::uint64_t index, std::uint8_t* packet) {
  constexpr double kDegToRad = 3.14159265358979 / 180.0;
  constexpr double kSensorHeight = 1.8;
  for (std::uint32_t block = 0; block < kBlocksPerPacket; ++block) {
    std::uint8_t* data = packet + block * kBlockBytes;
    // 10 Hz: ~0.2 degrees per 16-laser firing sequence, two sequences per block.
    const auto azimuth =
        static_cast<std::uint16_t>(((index * kBlocksPerPacket + block) * 40U) % 36000U);
    data[0] = 0xFFU;
    data[1] = 0xEEU;
    data[2] = static_cast<std::uint8_t>(azimuth & 0xFFU);
    data[3] = static_cast<std::uint8_t>(azimuth >> 8U);
    for (std::uint32_t record = 0; record < kRecordsPerBlock; ++record) {
      const double yaw = (azimuth * 0.01 + (record >= kLasers ? 0.2 : 0.0)) * kDegToRad;
      const double pitch = laser_pitch_degrees(record % kLasers) * kDegToRad;
      const double dx = std::sin(yaw);
      const double dy = std::cos(yaw);
      // Horizontal distance to the room walls, shortened by a 1 m pillar at (4, 5).
      double horizontal =
          std::min(dx != 0.0 ? 12.0 / std::abs(dx) : 1e9, dy != 0.0 ? 8.0 / std::abs(dy) : 1e9);
      const double along = 4.0 * dx + 5.0 * dy;
      const double across = std::abs(4.0 * dy - 5.0 * dx);
      if (along > 0.0 && across < 0.5) horizontal = std::min(horizontal, along - 0.5);
      double range = horizontal / std::cos(pitch);
      if (pitch < 0.0) range = std::min(range, kSensorHeight / std::sin(-pitch));
      if (range > 100.0) range = 0.0;  // no return
      const auto ticks = static_cast<std::uint16_t>(std::lround(range / kDistanceUnitMeters));
      std::uint8_t* rec = data + 4U + record * kRecordBytes;
      rec[0] = static_cast<std::uint8_t>(ticks & 0xFFU);
      rec[1] = static_cast<std::uint8_t>(ticks >> 8U);
      rec[2] = static_cast<std::uint8_t>(16U * (record % kLasers) + 15U);
    }
  }
  const auto micros = static_cast<std::uint32_t>((index * 1327U) % 3600000000ULL);
  for (std::uint32_t byte = 0; byte < 4U; ++byte) {
    packet[1200U + byte] = static_cast<std::uint8_t>(micros >> (8U * byte));
  }
  packet[kReturnModeOffset] = 0x37U;  // strongest return
  packet[kModelOffset] = kModelVLP16;
}

}  // namespace velodyne_lidar
