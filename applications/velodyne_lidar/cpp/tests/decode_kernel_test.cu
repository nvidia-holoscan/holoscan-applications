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

// Compare the CUDA packet decoder against the host reference.

#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "vlp16_decode_kernel.hpp"
#include "vlp16_packet.hpp"
#include "vlp16_synthetic.hpp"

namespace vl = velodyne_lidar;

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::cout << "SKIP: no CUDA device\n";
    return 77;
  }
  constexpr std::uint32_t kPackets = 8U;
  constexpr std::uint32_t kRing = 10U;
  constexpr std::uint32_t kFirstSlot = 6U;  // wraps around the ring end
  std::vector<std::uint8_t> packets(kPackets * vl::kPacketBytes);
  for (std::uint32_t i = 0; i < kPackets; ++i) {
    vl::make_synthetic_packet(100U + i, packets.data() + i * vl::kPacketBytes);
  }
  const std::size_t ring_floats = std::size_t{kRing} * vl::kPointsPerPacket * 4U;
  std::uint8_t* device_packets = nullptr;
  float* device_ring = nullptr;
  bool ok = cudaMalloc(&device_packets, packets.size()) == cudaSuccess &&
            cudaMalloc(&device_ring, ring_floats * sizeof(float)) == cudaSuccess &&
            cudaMemcpy(device_packets, packets.data(), packets.size(), cudaMemcpyHostToDevice) ==
                cudaSuccess &&
            cudaMemset(device_ring, 0, ring_floats * sizeof(float)) == cudaSuccess;
  ok = ok && vl::launch_decode_packets(device_packets, kPackets, device_ring, kFirstSlot, kRing,
                                       nullptr) == cudaSuccess;
  ok = ok && vl::launch_decode_packets(device_packets, kRing + 1U, device_ring, 0U, kRing,
                                       nullptr) == cudaErrorInvalidValue;
  std::vector<float> ring(ring_floats);
  ok = ok && cudaMemcpy(ring.data(), device_ring, ring_floats * sizeof(float),
                        cudaMemcpyDeviceToHost) == cudaSuccess;
  if (!ok) {
    std::cerr << "FAIL: CUDA setup or launch\n";
    return 1;
  }
  std::size_t mismatches = 0;
  for (std::uint32_t slot = 0; slot < kRing; ++slot) {
    const std::uint32_t packet = (slot + kRing - kFirstSlot) % kRing;
    for (std::uint32_t point = 0; point < vl::kPointsPerPacket; ++point) {
      std::array<float, 4> expected{};
      if (packet < kPackets) {
        vl::decode_point(packets.data() + packet * vl::kPacketBytes, point / vl::kRecordsPerBlock,
                         point % vl::kRecordsPerBlock, expected.data());
      }
      const float* actual = ring.data() + (std::size_t{slot} * vl::kPointsPerPacket + point) * 4U;
      for (int c = 0; c < 4; ++c) {
        if (std::abs(actual[c] - expected[c]) > 1e-3F * (1.0F + std::abs(expected[c]))) {
          ++mismatches;
        }
      }
    }
  }
  cudaFree(device_packets);
  cudaFree(device_ring);
  if (mismatches != 0U) {
    std::cerr << "FAIL: " << mismatches << " decoded values differ from the host reference\n";
    return 1;
  }
  std::cout << "velodyne_decode_kernel_test: PASS\n";
  return 0;
}
