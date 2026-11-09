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

#include <cuda_runtime_api.h>

#include <cstdint>

namespace velodyne_lidar {

/// Decode \p count device-resident packets into a ring of float4 points (x, y, z, intensity).
///
/// Packet i lands in ring slot (first_slot + i) % ring_packets; each slot holds 384 points.
/// Requires count <= ring_packets so no two packets of one launch share a slot.
cudaError_t launch_decode_packets(const std::uint8_t* packets, std::uint32_t count, float* ring,
                                  std::uint32_t first_slot, std::uint32_t ring_packets,
                                  cudaStream_t stream);

}  // namespace velodyne_lidar
