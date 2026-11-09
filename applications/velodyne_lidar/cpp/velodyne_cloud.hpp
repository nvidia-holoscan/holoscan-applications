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

#include <array>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>

#include "packet_batch.hpp"

namespace velodyne_lidar {

inline constexpr std::uint32_t kMaxCloudPackets = 256U;  // ~3.4 revolutions at 10 Hz

struct VelodyneCloudParams {
  std::uint32_t packet_buffer_size{80U};  // ~1.05 revolutions at 10 Hz
};

HOLOSCAN_PARAM_SCHEMA(
    VelodyneCloudParams,
    holoscan::param_schema("holoscan.applications.velodyne_lidar.VelodyneCloudParams", 1),
    holoscan::field<&VelodyneCloudParams::packet_buffer_size>("packet_buffer_size")
        .description("Packets kept in the rolling point cloud (384 points each).")
        .range(1U, kMaxCloudPackets));

/// Device float32 [packet_buffer_size * 384, 4] point cloud: x, y, z in meters and intensity.
inline constexpr holoscan::TensorRepresentation kPointCloud{
    .memory_kind = holoscan::MemoryKind::kCudaDevice,
    .dtype = kFloat32,
    .rank = 2U,
    .strides = holoscan::TensorStridePolicy::kContiguousOnly,
};

/// Convert VLP-16 packet batches to XYZ on the GPU and emit a snapshot of the rolling cloud.
class VelodyneCloudOp final : public holoscan::Operator<VelodyneCloudParams> {
 public:
  explicit VelodyneCloudOp(std::shared_ptr<RunProgress> progress)
      : progress_(std::move(progress)) {}

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Input<holoscan::Tensor> packets;
  holoscan::Output<holoscan::Tensor> cloud;

 private:
  static constexpr std::uint32_t kStagingSlots = 4U;

  struct DeviceFree {
    void operator()(void* pointer) const noexcept { static_cast<void>(cudaFree(pointer)); }
  };
  struct HostFree {
    void operator()(void* pointer) const noexcept { static_cast<void>(cudaFreeHost(pointer)); }
  };
  struct EventDestroy {
    void operator()(cudaEvent_t event) const noexcept {
      static_cast<void>(cudaEventDestroy(event));
    }
  };
  using DeviceBuffer = std::unique_ptr<std::uint8_t, DeviceFree>;
  using PinnedBuffer = std::unique_ptr<std::uint8_t, HostFree>;
  using Event = std::unique_ptr<std::remove_pointer_t<cudaEvent_t>, EventDestroy>;

  std::shared_ptr<RunProgress> progress_;
  DeviceBuffer ring_;
  std::array<PinnedBuffer, kStagingSlots> host_staging_;
  std::array<DeviceBuffer, kStagingSlots> device_staging_;
  std::array<Event, kStagingSlots> staging_free_;
  Event ring_written_;
  std::uint32_t ring_packets_{};
  std::uint32_t ring_index_{};
  std::uint32_t next_slot_{};
};

}  // namespace velodyne_lidar
