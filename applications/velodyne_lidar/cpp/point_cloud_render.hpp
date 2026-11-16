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
#include <string>
#include <utility>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include "packet_batch.hpp"

namespace velodyne_lidar {

inline constexpr std::uint32_t kMaxImageSide = 2048U;

struct PointCloudRenderParams {
  std::string view{"top"};  // "top" or "chase"
  std::uint32_t width{720U};
  std::uint32_t height{720U};
  std::uint32_t period_ms{33U};
  std::uint32_t point_size{2U};
  float range_m{30.0F};  // top view: half the visible width
  float eye_x{0.0F};
  float eye_y{-14.0F};
  float eye_z{7.0F};
  float target_x{0.0F};
  float target_y{6.0F};
  float target_z{0.0F};
  float fov_deg{60.0F};
  float orbit_deg_per_s{0.0F};
  float z_min{-2.0F};  // height color map range in meters
  float z_max{3.0F};
};

HOLOSCAN_PARAM_SCHEMA(
    PointCloudRenderParams,
    holoscan::param_schema("holoscan.applications.velodyne_lidar.PointCloudRenderParams", 1),
    holoscan::field<&PointCloudRenderParams::view>("view").description(
        "Fixed camera: top (plan view) or chase (perspective)."),
    holoscan::field<&PointCloudRenderParams::width>("width").range(16U, kMaxImageSide),
    holoscan::field<&PointCloudRenderParams::height>("height").range(16U, kMaxImageSide),
    holoscan::field<&PointCloudRenderParams::period_ms>("period_ms")
        .description("Render period in milliseconds.")
        .range(1U, 1000U),
    holoscan::field<&PointCloudRenderParams::point_size>("point_size").range(1U, 8U),
    holoscan::field<&PointCloudRenderParams::range_m>("range_m").range(1.0F, 200.0F),
    holoscan::field<&PointCloudRenderParams::eye_x>("eye_x"),
    holoscan::field<&PointCloudRenderParams::eye_y>("eye_y"),
    holoscan::field<&PointCloudRenderParams::eye_z>("eye_z"),
    holoscan::field<&PointCloudRenderParams::target_x>("target_x"),
    holoscan::field<&PointCloudRenderParams::target_y>("target_y"),
    holoscan::field<&PointCloudRenderParams::target_z>("target_z"),
    holoscan::field<&PointCloudRenderParams::fov_deg>("fov_deg").range(10.0F, 150.0F),
    holoscan::field<&PointCloudRenderParams::orbit_deg_per_s>("orbit_deg_per_s")
        .description("Chase view only: rotate the eye around the target's vertical axis.")
        .range(-360.0F, 360.0F),
    holoscan::field<&PointCloudRenderParams::z_min>("z_min"),
    holoscan::field<&PointCloudRenderParams::z_max>("z_max"));

/// Fixed-camera projection used by the render kernels.
struct RenderCamera {
  std::uint32_t width{};
  std::uint32_t height{};
  std::uint32_t point_size{};
  bool top{};
  float scale{};  // top: pixels per meter; chase: focal length in pixels
  float eye[3]{};
  float right[3]{};
  float up[3]{};
  float forward[3]{};
  float z_min{};
  float z_max{};
};

/// Build the camera for \p params after \p seconds of orbit.
[[nodiscard]] RenderCamera make_camera(const PointCloudRenderParams& params, double seconds);

/// Render \p points float4 points into an RGBA8 image using \p depth (width * height uint64).
cudaError_t launch_render(const float* cloud, std::uint32_t points, const RenderCamera& camera,
                          std::uint64_t* depth, std::uint8_t* rgba, cudaStream_t stream);

/// Render the newest point cloud from a fixed top-down or chase camera into an RGBA image.
class PointCloudRenderOp final : public holoscan::Operator<PointCloudRenderParams> {
 public:
  explicit PointCloudRenderOp(std::shared_ptr<RunProgress> progress)
      : progress_(std::move(progress)) {}

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Input<holoscan::Tensor> cloud;
  holoscan::Output<holoscan::schema::ImageT> image;

 private:
  struct DeviceFree {
    void operator()(void* pointer) const noexcept { static_cast<void>(cudaFree(pointer)); }
  };

  std::shared_ptr<RunProgress> progress_;
  std::unique_ptr<std::uint64_t, DeviceFree> depth_;
  std::uint64_t frame_index_{};
};

}  // namespace velodyne_lidar
