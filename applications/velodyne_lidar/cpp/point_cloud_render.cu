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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>
#include <utility>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

#include "point_cloud_render.hpp"

namespace velodyne_lidar {
namespace {

constexpr std::uint64_t kEmpty = ~std::uint64_t{0};
constexpr DLDataType kRgbaU8{kDLUInt, 8U, 1U};

__device__ std::uint32_t height_color(float z, float z_min, float z_max) {
  // Blue -> cyan -> green -> yellow -> red.
  const float t = fminf(fmaxf((z - z_min) / fmaxf(z_max - z_min, 1e-3F), 0.0F), 1.0F) * 4.0F;
  const float r = fminf(fmaxf(t - 2.0F, 0.0F), 1.0F);
  const float g = t < 1.0F ? t : (t < 3.0F ? 1.0F : 4.0F - t);
  const float b = fminf(fmaxf(2.0F - t, 0.0F), 1.0F);
  return static_cast<std::uint32_t>(r * 255.0F) | (static_cast<std::uint32_t>(g * 255.0F) << 8U) |
         (static_cast<std::uint32_t>(b * 255.0F) << 16U) | (255U << 24U);
}

__global__ void clear_depth(std::uint64_t* depth, std::uint32_t pixels) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < pixels) depth[i] = kEmpty;
}

__global__ void splat_points(const float4* cloud, std::uint32_t points, RenderCamera camera,
                             std::uint64_t* depth) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= points) return;
  const float4 p = cloud[i];
  if (p.x * p.x + p.y * p.y + p.z * p.z < 0.0025F) return;  // no return
  float px = 0.0F;
  float py = 0.0F;
  float distance = 0.0F;
  if (camera.top) {
    px = 0.5F * camera.width + p.x * camera.scale;
    py = 0.5F * camera.height - p.y * camera.scale;
    distance = 1000.0F - p.z;  // higher points on top
  } else {
    const float d[3] = {p.x - camera.eye[0], p.y - camera.eye[1], p.z - camera.eye[2]};
    distance = d[0] * camera.forward[0] + d[1] * camera.forward[1] + d[2] * camera.forward[2];
    if (distance < 0.1F) return;
    const float x = d[0] * camera.right[0] + d[1] * camera.right[1] + d[2] * camera.right[2];
    const float y = d[0] * camera.up[0] + d[1] * camera.up[1] + d[2] * camera.up[2];
    px = 0.5F * camera.width + camera.scale * x / distance;
    py = 0.5F * camera.height - camera.scale * y / distance;
  }
  if (!(distance > 0.0F)) return;
  const int x0 = static_cast<int>(floorf(px)) - static_cast<int>(camera.point_size / 2U);
  const int y0 = static_cast<int>(floorf(py)) - static_cast<int>(camera.point_size / 2U);
  const std::uint64_t key = (static_cast<std::uint64_t>(__float_as_uint(distance)) << 32U) |
                            height_color(p.z, camera.z_min, camera.z_max);
  for (int dy = 0; dy < static_cast<int>(camera.point_size); ++dy) {
    for (int dx = 0; dx < static_cast<int>(camera.point_size); ++dx) {
      const int x = x0 + dx;
      const int y = y0 + dy;
      if (x < 0 || y < 0 || x >= static_cast<int>(camera.width) ||
          y >= static_cast<int>(camera.height)) {
        continue;
      }
      // CUDA exposes 64-bit atomicMin only for unsigned long long.
      atomicMin(reinterpret_cast<unsigned long long*>(  // NOLINT(runtime/int)
                    depth + static_cast<std::size_t>(y) * camera.width + x),
                static_cast<unsigned long long>(key));  // NOLINT(runtime/int)
    }
  }
}

__global__ void resolve(const std::uint64_t* depth, RenderCamera camera, uchar4* rgba) {
  const std::uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= camera.width || y >= camera.height) return;
  const std::size_t i = static_cast<std::size_t>(y) * camera.width + x;
  const std::uint64_t key = depth[i];
  if (key != kEmpty) {
    const auto color = static_cast<std::uint32_t>(key & 0xFFFFFFFFU);
    rgba[i] = make_uchar4(color & 0xFFU, (color >> 8U) & 0xFFU, (color >> 16U) & 0xFFU, 255U);
    return;
  }
  uchar4 background = make_uchar4(12U, 12U, 18U, 255U);
  if (camera.top) {
    // 10 m range rings around the sensor.
    const float dx = (x + 0.5F - 0.5F * camera.width) / camera.scale;
    const float dy = (y + 0.5F - 0.5F * camera.height) / camera.scale;
    const float ring = fmodf(sqrtf(dx * dx + dy * dy), 10.0F);
    if (ring < 1.0F / camera.scale || 10.0F - ring < 1.0F / camera.scale) {
      background = make_uchar4(60U, 60U, 70U, 255U);
    }
  }
  rgba[i] = background;
}

void normalize(float* v) {
  const float length = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  if (length > 0.0F) {
    for (int i = 0; i < 3; ++i) v[i] /= length;
  }
}

void cross(const float* a, const float* b, float* out) {
  out[0] = a[1] * b[2] - a[2] * b[1];
  out[1] = a[2] * b[0] - a[0] * b[2];
  out[2] = a[0] * b[1] - a[1] * b[0];
}

}  // namespace

RenderCamera make_camera(const PointCloudRenderParams& params, double seconds) {
  RenderCamera camera;
  camera.width = params.width;
  camera.height = params.height;
  camera.point_size = params.point_size;
  camera.top = params.view == "top";
  camera.z_min = params.z_min;
  camera.z_max = params.z_max;
  if (camera.top) {
    camera.scale =
        0.5F * static_cast<float>(std::min(params.width, params.height)) / params.range_m;
    return camera;
  }
  const double angle = params.orbit_deg_per_s * seconds * 3.14159265358979 / 180.0;
  const float ox = params.eye_x - params.target_x;
  const float oy = params.eye_y - params.target_y;
  camera.eye[0] = params.target_x + static_cast<float>(ox * std::cos(angle) - oy * std::sin(angle));
  camera.eye[1] = params.target_y + static_cast<float>(ox * std::sin(angle) + oy * std::cos(angle));
  camera.eye[2] = params.eye_z;
  camera.forward[0] = params.target_x - camera.eye[0];
  camera.forward[1] = params.target_y - camera.eye[1];
  camera.forward[2] = params.target_z - camera.eye[2];
  normalize(camera.forward);
  const float world_up[3] = {0.0F, 0.0F, 1.0F};
  cross(camera.forward, world_up, camera.right);
  normalize(camera.right);
  cross(camera.right, camera.forward, camera.up);
  camera.scale = 0.5F * static_cast<float>(params.height) /
                 static_cast<float>(std::tan(0.5 * params.fov_deg * 3.14159265358979 / 180.0));
  return camera;
}

cudaError_t launch_render(const float* cloud, std::uint32_t points, const RenderCamera& camera,
                          std::uint64_t* depth, std::uint8_t* rgba, cudaStream_t stream) {
  const std::uint32_t pixels = camera.width * camera.height;
  constexpr std::uint32_t kThreads = 256U;
  clear_depth<<<(pixels + kThreads - 1U) / kThreads, kThreads, 0, stream>>>(depth, pixels);
  if (points > 0U) {
    splat_points<<<(points + kThreads - 1U) / kThreads, kThreads, 0, stream>>>(
        reinterpret_cast<const float4*>(cloud), points, camera, depth);
  }
  const dim3 block(16U, 16U);
  const dim3 grid((camera.width + 15U) / 16U, (camera.height + 15U) / 16U);
  resolve<<<grid, block, 0, stream>>>(depth, camera, reinterpret_cast<uchar4*>(rgba));
  return cudaPeekAtLastError();
}

void PointCloudRenderOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(cloud, "cloud")
      .queue_depth(1U)
      .expects_tensor(
          holoscan::TensorInputSpec{.representation = holoscan::TensorRepresentation{
                                        .memory_kind = holoscan::MemoryKind::kCudaDevice,
                                        .dtype = kFloat32,
                                        .rank = 2U,
                                        .strides = holoscan::TensorStridePolicy::kContiguousOnly,
                                    }});
  spec.output(image, "image")
      .produces_tensor(holoscan::TensorOutputSpec{
          .representation =
              holoscan::TensorRepresentation{
                  .memory_kind = holoscan::MemoryKind::kCudaDevice,
                  .dtype = kRgbaU8,
                  .rank = 3U,
                  .strides = holoscan::TensorStridePolicy::kContiguousOnly,
              },
          .bounds = holoscan::tensor_bounds(std::size_t{kMaxImageSide} * kMaxImageSide * 4U),
          .storage = holoscan::TensorOutputStorage::kRuntimePool,
      })
      .payload_options(holoscan::PayloadOptions{
          .schema = holoscan::schema_identity<holoscan::schema::ImageT>(),
      });
}

holoscan::Contract PointCloudRenderOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnClock{.period = std::chrono::milliseconds{params().period_ms}})
      .input(cloud, holoscan::LatestWithin{.max_age = std::chrono::seconds{1}})
      .on_missing(cloud, holoscan::Skip{});
  return result;
}

void PointCloudRenderOp::start() {
  const auto& p = params();
  void* pointer = nullptr;
  const cudaError_t error =
      cudaMalloc(&pointer, std::size_t{p.width} * p.height * sizeof(std::uint64_t));
  if (error != cudaSuccess) {
    throw holoscan::RuntimeError{
        holoscan::ErrorCode::kResourceExhausted,
        std::string("PointCloudRenderOp: cudaMalloc: ") + cudaGetErrorString(error)};
  }
  depth_.reset(static_cast<std::uint64_t*>(pointer));
  frame_index_ = 0U;
}

void PointCloudRenderOp::stop() {
  static_cast<void>(cudaDeviceSynchronize());
  depth_.reset();
}

holoscan::expected<void, holoscan::Error> PointCloudRenderOp::compute(
    holoscan::ExecutionContext& context) {
  const auto& p = context.params<PointCloudRenderParams>();
  auto sample = cloud.receive();
  if (!sample) return holoscan::make_unexpected(std::move(sample).error());
  const holoscan::Tensor& points = sample->data;
  const auto shape = points.shape_span();
  if (shape.size() != 2U || shape[1] != static_cast<std::int64_t>(kPointComponents)) {
    return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kInvalidArgument,
                                                     "PointCloudRenderOp: expected [N, 4] points"});
  }
  const std::array<std::int64_t, 3> image_shape{p.height, p.width, 4};
  auto loan =
      image.allocate_tensor(holoscan::TensorLoanRequest{.shape = image_shape, .dtype = kRgbaU8});
  if (!loan) return holoscan::make_unexpected(std::move(loan).error());
  auto writer = loan->write(context.cuda_stream());
  if (!writer) return holoscan::make_unexpected(std::move(writer).error());
  const double seconds = static_cast<double>(frame_index_++) * p.period_ms * 1e-3;
  const cudaError_t error = launch_render(
      points.data_as<float>(), static_cast<std::uint32_t>(shape[0]), make_camera(p, seconds),
      depth_.get(), writer->data_as<std::uint8_t>(), context.cuda_stream().get());
  if (error != cudaSuccess) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kFailure,
                        std::string("PointCloudRenderOp: render: ") + cudaGetErrorString(error)});
  }
  HOLOSCAN_TRY(std::move(*writer).commit());
  holoscan::schema::ImageT descriptor;
  descriptor.width = static_cast<std::int32_t>(p.width);
  descriptor.height = static_cast<std::int32_t>(p.height);
  descriptor.encoding = holoscan::schema::ImageEncoding_RGBA8;
  HOLOSCAN_TRY(image.emit_tensor(std::move(*loan), descriptor,
                                 holoscan::EmitOptions{.capture_time = context.activation_time(),
                                                       .frame_id = sample->metadata.frame_id}));
  progress_->frames.fetch_add(1U);
  return {};
}

}  // namespace velodyne_lidar
