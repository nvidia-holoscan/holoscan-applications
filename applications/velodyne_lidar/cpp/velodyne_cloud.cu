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
#include <string>
#include <utility>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

#include "velodyne_cloud.hpp"
#include "vlp16_decode_kernel.hpp"
#include "vlp16_packet.hpp"

namespace velodyne_lidar {
namespace {

__global__ void decode_packets_kernel(const std::uint8_t* packets, float4* ring,
                                      std::uint32_t first_slot, std::uint32_t ring_packets) {
  const std::uint32_t packet = blockIdx.x;
  const std::uint32_t record = threadIdx.x;
  const std::uint32_t block = threadIdx.y;
  float point[kPointComponents];
  decode_point(packets + std::size_t{packet} * kPacketBytes, block, record, point);
  const std::uint32_t slot = (first_slot + packet) % ring_packets;
  ring[std::size_t{slot} * kPointsPerPacket + block * kRecordsPerBlock + record] =
      make_float4(point[0], point[1], point[2], point[3]);
}

holoscan::Error cuda_error(const char* what, cudaError_t error) {
  return holoscan::Error{holoscan::ErrorCode::kFailure, std::string("VelodyneCloudOp: ") + what +
                                                            ": " + cudaGetErrorString(error)};
}

void throw_on_error(const char* what, cudaError_t error) {
  if (error != cudaSuccess) {
    throw holoscan::RuntimeError{holoscan::ErrorCode::kResourceExhausted,
                                 cuda_error(what, error).message};
  }
}

}  // namespace

cudaError_t launch_decode_packets(const std::uint8_t* packets, std::uint32_t count, float* ring,
                                  std::uint32_t first_slot, std::uint32_t ring_packets,
                                  cudaStream_t stream) {
  if (count == 0U) return cudaSuccess;
  if (count > ring_packets) return cudaErrorInvalidValue;
  decode_packets_kernel<<<count, dim3(kRecordsPerBlock, kBlocksPerPacket), 0, stream>>>(
      packets, reinterpret_cast<float4*>(ring), first_slot, ring_packets);
  return cudaPeekAtLastError();
}

void VelodyneCloudOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(packets, "packets")
      .queue_depth(8U)
      .expects_tensor(holoscan::TensorInputSpec{.representation = kPacketBatch});
  spec.output(cloud, "cloud")
      .produces_tensor(holoscan::TensorOutputSpec{
          .representation = kPointCloud,
          .bounds = holoscan::tensor_bounds(std::size_t{kMaxCloudPackets} * kPointsPerPacket *
                                            kPointComponents * sizeof(float)),
          .storage = holoscan::TensorOutputStorage::kRuntimePool,
      });
}

holoscan::Contract VelodyneCloudOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{packets});
  return result;
}

void VelodyneCloudOp::start() {
  ring_packets_ = params().packet_buffer_size;
  ring_index_ = 0U;
  next_slot_ = 0U;
  const std::size_t ring_bytes =
      std::size_t{ring_packets_} * kPointsPerPacket * kPointComponents * sizeof(float);
  const std::size_t staging_bytes = std::size_t{kMaxBatchPackets} * kPacketBytes;
  void* pointer = nullptr;
  throw_on_error("cudaMalloc(ring)", cudaMalloc(&pointer, ring_bytes));
  ring_.reset(static_cast<std::uint8_t*>(pointer));
  // Unfilled slots stay at the origin, which the renderer treats as "no return".
  throw_on_error("cudaMemset(ring)", cudaMemset(ring_.get(), 0, ring_bytes));
  for (std::uint32_t slot = 0; slot < kStagingSlots; ++slot) {
    throw_on_error("cudaMallocHost", cudaMallocHost(&pointer, staging_bytes));
    host_staging_[slot].reset(static_cast<std::uint8_t*>(pointer));
    throw_on_error("cudaMalloc(staging)", cudaMalloc(&pointer, staging_bytes));
    device_staging_[slot].reset(static_cast<std::uint8_t*>(pointer));
    cudaEvent_t event = nullptr;
    throw_on_error("cudaEventCreate", cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    staging_free_[slot].reset(event);
  }
  cudaEvent_t event = nullptr;
  throw_on_error("cudaEventCreate", cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  ring_written_.reset(event);
  throw_on_error("cudaEventRecord", cudaEventRecord(ring_written_.get(), nullptr));
}

void VelodyneCloudOp::stop() {
  static_cast<void>(cudaDeviceSynchronize());
  ring_written_.reset();
  for (std::uint32_t slot = 0; slot < kStagingSlots; ++slot) {
    staging_free_[slot].reset();
    device_staging_[slot].reset();
    host_staging_[slot].reset();
  }
  ring_.reset();
}

holoscan::expected<void, holoscan::Error> VelodyneCloudOp::compute(
    holoscan::ExecutionContext& context) {
  auto batch = packets.receive_data();
  if (!batch) return holoscan::make_unexpected(std::move(batch).error());
  const holoscan::Tensor& tensor = *batch;
  const auto shape = tensor.shape_span();
  if (shape.size() != 2U || shape[0] < 1 || shape[0] > kMaxBatchPackets ||
      shape[1] != static_cast<std::int64_t>(kPacketBytes) || !tensor.is_contiguous()) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kInvalidArgument, "VelodyneCloudOp: expected [N, 1206] packets"});
  }
  HOLOSCAN_TRY(tensor.synchronize_for_host());
  auto count = static_cast<std::uint32_t>(shape[0]);
  const std::uint8_t* source = tensor.data_as<std::uint8_t>();
  if (count > ring_packets_) {  // only the newest packets fit
    source += std::size_t{count - ring_packets_} * kPacketBytes;
    count = ring_packets_;
  }
  const std::size_t nbytes = std::size_t{count} * kPacketBytes;
  const cudaStream_t stream = context.cuda_stream().get();

  // Reuse a pinned staging slot only after its previous upload finished; this waits only when
  // the GPU is kStagingSlots batches behind.
  const std::uint32_t slot = next_slot_;
  next_slot_ = (next_slot_ + 1U) % kStagingSlots;
  if (cudaEventQuery(staging_free_[slot].get()) == cudaErrorNotReady) {
    if (const auto error = cudaEventSynchronize(staging_free_[slot].get()); error != cudaSuccess) {
      return holoscan::make_unexpected(cuda_error("staging wait", error));
    }
  }
  std::copy(source, source + nbytes, host_staging_[slot].get());
  // The ring persists across activations, so order this activation after the previous writer.
  cudaError_t error = cudaStreamWaitEvent(stream, ring_written_.get(), 0);
  if (error == cudaSuccess) {
    error = cudaMemcpyAsync(device_staging_[slot].get(), host_staging_[slot].get(), nbytes,
                            cudaMemcpyHostToDevice, stream);
  }
  if (error == cudaSuccess) error = cudaEventRecord(staging_free_[slot].get(), stream);
  if (error == cudaSuccess) {
    error = launch_decode_packets(device_staging_[slot].get(), count,
                                  reinterpret_cast<float*>(ring_.get()), ring_index_, ring_packets_,
                                  stream);
  }
  if (error != cudaSuccess) return holoscan::make_unexpected(cuda_error("decode", error));
  ring_index_ = (ring_index_ + count) % ring_packets_;

  const std::array<std::int64_t, 2> cloud_shape{
      static_cast<std::int64_t>(std::size_t{ring_packets_} * kPointsPerPacket),
      static_cast<std::int64_t>(kPointComponents)};
  auto loan =
      cloud.allocate_tensor(holoscan::TensorLoanRequest{.shape = cloud_shape, .dtype = kFloat32});
  if (!loan) return holoscan::make_unexpected(std::move(loan).error());
  auto writer = loan->write(context.cuda_stream());
  if (!writer) return holoscan::make_unexpected(std::move(writer).error());
  error = cudaMemcpyAsync(writer->data(), ring_.get(), writer->byte_size(),
                          cudaMemcpyDeviceToDevice, stream);
  if (error == cudaSuccess) error = cudaEventRecord(ring_written_.get(), stream);
  if (error != cudaSuccess) return holoscan::make_unexpected(cuda_error("snapshot", error));
  HOLOSCAN_TRY(std::move(*writer).commit());
  // frame_id counts integrated batches, so a renderer can tell which batches a frame shows.
  const std::uint64_t generation = progress_->integrated_batches.load() + 1U;
  HOLOSCAN_TRY(cloud.emit(
      std::move(*loan),
      holoscan::EmitOptions{.capture_time = context.activation_time(), .frame_id = generation}));
  progress_->integrated_batches.store(generation);
  return {};
}

}  // namespace velodyne_lidar
