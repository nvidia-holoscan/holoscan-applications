/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Runtime / functional test for VideoFileReaderOp::compute().

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>
#include <dlpack/dlpack.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "synthetic_mp4.hpp"
#include "video_file_reader.hpp"

namespace holoscan::ops {
namespace {

using namespace std::chrono_literals;

/// @brief Captures every tensor the sink sees so the test can inspect shape / dtype.
struct SinkObservation {
  static void reset() {
    const std::lock_guard lock(mutex);
    ticks = 0U;
    last_shape.clear();
    last_dtype = DLDataType{};
  }
  static void record(const std::vector<int64_t>& shape, const DLDataType& dtype) {
    {
      const std::lock_guard lock(mutex);
      ++ticks;
      last_shape = shape;
      last_dtype = dtype;
    }
    ready.notify_all();
  }
  [[nodiscard]] static bool wait_for(std::size_t expected, std::chrono::seconds timeout) {
    std::unique_lock lock(mutex);
    return ready.wait_for(lock, timeout, [expected] { return ticks >= expected; });
  }
  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::size_t ticks{};
  static inline std::vector<int64_t> last_shape;
  static inline DLDataType last_dtype{};
};

/// @brief Minimal consumer that records what the reader emits.
class TensorRecorder final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(4U)
        .expects_tensor(holoscan::TensorInputSpec{
            .representation = holoscan::TensorRepresentation{
                .memory_kind = holoscan::MemoryKind::kCudaDevice,
            },
        });
  }
  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto sample = input.receive_data();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    const holoscan::Tensor& t = *sample;
    SinkObservation::record(t.shape(), t.dtype());
    return {};
  }
  holoscan::Input<holoscan::Tensor> input;
};

[[nodiscard]] bool run_reader_pipeline(const VideoFileReaderOpParams& params,
                                        std::size_t expected_frames,
                                        std::chrono::seconds timeout,
                                        std::string& err) {
  SinkObservation::reset();
  try {
    holoscan::Graph graph{"video-file-reader-runtime-test"};
    const auto reader = graph.op<VideoFileReaderOp>("reader", params);
    const auto sink = graph.op<TensorRecorder>("sink");
    graph.add_flow(*reader->output_port(), sink->input,
                   holoscan::ConnectionOptions{.queue_depth = 4U});
    graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

    holoscan::CompileOptions options;
    options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = "reader",
        .output_port = params.tensor_port_name,
        .device = holoscan::DeviceId{0},
    });

    const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
    if (!plan.ok()) {
      err = "compile failed: " + std::string(plan.json());
      return false;
    }
    holoscan::RunSession session = holoscan::run_async(plan);
    const bool complete = SinkObservation::wait_for(expected_frames, timeout);
    session.request_stop();
    session.wait();
    if (!complete) {
      err = "sink did not observe " + std::to_string(expected_frames) + " frames within timeout";
      return false;
    }
    return true;
  } catch (const std::exception& e) {
    err = std::string("exception: ") + e.what();
    return false;
  }
}

}  // namespace

// =============================================================================================
// Fixture generates the synthetic MP4 once per test case under a per-test tmp dir, cleans up
// in TearDown so parallel ctest invocations don't collide.
// =============================================================================================

class VideoFileReaderRuntimeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    tmp_dir_ = std::filesystem::temp_directory_path() /
               ("video_file_reader_rt_" +
                std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(tmp_dir_);
    mp4_path_ = tmp_dir_ / "synthetic.mp4";
    ASSERT_NO_THROW(surgical_deid::test::encode_synthetic_mp4(
        mp4_path_.string(), /*width=*/128, /*height=*/128, /*frame_count=*/10));
    ASSERT_TRUE(std::filesystem::exists(mp4_path_));
    ASSERT_GT(std::filesystem::file_size(mp4_path_), 0U);
  }
  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove_all(tmp_dir_, ec);
  }

  std::filesystem::path tmp_dir_;
  std::filesystem::path mp4_path_;
};

TEST_F(VideoFileReaderRuntimeTest, Decodes_SyntheticMp4_AsRgb24Nhwc) {
  VideoFileReaderOpParams params;
  params.path = mp4_path_;
  params.tensor_port_name = "frame";
  params.output_pix_fmt = "rgb24";
  params.output_layout = "nhwc";
  params.prepend_batch_dim = false;
  params.frame_count = 5;  // Stop after 5 frames.
  params.memory_kind = holoscan::MemoryKind::kCudaDevice;

  std::string err;
  ASSERT_TRUE(run_reader_pipeline(params, 5U, 30s, err)) << err;

  const std::lock_guard lock(SinkObservation::mutex);
  EXPECT_GE(SinkObservation::ticks, 5U);
  // rank-3 [H, W, C] because prepend_batch_dim=false and layout=nhwc.
  ASSERT_EQ(SinkObservation::last_shape.size(), 3U);
  EXPECT_EQ(SinkObservation::last_shape[0], 128);  // H
  EXPECT_EQ(SinkObservation::last_shape[1], 128);  // W
  EXPECT_EQ(SinkObservation::last_shape[2], 3);    // C (RGB)
  EXPECT_EQ(SinkObservation::last_dtype.code, kDLUInt);
  EXPECT_EQ(SinkObservation::last_dtype.bits, 8U);
}

TEST_F(VideoFileReaderRuntimeTest, Decodes_SyntheticMp4_AsRgb24Nchw) {
  VideoFileReaderOpParams params;
  params.path = mp4_path_;
  params.output_pix_fmt = "rgb24";
  params.output_layout = "nchw";
  params.prepend_batch_dim = true;
  params.frame_count = 3;
  params.memory_kind = holoscan::MemoryKind::kCudaDevice;

  std::string err;
  ASSERT_TRUE(run_reader_pipeline(params, 3U, 30s, err)) << err;

  const std::lock_guard lock(SinkObservation::mutex);
  EXPECT_GE(SinkObservation::ticks, 3U);
  // rank-4 [1, C, H, W] because prepend_batch_dim=true and layout=nchw.
  ASSERT_EQ(SinkObservation::last_shape.size(), 4U);
  EXPECT_EQ(SinkObservation::last_shape[0], 1);     // N
  EXPECT_EQ(SinkObservation::last_shape[1], 3);     // C
  EXPECT_EQ(SinkObservation::last_shape[2], 128);   // H
  EXPECT_EQ(SinkObservation::last_shape[3], 128);   // W
  EXPECT_EQ(SinkObservation::last_dtype.code, kDLUInt);
  EXPECT_EQ(SinkObservation::last_dtype.bits, 8U);
}

TEST_F(VideoFileReaderRuntimeTest, Decodes_SyntheticMp4_AsRgbFp32Nhwc_WithDefaultDtype) {
  VideoFileReaderOpParams params;
  params.path = mp4_path_;
  params.tensor_port_name = "frame";
  params.output_pix_fmt = "rgb_fp32";
  params.output_layout = "nhwc";
  params.prepend_batch_dim = false;
  params.frame_count = 3;
  params.memory_kind = holoscan::MemoryKind::kCudaDevice;

  std::string err;
  ASSERT_TRUE(run_reader_pipeline(params, 3U, 30s, err)) << err;

  const std::lock_guard lock(SinkObservation::mutex);
  EXPECT_GE(SinkObservation::ticks, 3U);
  ASSERT_EQ(SinkObservation::last_shape.size(), 3U);
  EXPECT_EQ(SinkObservation::last_shape[0], 128);  // H
  EXPECT_EQ(SinkObservation::last_shape[1], 128);  // W
  EXPECT_EQ(SinkObservation::last_shape[2], 3);    // C (RGB)
  EXPECT_EQ(SinkObservation::last_dtype.code, kDLFloat);
  EXPECT_EQ(SinkObservation::last_dtype.bits, 32U);
}

TEST_F(VideoFileReaderRuntimeTest, Resizes_WhenOutputWidthHeightSet) {
  VideoFileReaderOpParams params;
  params.path = mp4_path_;
  params.output_width = 64;
  params.output_height = 48;
  params.output_pix_fmt = "rgb24";
  params.output_layout = "nhwc";
  params.prepend_batch_dim = false;
  params.frame_count = 2;
  params.memory_kind = holoscan::MemoryKind::kCudaDevice;

  std::string err;
  ASSERT_TRUE(run_reader_pipeline(params, 2U, 30s, err)) << err;

  const std::lock_guard lock(SinkObservation::mutex);
  ASSERT_EQ(SinkObservation::last_shape.size(), 3U);
  EXPECT_EQ(SinkObservation::last_shape[0], 48);  // H (user-requested)
  EXPECT_EQ(SinkObservation::last_shape[1], 64);  // W (user-requested)
}

}  // namespace holoscan::ops
