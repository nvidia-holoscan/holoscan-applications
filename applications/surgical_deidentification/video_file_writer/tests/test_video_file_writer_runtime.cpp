/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// End-to-end runtime smoke test for VideoFileWriterOp. Builds a two-op Fragment that pushes a
// handful of synthetic uint8 HWC frames into the writer, runs the graph, then probes the output
// MP4 with libavformat to verify width/height/encoded-frame-count match configuration.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <dlpack/dlpack.h>
#include <gtest/gtest.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

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
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "video_file_writer.hpp"

namespace holoscan::ops {
namespace {

using namespace std::chrono_literals;

constexpr std::chrono::nanoseconds kFeedPeriod = 10ms;
constexpr std::chrono::seconds kRunTimeout = 30s;
constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};

/// @brief Emit a fixed set of uint8 HWC frames per tick, filled with a known gradient value so
/// the encoder has non-degenerate content to compress.
class FrameFeeder final : public holoscan::Operator<> {
 public:
  FrameFeeder(std::string port_name, int32_t height, int32_t width, int32_t frames)
      : port_name_(std::move(port_name)), height_(height), width_(width), frames_(frames) {}

  void setup(holoscan::OperatorSpec& spec) override {
    const std::size_t byte_span =
        static_cast<std::size_t>(height_) * static_cast<std::size_t>(width_) * 3U;
    spec.output(output_, port_name_)
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation =
                holoscan::TensorRepresentation{
                    .memory_kind = holoscan::MemoryKind::kCudaDevice,
                    .dtype = kUInt8,
                    .rank = 3U,
                },
            .bounds = holoscan::TensorBounds{.max_rank = 3U, .max_byte_span = byte_span},
            .storage = holoscan::TensorOutputStorage::kRuntimePool,
        });
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kFeedPeriod});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (sent_ >= frames_) {
      return {};
    }
    const cudaStream_t stream = context.cuda_stream().get();
    if (stream == nullptr) {
      return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kNotSupported});
    }
    const std::vector<int64_t> shape{height_, width_, 3};
    auto loan =
        output_.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = kUInt8});
    if (!loan) {
      return holoscan::make_unexpected(std::move(loan).error());
    }
    auto writer = loan->write(context.cuda_stream());
    if (!writer) {
      return holoscan::make_unexpected(std::move(writer).error());
    }
    // Fill with (sent_ * 11) mod 256 so each frame is distinguishable by codec stats.
    const std::uint8_t fill = static_cast<std::uint8_t>((sent_ * 11) & 0xFF);
    if (cudaMemsetAsync(writer->data(), fill, writer->byte_size(), stream) != cudaSuccess) {
      return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kFailure});
    }
    auto committed = std::move(*writer).commit();
    if (!committed) {
      return holoscan::make_unexpected(std::move(committed).error());
    }
    ++sent_;
    return output_.emit(std::move(*loan),
                        holoscan::EmitOptions{.capture_time = context.activation_time()});
  }

  [[nodiscard]] holoscan::Output<holoscan::Tensor>* output_port() const {
    return const_cast<holoscan::Output<holoscan::Tensor>*>(&output_);
  }

 private:
  std::string port_name_;
  int32_t height_{};
  int32_t width_{};
  int32_t frames_{};
  int32_t sent_{};
  holoscan::Output<holoscan::Tensor> output_;
};

/// @brief Open the given MP4 with libavformat and return (width, height, encoded_frame_count).
struct ProbeResult {
  bool opened{false};
  int width{};
  int height{};
  int frame_count{};
};

[[nodiscard]] ProbeResult probe_mp4(const std::string& path) {
  ProbeResult r;
  AVFormatContext* ctx = nullptr;
  if (avformat_open_input(&ctx, path.c_str(), nullptr, nullptr) < 0) {
    return r;
  }
  r.opened = true;
  if (avformat_find_stream_info(ctx, nullptr) < 0) {
    avformat_close_input(&ctx);
    return r;
  }
  int video_idx = -1;
  for (unsigned i = 0; i < ctx->nb_streams; ++i) {
    if (ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
      video_idx = static_cast<int>(i);
      break;
    }
  }
  if (video_idx < 0) {
    avformat_close_input(&ctx);
    return r;
  }
  const auto* par = ctx->streams[video_idx]->codecpar;
  r.width = par->width;
  r.height = par->height;
  // Count packets on the video stream (one packet ≈ one frame for the codecs we exercise).
  AVPacket* pkt = av_packet_alloc();
  while (av_read_frame(ctx, pkt) >= 0) {
    if (pkt->stream_index == video_idx) {
      ++r.frame_count;
    }
    av_packet_unref(pkt);
  }
  av_packet_free(&pkt);
  avformat_close_input(&ctx);
  return r;
}

[[nodiscard]] bool run_writer_pipeline(VideoFileWriterOpParams params, int32_t frames_to_send,
                                       std::string& err_message) {
  try {
    holoscan::Graph graph{"video-file-writer-runtime-test"};
    const auto writer = graph.op<VideoFileWriterOp>("writer", params);
    const auto feeder = graph.op<FrameFeeder>("feeder", params.tensor_port_name, params.height,
                                              params.width, frames_to_send);
    graph.add_flow(*feeder->output_port(), *writer->input_port(),
                   holoscan::ConnectionOptions{.queue_depth = 4U});
    graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

    holoscan::CompileOptions options;
    options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = "feeder",
        .output_port = params.tensor_port_name,
        .device = holoscan::DeviceId{0},
    });

    const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
    if (!plan.ok()) {
      err_message = "compile failed: " + std::string(plan.json());
      return false;
    }
    holoscan::RunSession session = holoscan::run_async(plan);
    // Pace: kFeedPeriod per frame + 1s warmup + 1s flush headroom. The writer flushes the
    // encoder on stop() so the trailer gets written.
    const auto run_ms = 1000 + frames_to_send * 20 + 1000;
    std::this_thread::sleep_for(std::chrono::milliseconds(run_ms));
    session.request_stop();
    session.wait();
    return true;
  } catch (const std::exception& e) {
    err_message = std::string("exception: ") + e.what();
    return false;
  }
}

}  // namespace

TEST(VideoFileWriterRuntime, WritesMp4WithExpectedDimensionsAndFrameCount) {
  const auto tmp = std::filesystem::temp_directory_path() / "video_file_writer_runtime_test";
  std::filesystem::create_directories(tmp);
  const auto path = tmp / "out.mp4";
  if (std::filesystem::exists(path)) {
    std::filesystem::remove(path);
  }

  VideoFileWriterOpParams params;
  params.path = path;
  params.tensor_port_name = "frame";
  params.codec = "mpeg4";  // mpeg4 is in every FFmpeg build; libx264 is optional.
  params.input_pix_fmt = "rgb24";
  params.input_layout = "nhwc";
  params.strip_batch_dim = false;
  params.width = 128;
  params.height = 128;
  params.fps = 25.0F;
  params.bitrate = 1'000'000;
  params.frame_count = 10;

  std::string err;
  ASSERT_TRUE(run_writer_pipeline(params, params.frame_count, err)) << err;

  ASSERT_TRUE(std::filesystem::exists(path)) << "writer never produced " << path;
  EXPECT_GT(std::filesystem::file_size(path), 0U);

  const ProbeResult probe = probe_mp4(path.string());
  EXPECT_TRUE(probe.opened);
  EXPECT_EQ(probe.width, 128);
  EXPECT_EQ(probe.height, 128);
  // Codec may emit a startup prefix with fewer-than-N packets if the stop signal races the
  // encoder flush; allow +/- 2 packets around the configured count.
  EXPECT_GE(probe.frame_count, params.frame_count - 2);
  EXPECT_LE(probe.frame_count, params.frame_count + 2);

  std::filesystem::remove_all(tmp);
}

}  // namespace holoscan::ops
