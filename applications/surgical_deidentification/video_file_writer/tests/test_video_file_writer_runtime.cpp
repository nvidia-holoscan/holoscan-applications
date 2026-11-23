/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// End-to-end runtime smoke test for VideoFileWriterOp. Builds a two-op Fragment that pushes a
// handful of synthetic uint8 HWC frames into the writer, runs the graph, then probes the output
// MP4 with libavformat to verify width/height/encoded-frame-count match configuration.

#include <cuda_runtime_api.h>
#include <dlpack/dlpack.h>
#include <gtest/gtest.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

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

using std::chrono_literals::operator""ms;
using std::chrono_literals::operator""s;

constexpr std::chrono::nanoseconds kFeedPeriod = 10ms;
constexpr std::chrono::seconds kRunTimeout = 30s;
constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};

/// @brief Emit a fixed set of uint8 HWC frames per tick, filled with a known gradient value so
/// the encoder has non-degenerate content to compress.
class FrameFeeder final : public holoscan::Operator<> {
 public:
  FrameFeeder(std::string port_name, int32_t height, int32_t width, int32_t frames,
              DLDataType dtype = kUInt8, float float_value = 0.0F)
      : port_name_(std::move(port_name)),
        height_(height),
        width_(width),
        frames_(frames),
        dtype_(dtype),
        float_pixels_(dtype.code == kDLFloat ? static_cast<std::size_t>(height) * width * 3U : 0U,
                      float_value) {}

  void setup(holoscan::OperatorSpec& spec) override {
    const std::size_t byte_span = static_cast<std::size_t>(height_) *
                                  static_cast<std::size_t>(width_) * 3U * (dtype_.bits / 8U) *
                                  dtype_.lanes;
    spec.output(output_, port_name_)
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation =
                holoscan::TensorRepresentation{
                    .memory_kind = holoscan::MemoryKind::kCudaDevice,
                    .dtype = dtype_,
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
        output_.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = dtype_});
    if (!loan) {
      return holoscan::make_unexpected(std::move(loan).error());
    }
    auto writer = loan->write(context.cuda_stream());
    if (!writer) {
      return holoscan::make_unexpected(std::move(writer).error());
    }
    // Fill with (sent_ * 11) mod 256 so each frame is distinguishable by codec stats.
    // Float regression inputs include non-finite values that memset cannot represent.
    const std::uint8_t fill =
        dtype_.bits == 8U ? static_cast<std::uint8_t>((sent_ * 11) & 0xFF) : 0U;
    const auto status = dtype_.code == kDLFloat
                            ? cudaMemcpyAsync(writer->data(), float_pixels_.data(),
                                              writer->byte_size(), cudaMemcpyHostToDevice, stream)
                            : cudaMemsetAsync(writer->data(), fill, writer->byte_size(), stream);
    if (status != cudaSuccess) {
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
  DLDataType dtype_{};
  std::vector<float> float_pixels_;
  holoscan::Output<holoscan::Tensor> output_;
};

/// @brief Open the given MP4 with libavformat and return (width, height, encoded_frame_count).
struct ProbeResult {
  bool opened{false};
  int width{};
  int height{};
  int frame_count{};
  int first_luma{-1};
};

[[nodiscard]] ProbeResult probe_mp4(const std::string& path, bool decode_first_frame = false) {
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
  AVCodecContext* decoder = nullptr;
  AVFrame* decoded_frame = nullptr;
  if (decode_first_frame) {
    const AVCodec* codec = avcodec_find_decoder(par->codec_id);
    decoder = avcodec_alloc_context3(codec);
    if (decoder != nullptr && avcodec_parameters_to_context(decoder, par) >= 0 &&
        avcodec_open2(decoder, codec, nullptr) >= 0) {
      decoded_frame = av_frame_alloc();
    }
  }
  // Count packets on the video stream (one packet ≈ one frame for the codecs we exercise).
  AVPacket* pkt = av_packet_alloc();
  while (av_read_frame(ctx, pkt) >= 0) {
    if (pkt->stream_index == video_idx) {
      ++r.frame_count;
      if (decoded_frame != nullptr && r.first_luma < 0 && avcodec_send_packet(decoder, pkt) >= 0 &&
          avcodec_receive_frame(decoder, decoded_frame) >= 0 &&
          decoded_frame->format == AV_PIX_FMT_YUV420P) {
        r.first_luma = decoded_frame->data[0][0];
      }
    }
    av_packet_unref(pkt);
  }
  // Flush decoder delay for a one-frame clip.
  if (decoded_frame != nullptr && r.first_luma < 0 && avcodec_send_packet(decoder, nullptr) >= 0 &&
      avcodec_receive_frame(decoder, decoded_frame) >= 0 &&
      decoded_frame->format == AV_PIX_FMT_YUV420P) {
    r.first_luma = decoded_frame->data[0][0];
  }
  av_packet_free(&pkt);
  av_frame_free(&decoded_frame);
  avcodec_free_context(&decoder);
  avformat_close_input(&ctx);
  return r;
}

struct WriterRunResult {
  holoscan::RunTermination termination{};
  std::vector<holoscan::Diagnostic> diagnostics;
  bool finalize_failed{};
};

[[nodiscard]] bool run_writer_pipeline(VideoFileWriterOpParams params, int32_t frames_to_send,
                                       std::string& err_message, DLDataType dtype = kUInt8,
                                       WriterRunResult* result = nullptr,
                                       float float_value = 0.0F) {
  try {
    holoscan::Graph graph{"video-file-writer-runtime-test"};
    const auto writer = graph.op<VideoFileWriterOp>("writer", params);
    const auto feeder = graph.op<FrameFeeder>("feeder", params.tensor_port_name, params.height,
                                              params.width, frames_to_send, dtype, float_value);
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
    if (result != nullptr) {
      result->termination = session.termination();
      result->diagnostics.assign(session.diagnostics().begin(), session.diagnostics().end());
      result->finalize_failed = writer->finalize_failed();
    } else if (!session.diagnostics().empty() ||
               session.termination() == holoscan::RunTermination::kFailed ||
               writer->finalize_failed()) {
      err_message = "writer pipeline reported a runtime or finalization error";
      return false;
    }
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

// Writes `frame_count` frames with the given codec/container/size and checks that the output
// opens with the configured dimensions. Covers encoder startup (MJPEG needs full-range YUV) and
// packed source rows whose width is not a multiple of the swscale SIMD block.
static void expect_writes_video(const std::string& codec, const std::string& container,
                                const std::string& file_name, int32_t width, int32_t height) {
  const auto tmp =
      std::filesystem::temp_directory_path() / ("video_file_writer_runtime_test_" + codec);
  std::filesystem::remove_all(tmp);
  std::filesystem::create_directories(tmp);
  const auto path = tmp / file_name;

  VideoFileWriterOpParams params;
  params.path = path;
  params.tensor_port_name = "frame";
  params.codec = codec;
  params.output_container = container;
  params.input_pix_fmt = "rgb24";
  params.input_layout = "nhwc";
  params.strip_batch_dim = false;
  params.width = width;
  params.height = height;
  params.fps = 30.0F;
  params.bitrate = 1'000'000;
  params.frame_count = 5;

  std::string err;
  ASSERT_TRUE(run_writer_pipeline(params, params.frame_count, err)) << err;
  ASSERT_TRUE(std::filesystem::exists(path)) << "writer never produced " << path;
  EXPECT_GT(std::filesystem::file_size(path), 0U);

  const ProbeResult probe = probe_mp4(path.string());
  EXPECT_TRUE(probe.opened);
  EXPECT_EQ(probe.width, width);
  EXPECT_EQ(probe.height, height);
  EXPECT_GT(probe.frame_count, 0);

  std::filesystem::remove_all(tmp);
}

TEST(VideoFileWriterRuntime, WritesMjpegWithFullRangeYuv) {
  expect_writes_video("mjpeg", "avi", "out.avi", 128, 128);
}

TEST(VideoFileWriterRuntime, WritesWidthNotAlignedToSimdBlock) {
  expect_writes_video("mpeg4", "mp4", "out.mp4", 1912, 16);
}

TEST(VideoFileWriterRuntime, RejectsInt32TensorForRgbFp32) {
  const auto tmp = std::filesystem::temp_directory_path() / "video_file_writer_dtype_rejection";
  std::filesystem::create_directories(tmp);
  constexpr DLDataType kInt32{kDLInt, 32U, 1U};

  // The inferred pixel dtype must hold whether expected_dtype is omitted or explicitly agrees
  // with the incompatible source. Shape and byte count deliberately match a float32 frame.
  for (const bool set_expected_dtype : {false, true}) {
    SCOPED_TRACE(set_expected_dtype);
    VideoFileWriterOpParams params;
    params.path = tmp / "out.mp4";
    params.input_pix_fmt = "rgb_fp32";
    params.input_layout = "nhwc";
    params.width = 128;
    params.height = 128;
    if (set_expected_dtype) {
      params.expected_dtype = kInt32;
    }
    WriterRunResult result;
    std::string err;
    ASSERT_TRUE(run_writer_pipeline(params, 1, err, kInt32, &result)) << err;
    EXPECT_EQ(result.termination, holoscan::RunTermination::kFailed);
    EXPECT_TRUE(std::any_of(result.diagnostics.begin(), result.diagnostics.end(),
                            [](const holoscan::Diagnostic& diagnostic) {
                              return diagnostic.severity == holoscan::DiagnosticSeverity::kError &&
                                     diagnostic.detail &&
                                     diagnostic.detail->operator_path == "writer" &&
                                     diagnostic.detail->error.code ==
                                         holoscan::ErrorCode::kInvalidArgument;
                            }));
  }
  std::filesystem::remove_all(tmp);
}

TEST(VideoFileWriterRuntime, WritesFloat32TensorForRgbFp32) {
  const auto tmp = std::filesystem::temp_directory_path() / "video_file_writer_fp32";
  std::filesystem::create_directories(tmp);
  VideoFileWriterOpParams params;
  params.path = tmp / "out.mp4";
  params.input_pix_fmt = "rgb_fp32";
  params.input_layout = "nhwc";
  params.width = 128;
  params.height = 128;

  WriterRunResult result;
  std::string err;
  ASSERT_TRUE(run_writer_pipeline(params, 3, err, DLDataType{kDLFloat, 32U, 1U}, &result)) << err;
  EXPECT_EQ(result.termination, holoscan::RunTermination::kStopped);
  EXPECT_TRUE(result.diagnostics.empty());
  EXPECT_FALSE(result.finalize_failed);
  const ProbeResult probe = probe_mp4(params.path.string());
  ASSERT_TRUE(probe.opened);
  EXPECT_EQ(probe.width, 128);
  EXPECT_EQ(probe.height, 128);
  EXPECT_EQ(probe.frame_count, 3);
  std::filesystem::remove_all(tmp);
}

// Run this case with -fsanitize=float-cast-overflow to catch a NaN-to-integer conversion even
// on architectures where the unsanitized conversion happens to yield zero.
TEST(VideoFileWriterRuntime, WritesNaNFloat32TensorAsDecodableBlackFrame) {
  const auto tmp = std::filesystem::temp_directory_path() / "video_file_writer_nan";
  std::filesystem::create_directories(tmp);
  VideoFileWriterOpParams params;
  params.path = tmp / "out.mp4";
  params.input_pix_fmt = "rgb_fp32";
  params.input_layout = "nhwc";
  params.width = 128;
  params.height = 128;

  std::string err;
  ASSERT_TRUE(run_writer_pipeline(params, 1, err, DLDataType{kDLFloat, 32U, 1U}, nullptr,
                                  std::numeric_limits<float>::quiet_NaN()))
      << err;
  const ProbeResult probe = probe_mp4(params.path.string(), true);
  ASSERT_TRUE(probe.opened);
  EXPECT_EQ(probe.frame_count, 1);
  EXPECT_EQ(probe.first_luma, 16);  // Black in the encoder's limited-range YUV420P format.
  std::filesystem::remove_all(tmp);
}

}  // namespace holoscan::ops
