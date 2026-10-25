/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <dlpack/dlpack.h>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

// Forward declarations for libavcodec/libavformat/libswscale C types -- see the identical comment
// in video_file_reader.hpp.
extern "C" {
struct AVFormatContext;
struct AVCodecContext;
struct AVFrame;
struct AVPacket;
struct SwsContext;
struct AVStream;
}

namespace YAML {
class Node;
}  // namespace YAML

namespace holoscan::ops {

/**
 * @brief Compile-time-authored configuration for VideoFileWriterOp.
 *
 * Mirrors RawTensorWriterOpParams (see raw_tensor_writer.hpp) and is VideoFileReaderOpParams's
 * dual: instead of a flat `.raw` file, received Tensor frames are resampled via libswscale and
 * encoded into a compressed video container via libavcodec/libavformat.
 */
struct VideoFileWriterOpParams {
  /// Path to the output video file. Relative paths are resolved against the owning YAML config's
  /// directory by `from_yaml_node()`.
  std::filesystem::path path;
  /// Name of the single declared `Input<Tensor>` port. Must match the tensor name the upstream
  /// producer (typically `InferenceOp`) emits on its matching `Output<Tensor>` port.
  std::string tensor_port_name{"frame"};
  /// libavcodec encoder name: `libx264`, `libx265`, `mpeg4`, `mjpeg`, or `ffv1` (lossless).
  std::string codec{"libx264"};
  /// libavformat output format/muxer short name. Empty (the default) infers the muxer from
  /// `path`'s extension via `avformat_alloc_output_context2()`.
  std::string output_container;
  /// Pixel format the incoming tensor represents: same enum as
  /// `VideoFileReaderOpParams::output_pix_fmt` (`rgb24`/`bgr24`/`rgb_fp32`/`bgr_fp32`/
  /// `grayscale_u8`/`grayscale_fp32`).
  std::string input_pix_fmt{"rgb24"};
  /// Channel layout of the incoming tensor: `nchw` or `nhwc`.
  std::string input_layout{"nchw"};
  /// When `true`, strip a leading extent of `1` (a batch dimension) from the received Tensor's
  /// shape before treating the remainder as one image.
  bool strip_batch_dim{true};
  /// Required output frame width in pixels; also the expected tensor spatial width.
  int32_t width{};
  /// Required output frame height in pixels; also the expected tensor spatial height.
  int32_t height{};
  /// Encoder frame rate in frames per second.
  float fps{30.0F};
  /// Target encoder bitrate in bits per second.
  int64_t bitrate{4000000};
  /// Number of frames to write before compute() stops accepting input; `0` writes every received
  /// frame for the life of the run.
  int32_t frame_count{0};
  /// When set, compute() errors out (instead of writing) if a received Tensor's dtype differs.
  std::optional<DLDataType> expected_dtype;
  /// When set, compute() errors out (instead of writing) if a received Tensor's shape differs.
  std::optional<std::vector<int64_t>> expected_shape;

  /**
   * @brief Populate one `VideoFileWriterOpParams` from a `tensor_sinks:` list entry.
   *
   * ```yaml
   * tensor_sinks:
   *   - type: video_file                 # required discriminant (see holoinfer_example.cpp)
   *     path: ../data/output_video.mp4    # required; resolved relative to config_dir if relative
   *     tensor_port_name: tensor_out       # default: "frame"
   *     codec: libx264                     # default: "libx264"
   *     input_pix_fmt: rgb_fp32            # default: "rgb24"
   *     input_layout: nchw                 # default: "nchw"
   *     strip_batch_dim: true              # default: true
   *     width: 256                         # required
   *     height: 256                        # required
   *     fps: 30                            # default: 30
   *     bitrate: 4000000                   # default: 4000000 (4 Mbps)
   *     frame_count: 0                     # default: 0 (write until the run stops)
   * ```
   *
   * @param node One entry of a `tensor_sinks:` sequence.
   * @param config_dir Directory the owning YAML file was loaded from; used to resolve a relative
   *   `path`.
   * @return The populated `VideoFileWriterOpParams`.
   * @throws std::runtime_error if `path`/`width`/`height` is missing, or `input_pix_fmt`/
   *   `input_layout` names an unsupported value.
   */
  [[nodiscard]] static VideoFileWriterOpParams from_yaml_node(
      const YAML::Node& node, const std::filesystem::path& config_dir);
};

/**
 * @brief Receive tensor frames, resample via libswscale, and encode them into a compressed video
 * file via libavcodec/libavformat.
 *
 * The compressed-video dual of RawTensorWriterOp. Pairs with VideoFileReaderOp
 * (video_file_reader.hpp) for a real-video-in -> InferenceOp -> real-video-out pipeline (see
 * examples/holoinfer/README.md's config 10 walkthrough). Unlike the raw `.raw` round trip in
 * config 09, this path is not byte-exact: video codecs are lossy (or, even for a lossless codec
 * like `ffv1`, the container/timestamp bytes still differ run to run).
 */
class VideoFileWriterOp : public holoscan::Operator<VideoFileWriterOpParams> {
 public:
  VideoFileWriterOp() = default;

  /// @brief Construct with `params` already bound (mirrors `InferenceOp(InferenceOpParams)`).
  /// @throws std::runtime_error if `params` fails aggregate validation.
  explicit VideoFileWriterOp(VideoFileWriterOpParams params);

  ~VideoFileWriterOp() override;

  VideoFileWriterOp(const VideoFileWriterOp&) = delete;
  VideoFileWriterOp& operator=(const VideoFileWriterOp&) = delete;

  void setup(OperatorSpec& spec) override;
  void start() override;
  void stop() override;
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;

  [[nodiscard]] Contract contract() const override;

  /// @brief The single dynamically-named input port (declared in setup() as
  /// `params().tensor_port_name`).
  [[nodiscard]] Input<Tensor>* input_port() const { return &input_; }

 private:
  // Mutable so the const input_port() accessor (needed because graph.op<>() returns a
  // const-qualified OperatorHandle -- see RawTensorWriterOp::input_port() for the identical
  // pattern) can still hand out a mutable Input<Tensor>* for graph.add_flow() to bind at compile
  // time.
  mutable Input<Tensor> input_;

  AVFormatContext* format_ctx_{};
  AVCodecContext* codec_ctx_{};
  AVStream* stream_{};
  AVFrame*
      src_frame_{};  ///< HWC uint8 frame in the encoder's expected pixel format (post-swscale).
  AVPacket* packet_{};
  SwsContext* sws_ctx_{};
  bool header_written_{};

  std::vector<std::byte> host_buffer_;       ///< Raw copy_to_host() staging of the received Tensor.
  std::vector<std::uint8_t> hwc_u8_buffer_;  ///< HWC uint8, ready for sws_scale() as the src frame.

  int32_t frames_written_{};
  int64_t next_pts_{};

  /// @brief Send one frame to the encoder and drain every packet it returns via
  /// av_interleaved_write_frame(). Also used by stop() with `frame == nullptr` to flush.
  [[nodiscard]] expected<void, Error> send_and_drain(AVFrame* frame);
};

}  // namespace holoscan::ops
