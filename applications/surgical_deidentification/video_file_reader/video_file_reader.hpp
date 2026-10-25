/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
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

// Forward declarations for libavcodec/libavformat/libswscale C types, so this public header does
// not force every downstream consumer of holoscan::core to pull in <libavformat/avformat.h> etc.
// (mirrors how raw_tensor_reader.hpp forward-declares YAML::Node instead of including
// <yaml-cpp/yaml.h>). The full definitions are only needed by video_file_reader.cpp.
extern "C" {
struct AVFormatContext;
struct AVCodecContext;
struct AVFrame;
struct AVPacket;
struct SwsContext;
}

namespace YAML {
class Node;
}  // namespace YAML

namespace holoscan::ops {

/**
 * @brief Compile-time-authored configuration for VideoFileReaderOp.
 *
 * Mirrors RawTensorReaderOpParams (see raw_tensor_reader.hpp), but the underlying source is a
 * compressed video container (`.mp4`, `.avi`, `.mkv`, ...; anything libavformat can demux and
 * libavcodec can decode) decoded frame-by-frame via libavcodec/libavformat, resampled to a target
 * pixel format/resolution via libswscale, and emitted as a Tensor. Because libavformat's demuxer
 * accepts any URL its registered protocols understand, `path` may also be an `rtsp://` or
 * `http(s)://` live-stream URL with no code change (see examples/holoinfer/README.md's "Live
 * sources" note) -- naming it `path` (not `url`) keeps the YAML surface consistent with
 * RawTensorReaderOpParams.
 */
struct VideoFileReaderOpParams {
  /// Path (or libavformat-supported URL, e.g. `rtsp://...`) to the input video. Relative
  /// filesystem paths are resolved against the owning YAML config's directory by
  /// `from_yaml_node()`; anything containing a `://` scheme is left untouched.
  std::filesystem::path path;
  /// Name of the single declared `Output<Tensor>` port. Must match the tensor name the downstream
  /// consumer (typically `InferenceOp`) expects on its matching `Input<Tensor>` port.
  std::string tensor_port_name{"frame"};
  /// Optional output width in pixels. When unset, the decoded stream's native width is used.
  std::optional<int32_t> output_width;
  /// Optional output height in pixels. When unset, the decoded stream's native height is used.
  std::optional<int32_t> output_height;
  /// Pixel format of the emitted Tensor: one of `rgb24`/`bgr24` (interleaved uint8),
  /// `rgb_fp32`/`bgr_fp32` (interleaved float, normalized to `[0, 1]`), `grayscale_u8`, or
  /// `grayscale_fp32`.
  std::string output_pix_fmt{"rgb24"};
  /// Channel layout of the emitted Tensor: `nchw` (channel-first) or `nhwc` (channel-last).
  std::string output_layout{"nchw"};
  /// When `true`, prepend a leading extent of `1` to the emitted Tensor's shape (a batch
  /// dimension), matching InferenceOp's typical `[1, C, H, W]` / `[1, H, W, C]` expectation.
  bool prepend_batch_dim{true};
  /// DLPack element type of the emitted Tensor. Derived from `output_pix_fmt` by
  /// `from_yaml_node()` (uint8 for `*_u8`/`rgb24`/`bgr24`, float32 for `*_fp32`). Callers
  /// constructing the params in code may set this field directly; there is no `output_dtype:`
  /// YAML key.
  DLDataType output_dtype{kDLUInt, 8U, 1U};
  /// Memory kind of the emitted Tensor. `kCudaDevice` stages the frame through a host buffer and
  /// copies it to device memory; `kHost` (or a pinned/managed variant) copies it directly.
  holoscan::MemoryKind memory_kind{holoscan::MemoryKind::kCudaDevice};
  /// Number of frames to emit before compute() stops emitting; `0` decodes until end-of-stream
  /// (or forever, if `loop` is set).
  int32_t frame_count{0};
  /// Target emission rate in Hz. `0` (the default) emits as fast as decoding allows; a positive
  /// value paces emission to approximately `1 / frame_rate` seconds between frames.
  float frame_rate{0.0F};
  /// When the stream is exhausted, seek back to the first frame and keep emitting instead of
  /// stopping. Not meaningful (and not attempted) for a non-seekable live source.
  bool loop{false};
  /// Index of the video stream to decode within the container. `-1` (the default) selects the
  /// first video stream libavformat reports.
  int32_t stream_index{-1};

  /**
   * @brief Populate one `VideoFileReaderOpParams` from a `tensor_sources:` list entry.
   *
   * ```yaml
   * tensor_sources:
   *   - type: video_file                 # required discriminant (see holoinfer_example.cpp)
   *     path: ../data/sample_video.mp4    # required; resolved relative to config_dir if relative
   *     tensor_port_name: tensor_in        # default: "frame"
   *     output_pix_fmt: rgb_fp32           # default: "rgb24"
   *     output_layout: nchw                # default: "nchw"
   *     output_width: 256                  # default: unset (native decoded width)
   *     output_height: 256                 # default: unset (native decoded height)
   *     prepend_batch_dim: true            # default: true
   *     memory_kind: cuda_device           # default: cuda_device
   *     frame_count: 0                     # default: 0 (decode until end of stream)
   *     frame_rate: 0                      # default: 0 (as fast as possible)
   *     loop: false                        # default: false
   *     stream_index: -1                   # default: -1 (first video stream)
   * ```
   *
   * @param node One entry of a `tensor_sources:` sequence.
   * @param config_dir Directory the owning YAML file was loaded from; used to resolve a relative
   *   `path`.
   * @return The populated `VideoFileReaderOpParams`.
   * @throws std::runtime_error if `path` is missing, or `output_pix_fmt`/`output_layout` names an
   *   unsupported value.
   */
  [[nodiscard]] static VideoFileReaderOpParams from_yaml_node(
      const YAML::Node& node, const std::filesystem::path& config_dir);
};

/**
 * @brief Decode frames from a compressed video file (or live stream URL) via
 * libavformat/libavcodec, resample via libswscale, and emit each as a Tensor.
 *
 * The compressed-video dual of RawTensorReaderOp: instead of a flat, headerless `.raw` file of
 * pre-shaped frames, the source is any container/codec libavformat/libavcodec can open (MP4, AVI,
 * MKV, MPEG-TS, or a live RTSP/HTTP URL). One instance decodes one video stream; a multi-input
 * model needs one instance per `tensor_sources:` entry (see holoinfer_example.cpp's driver-side
 * wiring, which dispatches on each entry's `type:` key).
 */
class VideoFileReaderOp : public holoscan::Operator<VideoFileReaderOpParams> {
 public:
  VideoFileReaderOp() = default;

  /// @brief Construct with `params` already bound (mirrors `InferenceOp(InferenceOpParams)`).
  /// @throws std::runtime_error if `params` fails aggregate validation.
  explicit VideoFileReaderOp(VideoFileReaderOpParams params);

  ~VideoFileReaderOp() override;

  VideoFileReaderOp(const VideoFileReaderOp&) = delete;
  VideoFileReaderOp& operator=(const VideoFileReaderOp&) = delete;

  void setup(OperatorSpec& spec) override;
  void start() override;
  void stop() override;
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;

  /// @brief Poll frequently; actual pacing (when `frame_rate > 0`) is enforced inside compute().
  [[nodiscard]] Contract contract() const override;

  /// @brief The single dynamically-named output port (declared in setup() as
  /// `params().tensor_port_name`).
  [[nodiscard]] Output<Tensor>* output_port() const { return &output_; }

 private:
  // Mutable so the const output_port() accessor (needed because graph.op<>() returns a
  // const-qualified OperatorHandle -- see RawTensorReaderOp::output_port() for the identical
  // pattern) can still hand out a mutable Output<Tensor>* for graph.add_flow() to bind at compile
  // time.
  mutable Output<Tensor> output_;

  // Resolved output geometry: either params().output_width/output_height verbatim, or the source
  // stream's native resolution, probed by setup() (a lightweight avformat_open_input() +
  // avformat_find_stream_info() + avformat_close_input() with no decode) so the declared
  // Output<Tensor> port's TensorAllocationBounds is known before start() opens the stream for
  // real.
  int32_t resolved_width_{};
  int32_t resolved_height_{};
  int32_t channels_{};
  std::size_t frame_bytes_{};
  std::uint32_t rank_{};
  std::vector<int64_t> shape_;
  DLDataType effective_dtype_{};

  // libav* handles, opened in start() and released in stop()/the destructor. Forward-declared C
  // types (see the top of this header) require these to stay behind pointers.
  AVFormatContext* format_ctx_{};
  AVCodecContext* codec_ctx_{};
  AVFrame* decoded_frame_{};
  AVFrame* scaled_frame_{};
  AVPacket* packet_{};
  SwsContext* sws_ctx_{};
  int video_stream_index_{-1};

  std::vector<std::uint8_t> scaled_buffer_;  ///< Backing store for scaled_frame_'s HWC uint8 data.
  std::vector<std::byte> host_buffer_;       ///< Final (possibly transposed/normalized) frame.

  int32_t frames_emitted_{};
  bool eos_reached_{};
  std::chrono::steady_clock::time_point next_emit_time_{};
  bool have_next_emit_time_{};

  /// @brief Open format_ctx_/codec_ctx_/sws_ctx_ and (re)allocate frame_bytes_-sized buffers.
  /// Shared by start() and by the loop-triggered re-open after end-of-stream.
  /// @throws std::runtime_error on any libav* failure.
  void open_stream();
  /// @brief Release format_ctx_/codec_ctx_/sws_ctx_/decoded_frame_/scaled_frame_/packet_.
  void close_stream();
  /// @brief Read and decode exactly one frame into decoded_frame_.
  /// @return `true` on success, `false` on clean end-of-stream, an error on any other failure.
  [[nodiscard]] expected<bool, Error> decode_one_frame();
};

}  // namespace holoscan::ops
