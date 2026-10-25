/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "video_file_reader.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

#include <cuda_runtime_api.h>

#include <holoscan/core/graph.hpp>
#include <yaml-cpp/yaml.h>

namespace holoscan::ops {

namespace {

[[nodiscard]] holoscan::MemoryKind memory_kind_from_string(const std::string& name) {
  if (name == "cuda_device") return holoscan::MemoryKind::kCudaDevice;
  if (name == "host") return holoscan::MemoryKind::kHost;
  if (name == "pinned_host") return holoscan::MemoryKind::kPinnedHost;
  if (name == "cuda_managed") return holoscan::MemoryKind::kCudaManaged;
  throw std::runtime_error("unsupported memory_kind '" + name +
                           "' (expected one of cuda_device/host/pinned_host/cuda_managed)");
}

/// @brief Everything derived from `output_pix_fmt`: the libswscale destination format actually
/// requested (always an interleaved uint8 format -- fp32 variants normalize on the host side
/// after sws_scale, since libswscale's floating-point packed formats are not universally
/// available across libswscale builds), channel count, and DLPack dtype.
struct PixFmtInfo {
  AVPixelFormat sws_dst_fmt;
  int32_t channels;
  DLDataType dtype;
  bool is_float;
};

[[nodiscard]] PixFmtInfo pix_fmt_info(const std::string& name) {
  if (name == "rgb24") return {AV_PIX_FMT_RGB24, 3, DLDataType{kDLUInt, 8U, 1U}, false};
  if (name == "bgr24") return {AV_PIX_FMT_BGR24, 3, DLDataType{kDLUInt, 8U, 1U}, false};
  if (name == "rgb_fp32") return {AV_PIX_FMT_RGB24, 3, DLDataType{kDLFloat, 32U, 1U}, true};
  if (name == "bgr_fp32") return {AV_PIX_FMT_BGR24, 3, DLDataType{kDLFloat, 32U, 1U}, true};
  if (name == "grayscale_u8") return {AV_PIX_FMT_GRAY8, 1, DLDataType{kDLUInt, 8U, 1U}, false};
  if (name == "grayscale_fp32") return {AV_PIX_FMT_GRAY8, 1, DLDataType{kDLFloat, 32U, 1U}, true};
  throw std::runtime_error("unsupported output_pix_fmt '" + name +
                           "' (expected one of rgb24/bgr24/rgb_fp32/bgr_fp32/grayscale_u8/"
                           "grayscale_fp32)");
}

[[nodiscard]] bool is_nchw(const std::string& layout) {
  if (layout == "nchw") return true;
  if (layout == "nhwc") return false;
  throw std::runtime_error("unsupported output_layout '" + layout + "' (expected nchw or nhwc)");
}

[[nodiscard]] bool looks_like_url(const std::filesystem::path& path) {
  const std::string s = path.string();
  return s.find("://") != std::string::npos;
}

/// @brief Lightweight probe: open just enough of `path` to learn the selected video stream's
/// native width/height, then close it. No decoder is opened and no frame is decoded. Used by
/// VideoFileReaderOp::setup() to size the declared Output<Tensor> port's TensorAllocationBounds
/// before output_width/output_height (if unset) can be known any other way, since setup() runs
/// before start() opens the stream for real.
struct ProbeResult {
  int32_t width;
  int32_t height;
};

[[nodiscard]] ProbeResult probe_native_resolution(const std::filesystem::path& path,
                                                  int32_t stream_index) {
  AVFormatContext* format_ctx = nullptr;
  const std::string url = path.string();
  if (avformat_open_input(&format_ctx, url.c_str(), nullptr, nullptr) < 0) {
    throw std::runtime_error("VideoFileReaderOp: could not open '" + url + "'");
  }
  if (avformat_find_stream_info(format_ctx, nullptr) < 0) {
    avformat_close_input(&format_ctx);
    throw std::runtime_error("VideoFileReaderOp: could not read stream info from '" + url + "'");
  }

  int selected = stream_index;
  if (selected < 0) {
    selected = av_find_best_stream(format_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
  }
  if (selected < 0 || static_cast<unsigned>(selected) >= format_ctx->nb_streams ||
      format_ctx->streams[selected]->codecpar->codec_type != AVMEDIA_TYPE_VIDEO) {
    avformat_close_input(&format_ctx);
    throw std::runtime_error("VideoFileReaderOp: '" + url + "' has no usable video stream");
  }

  const AVCodecParameters* params = format_ctx->streams[selected]->codecpar;
  const ProbeResult result{.width = params->width, .height = params->height};
  avformat_close_input(&format_ctx);
  if (result.width <= 0 || result.height <= 0) {
    throw std::runtime_error("VideoFileReaderOp: '" + url +
                             "' reports an invalid native "
                             "resolution");
  }
  return result;
}

}  // namespace

VideoFileReaderOpParams VideoFileReaderOpParams::from_yaml_node(
    const YAML::Node& node, const std::filesystem::path& config_dir) {
  if (!node["path"]) {
    throw std::runtime_error(
        "VideoFileReaderOpParams::from_yaml_node: missing required key 'path'");
  }

  VideoFileReaderOpParams params;

  std::filesystem::path path = node["path"].as<std::string>();
  if (!looks_like_url(path) && path.is_relative() && !config_dir.empty()) {
    path = (config_dir / path).lexically_normal();
  }
  params.path = std::move(path);

  if (const auto name_node = node["tensor_port_name"]) {
    params.tensor_port_name = name_node.as<std::string>();
  }
  if (const auto width_node = node["output_width"]) {
    params.output_width = width_node.as<int32_t>();
  }
  if (const auto height_node = node["output_height"]) {
    params.output_height = height_node.as<int32_t>();
  }
  if (const auto pix_fmt_node = node["output_pix_fmt"]) {
    params.output_pix_fmt = pix_fmt_node.as<std::string>();
  }
  if (const auto layout_node = node["output_layout"]) {
    params.output_layout = layout_node.as<std::string>();
  }
  if (const auto batch_node = node["prepend_batch_dim"]) {
    params.prepend_batch_dim = batch_node.as<bool>();
  }
  // output_dtype is derived from output_pix_fmt below; validate output_pix_fmt/output_layout now
  // so from_yaml_node() fails loudly on a bad value instead of deferring to setup()/start().
  params.output_dtype = pix_fmt_info(params.output_pix_fmt).dtype;
  static_cast<void>(is_nchw(params.output_layout));

  if (const auto memory_kind_node = node["memory_kind"]) {
    params.memory_kind = memory_kind_from_string(memory_kind_node.as<std::string>());
  }
  if (const auto frame_count_node = node["frame_count"]) {
    params.frame_count = frame_count_node.as<int32_t>();
  }
  if (const auto frame_rate_node = node["frame_rate"]) {
    params.frame_rate = frame_rate_node.as<float>();
  }
  if (const auto loop_node = node["loop"]) {
    params.loop = loop_node.as<bool>();
  }
  if (const auto stream_index_node = node["stream_index"]) {
    params.stream_index = stream_index_node.as<int32_t>();
  }

  return params;
}

VideoFileReaderOp::VideoFileReaderOp(VideoFileReaderOpParams params) {
  auto bound = holoscan::detail::GraphOperatorAccess::bind(*this, std::move(params));
  if (!bound) {
    throw std::runtime_error("VideoFileReaderOp: invalid VideoFileReaderOpParams: " +
                             bound.error().message);
  }
}

VideoFileReaderOp::~VideoFileReaderOp() { close_stream(); }

void VideoFileReaderOp::setup(OperatorSpec& spec) {
  const VideoFileReaderOpParams& config = params();
  const PixFmtInfo info = pix_fmt_info(config.output_pix_fmt);
  const bool nchw = is_nchw(config.output_layout);
  channels_ = info.channels;
  effective_dtype_ = info.dtype;

  if (config.output_width && config.output_height) {
    resolved_width_ = *config.output_width;
    resolved_height_ = *config.output_height;
  } else {
    // Native-resolution probe (no decode) -- see probe_native_resolution()'s comment.
    const ProbeResult probe = probe_native_resolution(config.path, config.stream_index);
    resolved_width_ = config.output_width.value_or(probe.width);
    resolved_height_ = config.output_height.value_or(probe.height);
  }

  shape_.clear();
  if (config.prepend_batch_dim) {
    shape_.push_back(1);
  }
  if (nchw) {
    shape_.push_back(channels_);
    shape_.push_back(resolved_height_);
    shape_.push_back(resolved_width_);
  } else {
    shape_.push_back(resolved_height_);
    shape_.push_back(resolved_width_);
    shape_.push_back(channels_);
  }
  rank_ = static_cast<std::uint32_t>(shape_.size());

  const std::size_t pixels =
      static_cast<std::size_t>(resolved_width_) * static_cast<std::size_t>(resolved_height_);
  const std::size_t element_bytes = (static_cast<std::size_t>(effective_dtype_.bits) + 7U) / 8U;
  frame_bytes_ = pixels * static_cast<std::size_t>(channels_) * element_bytes;

  spec.output(output_, config.tensor_port_name)
      .produces_tensor(TensorOutputSpec{
          .representation =
              TensorRepresentation{
                  .memory_kind = config.memory_kind, .dtype = effective_dtype_, .rank = rank_},
          .bounds = TensorBounds{.max_rank = rank_, .max_byte_span = frame_bytes_},
          .storage = TensorOutputStorage::kRuntimePool,
      });
}

Contract VideoFileReaderOp::contract() const {
  Contract result;
  result.trigger(OnClock{.period = std::chrono::milliseconds{1}});
  return result;
}

void VideoFileReaderOp::open_stream() {
  const VideoFileReaderOpParams& config = params();
  const std::string url = config.path.string();

  if (avformat_open_input(&format_ctx_, url.c_str(), nullptr, nullptr) < 0) {
    throw std::runtime_error("VideoFileReaderOp: could not open '" + url + "'");
  }
  if (avformat_find_stream_info(format_ctx_, nullptr) < 0) {
    throw std::runtime_error("VideoFileReaderOp: could not read stream info from '" + url + "'");
  }

  video_stream_index_ = config.stream_index;
  if (video_stream_index_ < 0) {
    video_stream_index_ = av_find_best_stream(format_ctx_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
  }
  if (video_stream_index_ < 0 ||
      static_cast<unsigned>(video_stream_index_) >= format_ctx_->nb_streams ||
      format_ctx_->streams[video_stream_index_]->codecpar->codec_type != AVMEDIA_TYPE_VIDEO) {
    throw std::runtime_error("VideoFileReaderOp: '" + url + "' has no usable video stream");
  }

  const AVCodecParameters* codecpar = format_ctx_->streams[video_stream_index_]->codecpar;
  const AVCodec* decoder = avcodec_find_decoder(codecpar->codec_id);
  if (decoder == nullptr) {
    throw std::runtime_error("VideoFileReaderOp: no decoder available for codec id " +
                             std::to_string(static_cast<int>(codecpar->codec_id)) + " in '" + url +
                             "'");
  }

  codec_ctx_ = avcodec_alloc_context3(decoder);
  if (codec_ctx_ == nullptr) {
    throw std::runtime_error("VideoFileReaderOp: avcodec_alloc_context3 failed");
  }
  if (avcodec_parameters_to_context(codec_ctx_, codecpar) < 0) {
    throw std::runtime_error("VideoFileReaderOp: avcodec_parameters_to_context failed for '" + url +
                             "'");
  }
  if (avcodec_open2(codec_ctx_, decoder, nullptr) < 0) {
    throw std::runtime_error("VideoFileReaderOp: avcodec_open2 failed for '" + url + "'");
  }

  decoded_frame_ = av_frame_alloc();
  scaled_frame_ = av_frame_alloc();
  packet_ = av_packet_alloc();
  if (decoded_frame_ == nullptr || scaled_frame_ == nullptr || packet_ == nullptr) {
    throw std::runtime_error("VideoFileReaderOp: libav frame/packet allocation failed");
  }

  const PixFmtInfo info = pix_fmt_info(config.output_pix_fmt);
  sws_ctx_ =
      sws_getContext(codec_ctx_->width, codec_ctx_->height, codec_ctx_->pix_fmt, resolved_width_,
                     resolved_height_, info.sws_dst_fmt, SWS_BILINEAR, nullptr, nullptr, nullptr);
  if (sws_ctx_ == nullptr) {
    throw std::runtime_error("VideoFileReaderOp: sws_getContext failed for '" + url + "'");
  }

  const int scaled_bytes =
      av_image_get_buffer_size(info.sws_dst_fmt, resolved_width_, resolved_height_, 1);
  if (scaled_bytes <= 0) {
    throw std::runtime_error("VideoFileReaderOp: av_image_get_buffer_size failed for '" + url +
                             "'");
  }
  scaled_buffer_.assign(static_cast<std::size_t>(scaled_bytes), 0U);
  if (av_image_fill_arrays(scaled_frame_->data, scaled_frame_->linesize, scaled_buffer_.data(),
                           info.sws_dst_fmt, resolved_width_, resolved_height_, 1) < 0) {
    throw std::runtime_error("VideoFileReaderOp: av_image_fill_arrays failed for '" + url + "'");
  }
  scaled_frame_->width = resolved_width_;
  scaled_frame_->height = resolved_height_;
  scaled_frame_->format = info.sws_dst_fmt;
}

void VideoFileReaderOp::close_stream() {
  if (sws_ctx_ != nullptr) {
    sws_freeContext(sws_ctx_);
    sws_ctx_ = nullptr;
  }
  if (packet_ != nullptr) {
    av_packet_free(&packet_);
  }
  if (decoded_frame_ != nullptr) {
    av_frame_free(&decoded_frame_);
  }
  if (scaled_frame_ != nullptr) {
    av_frame_free(&scaled_frame_);
  }
  if (codec_ctx_ != nullptr) {
    avcodec_free_context(&codec_ctx_);
  }
  if (format_ctx_ != nullptr) {
    avformat_close_input(&format_ctx_);
  }
  video_stream_index_ = -1;
}

void VideoFileReaderOp::start() {
  const VideoFileReaderOpParams& config = params();
  if (frame_bytes_ == 0U) {
    throw std::runtime_error("VideoFileReaderOp: computed frame byte span is zero");
  }
  open_stream();
  host_buffer_.assign(frame_bytes_, std::byte{0});
  frames_emitted_ = 0;
  eos_reached_ = false;
  have_next_emit_time_ = false;
  (void)config;
}

void VideoFileReaderOp::stop() {
  close_stream();
  host_buffer_.clear();
  scaled_buffer_.clear();
}

expected<bool, Error> VideoFileReaderOp::decode_one_frame() {
  const VideoFileReaderOpParams& config = params();

  for (;;) {
    const int receive_ret = avcodec_receive_frame(codec_ctx_, decoded_frame_);
    if (receive_ret == 0) {
      return true;
    }
    if (receive_ret != AVERROR(EAGAIN) && receive_ret != AVERROR_EOF) {
      return make_unexpected(Error{
          ErrorCode::kFailure,
          "VideoFileReaderOp: avcodec_receive_frame failed for '" + config.path.string() + "'"});
    }
    if (receive_ret == AVERROR_EOF) {
      return false;
    }

    // EAGAIN: decoder needs another input packet before it can produce a frame.
    const int read_ret = av_read_frame(format_ctx_, packet_);
    if (read_ret < 0) {
      // Clean (or otherwise unrecoverable) end of the demuxed stream: flush the decoder so any
      // frames it was internally buffering (B-frames) are drained via the next
      // avcodec_receive_frame() call at the top of this loop.
      avcodec_send_packet(codec_ctx_, nullptr);
      continue;
    }
    if (packet_->stream_index != video_stream_index_) {
      av_packet_unref(packet_);
      continue;
    }
    const int send_ret = avcodec_send_packet(codec_ctx_, packet_);
    av_packet_unref(packet_);
    if (send_ret < 0 && send_ret != AVERROR(EAGAIN)) {
      return make_unexpected(Error{
          ErrorCode::kFailure,
          "VideoFileReaderOp: avcodec_send_packet failed for '" + config.path.string() + "'"});
    }
  }
}

expected<void, Error> VideoFileReaderOp::compute(ExecutionContext& context) {
  const VideoFileReaderOpParams& config = params();

  if (config.frame_count > 0 && frames_emitted_ >= config.frame_count) {
    return {};
  }
  if (eos_reached_ && !config.loop) {
    return {};
  }

  if (config.frame_rate > 0.0F) {
    const auto period = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(1.0 / static_cast<double>(config.frame_rate)));
    const auto now = std::chrono::steady_clock::now();
    if (!have_next_emit_time_) {
      next_emit_time_ = now + period;
      have_next_emit_time_ = true;
    } else {
      if (now < next_emit_time_) {
        std::this_thread::sleep_for(next_emit_time_ - now);
      }
      next_emit_time_ += period;
    }
  }

  auto decoded = decode_one_frame();
  if (!decoded) {
    return make_unexpected(std::move(decoded).error());
  }
  if (!*decoded) {
    if (config.loop) {
      // Rewind to the beginning of the stream and flush decoder state, then try once more.
      if (av_seek_frame(format_ctx_, video_stream_index_, 0, AVSEEK_FLAG_BACKWARD) < 0) {
        return make_unexpected(Error{
            ErrorCode::kFailure,
            "VideoFileReaderOp: av_seek_frame rewind failed for '" + config.path.string() + "'"});
      }
      avcodec_flush_buffers(codec_ctx_);
      auto retried = decode_one_frame();
      if (!retried) {
        return make_unexpected(std::move(retried).error());
      }
      if (!*retried) {
        return make_unexpected(
            Error{ErrorCode::kFailure, "VideoFileReaderOp: '" + config.path.string() +
                                           "' produced no frames after loop rewind"});
      }
    } else {
      eos_reached_ = true;
      return {};
    }
  }

  const PixFmtInfo info = pix_fmt_info(config.output_pix_fmt);
  const int scaled_lines =
      sws_scale(sws_ctx_, decoded_frame_->data, decoded_frame_->linesize, 0, codec_ctx_->height,
                scaled_frame_->data, scaled_frame_->linesize);
  if (scaled_lines <= 0) {
    return make_unexpected(Error{ErrorCode::kFailure, "VideoFileReaderOp: sws_scale failed for '" +
                                                          config.path.string() + "'"});
  }

  const bool nchw = is_nchw(config.output_layout);
  const std::size_t height = static_cast<std::size_t>(resolved_height_);
  const std::size_t width = static_cast<std::size_t>(resolved_width_);
  const std::size_t channels = static_cast<std::size_t>(channels_);
  const std::size_t linesize = static_cast<std::size_t>(scaled_frame_->linesize[0]);
  const std::uint8_t* hwc_src = scaled_frame_->data[0];

  if (!info.is_float) {
    // uint8: either copy HWC verbatim (row-by-row, to strip any sws stride padding) or transpose
    // into CHW.
    auto* dst = reinterpret_cast<std::uint8_t*>(host_buffer_.data());
    if (nchw) {
      for (std::size_t c = 0; c < channels; ++c) {
        for (std::size_t y = 0; y < height; ++y) {
          const std::uint8_t* row = hwc_src + y * linesize;
          for (std::size_t x = 0; x < width; ++x) {
            dst[(c * height + y) * width + x] = row[x * channels + c];
          }
        }
      }
    } else {
      for (std::size_t y = 0; y < height; ++y) {
        std::memcpy(dst + y * width * channels, hwc_src + y * linesize, width * channels);
      }
    }
  } else {
    // fp32: normalize uint8 [0, 255] -> float [0, 1] while writing into the target layout.
    auto* dst = reinterpret_cast<float*>(host_buffer_.data());
    constexpr float kScale = 1.0F / 255.0F;
    if (nchw) {
      for (std::size_t c = 0; c < channels; ++c) {
        for (std::size_t y = 0; y < height; ++y) {
          const std::uint8_t* row = hwc_src + y * linesize;
          for (std::size_t x = 0; x < width; ++x) {
            dst[(c * height + y) * width + x] = static_cast<float>(row[x * channels + c]) * kScale;
          }
        }
      }
    } else {
      for (std::size_t y = 0; y < height; ++y) {
        const std::uint8_t* row = hwc_src + y * linesize;
        for (std::size_t x = 0; x < width * channels; ++x) {
          dst[y * width * channels + x] = static_cast<float>(row[x]) * kScale;
        }
      }
    }
  }

  auto loan = output_.allocate_tensor(
      TensorLoanRequest{.shape = TensorShapeView{shape_}, .dtype = effective_dtype_});
  if (!loan) {
    return make_unexpected(std::move(loan).error());
  }
  auto writer = loan->write(context.cuda_stream());
  if (!writer) {
    return make_unexpected(std::move(writer).error());
  }
  if (writer->byte_size() != frame_bytes_) {
    return make_unexpected(Error{
        ErrorCode::kFailure,
        "VideoFileReaderOp: allocated Tensor byte span (" + std::to_string(writer->byte_size()) +
            ") does not match frame byte span (" + std::to_string(frame_bytes_) + ")"});
  }

  if (config.memory_kind == holoscan::MemoryKind::kCudaDevice ||
      config.memory_kind == holoscan::MemoryKind::kCudaManaged) {
    const cudaStream_t stream = context.cuda_stream().get();
    if (stream == nullptr) {
      return make_unexpected(Error{ErrorCode::kNotSupported,
                                   "VideoFileReaderOp: no CUDA stream available for a "
                                   "device-memory tensor source"});
    }
    if (cudaMemcpyAsync(writer->data(), host_buffer_.data(), frame_bytes_, cudaMemcpyHostToDevice,
                        stream) != cudaSuccess) {
      return make_unexpected(
          Error{ErrorCode::kFailure, "VideoFileReaderOp: cudaMemcpyAsync host->device failed"});
    }
  } else {
    std::memcpy(writer->data(), host_buffer_.data(), frame_bytes_);
  }

  auto committed = std::move(*writer).commit();
  if (!committed) {
    return make_unexpected(std::move(committed).error());
  }

  auto emitted =
      output_.emit(std::move(*loan), EmitOptions{.capture_time = context.activation_time()});
  if (!emitted) {
    return make_unexpected(std::move(emitted).error());
  }

  ++frames_emitted_;
  return {};
}

}  // namespace holoscan::ops
