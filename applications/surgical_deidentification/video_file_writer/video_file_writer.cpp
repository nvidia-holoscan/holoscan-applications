/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "video_file_writer.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}

#include <holoscan/core/graph.hpp>
#include <yaml-cpp/yaml.h>

namespace holoscan::ops {

namespace {

[[nodiscard]] DLDataType dtype_from_string(const std::string& name) {
  if (name == "uint8") return DLDataType{kDLUInt, 8U, 1U};
  if (name == "int8") return DLDataType{kDLInt, 8U, 1U};
  if (name == "uint16") return DLDataType{kDLUInt, 16U, 1U};
  if (name == "int16") return DLDataType{kDLInt, 16U, 1U};
  if (name == "uint32") return DLDataType{kDLUInt, 32U, 1U};
  if (name == "int32") return DLDataType{kDLInt, 32U, 1U};
  if (name == "uint64") return DLDataType{kDLUInt, 64U, 1U};
  if (name == "int64") return DLDataType{kDLInt, 64U, 1U};
  if (name == "fp16") return DLDataType{kDLFloat, 16U, 1U};
  if (name == "fp32") return DLDataType{kDLFloat, 32U, 1U};
  if (name == "fp64") return DLDataType{kDLFloat, 64U, 1U};
  if (name == "bool") return DLDataType{kDLBool, static_cast<std::uint8_t>(sizeof(bool) * 8U), 1U};
  throw std::runtime_error("unsupported dtype '" + name +
                           "' (expected one of uint8/int8/uint16/int16/uint32/int32/uint64/"
                           "int64/fp16/fp32/fp64/bool)");
}

[[nodiscard]] bool dtype_equal(DLDataType lhs, DLDataType rhs) noexcept {
  return lhs.code == rhs.code && lhs.bits == rhs.bits && lhs.lanes == rhs.lanes;
}

// See the matching PixFmtInfo in video_file_reader.cpp: kept duplicated (rather than shared)
// because the port constraints on this task exclude adding a new public/src/utils helper, and
// the reader/writer libraries are independent CMake targets.
struct PixFmtInfo {
  AVPixelFormat packed_fmt;  ///< Packed uint8 pixel format used as the host-side staging layout.
  int32_t channels;
  bool is_float;
};

[[nodiscard]] PixFmtInfo pix_fmt_info(const std::string& name) {
  if (name == "rgb24") return {AV_PIX_FMT_RGB24, 3, false};
  if (name == "bgr24") return {AV_PIX_FMT_BGR24, 3, false};
  if (name == "rgb_fp32") return {AV_PIX_FMT_RGB24, 3, true};
  if (name == "bgr_fp32") return {AV_PIX_FMT_BGR24, 3, true};
  if (name == "grayscale_u8") return {AV_PIX_FMT_GRAY8, 1, false};
  if (name == "grayscale_fp32") return {AV_PIX_FMT_GRAY8, 1, true};
  throw std::runtime_error("unsupported input_pix_fmt '" + name +
                           "' (expected one of rgb24/bgr24/rgb_fp32/bgr_fp32/grayscale_u8/"
                           "grayscale_fp32)");
}

[[nodiscard]] bool is_nchw(const std::string& layout) {
  if (layout == "nchw") return true;
  if (layout == "nhwc") return false;
  throw std::runtime_error("unsupported input_layout '" + layout + "' (expected nchw or nhwc)");
}

}  // namespace

VideoFileWriterOpParams VideoFileWriterOpParams::from_yaml_node(
    const YAML::Node& node, const std::filesystem::path& config_dir) {
  if (!node["path"]) {
    throw std::runtime_error(
        "VideoFileWriterOpParams::from_yaml_node: missing required key 'path'");
  }
  if (!node["width"] || !node["height"]) {
    throw std::runtime_error(
        "VideoFileWriterOpParams::from_yaml_node: missing required key 'width'/'height'");
  }

  VideoFileWriterOpParams params;

  std::filesystem::path path = node["path"].as<std::string>();
  if (path.is_relative() && !config_dir.empty()) {
    path = (config_dir / path).lexically_normal();
  }
  params.path = std::move(path);

  if (const auto name_node = node["tensor_port_name"]) {
    params.tensor_port_name = name_node.as<std::string>();
  }
  if (const auto codec_node = node["codec"]) {
    params.codec = codec_node.as<std::string>();
  }
  if (const auto container_node = node["output_container"]) {
    params.output_container = container_node.as<std::string>();
  }
  if (const auto pix_fmt_node = node["input_pix_fmt"]) {
    params.input_pix_fmt = pix_fmt_node.as<std::string>();
  }
  if (const auto layout_node = node["input_layout"]) {
    params.input_layout = layout_node.as<std::string>();
  }
  // Validate now so from_yaml_node() fails loudly on a bad value instead of deferring to start().
  static_cast<void>(pix_fmt_info(params.input_pix_fmt));
  static_cast<void>(is_nchw(params.input_layout));

  if (const auto batch_node = node["strip_batch_dim"]) {
    params.strip_batch_dim = batch_node.as<bool>();
  }
  params.width = node["width"].as<int32_t>();
  params.height = node["height"].as<int32_t>();
  if (const auto fps_node = node["fps"]) {
    params.fps = fps_node.as<float>();
  }
  if (const auto bitrate_node = node["bitrate"]) {
    params.bitrate = bitrate_node.as<int64_t>();
  }
  if (const auto frame_count_node = node["frame_count"]) {
    params.frame_count = frame_count_node.as<int32_t>();
  }
  if (const auto expected_dtype_node = node["expected_dtype"]) {
    params.expected_dtype = dtype_from_string(expected_dtype_node.as<std::string>());
  }
  if (const auto expected_shape_node = node["expected_shape"]) {
    std::vector<int64_t> shape;
    for (const auto& dim : expected_shape_node) {
      shape.push_back(dim.as<int64_t>());
    }
    params.expected_shape = std::move(shape);
  }

  return params;
}

VideoFileWriterOp::VideoFileWriterOp(VideoFileWriterOpParams params) {
  auto bound = holoscan::detail::GraphOperatorAccess::bind(*this, std::move(params));
  if (!bound) {
    throw std::runtime_error("VideoFileWriterOp: invalid VideoFileWriterOpParams: " +
                             bound.error().message);
  }
}

VideoFileWriterOp::~VideoFileWriterOp() {
  if (packet_ != nullptr) av_packet_free(&packet_);
  if (src_frame_ != nullptr) av_frame_free(&src_frame_);
  if (sws_ctx_ != nullptr) sws_freeContext(sws_ctx_);
  if (codec_ctx_ != nullptr) avcodec_free_context(&codec_ctx_);
  if (format_ctx_ != nullptr) {
    if (format_ctx_->pb != nullptr && !(format_ctx_->oformat->flags & AVFMT_NOFILE)) {
      avio_closep(&format_ctx_->pb);
    }
    avformat_free_context(format_ctx_);
    format_ctx_ = nullptr;
  }
}

void VideoFileWriterOp::setup(OperatorSpec& spec) {
  const VideoFileWriterOpParams& config = params();
  spec.input(input_, config.tensor_port_name)
      .queue_depth(4U)
      .expects_tensor(TensorInputSpec{.bounds = TensorBounds{.max_rank = 8}});
}

Contract VideoFileWriterOp::contract() const {
  Contract result;
  result.trigger(OnEach{input_});
  return result;
}

void VideoFileWriterOp::start() {
  const VideoFileWriterOpParams& config = params();
  if (config.width <= 0 || config.height <= 0) {
    throw std::runtime_error("VideoFileWriterOp: params().width/height must be positive");
  }
  if (!config.path.parent_path().empty()) {
    std::error_code error_code;
    std::filesystem::create_directories(config.path.parent_path(), error_code);
  }

  const std::string url = config.path.string();
  const char* format_name =
      config.output_container.empty() ? nullptr : config.output_container.c_str();
  if (avformat_alloc_output_context2(&format_ctx_, nullptr, format_name, url.c_str()) < 0 ||
      format_ctx_ == nullptr) {
    throw std::runtime_error("VideoFileWriterOp: could not determine an output container for '" +
                             url +
                             "' (set output_container explicitly if the extension is "
                             "ambiguous)");
  }

  const AVCodec* encoder = avcodec_find_encoder_by_name(config.codec.c_str());
  if (encoder == nullptr) {
    throw std::runtime_error("VideoFileWriterOp: unknown/unavailable codec '" + config.codec + "'");
  }

  stream_ = avformat_new_stream(format_ctx_, nullptr);
  if (stream_ == nullptr) {
    throw std::runtime_error("VideoFileWriterOp: avformat_new_stream failed for '" + url + "'");
  }

  codec_ctx_ = avcodec_alloc_context3(encoder);
  if (codec_ctx_ == nullptr) {
    throw std::runtime_error("VideoFileWriterOp: avcodec_alloc_context3 failed");
  }

  // Pick a pixel format the encoder actually supports, preferring YUV420p (the widest-compatible
  // choice for libx264/libx265/mpeg4/mjpeg); fall back to the encoder's first advertised format
  // (e.g. ffv1 defaults to something else entirely).
  AVPixelFormat encode_fmt = AV_PIX_FMT_YUV420P;
  if (encoder->pix_fmts != nullptr) {
    bool supports_yuv420p = false;
    for (const AVPixelFormat* fmt = encoder->pix_fmts; *fmt != AV_PIX_FMT_NONE; ++fmt) {
      if (*fmt == AV_PIX_FMT_YUV420P) {
        supports_yuv420p = true;
        break;
      }
    }
    if (!supports_yuv420p) {
      encode_fmt = encoder->pix_fmts[0];
    }
  }

  codec_ctx_->width = config.width;
  codec_ctx_->height = config.height;
  codec_ctx_->time_base = AVRational{1, std::max(1, static_cast<int>(config.fps))};
  codec_ctx_->framerate = AVRational{std::max(1, static_cast<int>(config.fps)), 1};
  codec_ctx_->pix_fmt = encode_fmt;
  codec_ctx_->bit_rate = config.bitrate;
  codec_ctx_->gop_size = std::max(1, static_cast<int>(config.fps));
  if (format_ctx_->oformat->flags & AVFMT_GLOBALHEADER) {
    codec_ctx_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
  }

  if (avcodec_open2(codec_ctx_, encoder, nullptr) < 0) {
    throw std::runtime_error("VideoFileWriterOp: avcodec_open2 failed for codec '" + config.codec +
                             "'");
  }
  if (avcodec_parameters_from_context(stream_->codecpar, codec_ctx_) < 0) {
    throw std::runtime_error("VideoFileWriterOp: avcodec_parameters_from_context failed for '" +
                             url + "'");
  }
  stream_->time_base = codec_ctx_->time_base;

  if (!(format_ctx_->oformat->flags & AVFMT_NOFILE)) {
    if (avio_open(&format_ctx_->pb, url.c_str(), AVIO_FLAG_WRITE) < 0) {
      throw std::runtime_error("VideoFileWriterOp: could not open '" + url + "' for writing");
    }
  }
  if (avformat_write_header(format_ctx_, nullptr) < 0) {
    throw std::runtime_error("VideoFileWriterOp: avformat_write_header failed for '" + url + "'");
  }
  header_written_ = true;

  const PixFmtInfo info = pix_fmt_info(config.input_pix_fmt);
  sws_ctx_ = sws_getContext(config.width, config.height, info.packed_fmt, config.width,
                            config.height, encode_fmt, SWS_BILINEAR, nullptr, nullptr, nullptr);
  if (sws_ctx_ == nullptr) {
    throw std::runtime_error("VideoFileWriterOp: sws_getContext failed for '" + url + "'");
  }

  src_frame_ = av_frame_alloc();
  packet_ = av_packet_alloc();
  if (src_frame_ == nullptr || packet_ == nullptr) {
    throw std::runtime_error("VideoFileWriterOp: libav frame/packet allocation failed");
  }
  src_frame_->format = encode_fmt;
  src_frame_->width = config.width;
  src_frame_->height = config.height;
  if (av_frame_get_buffer(src_frame_, 0) < 0) {
    throw std::runtime_error("VideoFileWriterOp: av_frame_get_buffer failed for '" + url + "'");
  }

  frames_written_ = 0;
  next_pts_ = 0;
}

expected<void, Error> VideoFileWriterOp::send_and_drain(AVFrame* frame) {
  const VideoFileWriterOpParams& config = params();
  int send_ret = avcodec_send_frame(codec_ctx_, frame);
  if (send_ret < 0 && send_ret != AVERROR_EOF) {
    return make_unexpected(Error{ErrorCode::kFailure,
                                 "VideoFileWriterOp: avcodec_send_frame "
                                 "failed for '" +
                                     config.path.string() + "'"});
  }
  for (;;) {
    const int receive_ret = avcodec_receive_packet(codec_ctx_, packet_);
    if (receive_ret == AVERROR(EAGAIN) || receive_ret == AVERROR_EOF) {
      break;
    }
    if (receive_ret < 0) {
      return make_unexpected(Error{
          ErrorCode::kFailure,
          "VideoFileWriterOp: avcodec_receive_packet failed for '" + config.path.string() + "'"});
    }
    av_packet_rescale_ts(packet_, codec_ctx_->time_base, stream_->time_base);
    packet_->stream_index = stream_->index;
    if (av_interleaved_write_frame(format_ctx_, packet_) < 0) {
      av_packet_unref(packet_);
      return make_unexpected(
          Error{ErrorCode::kFailure, "VideoFileWriterOp: av_interleaved_write_frame failed for '" +
                                         config.path.string() + "'"});
    }
  }
  return {};
}

void VideoFileWriterOp::stop() {
  if (codec_ctx_ != nullptr && header_written_) {
    // Flush: send a null frame and drain every packet the encoder was still buffering.
    auto flushed = send_and_drain(nullptr);
    if (!flushed) {
      // Log-equivalent best effort is to just proceed to trailer/close so the file is still
      // finalized.
    }
  }
  if (format_ctx_ != nullptr && header_written_) {
    av_write_trailer(format_ctx_);
  }
  if (format_ctx_ != nullptr && format_ctx_->pb != nullptr &&
      !(format_ctx_->oformat->flags & AVFMT_NOFILE)) {
    avio_closep(&format_ctx_->pb);
  }
  if (packet_ != nullptr) av_packet_free(&packet_);
  if (src_frame_ != nullptr) av_frame_free(&src_frame_);
  if (sws_ctx_ != nullptr) {
    sws_freeContext(sws_ctx_);
    sws_ctx_ = nullptr;
  }
  if (codec_ctx_ != nullptr) avcodec_free_context(&codec_ctx_);
  if (format_ctx_ != nullptr) {
    avformat_free_context(format_ctx_);
    format_ctx_ = nullptr;
  }
  header_written_ = false;
  host_buffer_.clear();
  hwc_u8_buffer_.clear();
}

expected<void, Error> VideoFileWriterOp::compute(ExecutionContext& context) {
  const VideoFileWriterOpParams& config = params();

  auto sample = input_.receive_data();
  if (!sample) {
    return make_unexpected(std::move(sample).error());
  }
  const Tensor& tensor = *sample;

  // Drain (accept and discard) once the configured limit is reached, rather than refusing input.
  if (config.frame_count > 0 && frames_written_ >= config.frame_count) {
    return {};
  }

  if (config.expected_dtype && !dtype_equal(tensor.dtype(), *config.expected_dtype)) {
    return make_unexpected(
        Error{ErrorCode::kFailure,
              "VideoFileWriterOp: received Tensor dtype does not match configured expected_dtype"});
  }
  if (config.expected_shape && tensor.shape() != *config.expected_shape) {
    return make_unexpected(
        Error{ErrorCode::kFailure,
              "VideoFileWriterOp: received Tensor shape does not match configured expected_shape"});
  }

  const auto frame_bytes = static_cast<std::size_t>(tensor.nbytes());
  if (frame_bytes == 0U) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "VideoFileWriterOp: received an empty Tensor"});
  }
  host_buffer_.resize(frame_bytes);
  auto copied = tensor.copy_to_host(host_buffer_.data(), frame_bytes, context.cuda_stream());
  if (!copied) {
    return make_unexpected(std::move(copied).error());
  }

  const PixFmtInfo info = pix_fmt_info(config.input_pix_fmt);
  const bool nchw = is_nchw(config.input_layout);
  const std::size_t height = static_cast<std::size_t>(config.height);
  const std::size_t width = static_cast<std::size_t>(config.width);
  const std::size_t channels = static_cast<std::size_t>(info.channels);
  const std::size_t expected_elements = height * width * channels;
  const std::size_t element_bytes = info.is_float ? 4U : 1U;
  if (frame_bytes != expected_elements * element_bytes) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              "VideoFileWriterOp: received Tensor byte span (" + std::to_string(frame_bytes) +
                  ") does not match width*height*channels for the configured input_pix_fmt/width/"
                  "height (" +
                  std::to_string(expected_elements * element_bytes) + ")"});
  }

  hwc_u8_buffer_.resize(height * width * channels);

  if (!info.is_float) {
    const auto* src = reinterpret_cast<const std::uint8_t*>(host_buffer_.data());
    if (nchw) {
      for (std::size_t c = 0; c < channels; ++c) {
        for (std::size_t y = 0; y < height; ++y) {
          for (std::size_t x = 0; x < width; ++x) {
            hwc_u8_buffer_[(y * width + x) * channels + c] = src[(c * height + y) * width + x];
          }
        }
      }
    } else {
      std::memcpy(hwc_u8_buffer_.data(), src, hwc_u8_buffer_.size());
    }
  } else {
    // fp32 -> uint8: scale up by 255 and clamp to [0, 255].
    const auto* src = reinterpret_cast<const float*>(host_buffer_.data());
    auto to_u8 = [](float value) -> std::uint8_t {
      const float scaled = value * 255.0F;
      return static_cast<std::uint8_t>(std::clamp(scaled, 0.0F, 255.0F));
    };
    if (nchw) {
      for (std::size_t c = 0; c < channels; ++c) {
        for (std::size_t y = 0; y < height; ++y) {
          for (std::size_t x = 0; x < width; ++x) {
            hwc_u8_buffer_[(y * width + x) * channels + c] =
                to_u8(src[(c * height + y) * width + x]);
          }
        }
      }
    } else {
      for (std::size_t i = 0; i < hwc_u8_buffer_.size(); ++i) {
        hwc_u8_buffer_[i] = to_u8(src[i]);
      }
    }
  }

  const std::uint8_t* packed_data[4] = {hwc_u8_buffer_.data(), nullptr, nullptr, nullptr};
  const int packed_linesize[4] = {static_cast<int>(width * channels), 0, 0, 0};

  if (av_frame_make_writable(src_frame_) < 0) {
    return make_unexpected(
        Error{ErrorCode::kFailure, "VideoFileWriterOp: av_frame_make_writable failed"});
  }
  const int scaled_lines =
      sws_scale(sws_ctx_, packed_data, packed_linesize, 0, static_cast<int>(height),
                src_frame_->data, src_frame_->linesize);
  if (scaled_lines <= 0) {
    return make_unexpected(Error{ErrorCode::kFailure, "VideoFileWriterOp: sws_scale failed for '" +
                                                          config.path.string() + "'"});
  }

  src_frame_->pts = next_pts_++;
  auto sent = send_and_drain(src_frame_);
  if (!sent) {
    return sent;
  }

  ++frames_written_;
  return {};
}

}  // namespace holoscan::ops
