/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Header-only helper that encodes a short synthetic MP4 to disk for runtime / functional
// tests. Uses libavformat + libavcodec directly so the tests don't depend on any upstream
// test asset or dataset mount — a self-contained fixture.

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
}

namespace surgical_deid::test {

/// @brief Encode `frame_count` synthetic HWC uint8 frames into an MP4 at `path`.
///
/// Each frame is `width × height` and filled with a per-frame grayscale value so the encoder
/// has real content to compress (uniform frames can confuse some codecs' rate control). The
/// encoder is `mpeg4`, which is available in every stock FFmpeg build (unlike `libx264`,
/// which is a separate build-time dep).
///
/// @throws std::runtime_error on any libav failure, with a descriptive message.
inline void encode_synthetic_mp4(const std::string& path, int width, int height, int frame_count,
                                 int fps = 25) {
  if (width <= 0 || height <= 0 || frame_count <= 0 || fps <= 0) {
    throw std::runtime_error("encode_synthetic_mp4: non-positive dimension or count");
  }

  AVFormatContext* fmt_ctx = nullptr;
  if (avformat_alloc_output_context2(&fmt_ctx, nullptr, nullptr, path.c_str()) < 0 ||
      fmt_ctx == nullptr) {
    throw std::runtime_error("avformat_alloc_output_context2 failed for " + path);
  }

  const AVCodec* codec = avcodec_find_encoder(AV_CODEC_ID_MPEG4);
  if (codec == nullptr) {
    avformat_free_context(fmt_ctx);
    throw std::runtime_error("mpeg4 encoder not available in this FFmpeg build");
  }

  AVStream* stream = avformat_new_stream(fmt_ctx, nullptr);
  AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
  if (stream == nullptr || codec_ctx == nullptr) {
    avformat_free_context(fmt_ctx);
    throw std::runtime_error("stream / codec_ctx allocation failed");
  }

  codec_ctx->codec_id = AV_CODEC_ID_MPEG4;
  codec_ctx->width = width;
  codec_ctx->height = height;
  codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
  codec_ctx->time_base = AVRational{1, fps};
  codec_ctx->framerate = AVRational{fps, 1};
  codec_ctx->bit_rate = 400'000;
  codec_ctx->gop_size = fps;
  if (fmt_ctx->oformat->flags & AVFMT_GLOBALHEADER) {
    codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
  }

  if (avcodec_open2(codec_ctx, codec, nullptr) < 0 ||
      avcodec_parameters_from_context(stream->codecpar, codec_ctx) < 0) {
    avcodec_free_context(&codec_ctx);
    avformat_free_context(fmt_ctx);
    throw std::runtime_error("avcodec_open2 / parameters_from_context failed");
  }
  stream->time_base = codec_ctx->time_base;

  if (!(fmt_ctx->oformat->flags & AVFMT_NOFILE) &&
      avio_open(&fmt_ctx->pb, path.c_str(), AVIO_FLAG_WRITE) < 0) {
    avcodec_free_context(&codec_ctx);
    avformat_free_context(fmt_ctx);
    throw std::runtime_error("avio_open failed for " + path);
  }
  if (avformat_write_header(fmt_ctx, nullptr) < 0) {
    avcodec_free_context(&codec_ctx);
    avformat_free_context(fmt_ctx);
    throw std::runtime_error("avformat_write_header failed");
  }

  AVFrame* frame = av_frame_alloc();
  AVPacket* pkt = av_packet_alloc();
  frame->format = codec_ctx->pix_fmt;
  frame->width = width;
  frame->height = height;
  if (av_frame_get_buffer(frame, 0) < 0) {
    av_frame_free(&frame);
    av_packet_free(&pkt);
    avcodec_free_context(&codec_ctx);
    avformat_free_context(fmt_ctx);
    throw std::runtime_error("av_frame_get_buffer failed");
  }

  for (int i = 0; i < frame_count; ++i) {
    if (av_frame_make_writable(frame) < 0) {
      throw std::runtime_error("av_frame_make_writable failed");
    }
    // Fill Y plane with (i * 20) mod 256, chroma neutral — mid-gray gradient per frame.
    const std::uint8_t y_val = static_cast<std::uint8_t>((i * 20) & 0xFF);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        frame->data[0][y * frame->linesize[0] + x] = y_val;
      }
    }
    for (int y = 0; y < height / 2; ++y) {
      for (int x = 0; x < width / 2; ++x) {
        frame->data[1][y * frame->linesize[1] + x] = 128;
        frame->data[2][y * frame->linesize[2] + x] = 128;
      }
    }
    frame->pts = i;

    if (avcodec_send_frame(codec_ctx, frame) < 0) {
      throw std::runtime_error("avcodec_send_frame failed");
    }
    while (true) {
      const int r = avcodec_receive_packet(codec_ctx, pkt);
      if (r == AVERROR(EAGAIN) || r == AVERROR_EOF) {
        break;
      }
      if (r < 0) {
        throw std::runtime_error("avcodec_receive_packet failed");
      }
      pkt->stream_index = stream->index;
      av_packet_rescale_ts(pkt, codec_ctx->time_base, stream->time_base);
      if (av_interleaved_write_frame(fmt_ctx, pkt) < 0) {
        throw std::runtime_error("av_interleaved_write_frame failed");
      }
      av_packet_unref(pkt);
    }
  }

  // Flush.
  avcodec_send_frame(codec_ctx, nullptr);
  while (true) {
    const int r = avcodec_receive_packet(codec_ctx, pkt);
    if (r == AVERROR(EAGAIN) || r == AVERROR_EOF) {
      break;
    }
    if (r < 0) {
      break;
    }
    pkt->stream_index = stream->index;
    av_packet_rescale_ts(pkt, codec_ctx->time_base, stream->time_base);
    av_interleaved_write_frame(fmt_ctx, pkt);
    av_packet_unref(pkt);
  }

  av_write_trailer(fmt_ctx);
  av_frame_free(&frame);
  av_packet_free(&pkt);
  avcodec_free_context(&codec_ctx);
  if (!(fmt_ctx->oformat->flags & AVFMT_NOFILE)) {
    avio_closep(&fmt_ctx->pb);
  }
  avformat_free_context(fmt_ctx);
}

}  // namespace surgical_deid::test
