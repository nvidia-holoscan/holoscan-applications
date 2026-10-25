/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <filesystem>
#include <string>

#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

#include "video_file_reader.hpp"

namespace holoscan::ops {
namespace {

VideoFileReaderOpParams load(const std::string& yaml,
                             const std::filesystem::path& config_dir = {}) {
  const YAML::Node node = YAML::Load(yaml);
  return VideoFileReaderOpParams::from_yaml_node(node, config_dir);
}

}  // namespace

TEST(VideoFileReaderYaml, Minimal_PopulatesRequiredAndDefaults) {
  const auto p = load(R"(path: in.mp4
)");
  EXPECT_EQ(p.path, "in.mp4");
  EXPECT_EQ(p.tensor_port_name, "frame");
  EXPECT_FALSE(p.output_width.has_value());
  EXPECT_FALSE(p.output_height.has_value());
  EXPECT_EQ(p.output_pix_fmt, "rgb24");
  EXPECT_EQ(p.output_layout, "nchw");
  EXPECT_TRUE(p.prepend_batch_dim);
  EXPECT_EQ(p.output_dtype.code, kDLUInt);
  EXPECT_EQ(p.output_dtype.bits, 8U);
  EXPECT_EQ(p.memory_kind, holoscan::MemoryKind::kCudaDevice);
  EXPECT_EQ(p.frame_count, 0);
  EXPECT_FLOAT_EQ(p.frame_rate, 0.0F);
  EXPECT_FALSE(p.loop);
  EXPECT_EQ(p.stream_index, -1);
}

TEST(VideoFileReaderYaml, MissingPath_Throws) {
  EXPECT_THROW(load("tensor_port_name: raw_frame\n"), std::runtime_error);
}

TEST(VideoFileReaderYaml, InvalidPixFmt_Throws) {
  EXPECT_THROW(load(R"(path: in.mp4
output_pix_fmt: yuv420p
)"),
               std::runtime_error);
}

TEST(VideoFileReaderYaml, InvalidLayout_Throws) {
  EXPECT_THROW(load(R"(path: in.mp4
output_layout: hwcn
)"),
               std::runtime_error);
}

TEST(VideoFileReaderYaml, Fp32PixFmt_UpgradesDtype) {
  const auto p = load(R"(path: in.mp4
output_pix_fmt: rgb_fp32
)");
  EXPECT_EQ(p.output_dtype.code, kDLFloat);
  EXPECT_EQ(p.output_dtype.bits, 32U);
}

TEST(VideoFileReaderYaml, GrayscalePixFmt_Accepted) {
  const auto p = load(R"(path: in.mp4
output_pix_fmt: grayscale_u8
)");
  EXPECT_EQ(p.output_pix_fmt, "grayscale_u8");
}

TEST(VideoFileReaderYaml, AllFields_RoundTrip) {
  const auto p = load(R"(
path: /tmp/in.mp4
tensor_port_name: raw_frame
output_width: 640
output_height: 480
output_pix_fmt: bgr24
output_layout: nhwc
prepend_batch_dim: false
memory_kind: host
frame_count: 100
frame_rate: 24
loop: true
stream_index: 1
)");
  EXPECT_EQ(p.path, "/tmp/in.mp4");
  EXPECT_EQ(p.tensor_port_name, "raw_frame");
  ASSERT_TRUE(p.output_width.has_value());
  EXPECT_EQ(*p.output_width, 640);
  ASSERT_TRUE(p.output_height.has_value());
  EXPECT_EQ(*p.output_height, 480);
  EXPECT_EQ(p.output_pix_fmt, "bgr24");
  EXPECT_EQ(p.output_layout, "nhwc");
  EXPECT_FALSE(p.prepend_batch_dim);
  EXPECT_EQ(p.memory_kind, holoscan::MemoryKind::kHost);
  EXPECT_EQ(p.frame_count, 100);
  EXPECT_FLOAT_EQ(p.frame_rate, 24.0F);
  EXPECT_TRUE(p.loop);
  EXPECT_EQ(p.stream_index, 1);
}

TEST(VideoFileReaderYaml, RelativePath_ResolvesAgainstConfigDir) {
  const auto tmp = std::filesystem::temp_directory_path() / "video_file_reader_yaml_test";
  std::filesystem::create_directories(tmp);
  const auto p = load("path: videos/in.mp4\n", tmp);
  EXPECT_EQ(p.path.parent_path(), tmp / "videos");
  EXPECT_EQ(p.path.filename(), "in.mp4");
  std::filesystem::remove_all(tmp);
}

TEST(VideoFileReaderYaml, UrlPath_LeftUnchanged) {
  const auto p = load("path: rtsp://example/stream\n", "/some/other/dir");
  EXPECT_EQ(p.path, "rtsp://example/stream");
}

}  // namespace holoscan::ops
