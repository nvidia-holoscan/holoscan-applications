/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Pure-CPU unit tests for VideoFileWriterOpParams::from_yaml_node().
// Mirrors the SDK's test_yaml_config.cpp pattern — no Fragment, no CUDA, no FFmpeg runtime.

#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

#include "video_file_writer.hpp"

namespace holoscan::ops {
namespace {

VideoFileWriterOpParams load(const std::string& yaml,
                             const std::filesystem::path& config_dir = {}) {
  const YAML::Node node = YAML::Load(yaml);
  return VideoFileWriterOpParams::from_yaml_node(node, config_dir);
}

}  // namespace

// -------- Required fields -----------------------------------------------------------------

TEST(VideoFileWriterYaml, Minimal_PopulatesRequiredAndDefaults) {
  const auto p = load(R"(
path: out.mp4
width: 256
height: 256
)");
  EXPECT_EQ(p.path, "out.mp4");
  EXPECT_EQ(p.tensor_port_name, "frame");
  EXPECT_EQ(p.codec, "libx264");
  EXPECT_EQ(p.input_pix_fmt, "rgb24");
  EXPECT_EQ(p.input_layout, "nchw");
  EXPECT_TRUE(p.strip_batch_dim);
  EXPECT_EQ(p.width, 256);
  EXPECT_EQ(p.height, 256);
  EXPECT_FLOAT_EQ(p.fps, 30.0F);
  EXPECT_EQ(p.bitrate, 4000000);
  EXPECT_EQ(p.frame_count, 0);
}

TEST(VideoFileWriterYaml, MissingPath_Throws) {
  EXPECT_THROW(load(R"(width: 256
height: 256
)"),
               std::runtime_error);
}

TEST(VideoFileWriterYaml, MissingWidth_Throws) {
  EXPECT_THROW(load(R"(path: out.mp4
height: 256
)"),
               std::runtime_error);
}

TEST(VideoFileWriterYaml, MissingHeight_Throws) {
  EXPECT_THROW(load(R"(path: out.mp4
width: 256
)"),
               std::runtime_error);
}

// -------- Validation ---------------------------------------------------------------------

TEST(VideoFileWriterYaml, InvalidPixFmt_Throws) {
  EXPECT_THROW(load(R"(path: out.mp4
width: 256
height: 256
input_pix_fmt: yuv420p
)"),
               std::runtime_error);
}

TEST(VideoFileWriterYaml, InvalidLayout_Throws) {
  EXPECT_THROW(load(R"(path: out.mp4
width: 256
height: 256
input_layout: hwcn
)"),
               std::runtime_error);
}

// -------- Field round-trips --------------------------------------------------------------

TEST(VideoFileWriterYaml, AllFields_RoundTrip) {
  const auto p = load(R"(
path: /tmp/out.mp4
tensor_port_name: tensor_out
codec: libx265
output_container: mp4
input_pix_fmt: rgb_fp32
input_layout: nhwc
strip_batch_dim: false
width: 640
height: 480
fps: 60
bitrate: 8000000
frame_count: 120
)");
  EXPECT_EQ(p.path, "/tmp/out.mp4");
  EXPECT_EQ(p.tensor_port_name, "tensor_out");
  EXPECT_EQ(p.codec, "libx265");
  EXPECT_EQ(p.output_container, "mp4");
  EXPECT_EQ(p.input_pix_fmt, "rgb_fp32");
  EXPECT_EQ(p.input_layout, "nhwc");
  EXPECT_FALSE(p.strip_batch_dim);
  EXPECT_EQ(p.width, 640);
  EXPECT_EQ(p.height, 480);
  EXPECT_FLOAT_EQ(p.fps, 60.0F);
  EXPECT_EQ(p.bitrate, 8000000);
  EXPECT_EQ(p.frame_count, 120);
}

// -------- Path resolution ---------------------------------------------------------------

TEST(VideoFileWriterYaml, RelativePath_ResolvesAgainstConfigDir) {
  const auto tmp = std::filesystem::temp_directory_path() / "video_file_writer_yaml_test";
  std::filesystem::create_directories(tmp);
  const auto p = load(R"(
path: output/out.mp4
width: 128
height: 128
)",
                      tmp);
  EXPECT_EQ(p.path.parent_path(), tmp / "output");
  EXPECT_EQ(p.path.filename(), "out.mp4");
  std::filesystem::remove_all(tmp);
}

TEST(VideoFileWriterYaml, AbsolutePath_LeftUnchanged) {
  const auto p = load(R"(
path: /tmp/abs/out.mp4
width: 128
height: 128
)",
                      "/some/other/dir");
  EXPECT_EQ(p.path, "/tmp/abs/out.mp4");
}

}  // namespace holoscan::ops
