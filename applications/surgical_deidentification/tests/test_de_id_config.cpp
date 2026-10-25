/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// App-level config smoke test. Loads the shipped applications/surgical_deidentification/config.yaml
// and verifies each sub-section parses through the correct operator-side from_yaml_node() helper.

#include <gtest/gtest.h>

#include <filesystem>
#include <string>

#include <yaml-cpp/yaml.h>

#include <holoscan/operators/inference/inference.hpp>
#include <holoscan/operators/tensor_preprocessor/tensor_preprocessor.hpp>

#include "video_file_reader.hpp"

namespace {

// Resolves the shipped config.yaml path at test time via a build-time macro the test's
// CMakeLists sets with target_compile_definitions.
#ifndef SURGICAL_DEID_CONFIG_PATH
#error "SURGICAL_DEID_CONFIG_PATH must be set by the test CMakeLists"
#endif

[[nodiscard]] std::filesystem::path config_path() {
  return std::filesystem::path{SURGICAL_DEID_CONFIG_PATH};
}

[[nodiscard]] YAML::Node load_config() {
  const auto path = config_path();
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error("missing config.yaml at " + path.string());
  }
  return YAML::LoadFile(path.string());
}

}  // namespace

// -------- Fixture ------------------------------------------------------------------------

class SurgicalDeIdConfigTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_NO_THROW(root_ = load_config()) << "cannot load " << config_path();
    config_dir_ = config_path().parent_path();
  }

  YAML::Node root_;
  std::filesystem::path config_dir_;
};

// -------- Top-level structure ------------------------------------------------------------

TEST_F(SurgicalDeIdConfigTest, HasAllRequiredSections) {
  for (const char* key : {"inference", "video_source", "preprocessor"}) {
    EXPECT_TRUE(root_[key]) << "config.yaml missing required top-level section: " << key;
  }
}

// -------- Inference section --------------------------------------------------------------

TEST_F(SurgicalDeIdConfigTest, InferenceSection_ParsesCleanly) {
  const auto node = root_["inference"];
  ASSERT_TRUE(node);
  // model_path_map / input_map / output_map must all be present and non-empty.
  for (const char* key : {"model_path_map", "input_map", "output_map"}) {
    ASSERT_TRUE(node[key]) << "inference missing " << key;
    ASSERT_TRUE(node[key].IsMap()) << "inference." << key << " not a map";
    EXPECT_GT(node[key].size(), 0U) << "inference." << key << " must have at least one entry";
  }
  // The de_id model entry should be present and name an ONNX file.
  ASSERT_TRUE(node["model_path_map"]["de_id"]);
  const auto onnx = node["model_path_map"]["de_id"].as<std::string>();
  EXPECT_NE(onnx.find(".onnx"), std::string::npos);
}

// -------- video_source section (VideoFileReaderOp) ---------------------------------------

TEST_F(SurgicalDeIdConfigTest, VideoSourceSection_ParsesViaReaderYamlLoader) {
  const auto node = root_["video_source"];
  ASSERT_TRUE(node);
  // The shipped config references a dataset path that may not exist on this host; parse-only
  // check — no file I/O.
  EXPECT_NO_THROW({
    const auto p =
        holoscan::ops::VideoFileReaderOpParams::from_yaml_node(node, config_dir_);
    EXPECT_FALSE(p.tensor_port_name.empty());
    EXPECT_FALSE(p.output_pix_fmt.empty());
    EXPECT_FALSE(p.output_layout.empty());
  });
}

// -------- preprocessor section (TensorPreprocessorOp) ------------------------------------

TEST_F(SurgicalDeIdConfigTest, PreprocessorSection_ParsesViaSdkYamlLoader) {
  const auto node = root_["preprocessor"];
  ASSERT_TRUE(node);
  EXPECT_NO_THROW({
    const auto p =
        holoscan::ops::TensorPreprocessorOpParams::from_yaml_node(node, config_dir_);
    EXPECT_GT(p.output_width, 0);
    EXPECT_GT(p.output_height, 0);
  });
}

// -------- Cross-section wiring ----------------------------------------------------------

TEST_F(SurgicalDeIdConfigTest, PreprocessorInputPortMatchesVideoSourceTensorName) {
  const auto src_name = root_["video_source"]["tensor_port_name"].as<std::string>();
  const auto pre_in = root_["preprocessor"]["input_port_name"].as<std::string>();
  EXPECT_EQ(src_name, pre_in)
      << "video_source.tensor_port_name must match preprocessor.input_port_name for graph wiring";
}

TEST_F(SurgicalDeIdConfigTest, InferenceInputMapNamesPreprocessorOutput) {
  const auto pre_out = root_["preprocessor"]["output_port_name"].as<std::string>();
  const auto input_map = root_["inference"]["input_map"];
  ASSERT_TRUE(input_map);
  bool found = false;
  for (auto it = input_map.begin(); it != input_map.end(); ++it) {
    const auto tensors = it->second;
    for (const auto& t : tensors) {
      if (t.as<std::string>() == pre_out) {
        found = true;
        break;
      }
    }
    if (found) {
      break;
    }
  }
  EXPECT_TRUE(found) << "preprocessor.output_port_name '" << pre_out
                     << "' must appear in inference.input_map for the graph to connect";
}
