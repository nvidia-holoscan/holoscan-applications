// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Exercise the production cache code in the same translation unit, keeping its
// implementation details private to the application.
#include <cstdio>
#include <iostream>

#include "../cpp/lstm_inference_v5.cpp"

int main() {
  using namespace endoscopy_tool_tracking;
  auto require = [](bool condition, const char* message) {
    if (!condition) {
      throw std::runtime_error(message);
    }
  };
  const auto directory =
      std::filesystem::temp_directory_path() / ("endoscopy-cache-test-" + std::to_string(getpid()));
  std::filesystem::create_directory(directory);
  struct Cleanup {
    std::filesystem::path directory;
    ~Cleanup() {
      std::error_code ignored;
      std::filesystem::remove_all(directory, ignored);
    }
  } cleanup{directory};
  LstmInferenceParams params{.model_path = (directory / "model.onnx").string()};
  auto write_model = [&](std::string_view content) {
    std::ofstream output(params.model_path, std::ios::binary | std::ios::trunc);
    output << content;
  };
  cudaDeviceProp device{};
  device.major = 12;
  device.minor = 1;
  device.multiProcessorCount = 48;
  std::snprintf(device.name, sizeof(device.name), "%s", "first GPU");
  write_model("model A");
  const auto initial = cache_filename(params, device);
  require(initial == cache_filename(params, device), "unchanged cache identity must be stable");
  write_model("model B");
  require(initial != cache_filename(params, device),
          "replacing a same-size ONNX model must invalidate its cached engine");
  write_model("model A");
  params.enable_fp16 = true;
  require(initial != cache_filename(params, device), "FP16 changes must invalidate cache");
  params.enable_fp16 = false;
  params.max_workspace_size /= 2U;
  require(initial != cache_filename(params, device), "workspace changes must invalidate cache");
  params.max_workspace_size *= 2U;
  std::snprintf(device.name, sizeof(device.name), "%s", "second GPU");
  require(initial != cache_filename(params, device),
          "different GPU models with equal compute capability and SM count must not share cache");

  const auto cache = directory / "test.engine";
  write_binary(cache, "first", 5U);
  write_binary(cache, "replacement", 11U);
  std::vector<char> content;
  require(
      read_binary(cache, content) && std::string(content.begin(), content.end()) == "replacement",
      "atomic cache replacement must publish the entire new engine");
  for (const auto& entry : std::filesystem::directory_iterator(directory)) {
    require(entry.path().filename().string().find(".tmp.") == std::string::npos,
            "successful cache writes must remove their temporary files");
  }
  std::cout << "inference cache tests passed\n";
}
