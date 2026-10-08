// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>

// Compile this translation unit instead of tool_tracking_postprocessor_v5.cu so
// the regression exercises the exact application kernel without a test-only API.
#include "../cpp/tool_tracking_postprocessor_v5.cu"  // NOLINT(build/include)

int main() {
  float* probabilities{};
  float* masks{};
  uchar4* output{};
  constexpr std::size_t pixels = 60U * 107U;
  auto check = [](cudaError_t error) {
    if (error != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(error));
    }
  };
  check(cudaMallocManaged(&probabilities, 7U * sizeof(float)));
  check(cudaMallocManaged(&masks, 7U * pixels * sizeof(float)));
  check(cudaMallocManaged(&output, pixels * sizeof(uchar4)));
  std::fill_n(probabilities, 7U, 0.0F);
  std::fill_n(masks, 7U * pixels, 1.0F);
  auto run = [&] {
    endoscopy_tool_tracking::compose_masks<<<(pixels + 255U) / 256U, 256U>>>(probabilities, masks,
                                                                             0.5F, output);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());
  };
  auto require = [](bool condition, const char* message) {
    if (!condition) {
      throw std::runtime_error(message);
    }
  };

  std::array<float, 7U> host_probs{};
  std::array<float, 14U> host_coords{};
  require(endoscopy_tool_tracking::valid_tracking_outputs(host_probs, host_coords),
          "finite zero probabilities are valid no-detection output");
  for (const float invalid : {std::numeric_limits<float>::quiet_NaN(),
                              std::numeric_limits<float>::infinity(), -0.1F, 1.1F}) {
    host_probs[6] = invalid;
    require(!endoscopy_tool_tracking::valid_tracking_outputs(host_probs, host_coords),
            "invalid probabilities must fail instead of silently hiding every tool");
  }
  host_probs[6] = 0.9F;
  host_coords[13] = std::numeric_limits<float>::quiet_NaN();
  require(!endoscopy_tool_tracking::valid_tracking_outputs(host_probs, host_coords),
          "non-finite coordinates must fail instead of reaching HoloViz");
  host_coords[13] = 0.5F;
  require(endoscopy_tool_tracking::valid_tracking_outputs(host_probs, host_coords),
          "valid detections must be accepted");

  probabilities[0] = std::numeric_limits<float>::quiet_NaN();
  probabilities[1] = 0.5F;
  run();
  require(output[0].w == 0U && output[pixels - 1U].w == 0U,
          "NaN and threshold-equal probabilities must produce a transparent mask");

  probabilities[0] = 1.0F;
  run();
  require(output[0].x == 30U && output[0].y == 119U && output[0].z == 181U && output[0].w == 178U,
          "one tool must preserve its straight RGB color and clamped alpha");

  masks[0] = 0.0F;
  masks[pixels - 1U] = 0.3F;
  run();
  require(output[0].w == 0U && output[pixels - 1U].w == 0U,
          "mask values at or below the lower clamp must be transparent");
  require(output[1].w == 178U, "mask indexing must preserve neighboring pixels");

  probabilities[0] = 0.0F;
  probabilities[6] = 1.0F;
  run();
  require(output[0].x == 165U && output[0].y == 206U && output[0].z == 226U && output[0].w == 178U,
          "class-major masks must address the seventh class plane");

  probabilities[6] = 0.0F;
  probabilities[0] = 1.0F;
  probabilities[1] = 1.0F;
  run();
  require(output[1].x == 46U && output[1].y == 151U && output[1].z == 75U && output[1].w == 232U,
          "overlapping masks must composite in class order with straight-alpha output");

  check(cudaFree(output));
  check(cudaFree(masks));
  check(cudaFree(probabilities));
  std::cout << "postprocessor kernel tests passed\n";
}
