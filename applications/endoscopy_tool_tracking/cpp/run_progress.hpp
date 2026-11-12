// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <atomic>
#include <cstdint>

#include <holoscan/core/execution_context.hpp>

namespace endoscopy_tool_tracking {

// Shared by terminal consumers. Completion is requested only after every active lossless
// branch has consumed every frame; stopping at source EOF would discard queued work.
struct RunProgress {
  std::uint32_t expected{};
  bool inference{};
  bool cae_policy{};
  std::atomic<std::uint32_t> consumed{};
  std::atomic<std::uint32_t> tracked{};
  std::atomic<std::uint32_t> policies{};

  void finish_if_ready(holoscan::ExecutionContext& context) const {
    if (consumed.load() == expected && (!inference || tracked.load() == expected) &&
        (!cae_policy || policies.load() == expected)) {
      context.request_completion();
    }
  }
};

}  // namespace endoscopy_tool_tracking
