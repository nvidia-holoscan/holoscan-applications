// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <atomic>
#include <cstdint>

#include <holoscan/core/execution_context.hpp>

namespace endoscopy_tool_tracking {

// Shared by terminal consumers. Completion is requested only after both lossless
// branches have consumed every frame; stopping at source EOF would discard queued work.
struct RunProgress {
  std::uint32_t expected{};
  bool inference{};
  std::atomic<std::uint32_t> consumed{};
  std::atomic<std::uint32_t> tracked{};

  void finish_if_ready(holoscan::ExecutionContext& context) const {
    if (consumed.load() == expected && (!inference || tracked.load() == expected)) {
      context.request_completion();
    }
  }
};

}  // namespace endoscopy_tool_tracking
