// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// RenderSnapshotOp: consumes the frames Holoviz renders offscreen (--display
// headless), counts them, and writes the last one to a PPM file when asked.
// It lets the display path run, and be checked, without a screen.

#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace tag_detection {

struct RenderSnapshotConfig {
  // Where to write the last rendered frame; none when empty.
  std::filesystem::path output;
  // Frames rendered so far; read by main() after the run.
  std::shared_ptr<std::atomic<std::uint32_t>> rendered;
};

class RenderSnapshotOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if `rendered` is missing.
  explicit RenderSnapshotOp(RenderSnapshotConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // RGBA8 frames in CUDA device memory, from a HolovizOp with a CUDA sink.
  holoscan::Input<holoscan::schema::ImageT> frame;

 private:
  RenderSnapshotConfig config_;
  std::vector<std::uint8_t> last_;  // RGBA, rows packed.
  int width_{};
  int height_{};
};

}  // namespace tag_detection
