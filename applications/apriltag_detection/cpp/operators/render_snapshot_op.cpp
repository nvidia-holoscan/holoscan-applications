// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "render_snapshot_op.hpp"

#include <cuda_runtime_api.h>

#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

#include <holoscan/core/domain/tensor.hpp>

#include "detector/pnm_image.hpp"

namespace tag_detection {

RenderSnapshotOp::RenderSnapshotOp(RenderSnapshotConfig config) : config_(std::move(config)) {
  if (!config_.rendered) {
    throw std::invalid_argument("RenderSnapshotOp needs a frame counter");
  }
}

void RenderSnapshotOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(frame, "frame").queue_depth(2U);
}

holoscan::Contract RenderSnapshotOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{frame});
  return result;
}

holoscan::expected<void, holoscan::Error> RenderSnapshotOp::compute(
    holoscan::ExecutionContext& context) {
  auto sample = frame.receive();
  if (!sample) {
    return holoscan::make_unexpected(std::move(sample).error());
  }
  config_.rendered->fetch_add(1U);
  if (config_.output.empty() || !sample->data.data) {
    return {};
  }
  const holoscan::Tensor& tensor = *sample->data.data;
  const auto shape = tensor.shape_span();
  if (shape.size() != 3 || shape[2] != 4) {
    return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kInvalidArgument,
                                                     "RenderSnapshotOp expects RGBA frames"});
  }
  height_ = static_cast<int>(shape[0]);
  width_ = static_cast<int>(shape[1]);
  last_.resize(static_cast<std::size_t>(tensor.nbytes()));
  return tensor.copy_to_host(last_.data(), last_.size(), context.cuda_stream());
}

void RenderSnapshotOp::stop() {
  if (config_.output.empty() || last_.empty()) {
    return;
  }
  PnmImage image{width_, height_, 3, {}};
  image.pixels.reserve(static_cast<std::size_t>(width_) * height_ * 3);
  for (std::size_t i = 0; i + 3 < last_.size(); i += 4) {
    image.pixels.insert(image.pixels.end(), {last_[i], last_[i + 1], last_[i + 2]});
  }
  try {
    write_pnm(config_.output, image);
  } catch (const std::exception& error) {
    std::cerr << "RenderSnapshotOp: " << error.what() << '\n';
  }
}

}  // namespace tag_detection
