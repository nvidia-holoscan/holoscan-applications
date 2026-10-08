/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#include "lstm_inference_v5.hpp"

#include <NvInferPlugin.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/logger/logger.hpp>

namespace endoscopy_tool_tracking {
namespace {

constexpr std::string_view kInputVideo{"data_ph:0"};
constexpr std::string_view kInputCell{"cellstate_ph:0"};
constexpr std::string_view kInputHidden{"hiddenstate_ph:0"};
constexpr std::string_view kOutputCell{"Model/net_states:0"};
constexpr std::string_view kOutputHidden{"Model/net_hidden:0"};
constexpr std::string_view kOutputProbs{"probs:0"};
constexpr std::string_view kOutputCoords{"Localize/scaled_coords:0"};
constexpr std::string_view kOutputMasks{"Localize_1/binary_masks:0"};

constexpr std::size_t kClassCount = 7U;
constexpr std::size_t kMaskHeight = 60U;
constexpr std::size_t kMaskWidth = 107U;
constexpr std::size_t kStateElements = kClassCount * kMaskHeight * kMaskWidth;
constexpr std::size_t kStateBytes = kStateElements * sizeof(float);

[[nodiscard]] holoscan::Error cuda_error(cudaError_t error, std::string_view operation) {
  return holoscan::Error{holoscan::ErrorCode::kFailure,
                         std::string(operation) + ": " + cudaGetErrorString(error)};
}

template <typename T>
void destroy_trt(T*& object) noexcept {
  delete object;
  object = nullptr;
}

[[nodiscard]] bool read_binary(const std::filesystem::path& path, std::vector<char>& bytes) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return false;
  }
  bytes.assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
  return !bytes.empty();
}

// Deterministic content fingerprint for cache invalidation, not a security primitive.
[[nodiscard]] std::uint64_t model_fingerprint(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("cannot read ONNX model: " + path.string());
  }
  std::uint64_t hash = 14695981039346656037ULL;
  std::array<char, 65536> chunk{};
  while (input.read(chunk.data(), chunk.size()) || input.gcount() != 0) {
    for (std::streamsize i = 0; i < input.gcount(); ++i) {
      hash = (hash ^ static_cast<unsigned char>(chunk[i])) * 1099511628211ULL;
    }
  }
  if (!input.eof()) {
    throw std::runtime_error("failed reading ONNX model: " + path.string());
  }
  return hash;
}

[[nodiscard]] std::string cache_filename(const LstmInferenceParams& p,
                                         const cudaDeviceProp& properties) {
  std::uint64_t gpu = 14695981039346656037ULL;
  for (const unsigned char character : std::string_view{properties.name}) {
    gpu = (gpu ^ character) * 1099511628211ULL;
  }
  std::ostringstream name;
  name << std::filesystem::path(p.model_path).stem().string() << "_model" << std::hex
       << model_fingerprint(p.model_path) << "_gpu" << gpu << std::dec << "_sm" << properties.major
       << properties.minor << "_n" << properties.multiProcessorCount << "_trt" << NV_TENSORRT_MAJOR
       << '.' << NV_TENSORRT_MINOR << '.' << NV_TENSORRT_PATCH << '.' << NV_TENSORRT_BUILD << "_fp"
       << (p.enable_fp16 ? 16 : 32) << "_ws" << p.max_workspace_size << ".engine";
  return name.str();
}

void write_binary(const std::filesystem::path& path, const void* data, std::size_t bytes) {
  // Exclusive temporary creation and atomic rename prevent readers observing a
  // partial engine when multiple app instances share the cache directory.
  std::string temporary = path.string() + ".tmp.XXXXXX";
  const int descriptor = mkstemp(temporary.data());
  if (descriptor < 0) {
    throw std::runtime_error("failed to create engine cache temporary: " +
                             std::string(std::strerror(errno)));
  }
  bool open = true;
  try {
    const auto* cursor = static_cast<const char*>(data);
    while (bytes != 0) {
      const ssize_t written = ::write(descriptor, cursor, bytes);
      if (written < 0 && errno == EINTR) {
        continue;
      }
      if (written <= 0) {
        throw std::runtime_error("failed to write TensorRT engine cache: " + path.string());
      }
      cursor += written;
      bytes -= static_cast<std::size_t>(written);
    }
    open = false;
    if (::close(descriptor) != 0) {
      throw std::runtime_error("failed to close TensorRT engine cache: " + path.string());
    }
    std::filesystem::rename(temporary, path);
  } catch (...) {
    if (open) {
      (void)::close(descriptor);
    }
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    throw;
  }
}

void validate_engine(nvinfer1::ICudaEngine& engine) {
  struct Binding {
    std::string_view name;
    nvinfer1::TensorIOMode mode;
    std::vector<std::int64_t> shape;
  };
  const std::array<Binding, 8> bindings{{
      {kInputVideo, nvinfer1::TensorIOMode::kINPUT, {1, 480, 854, 3}},
      {kInputCell, nvinfer1::TensorIOMode::kINPUT, {1, 60, 107, 7}},
      {kInputHidden, nvinfer1::TensorIOMode::kINPUT, {1, 60, 107, 7}},
      {kOutputCell, nvinfer1::TensorIOMode::kOUTPUT, {1, 60, 107, 7}},
      {kOutputHidden, nvinfer1::TensorIOMode::kOUTPUT, {1, 60, 107, 7}},
      {kOutputProbs, nvinfer1::TensorIOMode::kOUTPUT, {1, 7}},
      {kOutputCoords, nvinfer1::TensorIOMode::kOUTPUT, {1, 7, 2}},
      {kOutputMasks, nvinfer1::TensorIOMode::kOUTPUT, {1, 7, 60, 107}},
  }};
  if (engine.getNbIOTensors() != static_cast<int>(bindings.size())) {
    throw std::runtime_error("LSTM engine must have exactly three inputs and five outputs");
  }
  for (const auto& binding : bindings) {
    const char* name = binding.name.data();
    if (engine.getTensorIOMode(name) != binding.mode ||
        engine.getTensorDataType(name) != nvinfer1::DataType::kFLOAT ||
        engine.getTensorLocation(name) != nvinfer1::TensorLocation::kDEVICE ||
        engine.getTensorFormat(name) != nvinfer1::TensorFormat::kLINEAR) {
      throw std::runtime_error("incompatible TensorRT LSTM binding: " + std::string(name));
    }
    const auto dimensions = engine.getTensorShape(name);
    if (dimensions.nbDims != static_cast<int>(binding.shape.size())) {
      throw std::runtime_error("incompatible TensorRT LSTM rank: " + std::string(name));
    }
    for (int axis = 0; axis < dimensions.nbDims; ++axis) {
      if (dimensions.d[axis] != -1 && dimensions.d[axis] != binding.shape[axis]) {
        throw std::runtime_error("incompatible TensorRT LSTM shape: " + std::string(name));
      }
    }
  }
  std::unique_ptr<nvinfer1::IExecutionContext> validation{engine.createExecutionContext()};
  if (!validation) {
    throw std::runtime_error("failed to create TensorRT validation context");
  }
  for (const auto& binding : bindings) {
    const char* name = binding.name.data();
    if (binding.mode == nvinfer1::TensorIOMode::kINPUT) {
      nvinfer1::Dims dimensions{};
      dimensions.nbDims = static_cast<int>(binding.shape.size());
      std::copy(binding.shape.begin(), binding.shape.end(), dimensions.d);
      if (!validation->setInputShape(name, dimensions)) {
        throw std::runtime_error("incompatible TensorRT LSTM input shape: " + std::string(name));
      }
    }
  }
  for (const auto& binding : bindings) {
    const auto dimensions = validation->getTensorShape(binding.name.data());
    if (dimensions.nbDims != static_cast<int>(binding.shape.size()) ||
        !std::equal(binding.shape.begin(), binding.shape.end(), dimensions.d)) {
      throw std::runtime_error("incompatible TensorRT LSTM shape: " + std::string(binding.name));
    }
  }
}

}  // namespace

class LstmInferenceOp::Logger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kERROR) {
      HOLOSCAN_LOG_ERROR("TensorRT: {}", message);
    } else if (severity == Severity::kWARNING) {
      HOLOSCAN_LOG_WARN("TensorRT: {}", message);
    } else if (severity == Severity::kINFO) {
      HOLOSCAN_LOG_INFO("TensorRT: {}", message);
    }
  }
};

LstmInferenceOp::LstmInferenceOp() : logger_(std::make_unique<Logger>()) {}

LstmInferenceOp::~LstmInferenceOp() { release(); }

void LstmInferenceOp::setup(holoscan::OperatorSpec& spec) {
  const auto device_float = [](std::uint32_t rank) {
    return holoscan::TensorRepresentation{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = kFloat32,
        .rank = rank,
        .strides = holoscan::TensorStridePolicy::kContiguousOnly,
    };
  };
  spec.input(source_video, "source_video")
      .queue_depth(2U)
      .expects_tensor(holoscan::TensorInputSpec{.representation = device_float(4U)});
  spec.output(probs, "probs")
      .produces_tensor(holoscan::TensorOutputSpec{
          .representation = device_float(2U),
          .bounds = holoscan::tensor_bounds(kClassCount * sizeof(float)),
          .storage = holoscan::TensorOutputStorage::kRuntimePool,
      });
  spec.output(scaled_coords, "scaled_coords")
      .produces_tensor(holoscan::TensorOutputSpec{
          .representation = device_float(3U),
          .bounds = holoscan::tensor_bounds(kClassCount * 2U * sizeof(float)),
          .storage = holoscan::TensorOutputStorage::kRuntimePool,
      });
  spec.output(binary_masks, "binary_masks")
      .produces_tensor(holoscan::TensorOutputSpec{
          .representation = device_float(4U),
          .bounds = holoscan::tensor_bounds(kStateBytes),
          .storage = holoscan::TensorOutputStorage::kRuntimePool,
      });
}

holoscan::Contract LstmInferenceOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{source_video});
  return result;
}

std::string LstmInferenceOp::engine_cache_path() const {
  cudaDeviceProp properties{};
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
    throw std::runtime_error("failed to query CUDA device for TensorRT cache identity");
  }
  return (std::filesystem::path(params().engine_cache_dir) / cache_filename(params(), properties))
      .string();
}

void LstmInferenceOp::load_or_build_engine() {
  const auto& p = params();
  if (!std::filesystem::is_regular_file(p.model_path)) {
    throw std::runtime_error("LSTM ONNX model not found: " + p.model_path);
  }
  std::error_code ec;
  std::filesystem::create_directories(p.engine_cache_dir, ec);
  if (ec) {
    throw std::runtime_error("failed to create TensorRT engine cache directory");
  }

  if (!initLibNvInferPlugins(logger_.get(), "")) {
    throw std::runtime_error("failed to initialize TensorRT plugins");
  }
  const std::filesystem::path cache_path = engine_cache_path();
  std::vector<char> serialized;
  if (!p.force_engine_update && read_binary(cache_path, serialized)) {
    runtime_ = nvinfer1::createInferRuntime(*logger_);
    if (runtime_ != nullptr) {
      engine_ = runtime_->deserializeCudaEngine(serialized.data(), serialized.size());
    }
    if (engine_ != nullptr) {
      try {
        validate_engine(*engine_);
        HOLOSCAN_LOG_INFO("Loaded TensorRT engine {}", cache_path.string());
        return;
      } catch (const std::exception& error) {
        HOLOSCAN_LOG_WARN("Rebuilding incompatible cached engine: {}", error.what());
      }
    }
  }
  destroy_trt(engine_);
  destroy_trt(runtime_);
  HOLOSCAN_LOG_INFO("Building TensorRT engine from {} (this may take several minutes)",
                    p.model_path);

  std::unique_ptr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(*logger_)};
  if (!builder) {
    throw std::runtime_error("failed to create TensorRT builder");
  }
#if NV_TENSORRT_MAJOR >= 10
  constexpr std::uint32_t kNetworkFlags = 0U;
#else
  constexpr std::uint32_t kNetworkFlags =
      1U << static_cast<std::uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
#endif
  std::unique_ptr<nvinfer1::INetworkDefinition> network{builder->createNetworkV2(kNetworkFlags)};
  std::unique_ptr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};
  if (!network || !config) {
    throw std::runtime_error("failed to create TensorRT network/configuration");
  }
  std::unique_ptr<nvonnxparser::IParser> parser{nvonnxparser::createParser(*network, *logger_)};
  if (!parser || !parser->parseFromFile(p.model_path.c_str(),
                                        static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
    throw std::runtime_error("failed to parse LSTM ONNX model");
  }
  if (network->getNbInputs() != 3 || network->getNbOutputs() != 5) {
    throw std::runtime_error("LSTM model must have exactly three inputs and five outputs");
  }
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, p.max_workspace_size);
  if (p.enable_fp16) {
    // Keep the supported TensorRT 10 flag despite its deprecation, as HoloInfer does.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
#pragma GCC diagnostic pop
  }

  // TensorRT's builder owns profiles. Supply the actual supported dimensions,
  // including dynamic spatial axes, rather than replacing every wildcard by 1.
  nvinfer1::IOptimizationProfile* profile = nullptr;
  for (int index = 0; index < network->getNbInputs(); ++index) {
    const nvinfer1::ITensor* input = network->getInput(index);
    const std::string_view name{input->getName()};
    const bool video = name == kInputVideo;
    if (!video && name != kInputCell && name != kInputHidden) {
      throw std::runtime_error("unexpected LSTM input binding: " + std::string(name));
    }
    const nvinfer1::Dims4 expected =
        video ? nvinfer1::Dims4{1, 480, 854, 3} : nvinfer1::Dims4{1, 60, 107, 7};
    const auto dims = input->getDimensions();
    if (dims.nbDims != expected.nbDims) {
      throw std::runtime_error("unexpected LSTM input rank: " + std::string(name));
    }
    bool dynamic = false;
    for (int axis = 0; axis < dims.nbDims; ++axis) {
      dynamic |= dims.d[axis] == -1;
      if (dims.d[axis] != -1 && dims.d[axis] != expected.d[axis]) {
        throw std::runtime_error("unexpected LSTM input dimensions: " + std::string(name));
      }
    }
    if (!dynamic) {
      continue;
    }
    if (profile == nullptr) {
      profile = builder->createOptimizationProfile();
    }
    if (profile == nullptr ||
        !profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, expected) ||
        !profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, expected) ||
        !profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, expected)) {
      throw std::runtime_error("failed to configure LSTM optimization profile");
    }
  }
  if (profile != nullptr && config->addOptimizationProfile(profile) < 0) {
    throw std::runtime_error("failed to add LSTM optimization profile");
  }

  std::unique_ptr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
  if (!plan || plan->data() == nullptr || plan->size() == 0U) {
    throw std::runtime_error("failed to build TensorRT engine");
  }
  runtime_ = nvinfer1::createInferRuntime(*logger_);
  engine_ =
      runtime_ != nullptr ? runtime_->deserializeCudaEngine(plan->data(), plan->size()) : nullptr;
  if (engine_ == nullptr) {
    throw std::runtime_error("failed to deserialize newly built TensorRT engine");
  }
  validate_engine(*engine_);
  write_binary(cache_path, plan->data(), plan->size());
}

void LstmInferenceOp::start() try {
  release();
  load_or_build_engine();
  execution_context_ = engine_->createExecutionContext();
  if (execution_context_ == nullptr) {
    release();
    throw holoscan::RuntimeError{holoscan::ErrorCode::kFailure,
                                 "failed to create TensorRT execution context"};
  }
  for (std::size_t index = 0; index < 2U; ++index) {
    if (cudaError_t error = cudaMalloc(&cell_state_[index], kStateBytes); error != cudaSuccess) {
      release();
      throw holoscan::RuntimeError{cuda_error(error, "allocate LSTM cell state")};
    }
    if (cudaError_t error = cudaMalloc(&hidden_state_[index], kStateBytes); error != cudaSuccess) {
      release();
      throw holoscan::RuntimeError{cuda_error(error, "allocate LSTM hidden state")};
    }
  }
  current_state_ = 0U;
  state_initialized_ = false;
} catch (...) {
  release();
  throw;
}

void LstmInferenceOp::release() noexcept {
  state_initialized_ = false;
  for (void*& state : cell_state_) {
    if (state != nullptr) {
      (void)cudaFree(state);
      state = nullptr;
    }
  }
  for (void*& state : hidden_state_) {
    if (state != nullptr) {
      (void)cudaFree(state);
      state = nullptr;
    }
  }
  destroy_trt(execution_context_);
  destroy_trt(engine_);
  destroy_trt(runtime_);
}

void LstmInferenceOp::stop() { release(); }

holoscan::expected<void, holoscan::Error> LstmInferenceOp::compute(
    holoscan::ExecutionContext& context) {
  auto sample = source_video.receive_data();
  if (!sample) {
    return holoscan::make_unexpected(std::move(sample).error());
  }
  const holoscan::Tensor& video = *sample;
  const auto shape = video.shape_span();
  if (shape.size() != 4U || shape[0] != 1 || shape[1] != 480 || shape[2] != 854 || shape[3] != 3 ||
      !video.is_contiguous() || video.data_as<float>() == nullptr) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kInvalidArgument,
                        "LstmInferenceOp expects CUDA float32 [1,480,854,3]"});
  }

  const std::array<std::int64_t, 2> probs_shape{1, 7};
  const std::array<std::int64_t, 3> coords_shape{1, 7, 2};
  const std::array<std::int64_t, 4> masks_shape{1, 7, 60, 107};
  auto probs_loan =
      probs.allocate_tensor(holoscan::TensorLoanRequest{.shape = probs_shape, .dtype = kFloat32});
  auto coords_loan = scaled_coords.allocate_tensor(
      holoscan::TensorLoanRequest{.shape = coords_shape, .dtype = kFloat32});
  auto masks_loan = binary_masks.allocate_tensor(
      holoscan::TensorLoanRequest{.shape = masks_shape, .dtype = kFloat32});
  if (!probs_loan || !coords_loan || !masks_loan) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kResourceExhausted,
                        "failed to allocate LSTM inference output tensors"});
  }
  auto probs_writer = probs_loan->write(context.cuda_stream());
  auto coords_writer = coords_loan->write(context.cuda_stream());
  auto masks_writer = masks_loan->write(context.cuda_stream());
  if (!probs_writer || !coords_writer || !masks_writer) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kResourceExhausted,
                        "failed to acquire LSTM inference output writers"});
  }

  if (!state_initialized_) {
    // Initialize only the first input state on the same stream as inference.
    // Subsequent activations reuse this partition's ordered runtime stream.
    for (void* state : {cell_state_[current_state_], hidden_state_[current_state_]}) {
      if (const auto error = cudaMemsetAsync(state, 0, kStateBytes, context.cuda_stream().get());
          error != cudaSuccess) {
        return holoscan::make_unexpected(cuda_error(error, "initialize recurrent state"));
      }
    }
    state_initialized_ = true;
  }
  const std::uint32_t next_state = 1U - current_state_;
  const nvinfer1::Dims4 video_dims{1, 480, 854, 3};
  const nvinfer1::Dims4 state_dims{1, 60, 107, 7};
  if (!execution_context_->setInputShape(kInputVideo.data(), video_dims) ||
      !execution_context_->setInputShape(kInputCell.data(), state_dims) ||
      !execution_context_->setInputShape(kInputHidden.data(), state_dims) ||
      !execution_context_->setTensorAddress(kInputVideo.data(), const_cast<void*>(video.data())) ||
      !execution_context_->setTensorAddress(kInputCell.data(), cell_state_[current_state_]) ||
      !execution_context_->setTensorAddress(kInputHidden.data(), hidden_state_[current_state_]) ||
      !execution_context_->setTensorAddress(kOutputCell.data(), cell_state_[next_state]) ||
      !execution_context_->setTensorAddress(kOutputHidden.data(), hidden_state_[next_state]) ||
      !execution_context_->setTensorAddress(kOutputProbs.data(), probs_writer->data()) ||
      !execution_context_->setTensorAddress(kOutputCoords.data(), coords_writer->data()) ||
      !execution_context_->setTensorAddress(kOutputMasks.data(), masks_writer->data())) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kFailure, "failed to bind LSTM TensorRT tensors"});
  }
  if (!execution_context_->enqueueV3(context.cuda_stream().get())) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kFailure, "TensorRT LSTM enqueue failed"});
  }
  current_state_ = next_state;

  auto probs_committed = std::move(*probs_writer).commit();
  auto coords_committed = std::move(*coords_writer).commit();
  auto masks_committed = std::move(*masks_writer).commit();
  if (!probs_committed || !coords_committed || !masks_committed) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kFailure, "failed to commit LSTM outputs"});
  }
  const holoscan::EmitOptions emit{.capture_time = context.activation_time()};
  if (auto result = probs.emit(std::move(*probs_loan), emit); !result) {
    return result;
  }
  if (auto result = scaled_coords.emit(std::move(*coords_loan), emit); !result) {
    return result;
  }
  return binary_masks.emit(std::move(*masks_loan), emit);
}

}  // namespace endoscopy_tool_tracking
