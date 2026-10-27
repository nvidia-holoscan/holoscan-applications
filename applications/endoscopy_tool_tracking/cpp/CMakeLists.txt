# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)

# Select the current GPU unless the caller supplied an architecture list.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to compile")
endif()
project(endoscopy_tool_tracking LANGUAGES CXX CUDA)

option(ENDOSCOPY_ENABLE_VISUALIZATION "Build the Holoscan Visualization display" ON)
set(ENDOSCOPY_DATA_DIR "" CACHE PATH "Prepared sample data for real-model integration tests")

# The wheel's CTest driver does not forward --local-sdk-root into CMake.
# Also honor the standard SDK environment variable for direct CMake consumers.
if(DEFINED ENV{HOLOSCAN_SDK_ROOT})
  list(PREPEND CMAKE_PREFIX_PATH "$ENV{HOLOSCAN_SDK_ROOT}")
elseif(DEFINED ENV{HOLOSCAN_LIB_PATH})
  # The CLI mounts the supplied SDK and exports its library path; derive the SDK
  # root from it when HOLOSCAN_SDK_ROOT is not set in the recursive command.
  get_filename_component(_endoscopy_sdk_root "$ENV{HOLOSCAN_LIB_PATH}" DIRECTORY)
  list(PREPEND CMAKE_PREFIX_PATH "${_endoscopy_sdk_root}")
endif()
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS core sensor_io)
find_package(CUDAToolkit REQUIRED)
if(ENDOSCOPY_ENABLE_VISUALIZATION)
  find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS visualization)
endif()

find_path(TENSORRT_INCLUDE_DIR NvInfer.h REQUIRED
  PATH_SUFFIXES x86_64-linux-gnu aarch64-linux-gnu)
find_library(TENSORRT_LIBRARY nvinfer REQUIRED)
find_library(TENSORRT_PLUGIN_LIBRARY nvinfer_plugin REQUIRED)
find_library(TENSORRT_ONNX_LIBRARY nvonnxparser REQUIRED)

add_executable(endoscopy_tool_tracking
  main.cpp lstm_inference_v5.cpp video_preprocessor.cu tool_tracking_postprocessor_v5.cu)
target_compile_features(endoscopy_tool_tracking PRIVATE cxx_std_20 cuda_std_20)
target_include_directories(endoscopy_tool_tracking PRIVATE ${TENSORRT_INCLUDE_DIR})
target_link_libraries(endoscopy_tool_tracking PRIVATE holoscan::core holoscan::sensor_io CUDA::cudart
  ${TENSORRT_LIBRARY} ${TENSORRT_PLUGIN_LIBRARY} ${TENSORRT_ONNX_LIBRARY})
target_compile_definitions(endoscopy_tool_tracking PRIVATE
  ENDOSCOPY_ENABLE_VISUALIZATION=$<BOOL:${ENDOSCOPY_ENABLE_VISUALIZATION}>)
if(ENDOSCOPY_ENABLE_VISUALIZATION)
  target_link_libraries(endoscopy_tool_tracking PRIVATE holoscan::visualization::op_holoviz)
endif()
set_target_properties(endoscopy_tool_tracking PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/.."
  CXX_EXTENSIONS OFF CUDA_EXTENSIONS OFF)
install(TARGETS endoscopy_tool_tracking RUNTIME DESTINATION bin)

if(BUILD_TESTING)
  # Keep tests in the C++ dependency scope and preserve their build directory.
  add_subdirectory(../tests "${CMAKE_CURRENT_BINARY_DIR}/../tests")
endif()
