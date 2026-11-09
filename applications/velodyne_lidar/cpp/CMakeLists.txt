# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)

# Select the current GPU unless the caller supplied an architecture list.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to compile")
endif()
project(velodyne_lidar LANGUAGES CXX CUDA)

option(VELODYNE_ENABLE_VISUALIZATION "Build the Holoscan Visualization display" ON)

# Honor the CLI's SDK location for direct CMake and CTest consumers.
if(DEFINED ENV{HOLOSCAN_SDK_ROOT})
  list(PREPEND CMAKE_PREFIX_PATH "$ENV{HOLOSCAN_SDK_ROOT}")
elseif(DEFINED ENV{HOLOSCAN_LIB_PATH})
  get_filename_component(_velodyne_sdk_root "$ENV{HOLOSCAN_LIB_PATH}" DIRECTORY)
  list(PREPEND CMAKE_PREFIX_PATH "${_velodyne_sdk_root}")
endif()
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS core sensor_io)
find_package(CUDAToolkit REQUIRED)
if(VELODYNE_ENABLE_VISUALIZATION)
  find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS visualization)
endif()

add_executable(velodyne_lidar main.cpp velodyne_cloud.cu point_cloud_render.cu)
target_compile_features(velodyne_lidar PRIVATE cxx_std_20 cuda_std_20)
target_link_libraries(velodyne_lidar PRIVATE holoscan::core holoscan::sensor_io CUDA::cudart)
target_compile_definitions(velodyne_lidar PRIVATE
  VELODYNE_ENABLE_VISUALIZATION=$<BOOL:${VELODYNE_ENABLE_VISUALIZATION}>)
if(VELODYNE_ENABLE_VISUALIZATION)
  target_link_libraries(velodyne_lidar PRIVATE holoscan::visualization::op_holoviz)
endif()
set_target_properties(velodyne_lidar PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/.."
  CXX_EXTENSIONS OFF CUDA_EXTENSIONS OFF)
install(TARGETS velodyne_lidar RUNTIME DESTINATION bin)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
