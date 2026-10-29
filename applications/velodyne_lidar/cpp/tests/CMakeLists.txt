# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(_velodyne_cpp ${CMAKE_CURRENT_SOURCE_DIR}/..)

# Host-only graphs: no GPU work, but the SDK runtime still needs a CUDA driver.
add_executable(velodyne_sources_test sources_test.cpp)
target_compile_features(velodyne_sources_test PRIVATE cxx_std_20)
target_include_directories(velodyne_sources_test PRIVATE ${_velodyne_cpp})
target_link_libraries(velodyne_sources_test PRIVATE holoscan::core holoscan::sensor_io)

add_executable(velodyne_decode_kernel_test decode_kernel_test.cu ${_velodyne_cpp}/velodyne_cloud.cu)
target_compile_features(velodyne_decode_kernel_test PRIVATE cxx_std_20 cuda_std_20)
target_include_directories(velodyne_decode_kernel_test PRIVATE ${_velodyne_cpp})
target_link_libraries(velodyne_decode_kernel_test PRIVATE
  holoscan::core holoscan::sensor_io CUDA::cudart)

add_test(NAME velodyne_sources COMMAND velodyne_sources_test)
add_test(NAME velodyne_decode_kernel COMMAND velodyne_decode_kernel_test)
add_test(NAME velodyne_application
  COMMAND ${CMAKE_COMMAND} -E env VELODYNE_EXECUTABLE=$<TARGET_FILE:velodyne_lidar>
    ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../../python/tests/test_application.py)
set_tests_properties(velodyne_sources PROPERTIES LABELS "velodyne" TIMEOUT 120)
set_tests_properties(velodyne_decode_kernel PROPERTIES
  LABELS "velodyne;gpu" TIMEOUT 60 SKIP_RETURN_CODE 77)
set_tests_properties(velodyne_application PROPERTIES LABELS "velodyne;gpu" TIMEOUT 300)
