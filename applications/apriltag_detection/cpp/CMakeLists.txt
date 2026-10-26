# SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)

# Select the current GPU unless the caller supplied an architecture list.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to compile")
endif()
project(apriltag_detection LANGUAGES C CXX CUDA)

option(APRILTAG_DETECTION_ENABLE_VISUALIZATION "Build the Holoscan Visualization display" ON)
option(APRILTAG_DETECTION_WITH_VPI "Build the VPI backend when VPI 4 is installed" ON)

# The wheel's CTest driver does not forward --local-sdk-root into CMake.
# Also honor the standard SDK environment variable for direct CMake consumers.
if(DEFINED ENV{HOLOSCAN_SDK_ROOT})
  list(PREPEND CMAKE_PREFIX_PATH "$ENV{HOLOSCAN_SDK_ROOT}")
elseif(DEFINED ENV{HOLOSCAN_LIB_PATH})
  get_filename_component(_apriltag_sdk_root "$ENV{HOLOSCAN_LIB_PATH}" DIRECTORY)
  list(PREPEND CMAKE_PREFIX_PATH "${_apriltag_sdk_root}")
endif()
set(_apriltag_holoscan_components core sensor_io)
if(APRILTAG_DETECTION_ENABLE_VISUALIZATION)
  list(APPEND _apriltag_holoscan_components visualization)
endif()
if(BUILD_TESTING)
  # The operator tests use the SDK's operator test harness.
  list(APPEND _apriltag_holoscan_components testing)
endif()
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS ${_apriltag_holoscan_components})
find_package(CUDAToolkit REQUIRED)

# AprilTag 3 (BSD-2-Clause): an installed copy if there is one, otherwise the
# pinned release, verified by its SHA-256 and built as a static library.
find_package(apriltag 3.4 QUIET CONFIG)
if(apriltag_FOUND)
  set(APRILTAG_TARGET apriltag::apriltag)
else()
  include(FetchContent)
  # The library's build options apply to it alone; its own tests stay out of
  # this application's suite.
  block(PROPAGATE apriltag_SOURCE_DIR)
    set(BUILD_SHARED_LIBS OFF)
    set(BUILD_EXAMPLES OFF)
    set(BUILD_PYTHON_WRAPPER OFF)
    set(BUILD_TESTING OFF)
    FetchContent_Declare(apriltag
      URL https://github.com/AprilRobotics/apriltag/archive/refs/tags/v3.4.5.tar.gz
      URL_HASH SHA256=3749f48220db78c29c375a373861f4e1f4138e5dbae3c298584e96cfcc556515
      EXCLUDE_FROM_ALL)
    FetchContent_MakeAvailable(apriltag)
  endblock()
  set(APRILTAG_TARGET apriltag)
  # The library is linked statically into the installed executable; its
  # BSD-2-Clause license requires the notice to travel with it.
  install(FILES "${apriltag_SOURCE_DIR}/LICENSE.md"
    DESTINATION share/apriltag_detection/licenses RENAME apriltag-LICENSE.md)
endif()

set(APRILTAG_DETECTION_VPI OFF)
if(APRILTAG_DETECTION_WITH_VPI)
  find_package(vpi 4 QUIET CONFIG PATHS /opt/nvidia/vpi4)
  if(vpi_FOUND)
    set(APRILTAG_DETECTION_VPI ON)
  else()
    message(STATUS "apriltag_detection: VPI 4 not found; building without the VPI backend")
  endif()
endif()

# The detector library: backends, pose, synthetic scenes. No Holoscan.
add_library(apriltag_detection_core STATIC
  detector/cpu_backend.cpp
  detector/cuda_backend.cpp
  detector/detector_backend.cpp
  detector/gpu_segmenter.cu
  detector/gray_converter.cu
  detector/pnm_image.cpp
  detector/pose.cpp
  detector/quad_fit.cpp
  detector/synthetic_scene.cpp
  detector/tag_decoder.cpp
  detector/tag_family.cpp)
target_compile_features(apriltag_detection_core PUBLIC cxx_std_20 cuda_std_20)
target_include_directories(apriltag_detection_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(apriltag_detection_core PUBLIC CUDA::cudart ${APRILTAG_TARGET})
target_compile_definitions(apriltag_detection_core PUBLIC
  APRILTAG_DETECTION_WITH_VPI=$<BOOL:${APRILTAG_DETECTION_VPI}>)
if(APRILTAG_DETECTION_VPI)
  target_sources(apriltag_detection_core PRIVATE detector/vpi_backend.cpp)
  target_link_libraries(apriltag_detection_core PUBLIC vpi)
endif()
set_target_properties(apriltag_detection_core PROPERTIES
  POSITION_INDEPENDENT_CODE ON CXX_EXTENSIONS OFF CUDA_EXTENSIONS OFF)

# The detections message, generated from its FlatBuffers schema.
holoscan_add_flatbuffer_schema(
  TARGET apriltag_detections_schema
  SCHEMA operators/apriltag_detections.fbs
  MAX_SERIALIZED_SIZE 16384)

# The Holoscan operators.
add_library(apriltag_detection_operators STATIC
  operators/apriltag_detector_op.cpp
  operators/detection_report_op.cpp
  operators/image_sources.cpp)
target_include_directories(apriltag_detection_operators PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(apriltag_detection_operators PUBLIC
  apriltag_detection_core apriltag_detections_schema holoscan::core holoscan::sensor_io)
if(APRILTAG_DETECTION_ENABLE_VISUALIZATION)
  target_sources(apriltag_detection_operators PRIVATE
    operators/apriltag_overlay_op.cpp operators/render_snapshot_op.cpp)
  target_link_libraries(apriltag_detection_operators PUBLIC
    holoscan::visualization::op_holoviz)
endif()
set_target_properties(apriltag_detection_operators PROPERTIES CXX_EXTENSIONS OFF)

add_executable(apriltag_detection main.cpp)
target_link_libraries(apriltag_detection PRIVATE apriltag_detection_operators)
target_compile_definitions(apriltag_detection PRIVATE
  APRILTAG_DETECTION_ENABLE_VISUALIZATION=$<BOOL:${APRILTAG_DETECTION_ENABLE_VISUALIZATION}>)
# Place the executable in the application's build directory, where the CLI
# starts the run commands from metadata.json.
set_target_properties(apriltag_detection PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/.."
  CXX_EXTENSIONS OFF)
install(TARGETS apriltag_detection RUNTIME DESTINATION bin)

if(BUILD_TESTING)
  # Keep tests in the C++ dependency scope and preserve their build directory.
  add_subdirectory(../tests "${CMAKE_CURRENT_BINARY_DIR}/../tests")
endif()
