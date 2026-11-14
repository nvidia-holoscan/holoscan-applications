# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to compile")
endif()
project(surgical_deidentification LANGUAGES CXX)

if(DEFINED ENV{HOLOSCAN_SDK_ROOT})
  list(PREPEND CMAKE_PREFIX_PATH "$ENV{HOLOSCAN_SDK_ROOT}")
elseif(DEFINED ENV{HOLOSCAN_LIB_PATH})
  get_filename_component(_sd_sdk_root "$ENV{HOLOSCAN_LIB_PATH}" DIRECTORY)
  list(PREPEND CMAKE_PREFIX_PATH "${_sd_sdk_root}")
endif()

find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS inference)
find_package(yaml-cpp REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(DEID_AVFORMAT REQUIRED IMPORTED_TARGET libavformat)
pkg_check_modules(DEID_AVCODEC  REQUIRED IMPORTED_TARGET libavcodec)
pkg_check_modules(DEID_AVUTIL   REQUIRED IMPORTED_TARGET libavutil)

if(NOT DEFINED HOLOHUB_DATA_DIR)
    set(HOLOHUB_DATA_DIR "${CMAKE_BINARY_DIR}/data"
        CACHE PATH "Root directory for application data")
endif()

option(SURGICAL_DEID_DOWNLOAD_DATA
    "Download the endoscopy_out_of_body_detection NGC resource during build" OFF)

add_subdirectory(../video_file_reader "${CMAKE_CURRENT_BINARY_DIR}/video_file_reader")
add_subdirectory(../video_file_writer "${CMAKE_CURRENT_BINARY_DIR}/video_file_writer")

add_executable(de_id ../de_id.cpp)
target_compile_features(de_id PRIVATE cxx_std_20)
target_link_libraries(de_id
    PRIVATE
        holoscan::core
        holoscan::op_inference
        holoscan::op_tensor_preprocessor
        holoscan::op_video_file_reader
        yaml-cpp::yaml-cpp
        PkgConfig::DEID_AVFORMAT
        PkgConfig::DEID_AVCODEC
        PkgConfig::DEID_AVUTIL
)
set_target_properties(de_id PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/.."
    CXX_EXTENSIONS OFF
    INSTALL_RPATH "$ORIGIN/../lib"
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

set(_sd_cfg_src "${CMAKE_CURRENT_SOURCE_DIR}/../config.yaml")
set(_sd_cfg_dst "${CMAKE_CURRENT_BINARY_DIR}/../config.yaml")
add_custom_command(
    OUTPUT  "${_sd_cfg_dst}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_sd_cfg_src}" "${_sd_cfg_dst}"
    DEPENDS "${_sd_cfg_src}"
    VERBATIM
)
add_custom_target(de_id_config DEPENDS "${_sd_cfg_dst}")
add_dependencies(de_id de_id_config)

install(TARGETS de_id op_video_file_reader op_video_file_writer
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
)
install(FILES ../config.yaml DESTINATION bin)

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/DownloadData.cmake)

if(BUILD_TESTING)
    add_subdirectory(../tests "${CMAKE_CURRENT_BINARY_DIR}/../tests")
endif()
