# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# This CMakeLists is normally pulled in via add_subdirectory() from the parent
# surgical_deidentification application. The standalone-build guard below lets `cmake -S
# ./video_file_writer` still work for operator-only development.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.20)
    project(video_file_writer CXX)
    # Holoscan SDK 5.x installs to /usr by default (/opt/nvidia/holoscan was the legacy
    # 4.x prefix). Keep both on the PATHS list for backwards compatibility with sites that
    # still ship the SDK under /opt.
    find_package(holoscan 5.0 REQUIRED CONFIG
                 PATHS "/usr" "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install"
                 COMPONENTS inference)
    find_package(yaml-cpp REQUIRED)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(AVFORMAT REQUIRED IMPORTED_TARGET libavformat)
pkg_check_modules(AVCODEC  REQUIRED IMPORTED_TARGET libavcodec)
pkg_check_modules(AVUTIL   REQUIRED IMPORTED_TARGET libavutil)
pkg_check_modules(SWSCALE  REQUIRED IMPORTED_TARGET libswscale)

add_library(op_video_file_writer SHARED
    video_file_writer.cpp
    video_file_writer.hpp
)
add_library(holoscan::op_video_file_writer ALIAS op_video_file_writer)

target_include_directories(op_video_file_writer
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(op_video_file_writer
    PUBLIC
        holoscan::core
    PRIVATE
        CUDA::cudart
        yaml-cpp::yaml-cpp
        PkgConfig::AVFORMAT
        PkgConfig::AVCODEC
        PkgConfig::AVUTIL
        PkgConfig::SWSCALE
)

set_target_properties(op_video_file_writer PROPERTIES
    OUTPUT_NAME holoscan_op_video_file_writer
    POSITION_INDEPENDENT_CODE ON
)

if(BUILD_TESTING)
    include(CTest)
    add_subdirectory(tests)
endif()
