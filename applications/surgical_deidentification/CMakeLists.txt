# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)
project(surgical_deidentification_apps LANGUAGES NONE)
include(CTest)

option(HOLOHUB_BUILD_CPP "Build support for C++" ON)

if(HOLOHUB_BUILD_CPP)
    add_subdirectory(cpp)
endif()

# CPU-only operator unit tests (schema / YAML round-trip). Build whenever CTest is enabled
# so they are reachable by hosted CI, which runs with HOLOHUB_BUILD_CPP=OFF and does not
# otherwise configure the cpp/ subdirectory.
if(BUILD_TESTING AND NOT HOLOHUB_BUILD_CPP)
    enable_language(CXX)
    find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS inference)
    find_package(yaml-cpp REQUIRED)
    add_subdirectory(video_file_reader "${CMAKE_CURRENT_BINARY_DIR}/video_file_reader_ci")
    add_subdirectory(video_file_writer "${CMAKE_CURRENT_BINARY_DIR}/video_file_writer_ci")
endif()

install(FILES README.md LICENSE DESTINATION share/surgical_deidentification)
