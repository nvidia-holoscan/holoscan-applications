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
# otherwise configure the cpp/ subdirectory. Hosted runners without the Holoscan SDK installed
# skip this block entirely rather than failing configure.
if(BUILD_TESTING AND NOT HOLOHUB_BUILD_CPP)
    enable_language(CXX)
    find_package(holoscan 5.0 QUIET CONFIG COMPONENTS inference)
    find_package(yaml-cpp QUIET)
    if(holoscan_FOUND AND yaml-cpp_FOUND)
        add_subdirectory(video_file_reader "${CMAKE_CURRENT_BINARY_DIR}/video_file_reader_ci")
        add_subdirectory(video_file_writer "${CMAKE_CURRENT_BINARY_DIR}/video_file_writer_ci")
    else()
        message(STATUS
            "surgical_deidentification: Holoscan SDK or yaml-cpp not found; "
            "skipping CPU operator unit tests.")
    endif()
endif()

install(FILES README.md LICENSE DESTINATION share/surgical_deidentification)
