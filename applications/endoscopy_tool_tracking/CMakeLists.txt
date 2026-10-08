# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)
project(endoscopy_tool_tracking_apps LANGUAGES NONE)
include(CTest)

# Also provide the CLI language switch for standalone application builds.
option(HOLOHUB_BUILD_CPP "Build support for C++" ON)

# Data-preparation tests use fixtures and run without the SDK, CUDA, or FFmpeg.
# Keep their registration outside the C++ dependency scope for hosted CI.
if(BUILD_TESTING)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_test(NAME endoscopy_data_preparation
    COMMAND ${Python3_EXECUTABLE} -m unittest discover
      -s ${CMAKE_CURRENT_SOURCE_DIR}/tests -p test_prepare_data.py -v)
  set_tests_properties(endoscopy_data_preparation PROPERTIES
    LABELS "endoscopy;data" TIMEOUT 60)
endif()

if(HOLOHUB_BUILD_CPP)
  add_subdirectory(cpp)
endif()

install(FILES README.md prepare_data.py LICENSE DESTINATION share/endoscopy_tool_tracking)
