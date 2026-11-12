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

  # The CAE policy configuration test is host-only C++ (no SDK or CUDA), so it is registered
  # here, outside the C++ scope, and runs in hosted CI whenever a C++ compiler is available.
  include(CheckLanguage)
  check_language(CXX)
  if(CMAKE_CXX_COMPILER)
    enable_language(CXX)
    add_executable(endoscopy_cae_policy_config_test tests/cae_policy_config_test.cpp)
    target_compile_features(endoscopy_cae_policy_config_test PRIVATE cxx_std_20)
    add_test(NAME endoscopy_cae_policy_config
      COMMAND endoscopy_cae_policy_config_test ${CMAKE_CURRENT_SOURCE_DIR}/cae_policy_default.json)
    set_tests_properties(endoscopy_cae_policy_config PROPERTIES
      LABELS "endoscopy;cae" TIMEOUT 60)
  endif()
endif()

if(HOLOHUB_BUILD_CPP)
  add_subdirectory(cpp)
endif()

install(FILES README.md prepare_data.py LICENSE cae_policy_default.json
  DESTINATION share/endoscopy_tool_tracking)
