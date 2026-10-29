# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)
project(bci_visualization_apps LANGUAGES NONE)
include(CTest)

option(HOLOHUB_BUILD_PYTHON "Build support for Python" ON)

if(BUILD_TESTING)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_test(NAME bci_visualization_data_preparation
    COMMAND ${Python3_EXECUTABLE} -m unittest discover
      -s ${CMAKE_CURRENT_SOURCE_DIR}/tests -p test_prepare_data.py -v)
  set_tests_properties(bci_visualization_data_preparation PROPERTIES
    LABELS "bci_visualization;data" TIMEOUT 60)
endif()

if(HOLOHUB_BUILD_PYTHON)
  add_subdirectory(python)
endif()
