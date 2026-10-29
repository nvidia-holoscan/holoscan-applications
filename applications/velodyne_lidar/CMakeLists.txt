# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)
project(velodyne_lidar_apps LANGUAGES NONE)
include(CTest)

# Also provide the CLI language switch for standalone application builds.
option(HOLOHUB_BUILD_CPP "Build support for C++" ON)

# The pcap fixture tests run without the SDK, CUDA, or a sensor, so hosted CI can run them.
if(BUILD_TESTING)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_test(NAME velodyne_pcap_fixture
    COMMAND ${Python3_EXECUTABLE} -m unittest discover
      -s ${CMAKE_CURRENT_SOURCE_DIR}/tests -p test_make_pcap.py -v)
  set_tests_properties(velodyne_pcap_fixture PROPERTIES LABELS "velodyne;data" TIMEOUT 60)
endif()

if(HOLOHUB_BUILD_CPP)
  add_subdirectory(cpp)
  if(BUILD_TESTING)
    add_subdirectory(tests)
  endif()
endif()

install(FILES README.md make_pcap.py vlp_inspect.py DESTINATION share/velodyne_lidar)
