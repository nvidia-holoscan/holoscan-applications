# SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)
project(apriltag_detection_apps LANGUAGES NONE)
include(CTest)

# Also provide the CLI language switch for standalone application builds.
option(HOLOHUB_BUILD_CPP "Build support for C++" ON)

# The webcam's YUYV conversion is header-only and needs no SDK or GPU. Keep its
# test outside the C++ dependency scope for hosted CI.
if(BUILD_TESTING)
  enable_language(CXX)
  add_executable(apriltag_yuyv_test tests/yuyv_test.cpp)
  target_include_directories(apriltag_yuyv_test PRIVATE cpp tests)
  target_compile_features(apriltag_yuyv_test PRIVATE cxx_std_20)
  add_test(NAME apriltag_yuyv COMMAND apriltag_yuyv_test)
  set_tests_properties(apriltag_yuyv PROPERTIES LABELS "apriltag" TIMEOUT 60)
endif()

if(HOLOHUB_BUILD_CPP)
  add_subdirectory(cpp)
endif()

install(FILES README.md LICENSE DESTINATION share/apriltag_detection)
