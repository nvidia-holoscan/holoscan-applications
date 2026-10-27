# SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)
project(apriltag_detection_apps LANGUAGES NONE)
include(CTest)

# Also provide the CLI language switch for standalone application builds.
option(HOLOHUB_BUILD_CPP "Build support for C++" ON)

# The overlay's axis drawing is header-only and needs no SDK or GPU. Keep its
# test outside the C++ dependency scope for hosted CI.
if(BUILD_TESTING)
  enable_language(CXX)
  add_executable(apriltag_tag_axes_test tests/tag_axes_test.cpp)
  target_include_directories(apriltag_tag_axes_test PRIVATE cpp tests)
  target_compile_features(apriltag_tag_axes_test PRIVATE cxx_std_20)
  add_test(NAME apriltag_tag_axes COMMAND apriltag_tag_axes_test)
  set_tests_properties(apriltag_tag_axes PROPERTIES LABELS "apriltag" TIMEOUT 60)
endif()

if(HOLOHUB_BUILD_CPP)
  add_subdirectory(cpp)
endif()

install(FILES README.md LICENSE DESTINATION share/apriltag_detection)
