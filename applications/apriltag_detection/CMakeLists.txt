# SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)
project(apriltag_detection_apps LANGUAGES NONE)
include(CTest)

# Also provide the CLI language switch for standalone application builds.
option(HOLOHUB_BUILD_CPP "Build support for C++" ON)

# The webcam's YUYV conversion and the overlay's axis drawing are header-only and
# need no SDK or GPU. Keep their tests outside the C++ dependency scope for hosted CI.
if(BUILD_TESTING)
  enable_language(CXX)
  foreach(_test IN ITEMS yuyv tag_axes)
    add_executable(apriltag_${_test}_test tests/${_test}_test.cpp)
    target_include_directories(apriltag_${_test}_test PRIVATE cpp tests)
    target_compile_features(apriltag_${_test}_test PRIVATE cxx_std_20)
    add_test(NAME apriltag_${_test} COMMAND apriltag_${_test}_test)
    set_tests_properties(apriltag_${_test} PROPERTIES LABELS "apriltag" TIMEOUT 60)
  endforeach()
endif()

if(HOLOHUB_BUILD_CPP)
  add_subdirectory(cpp)
endif()

install(FILES README.md LICENSE DESTINATION share/apriltag_detection)
