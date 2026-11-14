# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Prefer a system/package-installed GoogleTest; fall back to FetchContent only when neither
# gtest_main nor gmock target is already available. Avoids re-downloading GoogleTest on every
# clean build when a system install exists. See
# https://cmake.org/cmake/help/latest/module/FindGTest.html.
find_package(GTest 1.17 QUIET)
if(NOT TARGET GTest::gtest_main OR NOT TARGET GTest::gmock)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(googletest)
endif()

# Needed by gtest_discover_tests() below.
include(GoogleTest)

# ---------------------------------------------------------------------------------------------
# CPU unit tests — exercise VideoFileWriterOpParams::from_yaml_node() without the Fragment /
# FFmpeg runtime. Runs anywhere a yaml-cpp + GTest build works.
# ---------------------------------------------------------------------------------------------
add_executable(video_file_writer_yaml_test
    test_video_file_writer_yaml.cpp
)
target_link_libraries(video_file_writer_yaml_test
    PRIVATE
        op_video_file_writer
        yaml-cpp::yaml-cpp
        GTest::gtest_main
        GTest::gmock
)
gtest_discover_tests(video_file_writer_yaml_test
    DISCOVERY_MODE PRE_TEST
    PROPERTIES
        LABELS "cpu;unit;video_file_writer"
        TIMEOUT 60
)

# ---------------------------------------------------------------------------------------------
# Runtime smoke test — builds a Fragment, encodes a short mpeg4 clip, probes it with libav.
# ---------------------------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(TEST_AVFORMAT REQUIRED IMPORTED_TARGET libavformat)
pkg_check_modules(TEST_AVCODEC  REQUIRED IMPORTED_TARGET libavcodec)

add_executable(video_file_writer_runtime_test
    test_video_file_writer_runtime.cpp
)
target_link_libraries(video_file_writer_runtime_test
    PRIVATE
        op_video_file_writer
        holoscan::core
        CUDA::cudart
        PkgConfig::TEST_AVFORMAT
        PkgConfig::TEST_AVCODEC
        GTest::gtest_main
        GTest::gmock
)
gtest_discover_tests(video_file_writer_runtime_test
    DISCOVERY_MODE PRE_TEST
    PROPERTIES
        LABELS "gpu;runtime;video_file_writer"
        RESOURCE_LOCK "video_writer_tmpdir"
        TIMEOUT 120
)
