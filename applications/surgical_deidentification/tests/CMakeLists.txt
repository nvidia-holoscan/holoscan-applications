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
# App-level config schema smoke test. Loads the shipped config.yaml and dispatches each
# sub-section through the matching operator's from_yaml_node() helper, catching schema drift
# between the YAML and the operator param structs.
# ---------------------------------------------------------------------------------------------
add_executable(surgical_deid_config_test
    test_de_id_config.cpp
)
target_link_libraries(surgical_deid_config_test
    PRIVATE
        op_video_file_reader
        holoscan::core
        holoscan::op_inference
        holoscan::op_tensor_preprocessor
        yaml-cpp::yaml-cpp
        GTest::gtest_main
        GTest::gmock
)
target_compile_definitions(surgical_deid_config_test
    PRIVATE
        SURGICAL_DEID_CONFIG_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/../config.yaml\"
)
gtest_discover_tests(surgical_deid_config_test
    DISCOVERY_MODE PRE_TEST
    PROPERTIES
        LABELS "cpu;unit;surgical_deidentification"
        TIMEOUT 60
)

# ---------------------------------------------------------------------------------------------
# Binary smoke test — invokes the compiled de_id binary with no arguments and asserts the
# usage line, confirming the app links and loads its dependencies without crashing. Runs
# only once the de_id target exists (set as a dependency so this test rebuilds alongside).
# ---------------------------------------------------------------------------------------------
add_test(NAME surgical_deid_binary_smoke_test
         COMMAND $<TARGET_FILE:de_id>)
set_tests_properties(surgical_deid_binary_smoke_test
    PROPERTIES
        # The binary returns non-zero on missing --config and prints a usage line to stderr.
        # Match on the usage text — proves the binary ran its arg parser past the dynamic-lib
        # load step.
        PASS_REGULAR_EXPRESSION "usage: de_id --config"
        WILL_FAIL FALSE
        LABELS "cpu;smoke;surgical_deidentification"
        TIMEOUT 30
)

# ---------------------------------------------------------------------------------------------
# End-to-end functional test — invokes the actual de_id binary against the endoscopy stand-in
# model and sample video and asserts its stdout contains the classification lines emitted by
# ClassificationPrinterOp. Skips (CTest return code 77) when the endoscopy dataset is absent.
# ---------------------------------------------------------------------------------------------
if(NOT DEFINED HOLOHUB_DATA_DIR)
    set(HOLOHUB_DATA_DIR "${CMAKE_BINARY_DIR}/data")
endif()
# Paths exported by cmake/DownloadData.cmake; fall back to the stock layout when
# this test subdir is configured standalone.
if(NOT DEFINED SURGICAL_DEID_SAMPLE_MP4)
    set(SURGICAL_DEID_SAMPLE_MP4
        "${HOLOHUB_DATA_DIR}/endoscopy_out_of_body_detection/sample_clip_out_of_body_detection.mp4")
endif()
if(NOT DEFINED SURGICAL_DEID_MODEL_ONNX)
    set(SURGICAL_DEID_MODEL_ONNX
        "${HOLOHUB_DATA_DIR}/endoscopy_out_of_body_detection/out_of_body_detection.onnx")
endif()
set(ENDOSCOPY_SAMPLE_MP4_PATH "${SURGICAL_DEID_SAMPLE_MP4}")
set(ENDOSCOPY_MODEL_ONNX_PATH "${SURGICAL_DEID_MODEL_ONNX}")

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/config.test.yaml.in"
    "${CMAKE_CURRENT_BINARY_DIR}/config.test.yaml"
    @ONLY
)

add_test(
    NAME surgical_deid_binary_e2e_test
    COMMAND ${CMAKE_COMMAND}
        -DDE_ID=$<TARGET_FILE:de_id>
        -DCONFIG=${CMAKE_CURRENT_BINARY_DIR}/config.test.yaml
        -DEOB_MP4=${ENDOSCOPY_SAMPLE_MP4_PATH}
        -DEOB_ONNX=${ENDOSCOPY_MODEL_ONNX_PATH}
        -DTICKS=5
        -P ${CMAKE_CURRENT_SOURCE_DIR}/run_de_id_e2e.cmake
)
set_tests_properties(surgical_deid_binary_e2e_test
    PROPERTIES
        SKIP_RETURN_CODE 77
        PASS_REGULAR_EXPRESSION "Likely (in-body|out-of-body)"
        LABELS "gpu;functional;surgical_deidentification"
        TIMEOUT 240
)
