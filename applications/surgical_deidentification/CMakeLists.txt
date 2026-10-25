# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)

# Select the current GPU unless the caller supplied an architecture list.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to compile")
endif()
project(surgical_deidentification LANGUAGES CXX)

# The holoscan-cli test driver does not forward --local-sdk-root into CMake.
# Also honor the standard SDK environment variable for direct CMake consumers.
if(DEFINED ENV{HOLOSCAN_SDK_ROOT})
  list(PREPEND CMAKE_PREFIX_PATH "$ENV{HOLOSCAN_SDK_ROOT}")
elseif(DEFINED ENV{HOLOSCAN_LIB_PATH})
  # The CLI mounts the supplied SDK and exports its library path; derive the SDK
  # root from it when HOLOSCAN_SDK_ROOT is not set in the recursive command.
  get_filename_component(_sd_sdk_root "$ENV{HOLOSCAN_LIB_PATH}" DIRECTORY)
  list(PREPEND CMAKE_PREFIX_PATH "${_sd_sdk_root}")
endif()

# Pin to Holoscan SDK 5. The surgical_deidentification pipeline uses:
#   - holoscan::op_inference (TensorRT InferenceOp — ships with the SDK)
#   - holoscan::op_tensor_preprocessor (ships with the SDK inference module)
#   - holoscan::op_video_file_reader (bundled in-tree under ./video_file_reader)
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS inference)
find_package(yaml-cpp REQUIRED)

# Where the NGC endoscopy_out_of_body_detection resource lands. Override with
# -DHOLOHUB_DATA_DIR=... to point at a prepared location on the host.
if(NOT DEFINED HOLOHUB_DATA_DIR)
    set(HOLOHUB_DATA_DIR "${CMAKE_BINARY_DIR}/data"
        CACHE PATH "Root directory for application data")
endif()

# When ON, the dataset is fetched as part of the default build (chained onto de_id).
# When OFF (default), users can still fetch it on demand:
#     cmake --build . --target de_id_download_data
option(SURGICAL_DEID_DOWNLOAD_DATA
    "Download the endoscopy_out_of_body_detection NGC resource during build" OFF)

add_subdirectory(video_file_reader)
add_subdirectory(video_file_writer)

add_executable(de_id de_id.cpp)
target_compile_features(de_id PRIVATE cxx_std_20)
target_link_libraries(de_id
    PRIVATE
        holoscan::core
        holoscan::op_inference
        holoscan::op_tensor_preprocessor
        holoscan::op_video_file_reader
        yaml-cpp::yaml-cpp
)
set_target_properties(de_id PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    CXX_EXTENSIONS OFF
)

# Ship the YAML config alongside the binary so `./de_id --config config.yaml` works
# from the build output directory.
add_custom_command(
    TARGET de_id POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/config.yaml"
        "$<TARGET_FILE_DIR:de_id>/config.yaml"
)

install(TARGETS de_id RUNTIME DESTINATION bin)
install(FILES config.yaml DESTINATION bin)
install(FILES README.md LICENSE DESTINATION share/surgical_deidentification)

# Register the dataset-download target after `de_id` exists so SURGICAL_DEID_DOWNLOAD_DATA
# can chain onto it.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/DownloadData.cmake)

if(BUILD_TESTING)
    include(CTest)
    add_subdirectory(tests)
endif()
