# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Fetch the public NGC endoscopy_out_of_body_detection resource (ONNX model + sample MP4)
# into HOLOHUB_DATA_DIR/endoscopy_out_of_body_detection/. Both files are expected by
# config.yaml and the e2e test.
#
# Resource URL and pinned version mirror metadata.json "requirements.data".

set(SURGICAL_DEID_NGC_VERSION "20230127"
    CACHE STRING "NGC endoscopy_out_of_body_detection version")
set(SURGICAL_DEID_NGC_URL
    "https://api.ngc.nvidia.com/v2/resources/nvidia/clara-holoscan/endoscopy_out_of_body_detection/versions/${SURGICAL_DEID_NGC_VERSION}/zip"
    CACHE STRING "NGC resource download URL")

set(_sd_data_subdir "endoscopy_out_of_body_detection")
set(_sd_data_root   "${HOLOHUB_DATA_DIR}/${_sd_data_subdir}")
set(_sd_archive     "${CMAKE_CURRENT_BINARY_DIR}/${_sd_data_subdir}.zip")
set(_sd_mp4         "${_sd_data_root}/sample_clip_out_of_body_detection.mp4")
set(_sd_onnx        "${_sd_data_root}/out_of_body_detection.onnx")

# Build-time target: `cmake --build . --target de_id_download_data` fetches and
# extracts the resource. Idempotent — re-running is a no-op once both files exist.
add_custom_command(
    OUTPUT "${_sd_mp4}" "${_sd_onnx}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${_sd_data_root}"
    COMMAND ${CMAKE_COMMAND}
            -DURL=${SURGICAL_DEID_NGC_URL}
            -DARCHIVE=${_sd_archive}
            -DDEST=${_sd_data_root}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/fetch_and_extract.cmake"
    VERBATIM
    COMMENT "Downloading endoscopy_out_of_body_detection (${SURGICAL_DEID_NGC_VERSION}) from NGC"
)
add_custom_target(de_id_download_data DEPENDS "${_sd_mp4}" "${_sd_onnx}")

if(SURGICAL_DEID_DOWNLOAD_DATA)
    # Chain the download into the default build so `holoscan build` lands with
    # the dataset in place.
    add_dependencies(de_id de_id_download_data)
endif()

# Export the resolved paths for tests/CMakeLists.txt. include() inherits the
# caller's scope, so plain set() is enough.
set(SURGICAL_DEID_SAMPLE_MP4  "${_sd_mp4}")
set(SURGICAL_DEID_MODEL_ONNX  "${_sd_onnx}")
