# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(SURGICAL_DEID_NGC_VERSION "20230127"
    CACHE STRING "NGC endoscopy_out_of_body_detection version")

set(_sd_default_url
    "https://api.ngc.nvidia.com/v2/resources/nvidia/clara-holoscan/endoscopy_out_of_body_detection/versions/${SURGICAL_DEID_NGC_VERSION}/zip")
if(NOT DEFINED CACHE{SURGICAL_DEID_NGC_URL}
   OR "$CACHE{SURGICAL_DEID_NGC_URL}" STREQUAL "$CACHE{_SD_NGC_URL_PREV_DEFAULT}")
    set(SURGICAL_DEID_NGC_URL "${_sd_default_url}"
        CACHE STRING "NGC resource download URL" FORCE)
endif()
set(_SD_NGC_URL_PREV_DEFAULT "${_sd_default_url}" CACHE INTERNAL "")

set(_sd_data_subdir "endoscopy_out_of_body_detection")
set(_sd_data_root   "${HOLOHUB_DATA_DIR}/${_sd_data_subdir}")
set(_sd_archive     "${CMAKE_CURRENT_BINARY_DIR}/${_sd_data_subdir}-${SURGICAL_DEID_NGC_VERSION}.zip")
set(_sd_mp4         "${_sd_data_root}/sample_clip_out_of_body_detection.mp4")
set(_sd_onnx        "${_sd_data_root}/out_of_body_detection.onnx")
set(_sd_stamp       "${_sd_data_root}/.ngc_version")

add_custom_command(
    OUTPUT "${_sd_mp4}" "${_sd_onnx}" "${_sd_stamp}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${_sd_data_root}"
    COMMAND ${CMAKE_COMMAND}
            -DURL=${SURGICAL_DEID_NGC_URL}
            -DARCHIVE=${_sd_archive}
            -DDEST=${_sd_data_root}
            -DVERSION=${SURGICAL_DEID_NGC_VERSION}
            -DSTAMP=${_sd_stamp}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/fetch_and_extract.cmake"
    VERBATIM
    COMMENT "Downloading endoscopy_out_of_body_detection (${SURGICAL_DEID_NGC_VERSION}) from NGC"
)
add_custom_target(de_id_download_data DEPENDS "${_sd_mp4}" "${_sd_onnx}" "${_sd_stamp}")

if(SURGICAL_DEID_DOWNLOAD_DATA)
    add_dependencies(de_id de_id_download_data)
endif()

set(SURGICAL_DEID_SAMPLE_MP4  "${_sd_mp4}")
set(SURGICAL_DEID_MODEL_ONNX  "${_sd_onnx}")
