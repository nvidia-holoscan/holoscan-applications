# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(SURGICAL_DEID_NGC_VERSION "20230127"
    CACHE STRING "NGC endoscopy_out_of_body_detection version")

# SHA-256 of the NGC archive for version 20230127, captured from a verified download of the
# default URL. If SURGICAL_DEID_NGC_URL or SURGICAL_DEID_NGC_VERSION is overridden, override
# this to the matching hash (or set it to "" to opt out of verification, documented as a risk
# in SECURITY.md).
set(SURGICAL_DEID_NGC_SHA256
    "55f59e12f6cb9738d21e0cc1dea85d112d4278a02deeb234afbae9544af3afa3"
    CACHE STRING "Expected SHA-256 of the NGC archive (empty disables verification)")

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
# Key the cached archive filename by both version AND a short URL fingerprint, so switching
# SURGICAL_DEID_NGC_URL (e.g. to an internal mirror) cannot silently extract a stale archive
# that was fetched from a different URL at the same version.
string(SHA256 _sd_url_hash "${SURGICAL_DEID_NGC_URL}")
string(SUBSTRING "${_sd_url_hash}" 0 12 _sd_url_tag)
set(_sd_archive     "${CMAKE_CURRENT_BINARY_DIR}/${_sd_data_subdir}-${SURGICAL_DEID_NGC_VERSION}-${_sd_url_tag}.zip")
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
            -DEXPECTED_SHA256=${SURGICAL_DEID_NGC_SHA256}
            -DSTAMP=${_sd_stamp}
            -DMP4=${_sd_mp4}
            -DONNX=${_sd_onnx}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/fetch_and_extract.cmake"
    VERBATIM
    COMMENT "Downloading endoscopy_out_of_body_detection (${SURGICAL_DEID_NGC_VERSION}) from NGC"
)
add_custom_target(de_id_download_data DEPENDS "${_sd_mp4}" "${_sd_onnx}" "${_sd_stamp}")

if(SURGICAL_DEID_DOWNLOAD_DATA)
    add_dependencies(de_id de_id_download_data)
endif()

set(SURGICAL_DEID_SAMPLE_MP4  "${_sd_mp4}")
set(SURGICAL_DEID_MODEL_ONNX  "${_sd_onnx}")
