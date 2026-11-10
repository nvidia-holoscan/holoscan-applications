# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

foreach(var URL ARCHIVE DEST VERSION STAMP MP4 ONNX)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "fetch_and_extract.cmake: ${var} not set")
    endif()
endforeach()

# Expected archive SHA-256: set by the caller for integrity verification. Leave unset only
# during initial introduction of a new URL+version pair; always supply it in checked-in
# DownloadData.cmake.
if(NOT DEFINED EXPECTED_SHA256 OR EXPECTED_SHA256 STREQUAL "")
    set(EXPECTED_SHA256 "")
endif()

string(SHA256 _url_hash "${URL}")
set(_stamp_token "${VERSION}:${_url_hash}")

if(EXISTS "${STAMP}")
    file(READ "${STAMP}" _stamped_token)
    string(STRIP "${_stamped_token}" _stamped_token)
    if("${_stamped_token}" STREQUAL "${_stamp_token}"
       AND EXISTS "${MP4}"
       AND EXISTS "${ONNX}")
        message(STATUS "Dataset already at version ${VERSION} (URL unchanged); skipping download.")
        return()
    endif()
endif()

if(NOT EXISTS "${ARCHIVE}")
    message(STATUS "Downloading ${URL}")
    if(EXPECTED_SHA256 STREQUAL "")
        file(DOWNLOAD "${URL}" "${ARCHIVE}"
             SHOW_PROGRESS
             STATUS _dl_status
             TLS_VERIFY ON)
    else()
        file(DOWNLOAD "${URL}" "${ARCHIVE}"
             SHOW_PROGRESS
             STATUS _dl_status
             TLS_VERIFY ON
             EXPECTED_HASH "SHA256=${EXPECTED_SHA256}")
    endif()
    list(GET _dl_status 0 _dl_code)
    list(GET _dl_status 1 _dl_msg)
    if(NOT _dl_code EQUAL 0)
        file(REMOVE "${ARCHIVE}")
        message(FATAL_ERROR "Download failed (${_dl_code}): ${_dl_msg}")
    endif()
elseif(NOT EXPECTED_SHA256 STREQUAL "")
    file(SHA256 "${ARCHIVE}" _archive_hash)
    if(NOT _archive_hash STREQUAL EXPECTED_SHA256)
        file(REMOVE "${ARCHIVE}")
        message(FATAL_ERROR "Cached archive hash mismatch (expected ${EXPECTED_SHA256}, "
                            "got ${_archive_hash}); removed. Re-run the build to re-download.")
    endif()
endif()

file(MAKE_DIRECTORY "${DEST}")
message(STATUS "Extracting ${ARCHIVE} -> ${DEST}")
file(ARCHIVE_EXTRACT INPUT "${ARCHIVE}" DESTINATION "${DEST}")

file(WRITE "${STAMP}" "${_stamp_token}\n")
