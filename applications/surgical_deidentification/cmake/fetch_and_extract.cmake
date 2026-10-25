# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Script-mode helper invoked by DownloadData.cmake. Downloads ${URL} to ${ARCHIVE}
# (skipping if the archive already exists) and extracts it into ${DEST}.

foreach(var URL ARCHIVE DEST)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "fetch_and_extract.cmake: ${var} not set")
    endif()
endforeach()

if(NOT EXISTS "${ARCHIVE}")
    message(STATUS "Downloading ${URL}")
    file(DOWNLOAD "${URL}" "${ARCHIVE}"
         SHOW_PROGRESS
         STATUS _dl_status
         TLS_VERIFY ON)
    list(GET _dl_status 0 _dl_code)
    list(GET _dl_status 1 _dl_msg)
    if(NOT _dl_code EQUAL 0)
        file(REMOVE "${ARCHIVE}")
        message(FATAL_ERROR "Download failed (${_dl_code}): ${_dl_msg}")
    endif()
endif()

file(MAKE_DIRECTORY "${DEST}")
message(STATUS "Extracting ${ARCHIVE} -> ${DEST}")
file(ARCHIVE_EXTRACT INPUT "${ARCHIVE}" DESTINATION "${DEST}")
