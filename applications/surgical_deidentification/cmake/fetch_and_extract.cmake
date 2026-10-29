# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

foreach(var URL ARCHIVE DEST VERSION STAMP)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "fetch_and_extract.cmake: ${var} not set")
    endif()
endforeach()

if(EXISTS "${STAMP}")
    file(READ "${STAMP}" _stamped_version)
    string(STRIP "${_stamped_version}" _stamped_version)
    if("${_stamped_version}" STREQUAL "${VERSION}")
        message(STATUS "Dataset already at version ${VERSION}; skipping download.")
        return()
    endif()
endif()

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

file(WRITE "${STAMP}" "${VERSION}\n")
