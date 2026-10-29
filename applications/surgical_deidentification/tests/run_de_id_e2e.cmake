# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

foreach(var DE_ID CONFIG EOB_MP4 EOB_ONNX TICKS)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "run_de_id_e2e.cmake: ${var} not set")
    endif()
endforeach()

if(NOT EXISTS "${EOB_MP4}" OR NOT EXISTS "${EOB_ONNX}")
    message(STATUS "endoscopy_out_of_body_detection dataset missing:")
    message(STATUS "  mp4   : ${EOB_MP4}")
    message(STATUS "  model : ${EOB_ONNX}")
    message(STATUS "Build the endoscopy_out_of_body_detection application with "
                   "HOLOHUB_DOWNLOAD_DATASETS=ON to fetch them.")
    cmake_language(EXIT 77)
endif()

execute_process(
    COMMAND "${DE_ID}" --config "${CONFIG}" --ticks "${TICKS}"
    RESULT_VARIABLE rc
    OUTPUT_VARIABLE out
    ERROR_VARIABLE  err
)

if(NOT rc EQUAL 0)
    message(FATAL_ERROR "de_id exited with code ${rc}\nstdout:\n${out}\nstderr:\n${err}")
endif()

string(REGEX MATCH "Likely (in-body|out-of-body)" matched "${out}")
if(NOT matched)
    message(FATAL_ERROR
        "expected classification line ('Likely in-body' or 'Likely out-of-body') "
        "not found in de_id stdout:\n${out}")
endif()
