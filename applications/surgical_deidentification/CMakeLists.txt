# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)
project(surgical_deidentification_apps LANGUAGES NONE)
include(CTest)

option(HOLOHUB_BUILD_CPP "Build support for C++" ON)

if(HOLOHUB_BUILD_CPP)
    add_subdirectory(cpp)
endif()

install(FILES README.md LICENSE DESTINATION share/surgical_deidentification)
