# SPDX-FileCopyrightText: Copyright (c) {{cookiecutter.year}} {{cookiecutter.copyright_holder}}.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)
project({{ cookiecutter.project_slug }}_apps LANGUAGES NONE)
include(CTest)

# Also provide the CLI language switch for standalone application builds.
option(HOLOHUB_BUILD_CPP "Build support for C++" ON)

# Register SDK-free tests here, under BUILD_TESTING and outside the language
# scope below, so hosted CI runs them without the SDK, a GPU, or external data.
# Tests that need the SDK or the built executable belong in tests/CMakeLists.txt.

if(HOLOHUB_BUILD_CPP)
  add_subdirectory(cpp)
endif()
