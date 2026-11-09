# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The application runs from source; these tests need its requirements.txt installed.
if(BUILD_TESTING)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  set(_bci_tests ${CMAKE_CURRENT_SOURCE_DIR}/../tests)
  foreach(_bci_test IN ITEMS snirf kernel_sdk graph surface solver application)
    add_test(NAME bci_visualization_${_bci_test}
      COMMAND ${Python3_EXECUTABLE} -m unittest discover
        -s ${_bci_tests} -p test_${_bci_test}.py -v)
  endforeach()
  set_tests_properties(bci_visualization_snirf bci_visualization_kernel_sdk
    bci_visualization_graph bci_visualization_surface PROPERTIES
    LABELS "bci_visualization" TIMEOUT 120)
  set_tests_properties(bci_visualization_solver bci_visualization_application PROPERTIES
    LABELS "bci_visualization;gpu" TIMEOUT 300)
endif()

# Download the sample data at build time so `holoscan run bci_visualization` works on a fresh
# checkout. prepare_data.py keeps files whose checksums already match.
option(HOLOHUB_DOWNLOAD_DATASETS "Download datasets" ON)
set(BCI_VISUALIZATION_HEADSET "flow_2" CACHE STRING "Headset assets to download")
if(HOLOHUB_DOWNLOAD_DATASETS)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  if(NOT HOLOHUB_DATA_DIR)
    set(HOLOHUB_DATA_DIR "${CMAKE_SOURCE_DIR}/data")
  endif()
  get_filename_component(_bci_data_dir "${HOLOHUB_DATA_DIR}/bci_visualization"
    ABSOLUTE BASE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
  set(_bci_data_selection "${CMAKE_CURRENT_BINARY_DIR}/data-selection.txt")
  file(CONFIGURE OUTPUT "${_bci_data_selection}"
    CONTENT "${BCI_VISUALIZATION_HEADSET}\n${_bci_data_dir}\n" @ONLY)
  set(_bci_data_stamp "${CMAKE_CURRENT_BINARY_DIR}/data-prepared.stamp")
  add_custom_command(
    OUTPUT ${_bci_data_stamp}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../prepare_data.py
      --output ${_bci_data_dir} --headset ${BCI_VISUALIZATION_HEADSET}
    COMMAND ${CMAKE_COMMAND} -E copy ${_bci_data_selection} ${_bci_data_stamp}
    DEPENDS ${_bci_data_selection} ${CMAKE_CURRENT_SOURCE_DIR}/../prepare_data.py
      ${CMAKE_CURRENT_SOURCE_DIR}/../dataset_checksums.json
    COMMENT "Downloading bci_visualization sample data to ${_bci_data_dir}"
    VERBATIM)
  add_custom_target(bci_visualization_data ALL DEPENDS ${_bci_data_stamp})
endif()
