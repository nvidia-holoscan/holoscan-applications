# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# These tests include the production implementation directly; do not also link
# its source translation units into the test executables.
add_executable(endoscopy_cache_test inference_cache_test.cpp)
add_executable(endoscopy_malformed_engine_test malformed_engine_test.cpp)
foreach(test_target IN ITEMS endoscopy_cache_test endoscopy_malformed_engine_test)
  target_compile_features(${test_target} PRIVATE cxx_std_20)
  target_include_directories(${test_target} PRIVATE ${TENSORRT_INCLUDE_DIR})
  target_link_libraries(${test_target} PRIVATE
    holoscan::core holoscan::sensor_io CUDA::cudart
    ${TENSORRT_LIBRARY} ${TENSORRT_PLUGIN_LIBRARY} ${TENSORRT_ONNX_LIBRARY})
endforeach()

add_executable(endoscopy_postprocessor_test postprocessor_kernel_test.cu)
target_compile_features(endoscopy_postprocessor_test PRIVATE cxx_std_20 cuda_std_20)
target_link_libraries(endoscopy_postprocessor_test PRIVATE
  holoscan::core holoscan::sensor_io CUDA::cudart)

# CAE policy generation: the exact kernel and the operator inside a compiled graph. The host-only
# configuration test is registered in the application's top-level CMakeLists.txt.
add_executable(endoscopy_inference_to_policy_kernel_test inference_to_policy_kernel_test.cu)
add_executable(endoscopy_inference_to_policy_graph_test
  inference_to_policy_graph_test.cpp ../cpp/inference_to_policy.cu)
foreach(test_target IN ITEMS endoscopy_inference_to_policy_kernel_test
                             endoscopy_inference_to_policy_graph_test)
  target_compile_features(${test_target} PRIVATE cxx_std_20 cuda_std_20)
  target_link_libraries(${test_target} PRIVATE holoscan::core holoscan::sensor_io CUDA::cudart)
endforeach()

add_test(NAME endoscopy_cache COMMAND endoscopy_cache_test)
add_test(NAME endoscopy_malformed_engine COMMAND endoscopy_malformed_engine_test)
add_test(NAME endoscopy_postprocessor COMMAND endoscopy_postprocessor_test)
add_test(NAME endoscopy_inference_to_policy_kernel COMMAND endoscopy_inference_to_policy_kernel_test)
add_test(NAME endoscopy_inference_to_policy_graph COMMAND endoscopy_inference_to_policy_graph_test)
set_tests_properties(endoscopy_malformed_engine endoscopy_postprocessor
  PROPERTIES LABELS "gpu" TIMEOUT 120)
set_tests_properties(endoscopy_inference_to_policy_kernel endoscopy_inference_to_policy_graph
  PROPERTIES LABELS "endoscopy;gpu;cae" TIMEOUT 120)

add_test(NAME endoscopy_replay
  COMMAND ${CMAKE_COMMAND} -E env ENDOSCOPY_EXECUTABLE=$<TARGET_FILE:endoscopy_tool_tracking>
    ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_application.py)
set_tests_properties(endoscopy_replay PROPERTIES LABELS "endoscopy;replay" TIMEOUT 120)
if(ENDOSCOPY_DATA_DIR)
  add_test(NAME endoscopy_inference
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/inference_smoke.py
      --binary $<TARGET_FILE:endoscopy_tool_tracking> --data ${ENDOSCOPY_DATA_DIR})
  set_tests_properties(endoscopy_inference PROPERTIES
    LABELS "endoscopy;gpu;inference" TIMEOUT 900)
endif()
