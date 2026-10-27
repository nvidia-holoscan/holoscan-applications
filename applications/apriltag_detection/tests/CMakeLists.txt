# SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# These tests need the SDK, its CUDA toolchain and a GPU. The SDK-free YUYV
# test is registered by the application's top-level CMakeLists.txt.

add_executable(apriltag_backend_accuracy_test backend_accuracy_test.cpp)
target_link_libraries(apriltag_backend_accuracy_test PRIVATE apriltag_detection_core)

add_executable(apriltag_operator_test operator_test.cpp)
target_link_libraries(apriltag_operator_test PRIVATE
  apriltag_detection_operators holoscan::testing)
target_compile_definitions(apriltag_operator_test PRIVATE
  APRILTAG_DETECTION_ENABLE_VISUALIZATION=$<BOOL:${APRILTAG_DETECTION_ENABLE_VISUALIZATION}>)

add_executable(apriltag_write_test_frames write_test_frames.cpp)
target_link_libraries(apriltag_write_test_frames PRIVATE apriltag_detection_core)

add_library(apriltag_spawn_thread_preload SHARED spawn_thread_preload.cpp)

add_test(NAME apriltag_backend_accuracy COMMAND apriltag_backend_accuracy_test)
add_test(NAME apriltag_operators COMMAND apriltag_operator_test)

# Runs the application through run_and_match.cmake, which checks its exit
# status as well as its output: apriltag_app_test(<name> <exit status> <output
# regex> <arguments>...).
set(_run_and_match "${CMAKE_CURRENT_SOURCE_DIR}/run_and_match.cmake")
function(apriltag_app_test name status output)
  add_test(NAME ${name}
    COMMAND ${CMAKE_COMMAND} -DEXPECT_EXIT=${status} "-DEXPECT_OUTPUT=${output}"
      -P "${_run_and_match}" -- $<TARGET_FILE:apriltag_detection> ${ARGN})
endfunction()

# The application on synthetic frames, every compiled backend, from host and
# from GPU memory. Each run checks its detections against ground truth.
apriltag_app_test(apriltag_app_synthetic 0 "verdict=PASS"
  --backend all --count 30 --period-ms 10 --display false)
apriltag_app_test(apriltag_app_device_input 0 "verdict=PASS"
  --backend all --count 20 --period-ms 10 --device-input true --display false)
apriltag_app_test(apriltag_app_family 0 "verdict=PASS"
  --backend all --count 10 --period-ms 10 --family tagStandard41h12 --display false)

# A folder of images: gray and color frames written by the fixture, then read
# back with explicit intrinsics.
set(_frames "${CMAKE_CURRENT_BINARY_DIR}/frames")
add_test(NAME apriltag_frames_setup
  COMMAND $<TARGET_FILE:apriltag_write_test_frames> "${_frames}/gray" 6 gray)
add_test(NAME apriltag_frames_setup_rgb
  COMMAND $<TARGET_FILE:apriltag_write_test_frames> "${_frames}/rgb" 6 rgb)
set_tests_properties(apriltag_frames_setup apriltag_frames_setup_rgb
  PROPERTIES FIXTURES_SETUP apriltag_frames)
foreach(_kind IN ITEMS gray rgb)
  apriltag_app_test(apriltag_app_images_${_kind} 0 "backend=cuda frames=6/6 tags=24 .*verdict=PASS"
    --source images --images "${_frames}/${_kind}" --backend all
    --fx 1152 --fy 1152 --cx 639.5 --cy 359.5 --tag-size 0.12 --period-ms 10 --display false)
  set_tests_properties(apriltag_app_images_${_kind} PROPERTIES FIXTURES_REQUIRED apriltag_frames)
endforeach()

# Invalid input fails fast with a usage error (exit status 2) instead of
# starting a run.
apriltag_app_test(apriltag_app_rejects_unknown_backend 2 "^error: "
  --backend gpu --display false)
apriltag_app_test(apriltag_app_rejects_zero_count 2 "^error: " --count 0 --display false)
apriltag_app_test(apriltag_app_rejects_missing_images 2 "^error: "
  --source images --display false)
apriltag_app_test(apriltag_app_rejects_partial_intrinsics 2 "^error: "
  --source images --images "${_frames}/gray" --fx 900 --display false)
apriltag_app_test(apriltag_app_rejects_missing_camera 2 "^error: cannot open camera /dev/video99"
  --source camera --device /dev/video99)

# A run that fails once started: the second image of a folder has another
# channel count. It exits 1 and names the failing operator.
set(_mixed "${CMAKE_CURRENT_BINARY_DIR}/frames/mixed")
file(MAKE_DIRECTORY "${_mixed}")
add_test(NAME apriltag_frames_setup_mixed
  COMMAND ${CMAKE_COMMAND} -E copy "${_frames}/gray/frame_0000.pgm"
    "${_frames}/rgb/frame_0001.ppm" "${_mixed}")
set_tests_properties(apriltag_frames_setup_mixed PROPERTIES
  FIXTURES_SETUP apriltag_mixed FIXTURES_REQUIRED apriltag_frames)
apriltag_app_test(apriltag_app_failed_run 1 "every image must match.*verdict=FAIL"
  --source images --images "${_mixed}" --backend cuda --period-ms 10 --display false)
set_tests_properties(apriltag_app_failed_run PROPERTIES FIXTURES_REQUIRED apriltag_mixed)

# A stop request ends the run early with its own verdict, and the warning it
# leaves is shown. (A shell starts background jobs with SIGINT ignored, so the
# test sends SIGTERM.)
add_test(NAME apriltag_app_stopped
  COMMAND ${CMAKE_COMMAND} -DEXPECT_EXIT=1 "-DEXPECT_OUTPUT=warning RUNTIME_MANAGED_SIGNAL_RECEIVED.*frames=[0-9]+/1000 .*verdict=STOPPED"
    -P "${_run_and_match}" -- sh -c
    "\"$1\" --backend cuda --count 1000 --period-ms 10 --display false & sleep 3 && kill -TERM $! && wait $!"
    sh $<TARGET_FILE:apriltag_detection>)

# Another thread running before main(), as under a profiler or a sanitizer:
# the run goes ahead without managed signal handling, and says so.
add_test(NAME apriltag_app_without_signal_domain
  COMMAND ${CMAKE_COMMAND} -DEXPECT_EXIT=0
    "-DEXPECT_OUTPUT=without managed SIGINT/SIGTERM handling.*verdict=PASS"
    -P "${_run_and_match}" -- ${CMAKE_COMMAND} -E env
    LD_PRELOAD=$<TARGET_FILE:apriltag_spawn_thread_preload>
    $<TARGET_FILE:apriltag_detection> --backend cuda --count 10 --period-ms 10 --display false)

if(APRILTAG_DETECTION_ENABLE_VISUALIZATION)
  # The display graph compiles, with or without a display to run it.
  apriltag_app_test(apriltag_app_display_plan 0 "plan=OK"
    --backend all --display true --compile-only true)
endif()

get_property(_tests DIRECTORY PROPERTY TESTS)
set_tests_properties(${_tests} PROPERTIES LABELS "apriltag;gpu" TIMEOUT 600)
