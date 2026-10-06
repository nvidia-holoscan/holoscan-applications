# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

# Follow HoloHub's application registration convention. Dependencies are owned
# by each application and resolved in its language-specific CMakeLists.txt.
function(add_holoscan_application NAME)
  option(APP_${NAME} "Build the ${NAME} application" ${BUILD_ALL})
  if(APP_${NAME})
    add_subdirectory(${NAME})
  endif()
endfunction()
