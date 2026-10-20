# SPDX-FileCopyrightText: Copyright (c) {{cookiecutter.year}} {{cookiecutter.copyright_holder}}.
# SPDX-License-Identifier: Apache-2.0

# These tests run the built executable, so they need the SDK. Register
# SDK-free tests in the application's top-level CMakeLists.txt instead.
add_test(NAME {{ cookiecutter.project_slug }}_pipeline
  COMMAND {{ cookiecutter.project_slug }} --count 10 --period-ms 1)
add_test(NAME {{ cookiecutter.project_slug }}_rejects_zero_count
  COMMAND {{ cookiecutter.project_slug }} --count 0)
set_tests_properties({{ cookiecutter.project_slug }}_rejects_zero_count PROPERTIES WILL_FAIL TRUE)
set_tests_properties(
  {{ cookiecutter.project_slug }}_pipeline {{ cookiecutter.project_slug }}_rejects_zero_count
  PROPERTIES LABELS "{{ cookiecutter.project_slug }}" TIMEOUT 60)
