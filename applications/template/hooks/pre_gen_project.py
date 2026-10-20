# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reject template inputs that would generate an invalid application."""

import re
import sys

SLUG = "{{ cookiecutter.project_slug }}"
LANGUAGE = "{{ cookiecutter.language }}"
VERSION = "{{ cookiecutter.version }}"
HOLOSCAN_VERSION = "{{ cookiecutter.holoscan_version }}"
SEMVER = r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


if not re.fullmatch(r"[a-z][a-z0-9_]*", SLUG) or SLUG == "template":
    fail(
        f"project_slug {SLUG!r} must be snake_case: a lowercase letter followed by lowercase "
        "letters, digits, or underscores, and not 'template'."
    )
if LANGUAGE != "cpp":
    fail(
        f"language {LANGUAGE!r} is not supported: this template generates C++ applications "
        "for Holoscan SDK 5. Use --language cpp."
    )
if not re.fullmatch(SEMVER, VERSION):
    fail(f"version {VERSION!r} must be MAJOR.MINOR.PATCH, for example 0.1.0.")
sdk = re.fullmatch(SEMVER, HOLOSCAN_VERSION)
if not sdk or int(sdk.group(1)) < 5:
    fail(
        f"holoscan_version {HOLOSCAN_VERSION!r} must be 5.0.0 or later: the generated "
        "application uses the Holoscan SDK 5 API."
    )
