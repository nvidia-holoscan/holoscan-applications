# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate supplied metadata files using the installed Holoscan CLI schemas."""

import json
import sys
from pathlib import Path

from holoscan_cli.metadata.metadata_validator import validate_json


def main(filenames: list[str]) -> int:
    """Report parsing, reading, and schema errors without stopping the batch."""
    failed = False
    for filename in filenames:
        path = Path(filename)
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            valid, message = validate_json(document, path.parent)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            valid, message = False, error
        if not valid:
            print(f"{path}: {message}", file=sys.stderr)
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
