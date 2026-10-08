# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Require the Apache-2.0 SPDX identifier in each supplied file."""

import sys
from pathlib import Path


def main(filenames: list[str]) -> int:
    """Report all files missing the required identifier."""
    marker = "SPDX-License-Identifier: Apache-2.0"
    missing = []
    for filename in filenames:
        path = Path(filename)
        if marker not in path.read_text(encoding="utf-8", errors="replace"):
            missing.append(str(path))
    if missing:
        print(
            "Missing Apache-2.0 SPDX identifier:\n" + "\n".join(missing),
            file=sys.stderr,
        )
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
