# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reject internal staging references in the supplied repository files."""

import sys
from pathlib import Path


def main(filenames: list[str]) -> int:
    """Report every file containing an internal staging reference."""
    # Split the marker so this check does not flag its own source.
    marker = "nv" + "staging"
    matches = []
    for filename in filenames:
        path = Path(filename)
        if marker in path.read_text(encoding="utf-8", errors="replace").lower():
            matches.append(str(path))
    if matches:
        print(
            "Internal staging reference found:\n" + "\n".join(matches),
            file=sys.stderr,
        )
    return 1 if matches else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
