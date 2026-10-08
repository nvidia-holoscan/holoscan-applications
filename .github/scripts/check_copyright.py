# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check Apache-2.0 licensing, copyright notices, and changed-file years."""

import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

COPYRIGHT = re.compile(
    r"^[ \t]*(?:[#/*]+[ \t]*)?"
    r"(?:SPDX-FileCopyrightText:[ \t]*(?:Copyright[ \t]+(?:\(c\)[ \t]*)?)?"
    r"|Copyright[ \t]+(?:\(c\)[ \t]*)?)"
    r"(\d{4})(?:-(\d{4}))?,?[ \t]+[^\s*/#][^\r\n]*$",
    re.MULTILINE | re.IGNORECASE,
)


def changed_files() -> set[Path] | None:
    """Return changed paths, or None when every supplied file needs this year's notice."""
    base = os.environ.get("COPYRIGHT_BASE_REF", "")
    root = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, check=False)
    if root.returncode:
        if base:
            raise ValueError(f"Cannot resolve copyright base {base!r} outside a Git repository")
        return None
    repo = Path(os.fsdecode(root.stdout).rstrip("\n"))
    # GitHub uses an all-zero before SHA for the first push to a branch.
    if base and set(base) == {"0"}:
        return None
    reference = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "rev-parse",
            "--verify",
            "--end-of-options",
            f"{base or 'HEAD'}^{{commit}}",
        ],
        capture_output=True,
        check=False,
    )
    if reference.returncode:
        if base:
            raise ValueError(f"Cannot resolve copyright base {base!r}; fetch the required history")
        # An initial local commit has no HEAD yet.
        return None
    revision = os.fsdecode(reference.stdout).strip()
    changed = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "diff",
            "--name-only",
            "--no-renames",
            "--diff-filter=ACMRT",
            "-z",
            revision,
            "--",
        ],
        capture_output=True,
        check=True,
    ).stdout
    untracked = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "--others", "--exclude-standard", "-z"],
        capture_output=True,
        check=True,
    ).stdout
    return {
        (repo / os.fsdecode(name)).resolve() for name in (changed + untracked).split(b"\0") if name
    }


def main(filenames: list[str]) -> int:
    """Validate all notices; require the current year only for changed files."""
    if not filenames:
        return 0
    try:
        changed = changed_files()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Copyright check failed: {error}", file=sys.stderr)
        return 1
    year = datetime.now(UTC).year
    marker = "SPDX-License-Identifier: Apache-2.0"
    errors = []
    for filename in filenames:
        path = Path(filename)
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError as error:
            errors.append(f"{path}: {error}")
            continue
        if marker not in content:
            errors.append(f"{path}: Missing Apache-2.0 SPDX identifier")
        notices = COPYRIGHT.findall(content)
        if not notices:
            errors.append(f"{path}: Missing or malformed copyright notice")
            continue
        years = [(int(start), int(end or start)) for start, end in notices]
        if any(not 1 <= start <= end <= year for start, end in years):
            errors.append(f"{path}: Copyright year range is reversed, invalid, or in the future")
        elif (changed is None or path.resolve() in changed) and not any(
            end == year for _, end in years
        ):
            errors.append(
                f"{path}: Changed file's copyright notice must include the current year {year}"
            )
    if errors:
        print("\n".join(errors), file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
