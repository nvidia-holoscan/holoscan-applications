# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate the Holoscan CLI lifecycle contract for every discovered application."""

import json
import re
import shlex
import subprocess
import sys
from pathlib import Path


class DiscoveryError(ValueError):
    """The CLI returned an invalid or ambiguous application discovery document."""


def plan_commands(discovery: object) -> list[list[str]]:
    """Plan dry runs from ``holoscan list --json`` without executing project code."""
    if not isinstance(discovery, dict) or not isinstance(
        discovery.get("projects"), list
    ):
        raise DiscoveryError("discovery must contain a projects list")

    aliases = {"C++": "cpp", "cpp": "cpp", "Python": "python", "python": "python"}
    applications = {}
    for project in discovery["projects"]:
        if not isinstance(project, dict):
            raise DiscoveryError("each discovery project must be an object")
        if project.get("project_type") != "application":
            continue

        name = project.get("name")
        if not isinstance(name, str) or not re.fullmatch(
            r"[A-Za-z0-9_][A-Za-z0-9_.-]*(?:/[A-Za-z0-9_][A-Za-z0-9_.-]*)*", name
        ):
            raise DiscoveryError(f"invalid application selector: {name!r}")

        source = project.get("source_folder")
        if not isinstance(source, str) or not source.strip():
            raise DiscoveryError(f"invalid source_folder for {name}")

        declared_languages = project.get("language", [])
        if isinstance(declared_languages, str):
            declared_languages = [declared_languages]
        if not isinstance(declared_languages, list):
            raise DiscoveryError(f"invalid language list for {name}")
        languages = set()
        for language in declared_languages:
            if not isinstance(language, str) or language not in aliases:
                raise DiscoveryError(f"unsupported language {language!r} for {name}")
            languages.add(aliases[language])

        modes = project.get("modes", [])
        if not isinstance(modes, list) or any(
            not isinstance(mode, str) or not mode for mode in modes
        ):
            raise DiscoveryError(f"invalid modes list for {name}")
        has_ci = "ci" in modes
        for language in languages or {""}:
            key = (name, language)
            if key in applications:
                previous_ci, previous_source = applications[key]
                if previous_source != source:
                    raise DiscoveryError(
                        f"ambiguous selector {name} ({language or 'default'}): "
                        f"{previous_source!r} and {source!r}; use unique application names"
                    )
                if previous_ci != has_ci:
                    raise DiscoveryError(
                        f"conflicting ci modes for {name} ({language or 'default'})"
                    )
            applications[key] = (has_ci, source)

    commands = []
    for (name, language), (has_ci, _) in sorted(applications.items()):
        language_args = ["--language", language] if language else []
        if has_ci:
            commands.append(
                ["holoscan", "build", name, "ci", "--local", "--dryrun", *language_args]
            )
            commands.append(
                [
                    "holoscan",
                    "run",
                    name,
                    "ci",
                    "--local",
                    "--no-local-build",
                    "--dryrun",
                    *language_args,
                ]
            )
        commands.append(
            [
                "holoscan",
                "test",
                name,
                "--local",
                "--no-xvfb",
                "--dryrun",
                *language_args,
            ]
        )
    return commands


def main() -> int:
    """Discover applications and fail on the first invalid CLI dry run."""
    repo = Path(__file__).resolve().parents[1]
    try:
        discovered = subprocess.run(
            ["holoscan", "list", "--json"],
            cwd=repo,
            stdout=subprocess.PIPE,
            text=True,
            check=True,
        )
        print(discovered.stdout, end="", flush=True)
        commands = plan_commands(json.loads(discovered.stdout))
        if not commands:
            print("No applications discovered; no lifecycle commands to validate.")
            return 0
        for command in commands:
            print(f"+ {shlex.join(command)}", flush=True)
            subprocess.run(command, cwd=repo, check=True)
        print(f"Validated {len(commands)} dry-run lifecycle commands.")
        return 0
    except subprocess.CalledProcessError as error:
        print(
            f"CI error: {shlex.join(error.cmd)} failed (exit {error.returncode})",
            file=sys.stderr,
        )
        return error.returncode if error.returncode > 0 else 1
    except (OSError, ValueError) as error:
        print(f"CI error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
