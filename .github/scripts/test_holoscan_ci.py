# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for application discovery and the shared CLI contract."""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from holoscan_ci import plan_commands


def application(name="demo", language=None, modes=None):
    return {
        "name": name,
        "project_type": "application",
        "source_folder": f"/repo/applications/{name}",
        "language": [] if language is None else language,
        "modes": [] if modes is None else modes,
    }


def discovery(*projects):
    return {"schema_version": 1, "projects": list(projects)}


class PlanCommandsTest(unittest.TestCase):
    def test_empty_repository_needs_no_lifecycle_commands(self):
        self.assertEqual(plan_commands(discovery()), [])

    def test_non_applications_are_excluded(self):
        operator = application("operator", ["unsupported"])
        operator["project_type"] = "operator"
        self.assertEqual(plan_commands(discovery(operator)), [])

    def test_application_without_language_gets_one_test_command(self):
        self.assertEqual(
            plan_commands(discovery(application())),
            [["holoscan", "test", "demo", "--local", "--no-xvfb", "--dryrun"]],
        )

    def test_missing_language_and_modes_use_the_universal_contract(self):
        project = application()
        del project["language"]
        del project["modes"]
        self.assertEqual(
            plan_commands(discovery(project)),
            [["holoscan", "test", "demo", "--local", "--no-xvfb", "--dryrun"]],
        )

    def test_all_declared_languages_are_normalized_and_sorted(self):
        self.assertEqual(
            plan_commands(discovery(application(language=["Python", "C++"]))),
            [
                [
                    "holoscan",
                    "test",
                    "demo",
                    "--local",
                    "--no-xvfb",
                    "--dryrun",
                    "--language",
                    "cpp",
                ],
                [
                    "holoscan",
                    "test",
                    "demo",
                    "--local",
                    "--no-xvfb",
                    "--dryrun",
                    "--language",
                    "python",
                ],
            ],
        )

    def test_ci_mode_run_includes_the_build_before_testing(self):
        self.assertEqual(
            plan_commands(
                discovery(application(language=["C++"], modes=["replay", "ci"]))
            ),
            [
                [
                    "holoscan",
                    "run",
                    "demo",
                    "ci",
                    "--local",
                    "--dryrun",
                    "--language",
                    "cpp",
                ],
                [
                    "holoscan",
                    "test",
                    "demo",
                    "--local",
                    "--no-xvfb",
                    "--dryrun",
                    "--language",
                    "cpp",
                ],
            ],
        )

    def test_non_ci_modes_do_not_opt_in_to_build_and_run(self):
        self.assertEqual(
            plan_commands(discovery(application(modes=["replay", "smoke"]))),
            [["holoscan", "test", "demo", "--local", "--no-xvfb", "--dryrun"]],
        )

    def test_duplicate_records_and_languages_are_checked_once_in_name_order(self):
        self.assertEqual(
            plan_commands(
                discovery(
                    application("zeta", ["Python", "python"]),
                    application("alpha", "C++"),
                    application("alpha", ["cpp"]),
                )
            ),
            [
                [
                    "holoscan",
                    "test",
                    "alpha",
                    "--local",
                    "--no-xvfb",
                    "--dryrun",
                    "--language",
                    "cpp",
                ],
                [
                    "holoscan",
                    "test",
                    "zeta",
                    "--local",
                    "--no-xvfb",
                    "--dryrun",
                    "--language",
                    "python",
                ],
            ],
        )

    def test_disagreeing_duplicate_ci_modes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "conflicting.*demo"):
            plan_commands(discovery(application(modes=["ci"]), application()))

    def test_distinct_sources_cannot_share_a_selector_and_language(self):
        first = application(language=["Python"])
        second = application(language=["Python"])
        first["source_folder"] = "/repo/applications/one/demo/python"
        second["source_folder"] = "/repo/applications/two/demo/python"
        with self.assertRaisesRegex(ValueError, "ambiguous.*demo"):
            plan_commands(discovery(first, second))

    def test_missing_or_invalid_sources_are_rejected(self):
        for source in (None, 7, "", " "):
            project = application()
            project["source_folder"] = source
            with (
                self.subTest(source=source),
                self.assertRaisesRegex(ValueError, "source_folder.*demo"),
            ):
                plan_commands(discovery(project))

    def test_language_specific_records_keep_their_own_ci_modes(self):
        commands = plan_commands(
            discovery(
                application(language=["Python"], modes=["ci"]),
                application(language=["C++"]),
            )
        )
        self.assertEqual([command[1] for command in commands], ["test", "run", "test"])
        self.assertEqual(
            [command[-1] for command in commands], ["cpp", "python", "python"]
        )

    def test_invalid_discovery_shapes_fail_clearly(self):
        for value in (
            None,
            [],
            {},
            {"projects": None},
            {"projects": {}},
            {"projects": [None]},
        ):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "discovery|projects|project"),
            ):
                plan_commands(value)

    def test_invalid_application_selectors_are_rejected(self):
        for name in (None, 12, "", " ", "--help", "has space", "app;command"):
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(ValueError, "selector"),
            ):
                plan_commands(discovery(application(name)))

    def test_unsupported_or_malformed_languages_are_rejected(self):
        for language in (["Rust"], [None], [""], {}, 7, None):
            project = application()
            project["language"] = language
            with (
                self.subTest(language=language),
                self.assertRaisesRegex(ValueError, "language.*demo"),
            ):
                plan_commands(discovery(project))

    def test_malformed_mode_lists_are_rejected(self):
        for modes in ("ci", {"ci": {}}, [None], 7):
            with (
                self.subTest(modes=modes),
                self.assertRaisesRegex(ValueError, "modes.*demo"),
            ):
                plan_commands(discovery(application(modes=modes)))


class DriverTest(unittest.TestCase):
    def run_driver(
        self, payload, fail="", missing_cli=False, args=(), always_build="true"
    ):
        """Exercise the entry point against a controlled CLI subprocess."""
        repo = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(prefix="holoscan ci ") as tmp:
            root = Path(tmp)
            (root / "discovery.json").write_text(payload, encoding="utf-8")
            executable = root / "holoscan"
            if not missing_cli:
                executable.write_text(
                    f"#!{sys.executable}\n"
                    + textwrap.dedent("""\
                        import json
                        import os
                        from pathlib import Path
                        import sys

                        root = Path(__file__).parent
                        with (root / "calls.jsonl").open("a") as log:
                            log.write(json.dumps({
                                "args": sys.argv[1:], "cwd": os.getcwd(),
                                "always_build": os.environ.get("HOLOSCAN_CLI_ALWAYS_BUILD"),
                            }) + "\\n")
                        if sys.argv[1] == os.environ.get("CI_TEST_FAIL"):
                            print("fixture CLI failure", file=sys.stderr)
                            raise SystemExit(7)
                        if sys.argv[1:] == ["list", "--json"]:
                            print((root / "discovery.json").read_text())
                    """),
                    encoding="utf-8",
                )
                executable.chmod(0o755)
            result = subprocess.run(
                [
                    sys.executable,
                    str(repo / ".github" / "scripts" / "holoscan_ci.py"),
                    *args,
                ],
                cwd=root,
                env={
                    **os.environ,
                    "PATH": str(root),
                    "CI_TEST_FAIL": fail,
                    "HOLOSCAN_CLI_ALWAYS_BUILD": always_build,
                },
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            log = root / "calls.jsonl"
            calls = (
                [json.loads(line) for line in log.read_text().splitlines()]
                if log.exists()
                else []
            )
            return result, calls

    def test_entry_point_discovers_then_validates_from_repository_root(self):
        result, calls = self.run_driver(
            json.dumps(discovery(application("nested/demo", ["Python"], ["ci"])))
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            [call["args"] for call in calls],
            [
                ["list", "--json"],
                [
                    "run",
                    "nested/demo",
                    "ci",
                    "--local",
                    "--dryrun",
                    "--language",
                    "python",
                ],
                [
                    "test",
                    "nested/demo",
                    "--local",
                    "--no-xvfb",
                    "--dryrun",
                    "--language",
                    "python",
                ],
            ],
        )
        self.assertEqual(
            {call["cwd"] for call in calls}, {str(Path(__file__).resolve().parents[2])}
        )
        self.assertIn("dry-run", result.stdout)

    def test_empty_repository_is_explicit_in_the_log(self):
        result, calls = self.run_driver(json.dumps(discovery()))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([call["args"] for call in calls], [["list", "--json"]])
        self.assertIn("No applications", result.stdout)

    def test_discovery_failure_is_visible_and_stops_the_driver(self):
        result, calls = self.run_driver(
            json.dumps(discovery(application())), fail="list"
        )
        self.assertEqual(result.returncode, 7)
        self.assertEqual(len(calls), 1)
        self.assertIn("fixture CLI failure", result.stderr)
        self.assertIn("holoscan list --json", result.stderr)

    def test_invalid_discovery_is_rejected_before_any_lifecycle_command(self):
        for payload in (
            "not JSON",
            json.dumps({"projects": {}}),
            json.dumps(discovery(application(), application("--help"))),
        ):
            with self.subTest(payload=payload):
                result, calls = self.run_driver(payload)
                self.assertEqual(result.returncode, 1)
                self.assertEqual([call["args"] for call in calls], [["list", "--json"]])
                self.assertIn("CI error", result.stderr)

    def test_first_lifecycle_failure_stops_later_commands(self):
        result, calls = self.run_driver(
            json.dumps(discovery(application(modes=["ci"]))), fail="run"
        )
        self.assertEqual(result.returncode, 7)
        self.assertEqual([call["args"][0] for call in calls], ["list", "run"])
        self.assertIn("fixture CLI failure", result.stderr)
        self.assertIn("holoscan run demo ci", result.stderr)

    def test_execute_runs_the_lifecycle_and_rejects_empty_ctest_suites(self):
        result, calls = self.run_driver(
            json.dumps(discovery(application(language=["C++"], modes=["ci"]))),
            args=("--execute",),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            [call["args"] for call in calls],
            [
                ["list", "--json"],
                ["run", "demo", "ci", "--local", "--language", "cpp"],
                [
                    "test",
                    "demo",
                    "--local",
                    "--no-xvfb",
                    "--ctest-options=--no-tests=error",
                    "--language",
                    "cpp",
                ],
            ],
        )
        self.assertIn("Executed", result.stdout)

    def test_execute_propagates_application_test_failures(self):
        result, calls = self.run_driver(
            json.dumps(discovery(application())), fail="test", args=("--execute",)
        )
        self.assertEqual(result.returncode, 7)
        self.assertEqual([call["args"][0] for call in calls], ["list", "test"])
        self.assertNotIn("--dryrun", calls[-1]["args"])

    def test_execute_does_not_claim_success_for_an_empty_repository(self):
        result, calls = self.run_driver(json.dumps(discovery()), args=("--execute",))
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual([call["args"] for call in calls], [["list", "--json"]])
        self.assertIn("No applications", result.stderr)

    def test_lifecycle_cannot_inherit_an_environment_that_skips_builds(self):
        for args in ((), ("--execute",)):
            with self.subTest(args=args):
                result, calls = self.run_driver(
                    json.dumps(discovery(application(modes=["ci"]))),
                    args=args,
                    always_build="0",
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(
                    [call["args"][0] for call in calls], ["list", "run", "test"]
                )
                self.assertEqual(
                    [call["always_build"] for call in calls[1:]], ["true", "true"]
                )

    def test_ambiguous_sources_fail_before_any_lifecycle_command(self):
        first = application(language=["Python"])
        second = application(language=["Python"])
        second["source_folder"] = "/repo/applications/other/demo/python"
        result, calls = self.run_driver(json.dumps(discovery(first, second)))
        self.assertEqual(result.returncode, 1)
        self.assertEqual([call["args"] for call in calls], [["list", "--json"]])
        self.assertIn("ambiguous", result.stderr)

    def test_missing_cli_is_an_actionable_error(self):
        result, calls = self.run_driver("", missing_cli=True)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(calls, [])
        self.assertIn("holoscan", result.stderr)
        self.assertIn("CI error", result.stderr)


if __name__ == "__main__":
    unittest.main()
