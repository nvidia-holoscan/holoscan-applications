# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate an application from applications/template with the installed CLI."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / ".github" / "scripts"
NAME = "template_check"
CONTEXT = (
    "--context",
    "description=Template check",
    "--context",
    "full_name=Template Author",
    "--context",
    "affiliation=Template Organization",
)
GENERATED = {
    "CMakeLists.txt",
    "README.md",
    "metadata.json",
    "cpp/CMakeLists.txt",
    "cpp/main.cpp",
    "tests/CMakeLists.txt",
}
IGNORED = shutil.ignore_patterns(
    ".git", ".venv", ".cache", ".nv", "__pycache__", "build", "build-*", "install*", "data"
)


class ApplicationTemplateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        cls.root = Path(directory.name) / "repository"
        shutil.copytree(REPO, cls.root, ignore=IGNORED)
        cls.app = cls.root / "applications" / NAME
        result = cls.holoscan("create", NAME, *cls.template_args())
        if result.returncode:
            raise AssertionError(f"holoscan create failed:\n{result.stdout}{result.stderr}")

    @staticmethod
    def template_args(*extra):
        return ("--template", "applications/template", "--interactive", "false", *CONTEXT, *extra)

    @classmethod
    def holoscan(cls, *args):
        return subprocess.run(
            ["holoscan", *args],
            cwd=cls.root,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )

    def run_script(self, script, *paths):
        return subprocess.run(
            [sys.executable, str(SCRIPTS / script), *map(str, paths)],
            cwd=self.root,
            env={**os.environ, "COPYRIGHT_BASE_REF": ""},
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

    def test_generates_complete_application_without_placeholders(self):
        files = {path.relative_to(self.app).as_posix() for path in self.app.rglob("*")}
        self.assertEqual({name for name in files if (self.app / name).is_file()}, GENERATED)
        for name in GENERATED:
            text = (self.app / name).read_text(encoding="utf-8")
            self.assertNotIn("{{", text, name)
            self.assertNotIn("{%", text, name)

    def test_generated_metadata_and_headers_pass_repository_hooks(self):
        result = self.run_script("validate_metadata.py", self.app / "metadata.json")
        self.assertEqual(result.returncode, 0, result.stderr)
        sources = [self.app / name for name in GENERATED if not name.endswith((".md", ".json"))]
        result = self.run_script("check_copyright.py", *sources)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cli_discovers_application_but_not_template(self):
        result = self.holoscan("list", "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        projects = json.loads(result.stdout)["projects"]
        template = self.root / "applications" / "template"
        for project in projects:
            self.assertFalse(Path(project["source_folder"]).is_relative_to(template), project)
        generated = [project for project in projects if project["name"] == NAME]
        self.assertEqual(len(generated), 1, projects)
        self.assertEqual(generated[0]["language"], ["C++"])
        self.assertEqual(set(generated[0]["modes"]), {"ci", "default"})

    def test_ci_lifecycle_commands_validate(self):
        for command in (
            ("run", NAME, "ci", "--local", "--dryrun", "--language", "cpp"),
            ("test", NAME, "--local", "--no-xvfb", "--dryrun", "--language", "cpp"),
        ):
            with self.subTest(command=command):
                result = self.holoscan(*command)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(shutil.which("cmake"), "CMake is not installed")
    def test_registered_application_configures_without_the_sdk(self):
        # Apply the template README's registration step: replace the line that
        # holoscan create appended with a sorted add_holoscan_application entry.
        cmakelists = self.root / "applications" / "CMakeLists.txt"
        lines = [
            line
            for line in cmakelists.read_text(encoding="utf-8").splitlines(keepends=True)
            if f"({NAME})" not in line
        ]
        entries = [line for line in lines if line.startswith("add_holoscan_application(")]
        others = [line for line in lines if line not in entries]
        entries = sorted([*entries, f"add_holoscan_application({NAME})\n"])
        cmakelists.write_text("".join(others + entries), encoding="utf-8")
        with tempfile.TemporaryDirectory() as build:
            result = subprocess.run(
                [
                    "cmake",
                    "-S",
                    str(self.root),
                    "-B",
                    build,
                    f"-DAPP_{NAME}=ON",
                    "-DHOLOHUB_BUILD_CPP=OFF",
                    "-DHOLOHUB_BUILD_PYTHON=OFF",
                ],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((Path(build) / "applications" / NAME).is_dir())

    def test_rejects_unsupported_language_and_sdk(self):
        for name, extra in (
            ("python_check", ("--language", "python")),
            ("sdk4_check", ("--context", "holoscan_version=4.6.0")),
        ):
            with self.subTest(name=name):
                result = self.holoscan("create", name, *self.template_args(*extra))
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertFalse((self.root / "applications" / name).exists())


if __name__ == "__main__":
    unittest.main()
