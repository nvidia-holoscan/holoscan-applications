# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise repository hooks with real files and the installed CLI schemas."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
METADATA = SCRIPTS.parents[1] / "applications" / "endoscopy_tool_tracking" / "metadata.json"


class RepositoryChecksTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def write_file(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode("utf-8") if isinstance(content, str) else content)
        return path

    def run_check(self, script, *paths, base_ref=""):
        return subprocess.run(
            [sys.executable, str(SCRIPTS / script), *map(str, paths)],
            cwd=self.root,
            env={**os.environ, "COPYRIGHT_BASE_REF": base_ref},
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

    def test_no_filenames_succeeds_without_checking_the_script_itself(self):
        for script in (
            "check_copyright.py",
            "check_internal_artifacts.py",
            "validate_metadata.py",
        ):
            with self.subTest(script=script):
                result = self.run_check(script)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout + result.stderr, "")

    def test_spdx_accepts_licensed_files_and_preserves_decode_tolerance(self):
        year = datetime.now(UTC).year
        paths = [
            self.write_file(
                "licensed file.py",
                f"# SPDX-FileCopyrightText: Copyright (c) {year} NVIDIA CORPORATION & AFFILIATES.\n"
                "# SPDX-License-Identifier: Apache-2.0\n",
            ),
            self.write_file(
                "encoded.cpp",
                f"// SPDX-FileCopyrightText: 2020-{year} Example contributors\n".encode()
                + b"// SPDX-License-Identifier: Apache-2.0\n// Invalid UTF-8: \xff\n",
            ),
            self.write_file(
                "legacy.c",
                f"/* Copyright (c) {year} Example contributors */\n"
                "/* SPDX-License-Identifier: Apache-2.0 */\n",
            ),
        ]
        result = self.run_check("check_copyright.py", *paths)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout + result.stderr, "")

    def test_spdx_reports_every_file_without_the_required_license(self):
        copyright_notice = f"# Copyright (c) {datetime.now(UTC).year} Example contributors\n"
        missing = self.write_file("missing license.py", copyright_notice + "print('hello')\n")
        wrong = self.write_file("wrong.py", copyright_notice + "# SPDX-License-Identifier: MIT\n")
        valid = self.write_file("valid.py", self.copyright_source())
        result = self.run_check("check_copyright.py", missing, valid, wrong)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn(str(missing), result.stderr)
        self.assertIn(str(wrong), result.stderr)
        self.assertNotIn(str(valid), result.stderr)

    def copyright_source(self, years=None):
        years = str(datetime.now(UTC).year) if years is None else years
        return (
            f"# SPDX-FileCopyrightText: Copyright (c) {years} NVIDIA CORPORATION & AFFILIATES.\n"
            "# SPDX-License-Identifier: Apache-2.0\n"
        )

    def test_spdx_rejects_missing_or_malformed_copyright_notices(self):
        year = datetime.now(UTC).year
        for notice in (
            "",
            "# SPDX-FileCopyrightText: NVIDIA\n",
            f"# Copyright {year}\n",
            f"/* Copyright (c) {year} */\n",
        ):
            with self.subTest(notice=notice):
                path = self.write_file(
                    "invalid.py", notice + "# SPDX-License-Identifier: Apache-2.0\n"
                )
                result = self.run_check("check_copyright.py", path)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn(str(path), result.stderr)
                self.assertIn("copyright", result.stderr.lower())

    def test_spdx_rejects_reversed_future_and_outdated_years(self):
        year = datetime.now(UTC).year
        for years in (f"{year}-{year - 1}", str(year + 1), f"2020-{year + 1}", str(year - 1)):
            with self.subTest(years=years):
                path = self.write_file("invalid.py", self.copyright_source(years))
                result = self.run_check("check_copyright.py", path)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn(str(path), result.stderr)
                self.assertIn("year", result.stderr.lower())

    def test_spdx_preserves_older_third_party_notices(self):
        source = "# Copyright (c) 2010-2020 Other contributors\n" + self.copyright_source()
        path = self.write_file("attribution.py", source)
        result = self.run_check("check_copyright.py", path)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(path.read_text(), source)

    def git(self, *args):
        return subprocess.run(
            ["git", *args],
            cwd=self.root,
            env={**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull},
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    def init_copyright_repository(self):
        self.git("init", "-q")
        self.git("config", "user.name", "CI test")
        self.git("config", "user.email", "ci@example.com")
        self.git("config", "commit.gpgsign", "false")
        old = str(datetime.now(UTC).year - 1)
        paths = [
            self.write_file(name, self.copyright_source(old)) for name in ("old.py", "edit me.py")
        ]
        self.git("add", ".")
        self.git("commit", "-qm", "Initial files")
        return paths

    def test_spdx_only_requires_current_year_for_local_changes(self):
        old, changed = self.init_copyright_repository()
        result = self.run_check("check_copyright.py", old, changed)
        self.assertEqual(result.returncode, 0, result.stderr)

        changed.write_text(changed.read_text() + "print('changed')\n")
        new = self.write_file("new.py", self.copyright_source(str(datetime.now(UTC).year - 1)))
        for staged in (False, True):
            with self.subTest(staged=staged):
                if staged:
                    self.git("add", ".")
                result = self.run_check("check_copyright.py", old, changed, new)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertNotIn(str(old), result.stderr)
                self.assertIn(str(changed), result.stderr)
                self.assertIn(str(new), result.stderr)

    def test_spdx_checks_committed_changes_against_ci_base(self):
        old, changed = self.init_copyright_repository()
        base = self.git("rev-parse", "HEAD")
        changed.write_text(changed.read_text() + "print('changed')\n")
        self.git("add", ".")
        self.git("commit", "-qm", "Modify a file")
        result = self.run_check("check_copyright.py", old, changed, base_ref=base)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertNotIn(str(old), result.stderr)
        self.assertIn(str(changed), result.stderr)

    def test_spdx_checks_all_files_for_initial_push(self):
        paths = self.init_copyright_repository()
        result = self.run_check("check_copyright.py", *paths, base_ref="0" * 40)
        self.assertEqual(result.returncode, 1, result.stderr)
        for path in paths:
            self.assertIn(str(path), result.stderr)

    def test_spdx_checks_new_local_repository_without_head(self):
        self.git("init", "-q")
        path = self.write_file("new.py", self.copyright_source(str(datetime.now(UTC).year - 1)))
        self.git("add", ".")
        result = self.run_check("check_copyright.py", path)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn(str(path), result.stderr)

    def test_spdx_validates_headers_even_when_unchanged(self):
        old, changed = self.init_copyright_repository()
        changed.write_text("# SPDX-License-Identifier: Apache-2.0\n")
        self.git("add", ".")
        self.git("commit", "-qm", "Missing copyright")
        result = self.run_check("check_copyright.py", old, changed)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertNotIn(str(old), result.stderr)
        self.assertIn(str(changed), result.stderr)

    def test_spdx_rejects_unavailable_ci_base(self):
        paths = self.init_copyright_repository()
        result = self.run_check("check_copyright.py", *paths, base_ref="missing-reference")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("missing-reference", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_internal_artifact_check_accepts_public_references(self):
        paths = [
            self.write_file("public.json", '{"image": "nvcr.io/nvidia/holoscan"}\n'),
            self.write_file("encoded.txt", b"public data\xff"),
        ]
        result = self.run_check("check_internal_artifacts.py", *paths)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout + result.stderr, "")

    def test_internal_artifact_check_reports_all_matches_case_insensitively(self):
        marker = "Nv" + "StAgInG"
        mixed = self.write_file("mixed case.txt", f"https://{marker}.example/image")
        lower = self.write_file("lower.txt", marker.lower())
        valid = self.write_file("valid.txt", "public image")
        result = self.run_check("check_internal_artifacts.py", mixed, valid, lower)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn(str(mixed), result.stderr)
        self.assertIn(str(lower), result.stderr)
        self.assertNotIn(str(valid), result.stderr)

    def test_metadata_accepts_valid_application_metadata(self):
        path = self.write_file("applications/valid app/metadata.json", METADATA.read_bytes())
        result = self.run_check("validate_metadata.py", path)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout + result.stderr, "")

    def test_metadata_reports_json_and_schema_errors_in_the_same_batch(self):
        malformed = self.write_file("applications/malformed/metadata.json", "{not JSON")
        document = json.loads(METADATA.read_text(encoding="utf-8"))
        document["application"]["name"] = 123
        invalid = self.write_file("applications/invalid/metadata.json", json.dumps(document))
        valid = self.write_file("applications/valid/metadata.json", METADATA.read_bytes())
        result = self.run_check("validate_metadata.py", malformed, valid, invalid)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn(str(malformed), result.stderr)
        self.assertIn(str(invalid), result.stderr)
        self.assertNotIn(str(valid), result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_metadata_reports_read_failures_and_continues_checking(self):
        missing = self.root / "applications/missing/metadata.json"
        malformed = self.write_file("applications/malformed/metadata.json", "{not JSON")
        result = self.run_check("validate_metadata.py", missing, malformed)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn(str(missing), result.stderr)
        self.assertIn(str(malformed), result.stderr)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
