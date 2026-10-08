# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise repository hooks with real files and the installed CLI schemas."""

import json
import subprocess
import sys
import tempfile
import unittest
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

    def run_check(self, script, *paths):
        return subprocess.run(
            [sys.executable, str(SCRIPTS / script), *map(str, paths)],
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

    def test_no_filenames_succeeds_without_checking_the_script_itself(self):
        for script in (
            "check_spdx.py",
            "check_internal_artifacts.py",
            "validate_metadata.py",
        ):
            with self.subTest(script=script):
                result = self.run_check(script)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout + result.stderr, "")

    def test_spdx_accepts_licensed_files_and_preserves_decode_tolerance(self):
        paths = [
            self.write_file("licensed file.py", "# SPDX-License-Identifier: Apache-2.0\n"),
            self.write_file("encoded.cpp", b"\xff// SPDX-License-Identifier: Apache-2.0\n"),
        ]
        result = self.run_check("check_spdx.py", *paths)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout + result.stderr, "")

    def test_spdx_reports_every_file_without_the_required_license(self):
        missing = self.write_file("missing license.py", "print('hello')\n")
        wrong = self.write_file("wrong.py", "# SPDX-License-Identifier: MIT\n")
        valid = self.write_file("valid.py", "# SPDX-License-Identifier: Apache-2.0\n")
        result = self.run_check("check_spdx.py", missing, valid, wrong)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn(str(missing), result.stderr)
        self.assertIn(str(wrong), result.stderr)
        self.assertNotIn(str(valid), result.stderr)

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
