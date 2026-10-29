# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the bci_visualization data preparation utility, using a local file:// dataset."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
SCRIPT = APP_DIR / "prepare_data.py"
sys.path.insert(0, str(APP_DIR))

from prepare_data import CHECKSUMS, HEADSETS, REVISION, dataset_files


class PrepareDataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.dataset = self.root / "dataset"
        self.output = self.root / "prepared"
        self.checksums = self.root / "checksums.json"
        self.manifest: dict[str, dict[str, object]] = {}

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _publish(self, headset: str, *, skip: str | None = None) -> None:
        for remote, _ in dataset_files(headset):
            if remote == skip:
                continue
            path = self.dataset / REVISION / remote
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(remote.encode())
            self.manifest[remote] = {
                "size": len(remote),
                "sha256": hashlib.sha256(remote.encode()).hexdigest(),
            }
        self.checksums.write_text(json.dumps(self.manifest))

    def _run(self, *arguments: object) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--base-url",
                self.dataset.as_uri(),
                "--output",
                str(self.output),
                "--checksums",
                str(self.checksums),
                *(str(argument) for argument in arguments),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_downloads_the_selected_headset_into_the_application_layout(self) -> None:
        self._publish("devkit_3")
        result = self._run("--headset", "devkit_3")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.output / "data.snirf").read_bytes(), b"data.snirf")
        self.assertEqual(
            (self.output / "flow_mega_jacobian.npy").read_bytes(),
            b"devkit_3/flow_mega_jacobian.npy",
        )
        self.assertEqual(
            (self.output / "voxel_info" / "ijk.npy").read_bytes(), b"devkit_3/voxel_info/ijk.npy"
        )

    def test_keeps_matching_files_unless_forced(self) -> None:
        self._publish("flow_2")
        self.assertEqual(self._run().returncode, 0)
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("Downloading", result.stdout)
        result = self._run("--force")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("Downloading"), len(dataset_files("flow_2")))

    def test_replaces_files_that_do_not_match_their_checksum(self) -> None:
        self._publish("flow_2")
        self.output.mkdir()
        (self.output / "data.snirf").write_bytes(b"local")
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.output / "data.snirf").read_bytes(), b"data.snirf")

    def test_replaces_headset_files_when_the_headset_changes(self) -> None:
        self._publish("flow_2")
        self._publish("flow_2_3_753")
        self.assertEqual(self._run("--headset", "flow_2_3_753").returncode, 0)
        result = self._run("--headset", "flow_2")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            (self.output / "flow_mega_jacobian.npy").read_bytes(), b"flow_2/flow_mega_jacobian.npy"
        )
        self.assertNotIn("Downloading data.snirf", result.stdout)

    def test_rejects_a_download_that_does_not_match_its_checksum(self) -> None:
        self._publish("flow_2")
        self.manifest["data.snirf"]["sha256"] = "0" * 64
        self.checksums.write_text(json.dumps(self.manifest))
        result = self._run()
        self.assertEqual(result.returncode, 1)
        self.assertIn("checksum mismatch", result.stderr)
        self.assertFalse((self.output / "data.snirf").exists())
        self.assertEqual(sorted(p.name for p in self.output.glob("tmp*")), [])

    def test_bundled_checksums_cover_every_headset(self) -> None:
        checksums = json.loads(CHECKSUMS.read_text())
        for headset in HEADSETS:
            for remote, _ in dataset_files(headset):
                self.assertRegex(checksums[remote]["sha256"], "^[0-9a-f]{64}$")

    def test_reports_a_missing_file_without_leaving_partial_output(self) -> None:
        self._publish("flow_2", skip="flow_2/flow_channel_map.json")
        result = self._run()
        self.assertEqual(result.returncode, 1)
        self.assertIn("flow_channel_map.json", result.stderr)
        self.assertFalse((self.output / "flow_channel_map.json").exists())
        self.assertEqual(sorted(p.name for p in self.output.glob("tmp*")), [])

    def test_rejects_an_unknown_headset(self) -> None:
        result = self._run("--headset", "unknown")
        self.assertEqual(result.returncode, 2)
        self.assertIn("invalid choice", result.stderr)


if __name__ == "__main__":
    unittest.main()
