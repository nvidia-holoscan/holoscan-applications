# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The application graph compiles with the SDK, without running on a GPU."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "python" / "bci_visualization.py"

SHORT_REPLAY = """
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv.pop(1)).parent))
from synthetic import write_synthetic_dataset
import bci_visualization as app
samples = int(sys.argv.pop(1))
with tempfile.TemporaryDirectory() as directory:
    data = write_synthetic_dataset(Path(directory), samples=samples)
    raise SystemExit(app.main(["--data", str(data), *sys.argv[1:]]))
"""


class GraphTest(unittest.TestCase):
    def test_rejects_one_frame_for_every_source(self) -> None:
        for source in ([], ["--source", "kernel"], ["--source", "kernel", "--simulate-device"]):
            with self.subTest(source=source):
                result = subprocess.run(
                    [sys.executable, str(APP), "--frames", "1", "--compile-only", *source],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=30,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("a baseline plus another sample", result.stderr)

    def test_validates_effective_recording_length(self) -> None:
        for samples in (0, 1, 2):
            for source in ([], ["--source", "kernel", "--simulate-device"]):
                for frames in (0, 8):
                    with self.subTest(samples=samples, source=source, frames=frames):
                        result = subprocess.run(
                            [
                                sys.executable,
                                "-c",
                                SHORT_REPLAY,
                                str(APP),
                                str(samples),
                                "--compile-only",
                                "--frames",
                                str(frames),
                                *source,
                            ],
                            capture_output=True,
                            text=True,
                            check=False,
                            timeout=120,
                        )
                        if samples < 2:
                            self.assertNotEqual(result.returncode, 0)
                            self.assertIn("a baseline plus another sample", result.stderr)
                        else:
                            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                            self.assertIn("compile: ok", result.stdout)

    def test_compiles_on_synthetic_data(self) -> None:
        result = subprocess.run(
            [sys.executable, str(APP), "--synthetic", "--headless", "--compile-only"],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("compile: ok", result.stdout)

    def test_compiles_with_a_simulated_headset(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(APP),
                "--synthetic",
                "--source",
                "kernel",
                "--simulate-device",
                "--headless",
                "--compile-only",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("compile: ok", result.stdout)

    def test_synthetic_headset_needs_simulation(self) -> None:
        result = subprocess.run(
            [sys.executable, str(APP), "--synthetic", "--source", "kernel", "--compile-only"],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--simulate-device", result.stderr)

    def test_reports_missing_data(self) -> None:
        result = subprocess.run(
            [sys.executable, str(APP), "--data", "/nonexistent", "--compile-only"],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("prepare_data.py", result.stderr)


if __name__ == "__main__":
    unittest.main()
