# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The application graph compiles with the SDK, without running on a GPU."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "python" / "bci_visualization.py"


class GraphTest(unittest.TestCase):
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
