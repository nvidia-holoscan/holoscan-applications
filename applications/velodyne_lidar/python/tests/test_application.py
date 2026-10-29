# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Black-box application checks; CTest supplies VELODYNE_EXECUTABLE."""

import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import make_pcap


def gpu_available():
    smi = shutil.which("nvidia-smi")
    return bool(smi) and subprocess.run([smi], capture_output=True, check=False).returncode == 0


@unittest.skipUnless(os.environ.get("VELODYNE_EXECUTABLE"), "requires the built application")
class ApplicationTest(unittest.TestCase):
    def run_app(self, *args, timeout=120):
        return subprocess.run(
            [os.environ["VELODYNE_EXECUTABLE"], "--display", "false", *map(str, args)],
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )

    def test_rejects_invalid_options(self):
        cases = [
            ("--count", "-1"),
            ("--snapshot", "y.ppm", "--source", "synthetic", "--count", "0"),
            ("--port", "70000"),
            ("--address", "lidar.local"),
            ("--view", "side"),
            ("--eye", "1,2"),
            ("--eye", "nan,0,1"),
            ("--eye", "inf,0,1"),
            ("--target", "0,nan,0"),
            ("--pcap", "/nonexistent.pcap"),
            ("--source", "daqiri"),
            ("--snapshot", "x.ppm"),  # live UDP is not finite
        ]
        for args in cases:
            with self.subTest(args=args):
                result = self.run_app(*args)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn("error:", result.stderr)

    @unittest.skipUnless(gpu_available(), "requires an NVIDIA GPU")
    def test_synthetic_run_renders_both_views(self):
        with tempfile.TemporaryDirectory() as tmp:
            for view in ("top", "chase"):
                snapshot = pathlib.Path(tmp) / f"{view}.ppm"
                result = self.run_app(
                    "--source",
                    "synthetic",
                    "--count",
                    "160",
                    "--source-period-ms",
                    "2",
                    "--view",
                    view,
                    "--width",
                    "320",
                    "--height",
                    "240",
                    "--snapshot",
                    snapshot,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("packets=160", result.stdout)
                self.assertIn("verdict=PASS", result.stdout)
                data = snapshot.read_bytes()
                header = b"P6\n320 240\n255\n"
                self.assertTrue(data.startswith(header))
                pixels = data[len(header) :]
                self.assertEqual(len(pixels), 320 * 240 * 3)
                lit = sum(1 for i in range(0, len(pixels), 3) if max(pixels[i : i + 3]) > 90)
                self.assertGreater(lit, 500, f"{view} view rendered too few points")

    @unittest.skipUnless(gpu_available(), "requires an NVIDIA GPU")
    def test_single_batch_runs_complete_with_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            pcap = pathlib.Path(tmp) / "one.pcap"
            make_pcap.write_pcap(pcap, 1)
            for name, source in (
                ("synthetic", ("--source", "synthetic", "--count", "1")),
                ("pcap", ("--pcap", pcap)),
            ):
                with self.subTest(source=name):
                    snapshot = pathlib.Path(tmp) / f"{name}.ppm"
                    result = self.run_app(
                        *source,
                        "--render-period-ms",
                        "200",
                        "--timeout-seconds",
                        "20",
                        "--snapshot",
                        snapshot,
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn("packets=1 ", result.stdout)
                    self.assertTrue(snapshot.is_file())

    @unittest.skipUnless(gpu_available(), "requires an NVIDIA GPU")
    def test_pcap_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "room.pcap"
            make_pcap.write_pcap(path, 100)
            result = self.run_app("--pcap", path)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("packets=100", result.stdout)
            self.assertIn("invalid=2", result.stdout)
            self.assertIn("verdict=PASS", result.stdout)


if __name__ == "__main__":
    unittest.main()
