# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the ci mode end to end and check the offscreen render (needs a GPU)."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "python" / "bci_visualization.py"


CHECKED_REPLAY = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv.pop(1)).parent))
import bci_visualization as app
from operators.vtk_volume_view import VTKVolumeViewOp
original_view_compute = VTKVolumeViewOp.compute
def checked_view_compute(self, context):
    samples = self.volume.selected_batch()
    original_view_compute(self, context)
    if samples and 0 <= self.last_frame <= samples[-1].metadata.frame_id:
        print("final_frame_rendered True", flush=True)
VTKVolumeViewOp.compute = checked_view_compute
"""

BACKDATED_REPLAY = """
import holoscan as hs
from operators.stream import StreamOperator
original_compute = StreamOperator.compute
class BackdatedContext:
    def __init__(self, context):
        self.context = context
        point = context.activation_time
        self.activation_time = hs.time.TimePoint(point.timestamp_ns - 2_000_000_000, point.clock_id)
    def __getattr__(self, name):
        return getattr(self.context, name)
def backdated_compute(self, context):
    return original_compute(self, BackdatedContext(context))
StreamOperator.compute = backdated_compute
"""


class ApplicationTest(unittest.TestCase):
    def test_two_samples_produce_one_volume_and_finish(self) -> None:
        for view in ("surface", "volume"):
            for source in ([], ["--source", "kernel", "--simulate-device"]):
                with self.subTest(view=view, source=source):
                    environment = dict(os.environ)
                    environment.setdefault("VTK_DEFAULT_OPENGL_WINDOW", "vtkEGLRenderWindow")
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(APP),
                            "--synthetic",
                            "--headless",
                            "--frames",
                            "2",
                            "--view",
                            view,
                            *source,
                        ],
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=240,
                        env=environment,
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertRegex(result.stdout, r"volumes 1\b")

    def test_synthetic_replay_renders_volumes(self) -> None:
        self._check_render(["--replay-rate", "realtime"])

    def test_simulated_headset_renders_volumes(self) -> None:
        self._check_render(["--source", "kernel", "--simulate-device"])

    def test_simulated_headset_stops_at_eof_without_a_frame_limit(self) -> None:
        self._check_render(["--source", "kernel", "--simulate-device", "--frames", "0"])

    @unittest.skipUnless(os.environ.get("DISPLAY"), "interactive replay needs an X display")
    def test_interactive_replay_stops_at_eof_without_a_frame_limit(self) -> None:
        for source in ([], ["--source", "kernel", "--simulate-device"]):
            with self.subTest(source=source):
                self._check_render([*source, "--frames", "0"], headless=False)

    def test_renders_and_finishes_with_old_capture_timestamps(self) -> None:
        for view in ("surface", "volume"):
            with self.subTest(view=view):
                self._render(["--view", view, "--replay-rate", "realtime"], backdate=True)

    def _check_render(self, options: list[str], *, headless: bool = True) -> None:
        for view in ("surface", "volume"):
            with self.subTest(view=view):
                self._render([*options, "--view", view], headless=headless)

    def _render(self, options: list[str], *, backdate: bool = False, headless: bool = True) -> None:
        import vtk

        with tempfile.TemporaryDirectory() as directory:
            screenshot = Path(directory) / "frame.png"
            environment = dict(os.environ)
            environment["VTK_DEFAULT_OPENGL_WINDOW"] = (
                "vtkEGLRenderWindow" if headless else "vtkXOpenGLRenderWindow"
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    CHECKED_REPLAY
                    + (BACKDATED_REPLAY if backdate else "")
                    + "\nraise SystemExit(app.main())",
                    str(APP),
                    "--synthetic",
                    *(["--headless"] if headless else []),
                    "--frames",
                    "8",
                    *options,
                    "--simulate-drag",
                    "--screenshot",
                    str(screenshot),
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=240,
                env=environment,
            )
            output = result.stdout + result.stderr
            self.assertEqual(result.returncode, 0, output)
            volumes = re.search(r"volumes (\d+)", result.stdout)
            self.assertIsNotNone(volumes, output)
            self.assertGreater(int(volumes.group(1)), 1, output)
            self.assertIn("final_frame_rendered True", result.stdout, output)
            self.assertIn("camera_moved True", result.stdout, output)
            self.assertIn("same_thread True", result.stdout, output)
            reader = vtk.vtkPNGReader()
            reader.SetFileName(str(screenshot))
            reader.Update()
            low, high = reader.GetOutput().GetPointData().GetScalars().GetRange()
            self.assertGreater(high, low, "the rendered frame is blank")


if __name__ == "__main__":
    unittest.main()
