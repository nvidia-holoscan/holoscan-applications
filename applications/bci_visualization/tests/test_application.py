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


BACKDATED_REPLAY = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv.pop(1)).parent))
import holoscan as hs
import bci_visualization as app
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
raise SystemExit(app.main())
"""


class ApplicationTest(unittest.TestCase):
    def test_synthetic_replay_renders_volumes(self) -> None:
        self._check_render(["--replay-rate", "realtime"])

    def test_renders_and_finishes_with_old_capture_timestamps(self) -> None:
        self._render(["--replay-rate", "realtime"], backdate=True)

    def _check_render(self, options: list[str]) -> None:
        self._render(options)

    def _render(self, options: list[str], *, backdate: bool = False) -> None:
        import vtk

        with tempfile.TemporaryDirectory() as directory:
            screenshot = Path(directory) / "frame.png"
            environment = dict(os.environ)
            environment.setdefault("VTK_DEFAULT_OPENGL_WINDOW", "vtkEGLRenderWindow")
            result = subprocess.run(
                [
                    *(
                        [sys.executable, "-c", BACKDATED_REPLAY, str(APP)]
                        if backdate
                        else [sys.executable, str(APP)]
                    ),
                    "--synthetic",
                    "--headless",
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
            self.assertIn("camera_moved True", result.stdout, output)
            self.assertIn("same_thread True", result.stdout, output)
            reader = vtk.vtkPNGReader()
            reader.SetFileName(str(screenshot))
            reader.Update()
            low, high = reader.GetOutput().GetPointData().GetScalars().GetRange()
            self.assertGreater(high, low, "the rendered frame is blank")


if __name__ == "__main__":
    unittest.main()
