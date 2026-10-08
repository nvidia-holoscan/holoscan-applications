# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the endoscopy sample-data preparation utility."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest
import zipfile
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
SCRIPT = APP_DIR / "prepare_data.py"
FRAME_BYTES = 854 * 480 * 3


class PrepareDataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.bin_dir = self.root / "bin"
        self.bin_dir.mkdir()
        self._write_fake_ffmpeg()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_fake_ffmpeg(self, *, fail: bool = False) -> None:
        fake_ffmpeg = self.bin_dir / "ffmpeg"
        body = f"""\
            #!{sys.executable}
            import pathlib
            import sys

            args = sys.argv[1:]
            if {fail!r}:
                print("fixture conversion failed", file=sys.stderr)
                raise SystemExit(9)
            required_pairs = (("-pix_fmt", "rgb24"), ("-f", "rawvideo"))
            for option, value in required_pairs:
                if option not in args or args[args.index(option) + 1] != value:
                    raise SystemExit(8)
            if "-vf" not in args or args[args.index("-vf") + 1] != "scale=854:480":
                raise SystemExit(8)
            if "-frames:v" in args and args[args.index("-frames:v") + 1] != "1":
                raise SystemExit(8)
            pathlib.Path(args[-1]).write_bytes(b"R" * {FRAME_BYTES})
            """
        fake_ffmpeg.write_text(textwrap.dedent(body), encoding="utf-8")
        fake_ffmpeg.chmod(fake_ffmpeg.stat().st_mode | stat.S_IXUSR)

    def _write_archive(
        self,
        *,
        members: list[tuple[str, bytes]] | None = None,
        name: str = "sample.zip",
    ) -> Path:
        archive = self.root / name
        if members is None:
            members = [
                ("nested/model/tool_loc_convlstm.onnx", b"onnx fixture"),
                ("nested/video/surgical_video.264", b"h264 fixture"),
            ]
        with zipfile.ZipFile(archive, "w") as bundle:
            for member_name, content in members:
                bundle.writestr(member_name, content)
        return archive

    def _run(self, *arguments: object) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["PATH"] = os.pathsep.join((str(self.bin_dir), environment.get("PATH", "")))
        return subprocess.run(
            [sys.executable, str(SCRIPT), *(str(argument) for argument in arguments)],
            cwd=APP_DIR,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_prepares_model_and_one_tightly_packed_rgb_frame_from_nested_members(
        self,
    ) -> None:
        archive = self._write_archive()
        output = self.root / "prepared"

        result = self._run("--archive", archive, "--output", output, "--frames", 1)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((output / "tool_loc_convlstm.onnx").read_bytes(), b"onnx fixture")
        rgb = (output / "surgical_video.rgb").read_bytes()
        self.assertEqual(len(rgb), FRAME_BYTES)
        self.assertEqual(rgb[:1], b"R")

    def test_rejects_negative_frame_limit_before_preparing_data(self) -> None:
        output = self.root / "prepared"

        result = self._run(
            "--archive", self.root / "missing.zip", "--output", output, "--frames", -1
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("non-negative", result.stderr)
        self.assertFalse(output.exists())

    def test_rejects_missing_duplicate_and_unsafe_required_members(self) -> None:
        cases = {
            "missing": [("data/tool_loc_convlstm.onnx", b"model")],
            "duplicate": [
                ("a/tool_loc_convlstm.onnx", b"model one"),
                ("b/tool_loc_convlstm.onnx", b"model two"),
                ("data/surgical_video.264", b"video"),
            ],
            "unsafe": [
                ("../tool_loc_convlstm.onnx", b"model"),
                ("data/surgical_video.264", b"video"),
            ],
        }
        for case, members in cases.items():
            with self.subTest(case=case):
                archive = self._write_archive(members=members, name=f"{case}.zip")
                output = self.root / case

                result = self._run("--archive", archive, "--output", output, "--frames", 1)

                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((output / "tool_loc_convlstm.onnx").exists())
                self.assertFalse((output / "surgical_video.rgb").exists())

    def test_existing_valid_assets_are_left_unchanged_without_force(self) -> None:
        output = self.root / "prepared"
        output.mkdir()
        model = output / "tool_loc_convlstm.onnx"
        video = output / "surgical_video.rgb"
        model.write_bytes(b"existing model")
        video.write_bytes(b"E" * FRAME_BYTES)

        result = self._run("--archive", self.root / "missing.zip", "--output", output)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(model.read_bytes(), b"existing model")
        self.assertEqual(video.read_bytes()[:1], b"E")

    def test_failed_forced_conversion_preserves_previous_valid_assets(self) -> None:
        archive = self._write_archive()
        output = self.root / "prepared"
        output.mkdir()
        model = output / "tool_loc_convlstm.onnx"
        video = output / "surgical_video.rgb"
        model.write_bytes(b"existing model")
        video.write_bytes(b"E" * FRAME_BYTES)
        self._write_fake_ffmpeg(fail=True)

        result = self._run("--archive", archive, "--output", output, "--frames", 1, "--force")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ffmpeg", result.stderr)
        self.assertEqual(model.read_bytes(), b"existing model")
        self.assertEqual(video.read_bytes()[:1], b"E")


if __name__ == "__main__":
    unittest.main()
