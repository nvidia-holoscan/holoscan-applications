# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Black-box replay regressions; CTest supplies ENDOSCOPY_EXECUTABLE."""

import os
import pathlib
import subprocess
import tempfile
import unittest


@unittest.skipUnless(os.environ.get("ENDOSCOPY_EXECUTABLE"), "requires the built application")
class ApplicationTest(unittest.TestCase):
    def run_app(self, *args):
        return subprocess.run(
            [os.environ["ENDOSCOPY_EXECUTABLE"], *map(str, args)],
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )

    def test_rejects_invalid_unsigned_options(self):
        for value in ("-1", "12bad", "4294967296", ""):
            with self.subTest(value=value):
                result = self.run_app("--count", value)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn("--count", result.stderr)

    def test_requires_model_for_tracking(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_app("--data", tmp, "--synthetic", "true", "--count", "1")
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("model", result.stderr.lower())

    def test_records_every_frame_and_replays_identical_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            result = self.run_app(
                "--synthetic",
                "true",
                "--inference",
                "false",
                "--display",
                "false",
                "--count",
                "5",
                "--width",
                "8",
                "--height",
                "6",
                "--period-ms",
                "1",
                "--record-dir",
                root / "first",
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            source = root / "first" / "input.rgb"
            data = source.read_bytes()
            self.assertEqual(len(data), 5 * 8 * 6 * 3)
            for frame in range(5):
                offset = frame * 144
                self.assertEqual(int.from_bytes(data[offset : offset + 4], "little"), frame)
                self.assertEqual(data[offset + 4], (4 + frame * 17) % 256)
            result = self.run_app(
                "--data",
                source.parent,
                "--basename",
                "input",
                "--count",
                "0",
                "--inference",
                "false",
                "--display",
                "false",
                "--width",
                "8",
                "--height",
                "6",
                "--period-ms",
                "1",
                "--record-dir",
                root / "second",
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual((root / "second" / "input.rgb").read_bytes(), data)

    def test_rejects_truncated_stream(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "surgical_video.rgb").write_bytes(b"x" * 143)
            result = self.run_app(
                "--data",
                root,
                "--inference",
                "false",
                "--display",
                "false",
                "--width",
                "8",
                "--height",
                "6",
            )
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("whole", result.stderr.lower())

    def test_recording_cannot_overwrite_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            data = b"x" * 144
            source = root / "input.rgb"
            source.write_bytes(data)
            result = self.run_app(
                "--data",
                root,
                "--basename",
                "input",
                "--inference",
                "false",
                "--display",
                "false",
                "--width",
                "8",
                "--height",
                "6",
                "--record-dir",
                root,
            )
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertEqual(source.read_bytes(), data)

    def test_count_exceeding_video_requires_repeat(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "surgical_video.rgb").write_bytes(bytes(range(144)))
            args = (
                "--data",
                root,
                "--inference",
                "false",
                "--display",
                "false",
                "--width",
                "8",
                "--height",
                "6",
                "--count",
                "3",
                "--period-ms",
                "1",
            )
            result = self.run_app(*args)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            result = self.run_app(*args, "--repeat", "true", "--record-dir", root / "record")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual((root / "record" / "input.rgb").read_bytes(), bytes(range(144)) * 3)

    def test_zero_synthetic_count_rejected(self):
        result = self.run_app("--synthetic", "true", "--inference", "false", "--count", "0")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)

    def test_cae_rejects_unknown_mode(self):
        result = self.run_app("--cae", "fast")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("--cae", result.stderr)

    def test_cae_encode_rejected_until_multimedia_is_available(self):
        result = self.run_app(
            "--synthetic", "true", "--inference", "false", "--count", "1", "--cae", "encode"
        )
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("holoscan-multimedia", result.stderr)

    def test_cae_encode_rejected_before_data_checks(self):
        result = self.run_app("--data", "/nonexistent", "--cae", "encode")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("holoscan-multimedia", result.stderr)

    def test_cae_paths_must_not_be_empty(self):
        for option in ("--cae-policy-config", "--cae-policy-dump"):
            with self.subTest(option=option):
                result = self.run_app("--cae", "policy", option, "")
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn(f"{option} requires a path", result.stderr)

    def test_cae_policy_requires_inference(self):
        result = self.run_app(
            "--synthetic", "true", "--inference", "false", "--count", "1", "--cae", "policy"
        )
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("--inference", result.stderr)

    def test_cae_options_require_policy_mode(self):
        for option in ("--cae-policy-config", "--cae-policy-dump"):
            with self.subTest(option=option):
                result = self.run_app(
                    "--synthetic", "true", "--inference", "false", "--count", "1", option, "x"
                )
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn("--cae policy", result.stderr)

    def test_cae_policy_config_errors_fail_before_running(self):
        configs = {
            "missing.json": None,
            "unknown_class.json": '{"schema_version": 1, "classes": {"forceps": "protect"}}',
            "bad_policy.json": '{"schema_version": 1, "classes": {"hook": "boost"}}',
            "duplicate.json": '{"schema_version": 1, "classes": {"0": "protect", "grasper": 1}}',
            "threshold.json": '{"schema_version": 1, "class_threshold": 2}',
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "tool_loc_convlstm.onnx").write_bytes(b"model placeholder")
            for name, text in configs.items():
                with self.subTest(config=name):
                    if text is not None:
                        (root / name).write_text(text)
                    result = self.run_app(
                        "--data",
                        root,
                        "--synthetic",
                        "true",
                        "--count",
                        "1",
                        "--display",
                        "false",
                        "--cae",
                        "policy",
                        "--cae-policy-config",
                        root / name,
                    )
                    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                    self.assertIn("policy config", result.stderr.lower())

    def test_cae_policy_dump_must_be_new(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "tool_loc_convlstm.onnx").write_bytes(b"model placeholder")
            result = self.run_app(
                "--data",
                root,
                "--synthetic",
                "true",
                "--count",
                "1",
                "--display",
                "false",
                "--cae",
                "policy",
                "--cae-policy-dump",
                root,
            )
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("already exists", result.stderr)

    def test_cae_policy_dump_must_be_creatable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "tool_loc_convlstm.onnx").write_bytes(b"model placeholder")
            (root / "a_file").write_text("not a directory")
            dump = root / "a_file" / "dump"
            result = self.run_app(
                "--data",
                root,
                "--synthetic",
                "true",
                "--count",
                "1",
                "--display",
                "false",
                "--cae",
                "policy",
                "--cae-policy-dump",
                dump,
            )
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("is not a writable directory", result.stderr)
            self.assertFalse(dump.exists())

    def test_cae_policy_dump_path_is_reported_absolute_and_normalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp).resolve()
            (root / "tool_loc_convlstm.onnx").write_bytes(b"model placeholder")
            (root / "existing").mkdir()
            result = subprocess.run(
                [
                    os.environ["ENDOSCOPY_EXECUTABLE"],
                    "--data",
                    str(root),
                    "--synthetic",
                    "true",
                    "--count",
                    "1",
                    "--display",
                    "false",
                    "--cae",
                    "policy",
                    "--cae-policy-dump",
                    "sub/./../existing/",
                ],
                cwd=root,
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn(f"already exists: {root / 'existing'}\n", result.stderr)


if __name__ == "__main__":
    unittest.main()
