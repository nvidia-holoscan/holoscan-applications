# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the bci_visualization data preparation utility, using a local file:// dataset."""

from __future__ import annotations

import hashlib
import json
import signal
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

APP_DIR = Path(__file__).resolve().parents[1]
SCRIPT = APP_DIR / "prepare_data.py"
sys.path.insert(0, str(APP_DIR))

import prepare_data as prepare_data_module
from prepare_data import (
    CHECKSUMS,
    DATASET_ID,
    DATASET_URL,
    HEADSETS,
    REVISION,
    PreparationError,
    dataset_files,
    download_from_hub,
    download_from_url,
)


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

    def test_hub_download_uses_the_pinned_dataset_and_publishes_verified_file(self) -> None:
        remote = "flow_2/flow_channel_map.json"
        payload = remote.encode()
        expected = {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        destination = self.output / "flow_channel_map.json"
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b"previous verified file")
        calls: list[dict[str, object]] = []

        def fake_hf_hub_download(**kwargs: object) -> str:
            calls.append(kwargs)
            staged = Path(str(kwargs["local_dir"])) / remote
            staged.parent.mkdir(parents=True)
            staged.write_bytes(payload)
            return str(staged)

        fake_module = types.ModuleType("huggingface_hub")
        fake_module.hf_hub_download = fake_hf_hub_download
        with patch.dict(sys.modules, {"huggingface_hub": fake_module}):
            download_from_hub(remote, destination, expected)

        self.assertEqual(destination.read_bytes(), payload)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["repo_id"], DATASET_ID)
        self.assertEqual(calls[0]["filename"], remote)
        self.assertEqual(calls[0]["repo_type"], "dataset")
        self.assertEqual(calls[0]["revision"], REVISION)
        self.assertTrue(Path(str(calls[0]["local_dir"])).parent == self.output)
        self.assertEqual(list(self.output.glob(".bci-download-*")), [])

    def test_default_preparation_routes_every_asset_through_hub(self) -> None:
        self._publish("flow_2")
        calls: list[str] = []

        def fake_download(
            remote: str, destination: Path, expected: dict[str, object], *limits: float
        ) -> None:
            calls.append(remote)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((self.dataset / REVISION / remote).read_bytes())

        options = prepare_data_module.parse_arguments(
            ["--output", str(self.output), "--checksums", str(self.checksums)]
        )
        self.assertEqual(options.base_url, DATASET_URL)
        with patch.object(prepare_data_module, "download_from_hub", side_effect=fake_download):
            prepare_data_module.prepare_data(options)

        self.assertEqual(calls, [remote for remote, _ in dataset_files("flow_2")])

    def test_bad_hub_download_preserves_existing_destination(self) -> None:
        remote = "flow_2/flow_channel_map.json"
        destination = self.output / "flow_channel_map.json"
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b"previous verified file")
        staged_payload = b"wrong contents"

        def fake_hf_hub_download(**kwargs: object) -> str:
            staged = Path(str(kwargs["local_dir"])) / remote
            staged.parent.mkdir(parents=True)
            staged.write_bytes(staged_payload)
            return str(staged)

        fake_module = types.ModuleType("huggingface_hub")
        fake_module.hf_hub_download = fake_hf_hub_download
        with (
            patch.dict(sys.modules, {"huggingface_hub": fake_module}),
            self.assertRaisesRegex(PreparationError, "checksum mismatch"),
        ):
            download_from_hub(remote, destination, {"size": 1, "sha256": "0" * 64})

        self.assertEqual(destination.read_bytes(), b"previous verified file")
        self.assertEqual(list(self.output.glob(".bci-download-*")), [])

    def test_hub_download_applies_timeouts_and_preserves_destination(self) -> None:
        destination = self.output / "flow_channel_map.json"
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b"previous verified file")
        calls: list[dict[str, object]] = []

        def stalled_hf_hub_download(**kwargs: object) -> str:
            calls.append(kwargs)
            time.sleep(2)
            return ""

        fake_module = types.ModuleType("huggingface_hub")
        fake_module.hf_hub_download = stalled_hf_hub_download
        handler = signal.getsignal(signal.SIGALRM)
        started = time.monotonic()
        with (
            patch.dict(sys.modules, {"huggingface_hub": fake_module}),
            self.assertRaisesRegex(PreparationError, "deadline"),
        ):
            download_from_hub(
                "flow_2/flow_channel_map.json", destination, {"size": 1, "sha256": "0" * 64}, 7, 0.1
            )

        self.assertLess(time.monotonic() - started, 1.5)
        self.assertEqual(calls[0]["etag_timeout"], 7)
        self.assertEqual(destination.read_bytes(), b"previous verified file")
        self.assertEqual(list(self.output.glob(".bci-download-*")), [])
        self.assertEqual(signal.getsignal(signal.SIGALRM), handler)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0, 0))

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


class DownloadTimeoutTest(unittest.TestCase):
    def test_timeout_arguments(self):
        from prepare_data import parse_arguments

        for flag in ("--network-timeout", "--download-deadline"):
            for value in ("0", "-1", "nan", "inf"):
                with self.subTest(flag=flag, value=value), self.assertRaises(SystemExit):
                    parse_arguments(["--output", "unused", flag, value])

    def test_network_stalls_and_overall_deadline(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                try:
                    if self.path == "/headers":
                        time.sleep(0.5)
                    self.send_response(200)
                    self.end_headers()
                    if self.path == "/stall":
                        time.sleep(0.5)
                    if self.path == "/trickle":
                        for _ in range(30):
                            self.wfile.write(b"x")
                            self.wfile.flush()
                            time.sleep(0.03)
                    else:
                        self.wfile.write(b"x")
                except (BrokenPipeError, ConnectionResetError):
                    pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                destination = Path(directory) / "data"
                expected = {"size": 1, "sha256": hashlib.sha256(b"x").hexdigest()}
                handler = signal.getsignal(signal.SIGALRM)
                for route, network, deadline in (
                    ("headers", 0.1, 2),
                    ("stall", 0.1, 2),
                    ("headers", 2, 0.15),
                    ("trickle", 0.2, 0.15),
                ):
                    with self.subTest(route=route, deadline=deadline):
                        destination.write_bytes(b"existing")
                        started = time.monotonic()
                        with self.assertRaisesRegex(PreparationError, "timed out|deadline"):
                            download_from_url(
                                f"http://127.0.0.1:{server.server_port}/{route}",
                                destination,
                                expected,
                                network,
                                deadline,
                            )
                        self.assertLess(time.monotonic() - started, 1.5)
                        self.assertEqual(destination.read_bytes(), b"existing")
                        self.assertEqual(list(Path(directory).iterdir()), [destination])
                        self.assertEqual(signal.getsignal(signal.SIGALRM), handler)
                        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0, 0))
                download_from_url(
                    f"http://127.0.0.1:{server.server_port}/ok", destination, expected, 1, 2
                )
                self.assertEqual(destination.read_bytes(), b"x")
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
