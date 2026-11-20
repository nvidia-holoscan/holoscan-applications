# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the actual CMake download target with tiny local archives, without the SDK."""

import hashlib
import shutil
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
MP4 = "sample_clip_out_of_body_detection.mp4"
ONNX = "out_of_body_detection.onnx"


class DownloadDataTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.source = self.root / "source"
        (self.source / "cpp").mkdir(parents=True)
        shutil.copytree(APP_DIR / "cmake", self.source / "cmake")
        (self.source / "cpp/CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 4.0)\n"
            "project(download_test NONE)\n"
            "include(../cmake/DownloadData.cmake)\n",
            encoding="utf-8",
        )
        self.data = self.root / "data/endoscopy_out_of_body_detection"

    def archive(self, name, contents):
        path = self.root / name
        with zipfile.ZipFile(path, "w") as archive:
            for member in (MP4, ONNX):
                # Fixed old timestamps ensure the build handles extracted files older than
                # its configured dependency, without repeatedly executing the fetch command.
                archive.writestr(zipfile.ZipInfo(member, (2020, 1, 1, 0, 0, 0)), contents)
        return path.as_uri(), hashlib.sha256(path.read_bytes()).hexdigest()

    def configure(self, build_name, version, url, sha):
        build = self.root / build_name
        result = subprocess.run(
            [
                "cmake",
                "-S",
                str(self.source / "cpp"),
                "-B",
                str(build),
                "-G",
                "Unix Makefiles",
                f"-DHOLOHUB_DATA_DIR={self.data.parent}",
                f"-DSURGICAL_DEID_NGC_VERSION={version}",
                f"-DSURGICAL_DEID_NGC_URL={url}",
                f"-DSURGICAL_DEID_NGC_SHA256={sha}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return build

    def build(self, build, *, succeeds=True):
        result = subprocess.run(
            ["cmake", "--build", str(build), "--target", "de_id_download_data"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode == 0, succeeds, result.stdout + result.stderr)
        return result.stdout + result.stderr

    def test_new_build_refreshes_shared_data_when_version_changes(self):
        url, _ = self.archive("sample.zip", "original")
        self.build(self.configure("first", "v1", url, ""))
        # Same URL, new version: replace the source archive while verification is disabled.
        self.archive("sample.zip", "updated")
        build = self.configure("second", "v2", url, "")
        self.build(build)
        self.assertEqual((self.data / ONNX).read_text(), "updated")
        self.assertEqual((self.data / MP4).read_text(), "updated")
        self.assertTrue((self.data / ".ngc_version").read_text().startswith("v2:"))
        self.assertNotIn("Downloading", self.build(build))

    def test_new_build_refreshes_shared_data_when_url_changes(self):
        url, _ = self.archive("first.zip", "original")
        self.build(self.configure("first", "v1", url, ""))
        url, _ = self.archive("second.zip", "updated")
        build = self.configure("second", "v1", url, "")
        self.build(build)
        self.assertEqual((self.data / ONNX).read_text(), "updated")
        self.assertEqual((self.data / MP4).read_text(), "updated")
        # Unchanged reconfiguration must not invalidate the configured dependency.
        self.configure("second", "v1", url, "")
        self.assertNotIn("Downloading", self.build(build))
        (self.data / MP4).unlink()
        self.build(build)
        self.assertEqual((self.data / MP4).read_text(), "updated")

    def test_new_build_checks_changed_checksum_before_accepting_shared_data(self):
        url, sha = self.archive("sample.zip", "original")
        self.build(self.configure("first", "v1", url, sha))
        build = self.configure("second", "v1", url, "0" * 64)
        self.build(build, succeeds=False)
        self.assertEqual((self.data / ONNX).read_text(), "original")


if __name__ == "__main__":
    unittest.main()
