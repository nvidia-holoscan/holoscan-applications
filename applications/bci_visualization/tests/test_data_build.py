# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Exercise real CMake dependency tracking with a tiny, offline data preparer."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

APP = Path(__file__).resolve().parents[1]


class DataBuildTest(unittest.TestCase):
    def test_selection_changes_and_failed_preparation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            (source / "python").mkdir(parents=True)
            for relative in ("CMakeLists.txt", "python/CMakeLists.txt"):
                shutil.copyfile(APP / relative, source / relative)
            (source / "dataset_checksums.json").write_text("{}")
            (source / "prepare_data.py").write_text(
                "import argparse\n"
                "from pathlib import Path\n"
                "p = argparse.ArgumentParser()\n"
                "p.add_argument('--output', type=Path)\n"
                "p.add_argument('--headset')\n"
                "a = p.parse_args()\n"
                "a.output.mkdir(parents=True, exist_ok=True)\n"
                "with (a.output / 'calls').open('a') as log:\n"
                "    log.write(a.headset + '\\n')\n"
                "if (a.output / 'fail').exists():\n"
                "    raise SystemExit(1)\n"
                "(a.output / 'headset').write_text(a.headset)\n"
            )
            build = root / "build"
            data = root / "data"

            def configure(headset="flow_2", enabled=True):
                subprocess.run(
                    [
                        "cmake",
                        "-S",
                        str(source),
                        "-B",
                        str(build),
                        "-DBUILD_TESTING=OFF",
                        f"-DBCI_VISUALIZATION_HEADSET={headset}",
                        f"-DHOLOHUB_DATA_DIR={data}",
                        f"-DHOLOHUB_DOWNLOAD_DATASETS={'ON' if enabled else 'OFF'}",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )

            def run(success=True):
                result = subprocess.run(
                    ["cmake", "--build", str(build)], check=False, capture_output=True, text=True
                )
                self.assertEqual(result.returncode == 0, success, result.stdout + result.stderr)

            for headset in ("flow_2", "devkit_3", "flow_2"):
                configure(headset)
                run()
                self.assertEqual((data / "bci_visualization/headset").read_text(), headset)
            calls = data / "bci_visualization/calls"
            self.assertEqual(calls.read_text().splitlines(), ["flow_2", "devkit_3", "flow_2"])
            configure()
            run()
            self.assertEqual(len(calls.read_text().splitlines()), 3)
            data = root / "other-data"
            configure()
            run()
            self.assertEqual((data / "bci_visualization/headset").read_text(), "flow_2")
            failure = data / "bci_visualization/fail"
            failure.touch()
            configure("devkit_3")
            run(success=False)
            self.assertEqual((data / "bci_visualization/headset").read_text(), "flow_2")
            failure.unlink()
            run()
            self.assertEqual((data / "bci_visualization/headset").read_text(), "devkit_3")
            configure("flow_2", enabled=False)
            run()
            self.assertEqual((data / "bci_visualization/headset").read_text(), "devkit_3")


if __name__ == "__main__":
    unittest.main()
