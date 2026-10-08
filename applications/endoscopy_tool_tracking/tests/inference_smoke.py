# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise real recurrent inference, cache reuse, and corrupt-cache recovery."""

import argparse
import subprocess
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="endoscopy-inference-") as cache:

        def run():
            completed = subprocess.run(
                [
                    args.binary,
                    "--data",
                    args.data,
                    "--count",
                    "3",
                    "--display",
                    "false",
                    "--engine-cache",
                    cache,
                ],
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
            if (
                completed.returncode
                or "consumed=3 tracked=3 inference=on verdict=PASS" not in completed.stdout
            ):
                raise AssertionError(completed.stdout + completed.stderr)

        run()
        engines = list(Path(cache).glob("*.engine"))
        if len(engines) != 1 or engines[0].stat().st_size == 0:
            raise AssertionError("uncached inference did not produce one engine")
        engine = engines[0]
        original = engine.read_bytes()
        written = engine.stat().st_mtime_ns
        run()
        if engine.stat().st_mtime_ns != written or engine.read_bytes() != original:
            raise AssertionError("cached inference rebuilt or changed its engine")
        engine.write_bytes(b"corrupt engine")
        run()
        if engine.stat().st_size <= len(b"corrupt engine"):
            raise AssertionError("corrupt-cache recovery did not replace the engine")
    print("Real inference passed: cold build, cache reuse, and corrupt-cache recovery")


if __name__ == "__main__":
    main()
