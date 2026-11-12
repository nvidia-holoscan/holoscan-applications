# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise real recurrent inference, cache reuse, and corrupt-cache recovery."""

import argparse
import json
import struct
import subprocess
import tempfile
from pathlib import Path


def run_cae_policy(args, cache):
    """Generate, validate and dump policy maps from real inference."""
    with tempfile.TemporaryDirectory(prefix="endoscopy-cae-") as tmp:
        root = Path(tmp)
        config = root / "protect_tools.json"
        config.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "class_threshold": 0.50000006,
                    "background_policy": "deprioritize",
                    "classes": {str(index): "protect" for index in range(7)},
                }
            )
        )
        dump = root / "dump"
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
                "--cae",
                "policy",
                "--cae-policy-config",
                str(config),
                "--cae-policy-dump",
                str(dump),
            ],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        output = completed.stdout + completed.stderr
        if (
            completed.returncode
            or "consumed=3 tracked=3 inference=on verdict=PASS" not in completed.stdout
            or "cae=policy policies=3 verdict=PASS" not in completed.stdout
        ):
            raise AssertionError(output)
        manifest = json.loads((dump / "manifest.json").read_text())
        if manifest["map"] != {
            "width": 107,
            "height": 60,
            "dtype": "uint8",
            "layout": "row-major",
            "file_format": "pgm",
        }:
            raise AssertionError(f"unexpected manifest map: {manifest['map']}")
        # 0.50000006 is the float after 0.5; six significant digits would record it as 0.5.
        threshold = manifest["config"]["class_threshold"]
        if struct.pack("f", threshold) != struct.pack("f", 0.50000006):
            raise AssertionError(f"class_threshold not recorded exactly: {threshold}")
        frames = [json.loads(line) for line in (dump / "frames.jsonl").read_text().splitlines()]
        if [frame["frame_id"] for frame in frames] != [1, 2, 3]:
            raise AssertionError(f"unexpected frame ids: {frames}")
        header = b"P5\n107 60\n255\n"
        for frame in frames:
            data = (dump / frame["file"]).read_bytes()
            if not data.startswith(header) or len(data) != len(header) + 107 * 60:
                raise AssertionError(f"malformed dump {frame['file']}")
            values = data[len(header) :]
            if set(values) - {1, 3}:
                raise AssertionError(f"unexpected policy values in {frame['file']}")
            if sum(frame["counts"].values()) != 107 * 60:
                raise AssertionError(f"counts do not cover the map in {frame['file']}")
            if frame["counts"]["protect"] != values.count(1):
                raise AssertionError(f"protect count does not match {frame['file']}")
        # The first sample frames show a grasper and a hook, so tools must be protected.
        if not all(frame["counts"]["protect"] for frame in frames):
            raise AssertionError(f"no protected cells; detection failed: {frames}")


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
        run_cae_policy(args, cache)
    print(
        "Real inference passed: cold build, cache reuse, corrupt-cache recovery, "
        "and CAE policy generation"
    )


if __name__ == "__main__":
    main()
