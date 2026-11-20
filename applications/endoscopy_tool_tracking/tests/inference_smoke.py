# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise real recurrent inference, cache reuse, and corrupt-cache recovery."""

import argparse
import json
import struct
import subprocess
import tempfile
import zlib
from pathlib import Path

PALETTE = {
    "neutral": [128, 128, 128],
    "protect": [100, 170, 255],
    "protect_strong": [0, 40, 160],
    "deprioritize": [255, 140, 0],
    "deprioritize_strong": [220, 0, 0],
}


def png_chunks(data):
    """Yield (type, payload) for each PNG chunk, checking every CRC."""
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise AssertionError("missing PNG signature")
    offset = 8
    while offset < len(data):
        (size,) = struct.unpack(">I", data[offset : offset + 4])
        kind = data[offset + 4 : offset + 8]
        payload = data[offset + 8 : offset + 8 + size]
        (crc,) = struct.unpack(">I", data[offset + 8 + size : offset + 12 + size])
        if zlib.crc32(kind + payload) != crc:
            raise AssertionError(f"bad CRC in {kind!r} chunk")
        yield kind, payload
        offset += 12 + size


def check_view(path, values, manifest):
    """The view must be an indexed PNG whose pixels are the policy values of the covering cells."""
    width, height = manifest["view"]["width"], manifest["view"]["height"]
    # The view is drawn at the source size, which --cae policy fixes at 854x480.
    if (width, height) != (854, 480) or manifest["source"] != {"width": 854, "height": 480}:
        raise AssertionError(f"unexpected view or source size: {manifest['view']}")
    if manifest["view"]["palette"] != PALETTE:
        raise AssertionError(f"unexpected view palette: {manifest['view']['palette']}")
    chunks = list(png_chunks(path.read_bytes()))
    kinds = [kind for kind, _ in chunks]
    if kinds != [b"IHDR", b"PLTE", b"IDAT", b"IEND"]:
        raise AssertionError(f"unexpected chunks in {path.name}: {kinds}")
    if chunks[3][1] != b"":
        raise AssertionError(f"IEND chunk is not empty in {path.name}")
    if chunks[0][1] != struct.pack(">IIBBBBB", width, height, 8, 3, 0, 0, 0):
        raise AssertionError(f"unexpected IHDR in {path.name}")
    if chunks[1][1] != bytes(c for rgb in PALETTE.values() for c in rgb):
        raise AssertionError(f"unexpected PLTE in {path.name}")
    raw = zlib.decompress(chunks[2][1])
    if len(raw) != height * (width + 1):
        raise AssertionError(f"unexpected image size in {path.name}")
    # Each pixel takes the cell that contains its centre.
    columns = [(2 * x + 1) * 107 // (2 * width) for x in range(width)]
    rows = [(2 * y + 1) * 60 // (2 * height) for y in range(height)]
    for y in range(height):
        line = raw[y * (width + 1) : (y + 1) * (width + 1)]
        expected = bytes([0] + [values[rows[y] * 107 + columns[x]] for x in range(width)])
        if line != expected:
            raise AssertionError(f"view row {y} does not match the map in {path.name}")


def check_readme(path):
    """The dump's README must describe the layout, legend and raw-value reading."""
    text = path.read_text()
    legend = [
        ("neutral", "No change", "gray"),
        ("protect", "Keep more quality", "light blue"),
        ("protect_strong", "Keep much more quality", "dark blue"),
        ("deprioritize", "Spend fewer bits", "orange"),
        ("deprioritize_strong", "Spend many fewer bits", "red"),
    ]
    for value, ((name, meaning, color), rgb) in enumerate(
        zip(legend, PALETTE.values(), strict=True)
    ):
        row = f"| {value} | `{name}` | {meaning} | {color} | {', '.join(map(str, rgb))} |"
        if row not in text:
            raise AssertionError(f"README.md lacks the legend row {row}")
    needles = (
        "policy/policy_NNNNNN.pgm",
        "view/view_NNNNNN.png",
        "plain PGM (`P2`)",
        "\n4           <- largest possible value\n",
        "numbers[4:]",
        "scale the values to 0-255",
        # In a code span, so Markdown does not read the asterisks as emphasis.
        "`floor((2x+1)*107/(2*854))`",
    )
    for needle in needles:
        if needle not in text:
            raise AssertionError(f"README.md does not mention {needle}")


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
            "file_format": "plain-pgm",
            "maxval": 4,
        }:
            raise AssertionError(f"unexpected manifest map: {manifest['map']}")
        # 0.50000006 is the float after 0.5; six significant digits would record it as 0.5.
        threshold = manifest["config"]["class_threshold"]
        if struct.pack("f", threshold) != struct.pack("f", 0.50000006):
            raise AssertionError(f"class_threshold not recorded exactly: {threshold}")
        check_readme(dump / "README.md")
        frames = [json.loads(line) for line in (dump / "frames.jsonl").read_text().splitlines()]
        if [frame["frame_id"] for frame in frames] != [1, 2, 3]:
            raise AssertionError(f"unexpected frame ids: {frames}")
        # Capture times must be recorded and increase; replay can skip clock ticks, so the gap
        # between frames is not fixed.
        times = [frame.get("capture_timestamp_ns") for frame in frames]
        if not all(type(stamp) is int for stamp in times) or times != sorted(set(times)):
            raise AssertionError(f"capture times are missing or not increasing: {times}")
        for frame in frames:
            name = f"{frame['frame_id']:06d}"
            if (frame["file"], frame["view"]) != (
                f"policy/policy_{name}.pgm",
                f"view/view_{name}.png",
            ):
                raise AssertionError(f"unexpected dump file names: {frame}")
            # Plain PGM: three header lines, then one line of 107 numbers per map row.
            lines = (dump / frame["file"]).read_text().splitlines()
            rows = [line.split() for line in lines[3:]]
            if lines[:3] != ["P2", "107 60", "4"] or len(rows) != 60:
                raise AssertionError(f"malformed dump {frame['file']}")
            if any(len(row) != 107 for row in rows):
                raise AssertionError(f"malformed row in {frame['file']}")
            values = bytes(int(value) for row in rows for value in row)
            if set(values) - {1, 3}:
                raise AssertionError(f"unexpected policy values in {frame['file']}")
            # Every policy's count must match the map, in value order 0..4.
            expected_counts = {name: values.count(value) for value, name in enumerate(PALETTE)}
            if frame["counts"] != expected_counts:
                raise AssertionError(f"counts do not match the map in {frame['file']}: {frame}")
            check_view(dump / frame["view"], values, manifest)
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
