#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Download the Kernel Flow sample recording and reconstruction assets for bci_visualization."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

DATASET_URL = "https://huggingface.co/datasets/KernelCo/holohub_bci_visualization/resolve"
REVISION = "fe75bc160f929e680d1e6ef04cd42139782cb8ec"
HEADSETS = (
    "devkit_3",
    "devkit_10",
    "devkit_12",
    "flow_2",
    "flow_2_1",
    "flow_2_2",
    "flow_2_3",
    "flow_2_3_753",
)
VOXEL_INFO = (
    "affine",
    "idxs_significant_voxels",
    "ijk",
    "mua",
    "musp",
    "resolution",
    "wavelengths",
    "xyz",
)
CHECKSUMS = Path(__file__).with_name("dataset_checksums.json")
CHUNK_SIZE = 8 * 1024 * 1024


class PreparationError(RuntimeError):
    """A user-facing data preparation failure."""


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    """Build and parse the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="destination directory")
    parser.add_argument(
        "--headset",
        choices=HEADSETS,
        default="flow_2",
        help="headset whose Jacobian and channel map to download (default: flow_2, which maps every channel in data.snirf)",
    )
    parser.add_argument("--base-url", default=DATASET_URL, help=argparse.SUPPRESS)
    parser.add_argument("--checksums", type=Path, default=CHECKSUMS, help=argparse.SUPPRESS)
    parser.add_argument(
        "--force", action="store_true", help="download every file, even ones that match"
    )
    return parser.parse_args(arguments)


def dataset_files(headset: str) -> list[tuple[str, str]]:
    """Return (dataset path, local path) pairs for one headset."""
    files = [
        ("data.snirf", "data.snirf"),
        ("bci_visualization/anatomy_labels_high_res.nii.gz", "anatomy_labels_high_res.nii.gz"),
    ]
    for name in (
        "extinction_coefficients_mua.csv",
        "flow_channel_map.json",
        "flow_mega_jacobian.npy",
    ):
        files.append((f"{headset}/{name}", name))
    for name in VOXEL_INFO:
        files.append((f"{headset}/voxel_info/{name}.npy", f"voxel_info/{name}.npy"))
    return files


def sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def matches(path: Path, expected: dict[str, object]) -> bool:
    """Return whether path has the expected size and SHA-256 checksum."""
    return (
        path.is_file()
        and path.stat().st_size == expected["size"]
        and sha256(path) == expected["sha256"]
    )


def download(url: str, destination: Path, expected: dict[str, object]) -> None:
    """Download url to destination through a verified temporary file in the same directory."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as staging:
        staged = Path(staging.name)
        try:
            with urllib.request.urlopen(url) as response:
                shutil.copyfileobj(response, staging, CHUNK_SIZE)
        except Exception as error:
            staged.unlink(missing_ok=True)
            raise PreparationError(f"failed to download {url}: {error}") from error
    if not matches(staged, expected):
        staged.unlink()
        raise PreparationError(f"checksum mismatch for {url}")
    staged.replace(destination)


def prepare_data(options: argparse.Namespace) -> None:
    """Download every file that is missing or doesn't match its checksum, or all with --force."""
    base = f"{options.base_url.rstrip('/')}/{REVISION}"
    checksums = json.loads(options.checksums.read_text())
    for remote, local in dataset_files(options.headset):
        destination = options.output / local
        expected = checksums[remote]
        if not options.force and destination.is_file():
            print(f"Verifying {destination}", flush=True)
            if matches(destination, expected):
                continue
        print(f"Downloading {remote}", flush=True)
        download(f"{base}/{remote}", destination, expected)
    print(f"Prepared {options.output} for headset {options.headset}")


def main(arguments: list[str] | None = None) -> int:
    """Run the command-line utility and convert expected errors to diagnostics."""
    options = parse_arguments(arguments)
    try:
        prepare_data(options)
    except (OSError, PreparationError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
