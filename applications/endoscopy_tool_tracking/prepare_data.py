#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Prepare bounded sample data for the endoscopy tool-tracking application."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

RESOURCE_URL = (
    "https://api.ngc.nvidia.com/v2/resources/nvidia/clara-holoscan/"
    "holoscan_endoscopy_sample_data/versions/20230222/zip"
)
MODEL_NAME = "tool_loc_convlstm.onnx"
SOURCE_VIDEO_NAME = "surgical_video.264"
OUTPUT_VIDEO_NAME = "surgical_video.rgb"
FRAME_BYTES = 854 * 480 * 3
DEFAULT_FRAMES = 300


class PreparationError(RuntimeError):
    """A user-facing sample-data preparation failure."""


def non_negative_integer(value: str) -> int:
    """Parse a non-negative integer for argparse."""
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    """Build and parse the command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Download or read the Holoscan endoscopy sample archive and prepare "
            "the model plus a contiguous 854x480 RGB24 video stream."
        )
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="destination directory"
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help="use a predownloaded NGC ZIP instead of downloading it",
    )
    parser.add_argument(
        "--frames",
        type=non_negative_integer,
        default=DEFAULT_FRAMES,
        metavar="N",
        help="maximum frames to convert (default: 300, about 10 seconds; 0 converts all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace existing prepared assets",
    )
    return parser.parse_args(arguments)


def valid_model(path: Path) -> bool:
    """Return whether path is a usable existing model asset."""
    return path.is_file() and path.stat().st_size > 0


def valid_rgb_video(path: Path) -> bool:
    """Return whether path contains one or more tightly packed RGB24 frames."""
    return (
        path.is_file()
        and path.stat().st_size >= FRAME_BYTES
        and path.stat().st_size % FRAME_BYTES == 0
    )


def safe_member_path(name: str) -> bool:
    """Accept relative POSIX ZIP paths without parent traversal."""
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts


def required_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    """Locate exactly one safe regular member for each required basename."""
    matches: dict[str, list[zipfile.ZipInfo]] = {
        MODEL_NAME: [],
        SOURCE_VIDEO_NAME: [],
    }
    for member in archive.infolist():
        basename = PurePosixPath(member.filename).name
        if basename in matches and not member.is_dir():
            if not safe_member_path(member.filename):
                raise PreparationError(f"unsafe archive member path: {member.filename}")
            matches[basename].append(member)

    selected: dict[str, zipfile.ZipInfo] = {}
    for basename, candidates in matches.items():
        if not candidates:
            raise PreparationError(f"archive is missing {basename}")
        if len(candidates) != 1:
            raise PreparationError(f"archive contains duplicate {basename} members")
        selected[basename] = candidates[0]
    return selected


def extract_required_files(archive_path: Path, staging: Path) -> tuple[Path, Path]:
    """Copy only the required archive members into controlled staging paths."""
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = required_members(archive)
            model = staging / MODEL_NAME
            source_video = staging / SOURCE_VIDEO_NAME
            for basename, destination in (
                (MODEL_NAME, model),
                (SOURCE_VIDEO_NAME, source_video),
            ):
                with (
                    archive.open(members[basename]) as source,
                    destination.open("wb") as output,
                ):
                    shutil.copyfileobj(source, output)
    except zipfile.BadZipFile as error:
        raise PreparationError(f"invalid ZIP archive: {archive_path}") from error

    if model.stat().st_size == 0:
        raise PreparationError(f"archive member {MODEL_NAME} is empty")
    if source_video.stat().st_size == 0:
        raise PreparationError(f"archive member {SOURCE_VIDEO_NAME} is empty")
    return model, source_video


def download_archive(destination: Path) -> None:
    """Download the pinned NGC resource to a staging path."""
    try:
        with (
            urllib.request.urlopen(RESOURCE_URL) as response,
            destination.open("wb") as output,
        ):
            shutil.copyfileobj(response, output)
    except Exception as error:
        # Redirect failures can include signed URLs, so keep this diagnostic generic.
        raise PreparationError("failed to download the NGC sample archive") from error


def convert_video(source: Path, destination: Path, frames: int) -> None:
    """Convert H.264 input to a contiguous, tightly packed RGB24 stream."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise PreparationError("ffmpeg was not found in PATH")

    command = [
        ffmpeg,
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(source),
        "-an",
        "-vf",
        "scale=854:480",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
    ]
    if frames:
        command.extend(("-frames:v", str(frames)))
    command.append(str(destination))

    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise PreparationError(
            f"ffmpeg conversion failed with exit code {result.returncode}"
        )
    if not valid_rgb_video(destination):
        raise PreparationError(
            "ffmpeg produced an empty or incomplete RGB24 frame stream"
        )
    if frames and destination.stat().st_size > frames * FRAME_BYTES:
        raise PreparationError("ffmpeg produced more frames than requested")


def prepare_data(options: argparse.Namespace) -> None:
    """Prepare the selected sample assets according to parsed options."""
    output = options.output
    model_destination = output / MODEL_NAME
    video_destination = output / OUTPUT_VIDEO_NAME

    model_is_valid = valid_model(model_destination)
    video_is_valid = valid_rgb_video(video_destination)
    if not options.force and model_is_valid and video_is_valid:
        print(f"Prepared assets already exist in {output}")
        return

    if not options.force:
        invalid_existing = [
            path
            for path, valid in (
                (model_destination, model_is_valid),
                (video_destination, video_is_valid),
            )
            if path.exists() and not valid
        ]
        if invalid_existing:
            joined = ", ".join(str(path) for path in invalid_existing)
            raise PreparationError(
                f"refusing to replace invalid existing assets without --force: {joined}"
            )

    output.mkdir(parents=True, exist_ok=True)
    if not output.is_dir():
        raise PreparationError(f"output is not a directory: {output}")

    with tempfile.TemporaryDirectory(prefix=".prepare-data-", dir=output) as temporary:
        staging = Path(temporary)
        archive_path = options.archive
        if archive_path is None:
            archive_path = staging / "sample.zip"
            download_archive(archive_path)
        elif not archive_path.is_file():
            raise PreparationError(f"archive does not exist: {archive_path}")

        staged_model, source_video = extract_required_files(archive_path, staging)
        staged_video = staging / OUTPUT_VIDEO_NAME
        convert_video(source_video, staged_video, options.frames)

        if options.force or not model_is_valid:
            staged_model.replace(model_destination)
        if options.force or not video_is_valid:
            staged_video.replace(video_destination)

    print(f"Prepared {model_destination} and {video_destination}")


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
