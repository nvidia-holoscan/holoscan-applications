# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kernel Flow BCI real-time reconstruction and interactive volume visualization.

The reconstruction operators pass Python objects holding CuPy arrays, and all of their GPU
work runs on CuPy's default stream, so each operator sees its upstream results in order.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from pathlib import Path

import h5py
import holoscan as hs
from operators.reconstruction import (
    BuildRHSOperator,
    ConvertToVoxelsOperator,
    NormalizeOperator,
    RegularizedSolverOperator,
)
from operators.reconstruction.solver_operator import REG_DEFAULT
from operators.stream import REPLAY_RATES, StreamOperator
from operators.voxel_stream_to_volume import VoxelStreamToVolumeOp
from operators.vtk_volume_view import VTKVolumeViewOp

REQUIRED_FILES = (
    "data.snirf",
    "anatomy_labels_high_res.nii.gz",
    "extinction_coefficients_mua.csv",
    "flow_channel_map.json",
    "flow_mega_jacobian.npy",
    *(
        f"voxel_info/{name}.npy"
        for name in (
            "idxs_significant_voxels",
            "ijk",
            "mua",
            "musp",
            "resolution",
            "wavelengths",
            "xyz",
        )
    ),
)


def parse_arguments(arguments=None):
    """Build and parse the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    default_data = os.environ.get("HOLOSCAN_INPUT_PATH", "data/bci_visualization")
    parser.add_argument("--data", type=Path, default=Path(default_data), help="data directory")
    parser.add_argument(
        "--synthetic", action="store_true", help="replay generated data instead of --data"
    )
    parser.add_argument(
        "--replay-rate",
        choices=REPLAY_RATES,
        default="fast",
        help="'fast' replays as fast as the pipeline runs; 'realtime' follows the recording",
    )
    parser.add_argument("--frames", type=int, default=0, help="samples to replay; 0 for all")
    parser.add_argument("--reg", type=float, default=REG_DEFAULT, help="regularization")
    parser.add_argument("--visualization-scale", type=float, default=10.0)
    parser.add_argument("--render-hz", type=int, default=30)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument(
        "--headless", action="store_true", help="render offscreen and exit when the replay ends"
    )
    parser.add_argument("--screenshot", default="", help="write the last frame to this PNG")
    parser.add_argument("--simulate-drag", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--compile-only", action="store_true", help="stop after hs.compile()")
    args = parser.parse_args(arguments)
    if args.frames < 0:
        parser.error("--frames must be non-negative")
    if not 1 <= args.render_hz <= 240:
        parser.error("--render-hz must be between 1 and 240")
    return args


def replay_length(snirf_path: Path, frames: int) -> int:
    """Return how many samples the replay emits."""
    with h5py.File(snirf_path, "r") as snirf:
        total = int(snirf["nirs/data1/dataTimeSeries"].shape[0])
    return min(frames, total) if frames else total


def build_graph(args, data_dir: Path) -> hs.Graph:
    """Compose the replay, reconstruction, and visualization pipeline."""
    graph = hs.Graph("bci_visualization", default_clock=hs.RealtimeClock)
    stream = graph.op(
        StreamOperator,
        "stream",
        snirf_path=str(data_dir / "data.snirf"),
        replay_rate=args.replay_rate,
        max_samples=args.frames,
    )
    build_rhs = graph.op(BuildRHSOperator, "build_rhs", data_dir=str(data_dir))
    normalize = graph.op(NormalizeOperator, "normalize")
    solver = graph.op(RegularizedSolverOperator, "solver", reg=args.reg)
    convert = graph.op(ConvertToVoxelsOperator, "convert_to_voxels", data_dir=str(data_dir))
    to_volume = graph.op(
        VoxelStreamToVolumeOp,
        "voxel_to_volume",
        mask_path=str(data_dir / "anatomy_labels_high_res.nii.gz"),
        visualization_scale=args.visualization_scale,
    )
    view = graph.op(
        VTKVolumeViewOp,
        "view",
        render_hz=args.render_hz,
        width=args.width,
        height=args.height,
        headless=args.headless,
        simulate_drag=args.simulate_drag,
        screenshot=args.screenshot,
        stop_at_last_frame=args.frames > 0,
        last_frame=replay_length(data_dir / "data.snirf", args.frames) - 1,
    )

    graph.add_flow(stream.samples, build_rhs.moments, queue_depth=2)
    graph.add_flow(build_rhs.batch, normalize.batch, queue_depth=2)
    graph.add_flow(normalize.normalized, solver.batch, queue_depth=2)
    graph.add_flow(solver.result, convert.result, queue_depth=2)
    graph.add_flow(convert.hb_volume, to_volume.hb_volume, queue_depth=2)
    # The view only needs the newest volume.
    graph.add_flow(
        to_volume.volume, view.volume, queue_policy=hs.QueuePolicy.DROP_OLDEST, queue_depth=1
    )
    return graph


def run(args, data_dir: Path) -> int:
    """Compile and run the graph, then print a summary."""
    missing = [name for name in REQUIRED_FILES if not (data_dir / name).is_file()]
    if missing:
        raise SystemExit(
            f"missing data in {data_dir}: {', '.join(missing)}\n"
            "Prepare it with prepare_data.py (see README.md) or pass --synthetic."
        )
    plan = hs.compile(build_graph(args, data_dir))
    if not plan.ok:
        raise SystemExit(plan.summary())
    print("compile: ok", flush=True)
    if args.compile_only:
        return 0
    if not args.headless and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise SystemExit("no display: set DISPLAY for the interactive window, or pass --headless")

    started = time.monotonic()
    with hs.run_async(plan) as session:
        try:
            session.wait()
        except KeyboardInterrupt:
            session.request_stop()
            session.wait()
    elapsed = time.monotonic() - started

    report = VTKVolumeViewOp.report
    threads = report.get("threads", {})
    print(
        f"renders {report.get('renders', 0)} volumes {report.get('volumes', 0)} "
        f"over {elapsed:.1f} s"
    )
    if args.simulate_drag:
        print(f"camera_moved {report.get('camera_moved')}")
    print(f"missing {report.get('missing', {})}")
    print(f"same_thread {len(threads) == 3 and len(set(threads.values())) == 1}")
    if session.diagnostics:
        print(f"diagnostics {session.diagnostics!r}")
    print(f"termination {session.termination}")
    return 1 if session.diagnostics else 0


def main(arguments=None) -> int:
    """Run on the prepared data directory or on a generated synthetic dataset."""
    args = parse_arguments(arguments)
    if not args.synthetic:
        return run(args, args.data.resolve())
    from synthetic import write_synthetic_dataset

    with tempfile.TemporaryDirectory(prefix="bci-synthetic-") as directory:
        samples = args.frames or 12
        return run(args, write_synthetic_dataset(Path(directory), samples=samples))


if __name__ == "__main__":
    raise SystemExit(main())
