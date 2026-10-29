"""
SPDX-FileCopyrightText: Copyright (c) 2026 Kernel.
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import time
from typing import NamedTuple

import holoscan as hs
import numpy as np
from streams.base_nirs import ChannelInfo
from streams.snirf import SNIRFStream

REPLAY_RATES = ("fast", "realtime")


class SampleOutput(NamedTuple):
    data: np.ndarray
    channels: ChannelInfo


class StreamOperator(hs.Operator):
    """Replay SNIRF moments as fast as the pipeline accepts them, or at the recorded rate."""

    samples = hs.Output("samples", hs.pyobject)
    snirf_path = hs.Param("", description="SNIRF recording to replay.")
    replay_rate = hs.Param(
        "fast",
        description="'fast' emits as soon as downstream has room; 'realtime' follows the "
        "recording's timestamps.",
        allowed=REPLAY_RATES,
    )
    max_samples = hs.Param(0, description="Stop after this many samples; 0 replays them all.")

    def setup(self, spec):
        spec.output(self.samples)

    def contract(self):
        # A short period leaves fast replay limited by downstream backpressure.
        return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1)))

    def start(self):
        self._stream = SNIRFStream(self.snirf_path)
        self._stream.start()
        self._channels = self._stream.get_channels()
        self._samples = self._stream.stream_nirs()
        self._pending = None
        self._sent = 0
        self._replay_origin = None

    def stop(self):
        self._stream.stop()

    def compute(self, ctx):
        if self._pending is None:
            item = None
            if not self.max_samples or self._sent < self.max_samples:
                item = next(self._samples, None)
            if item is None:
                return  # replay finished; the view decides when the run ends
            self._pending = item
        timestamp, data = self._pending
        if self.replay_rate == "realtime":
            now = time.monotonic()
            if self._replay_origin is None:
                self._replay_origin = now - timestamp
            if now < self._replay_origin + timestamp:
                return
        try:
            # Preserve capture time through reconstruction; the frame id marks the last sample.
            self.samples.emit(
                SampleOutput(data, self._channels),
                options=hs.EmitOptions(capture_time=ctx.activation_time, frame_id=self._sent),
            )
        except hs.errors.BackpressuredError:
            return  # retry the same sample on the next tick
        self._pending = None
        self._sent += 1
