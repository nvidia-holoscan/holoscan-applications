"""
SPDX-FileCopyrightText: Copyright (c) 2026 Kernel.
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import functools
import time
from typing import NamedTuple

import holoscan as hs
import numpy as np
from streams.base_nirs import ChannelInfo
from streams.kernel_sdk import KernelSDKStream, SNIRFFlowReceiver, kernel_sdk_receiver
from streams.snirf import SNIRFStream

REPLAY_RATES = ("fast", "realtime")
SOURCES = ("snirf", "kernel")


class SampleOutput(NamedTuple):
    data: np.ndarray
    channels: ChannelInfo


class StreamOperator(hs.Operator):
    """Emit moments from a SNIRF replay or, as they arrive, from a Kernel Flow headset."""

    samples = hs.Output("samples", hs.pyobject)
    snirf_path = hs.Param("", description="SNIRF recording to replay.")
    replay_rate = hs.Param(
        "fast",
        description="'fast' emits as soon as downstream has room; 'realtime' follows the "
        "recording's timestamps.",
        allowed=REPLAY_RATES,
    )
    final_frame = hs.Param(-1, description="Frame id marking finite simulated input completion.")
    max_samples = hs.Param(0, description="Stop after this many samples; 0 replays them all.")
    source = hs.Param(
        "snirf",
        description="'snirf' replays snirf_path; 'kernel' streams from Kortex.",
        allowed=SOURCES,
    )
    simulate_device = hs.Param(
        False, description="With source 'kernel', stream snirf_path instead of a headset."
    )

    def setup(self, spec):
        spec.output(self.samples)

    def contract(self):
        # A short period leaves fast replay limited by downstream backpressure.
        return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1)))

    def start(self):
        self._live = self.source == "kernel"
        # Live input, from a headset or from --simulate-device replaying the recording in
        # the Kernel SDK's layout, is queued by a receiver thread that drops the oldest
        # sample when the pipeline falls behind. Replay reads the file on demand instead,
        # so it never drops samples.
        if self._live:
            receiver = kernel_sdk_receiver
            if self.simulate_device:
                receiver = functools.partial(SNIRFFlowReceiver, self.snirf_path)
            self._stream = KernelSDKStream(receiver)
        else:
            self._stream = SNIRFStream(self.snirf_path)
        self._stream.start()
        self._channels = self._stream.get_channels()
        self._samples = None if self._live else self._stream.stream_nirs()
        self._pending = None
        self._lookahead = None
        self._sent = 0
        self._replay_origin = None

    def stop(self):
        self._stream.stop()
        if self._live and self._stream.dropped:
            print(f"dropped {self._stream.dropped} Kernel Flow samples", flush=True)

    def compute(self, ctx):
        if self._pending is None:
            item = None
            if not self.max_samples or self._sent < self.max_samples:
                if self._lookahead is not None:
                    item, self._lookahead = self._lookahead, None
                else:
                    item = self._stream.poll() if self._live else next(self._samples, None)
            if item is None:
                return  # nothing new, or the replay finished; the view ends the run
            self._pending = item
        timestamp, data = self._pending
        frame_id = self._sent
        if self.simulate_device and self._lookahead is None:
            self._lookahead = self._stream.poll()
            if self._lookahead is None:
                if not self._stream.ended:
                    return
                frame_id = self.final_frame
        if self.replay_rate == "realtime" and not self._live:
            now = time.monotonic()
            if self._replay_origin is None:
                self._replay_origin = now - timestamp
            if now < self._replay_origin + timestamp:
                return
        try:
            # Preserve capture time through reconstruction; the frame id marks the last sample.
            self.samples.emit(
                SampleOutput(data, self._channels),
                options=hs.EmitOptions(capture_time=ctx.activation_time, frame_id=frame_id),
            )
        except hs.errors.BackpressuredError:
            return  # retry the same sample on the next tick
        self._pending = None
        self._sent += 1
