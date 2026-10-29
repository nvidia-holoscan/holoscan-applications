# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kernel Flow streaming with fake receivers in place of the Kernel SDK."""

from __future__ import annotations

import functools
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from queue import Queue
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from streams import kernel_sdk
from streams.kernel_sdk import (
    SAMPLE_SHAPE,
    KernelSDKStream,
    SNIRFFlowReceiver,
    put_newest,
)
from streams.snirf import SNIRFStream
from synthetic import write_synthetic_dataset

CHANNELS = [(0, 0, 0, 2), (0, 1, 3, 5), (47, 2, 1, 0)]


def flow_sample(value: float) -> np.ndarray:
    sample = np.full(SAMPLE_SHAPE, np.nan)
    for index, channel in enumerate(CHANNELS):
        sample[channel] = value + index
    return sample


class FakeReceiver:
    def __init__(self, samples, error: BaseException | None = None, block: bool = False):
        self._samples = samples
        self._error = error
        self._block = block
        self.stopped = threading.Event()

    def samples(self):
        yield from self._samples
        if self._error is not None:
            raise self._error
        if self._block:
            self.stopped.wait()

    def stop(self):
        self.stopped.set()


class KernelSDKStreamTest(unittest.TestCase):
    def test_keeps_channels_valid_in_the_first_sample(self) -> None:
        stream = KernelSDKStream(lambda: FakeReceiver([flow_sample(1.0), flow_sample(5.0)]))
        stream.start()
        try:
            channels = stream.get_channels()
            samples = list(stream.stream_nirs())
        finally:
            stream.stop()
        self.assertEqual(
            list(
                zip(
                    channels.source_module,
                    channels.source_number,
                    channels.detector_module,
                    channels.detector_number,
                    strict=True,
                )
            ),
            CHANNELS,
        )
        self.assertEqual(len(samples), 2)
        for (_, moments), value in zip(samples, (1.0, 5.0), strict=True):
            self.assertEqual(moments.shape, (3, len(CHANNELS), 2))
            np.testing.assert_array_equal(moments[0, :, 0], value + np.arange(len(CHANNELS)))

    def test_snirf_receiver_matches_the_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snirf = write_synthetic_dataset(Path(directory), samples=3) / "data.snirf"
            replay = SNIRFStream(snirf)
            replay.start()
            try:
                expected_channels = replay.get_channels()
                expected = [moments for _, moments in replay.stream_nirs()]
            finally:
                replay.stop()
            stream = KernelSDKStream(functools.partial(SNIRFFlowReceiver, snirf))
            stream.start()
            try:
                channels = stream.get_channels()
                samples = [moments for _, moments in stream.stream_nirs()]
            finally:
                stream.stop()
        for field in expected_channels._fields:
            np.testing.assert_array_equal(
                getattr(channels, field), getattr(expected_channels, field)
            )
        self.assertEqual(len(samples), len(expected))
        for moments, reference in zip(samples, expected, strict=True):
            np.testing.assert_array_equal(moments, reference)

    def test_poll_returns_none_while_waiting(self) -> None:
        stream = KernelSDKStream(lambda: FakeReceiver([flow_sample(1.0)], block=True))
        stream.start()
        try:
            self.assertIsNotNone(stream.poll())
            self.assertIsNone(stream.poll())
            self.assertFalse(stream.ended)
        finally:
            stream.stop()

    def test_times_out_without_samples(self) -> None:
        stream = KernelSDKStream(lambda: FakeReceiver([], block=True), timeout=0.3)
        with self.assertRaisesRegex(RuntimeError, "no Kernel Flow sample"):
            stream.start()
        stream.stop()

    def test_reports_receiver_errors(self) -> None:
        receiver = FakeReceiver([flow_sample(1.0)], error=ConnectionError("lost Kortex"))
        stream = KernelSDKStream(lambda: receiver)
        stream.start()
        stream.poll()
        stream._thread.join(timeout=2.0)
        with self.assertRaisesRegex(RuntimeError, "receiver failed"):
            stream.poll()
        stream.stop()

    def test_rejects_a_first_sample_without_channels(self) -> None:
        stream = KernelSDKStream(lambda: FakeReceiver([np.full(SAMPLE_SHAPE, np.nan)]))
        with self.assertRaisesRegex(RuntimeError, "no valid channels"):
            stream.start()
        stream.stop()

    def test_put_newest_drops_the_oldest(self) -> None:
        queue: Queue[int] = Queue(2)
        self.assertFalse(put_newest(queue, 1))
        self.assertFalse(put_newest(queue, 2))
        self.assertTrue(put_newest(queue, 3))
        self.assertEqual([queue.get_nowait(), queue.get_nowait()], [2, 3])

    def test_explains_a_missing_kortex_config(self) -> None:
        missing = Path("/nonexistent/kortex.json")
        with (
            mock.patch.object(kernel_sdk, "KORTEX_CONFIG", missing),
            self.assertRaisesRegex(RuntimeError, "kortex.json not found"),
        ):
            kernel_sdk.kernel_sdk_receiver()


if __name__ == "__main__":
    unittest.main()
