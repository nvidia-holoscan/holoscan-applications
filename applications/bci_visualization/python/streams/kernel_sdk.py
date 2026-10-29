"""
SPDX-FileCopyrightText: Copyright (c) 2026 Kernel.
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
from typing import Protocol

import numpy as np

from .base_nirs import BaseNirsStream, ChannelInfo
from .snirf import NUM_MOMENTS, NUM_WAVELENGTHS, SNIRFStream

logger = logging.getLogger(__name__)

KORTEX_CONFIG = Path("/etc/kernel.com/kortex.json")
MAX_MODULE_COUNT = 48
NUM_SOURCES = 3
NUM_DETECTORS = 6
# One Flow sample: (source module, source, detector module, detector, moment, wavelength).
SAMPLE_SHAPE = (
    MAX_MODULE_COUNT,
    NUM_SOURCES,
    MAX_MODULE_COUNT,
    NUM_DETECTORS,
    NUM_MOMENTS,
    NUM_WAVELENGTHS,
)


class FlowReceiver(Protocol):
    """Source of Flow samples shaped SAMPLE_SHAPE, with NaN for channels that are off."""

    def samples(self) -> Iterator[np.ndarray]: ...

    def stop(self) -> None: ...


def put_newest(queue: Queue, item) -> bool:
    """Put item on queue, dropping the oldest entry if it is full; True if one was dropped."""
    dropped = False
    while True:
        try:
            queue.put_nowait(item)
            return dropped
        except Full:
            try:
                queue.get_nowait()
                dropped = True
            except Empty:
                pass


def kernel_sdk_receiver() -> FlowReceiver:
    """Connect to the Kortex service through the Kernel SDK."""
    if not KORTEX_CONFIG.is_file():
        raise RuntimeError(
            f"{KORTEX_CONFIG} not found: install and start Kortex, and mount the file into "
            "the container (see README.md)"
        )
    # kernel.sdk reports errors to Kernel's Sentry on import unless this is empty.
    os.environ.setdefault("SENTRY_DSN_SDK", "")
    from kernel.sdk import MomentNumber, SdkClient, Wavelength
    from kernel.sdk.socket import (
        Event as SdkEvent,
    )
    from kernel.sdk.socket import (
        check_faulted_on_command_failure,
        requires_flow_booted,
        requires_flow_connected,
    )

    moments = (MomentNumber.Zeroth, MomentNumber.First, MomentNumber.Second)
    wavelengths = (Wavelength.Red, Wavelength.IR)

    class KernelSDKReceiver(SdkClient):
        def __init__(self):
            # SdkClient.__init__ connects and calls stop() if that fails.
            self._stopped = Event()
            super().__init__()

        @check_faulted_on_command_failure
        @requires_flow_connected
        @requires_flow_booted
        def samples(self) -> Iterator[np.ndarray]:
            queue: Queue[np.ndarray] = Queue(5)
            urns = [
                [
                    self.MOMENTS_FIELD_URN_TEMPLATE.format(
                        moment_id=moment.value, wavelength=wavelength.value
                    )
                    for wavelength in wavelengths
                ]
                for moment in moments
            ]

            def on_moments(data: dict) -> None:
                missing = [urn for row in urns for urn in row if urn not in data]
                if missing:
                    logger.warning("Kernel SDK sample without %s; skipping it", missing)
                    return
                # Each field is (samples, source module, source, detector module, detector).
                stacked = np.stack(
                    [np.stack([data[urn] for urn in row], axis=-1) for row in urns], axis=-2
                )
                for sample in stacked:
                    put_newest(queue, sample)

            future = self._sdk.new_event(
                SdkEvent(device="flow", field_urns=[urn for row in urns for urn in row]),
                on_moments,
            )
            try:
                while not self._stopped.is_set():
                    try:
                        yield queue.get(timeout=0.5)
                    except Empty:
                        continue
            finally:
                future.cancel()

        def stop(self) -> None:
            self._stopped.set()
            super().stop()

    return KernelSDKReceiver()


class SNIRFFlowReceiver:
    """Replay a SNIRF recording at its recorded rate in the Kernel SDK's sample layout."""

    def __init__(self, snirf_path: Path | str) -> None:
        self._snirf = SNIRFStream(snirf_path)
        self._stopped = Event()

    def samples(self) -> Iterator[np.ndarray]:
        self._snirf.start()
        try:
            channels = self._snirf.get_channels()
            index = (
                channels.source_module,
                channels.source_number,
                channels.detector_module,
                channels.detector_number,
            )
            origin = time.monotonic()
            for timestamp, moments in self._snirf.stream_nirs():
                if self._stopped.wait(max(0.0, origin + timestamp - time.monotonic())):
                    return
                sample = np.full(SAMPLE_SHAPE, np.nan)
                sample[index] = moments.transpose(1, 0, 2)
                yield sample
        finally:
            self._snirf.stop()

    def stop(self) -> None:
        self._stopped.set()


class KernelSDKStream(BaseNirsStream):
    """Stream Flow moments, keeping the channels that are valid in the first sample."""

    def __init__(
        self,
        receiver_factory: Callable[[], FlowReceiver] = kernel_sdk_receiver,
        *,
        queue_size: int = 8,
        timeout: float = 10.0,
    ) -> None:
        if queue_size <= 0:
            raise ValueError("queue_size must be positive")
        self._receiver_factory = receiver_factory
        self._queue: Queue[tuple[float, np.ndarray]] = Queue(queue_size)
        self._timeout = timeout
        self._receiver: FlowReceiver | None = None
        self._thread: Thread | None = None
        self._error: BaseException | None = None
        self._channels: ChannelInfo | None = None
        self._good: np.ndarray | None = None
        self._first: tuple[float, np.ndarray] | None = None
        self._last_sample = 0.0
        self._ended = False
        self.dropped = 0

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("KernelSDKStream is already started")
        self._receiver = self._receiver_factory()
        self._thread = Thread(target=self._receive, name="KernelSDKReceiver", daemon=True)
        self._thread.start()
        timestamp, sample = self._wait(self._timeout)
        finite = np.isfinite(sample).all(axis=(4, 5)).reshape(-1)
        self._good = np.flatnonzero(finite)
        if not self._good.size:
            raise RuntimeError("the first Kernel Flow sample has no valid channels")
        source_module, source, detector_module, detector = np.unravel_index(
            self._good, SAMPLE_SHAPE[:4]
        )
        self._channels = ChannelInfo(
            detector_module=detector_module,
            detector_number=detector,
            source_module=source_module,
            source_number=source,
        )
        self._first = (timestamp, self._select(sample))
        logger.info("Streaming %d Kernel Flow channels", self._good.size)

    def stop(self) -> None:
        if self._receiver is not None:
            self._receiver.stop()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._receiver = None
        self._thread = None

    def get_channels(self) -> ChannelInfo:
        if self._channels is None:
            raise RuntimeError("KernelSDKStream.start() must be called first")
        return self._channels

    @property
    def ended(self) -> bool:
        """True once the receiver has finished and every sample has been taken."""
        return self._ended and self._first is None and self._queue.empty()

    def poll(self) -> tuple[float, np.ndarray] | None:
        """Return the oldest queued (receive time, moments) pair, or None if none is queued."""
        if self._first is not None:
            first, self._first = self._first, None
            return first
        try:
            timestamp, sample = self._queue.get_nowait()
        except Empty:
            self._check_alive()
            return None
        return timestamp, self._select(sample)

    def stream_nirs(self) -> Iterator[tuple[float, np.ndarray]]:
        while True:
            item = self.poll()
            if item is not None:
                yield item
            elif self.ended:
                return
            else:
                time.sleep(0.01)

    def _select(self, sample: np.ndarray) -> np.ndarray:
        """(SAMPLE_SHAPE) -> (moments, channels, wavelengths) for the kept channels."""
        flat = sample.reshape(-1, NUM_MOMENTS, NUM_WAVELENGTHS)
        return flat[self._good].transpose(1, 0, 2)

    def _wait(self, timeout: float) -> tuple[float, np.ndarray]:
        deadline = time.monotonic() + timeout
        while True:
            try:
                return self._queue.get(timeout=0.1)
            except Empty:
                self._check_alive()
                if self._ended:
                    raise RuntimeError("the Kernel Flow stream ended") from None
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        f"no Kernel Flow sample for {timeout:g} s; check that the headset "
                        "is connected and streaming"
                    ) from None

    def _check_alive(self) -> None:
        if self._error is not None:
            raise RuntimeError("the Kernel SDK receiver failed") from self._error
        if self._ended:
            return
        if self._last_sample and time.monotonic() - self._last_sample > self._timeout:
            raise RuntimeError(
                f"no Kernel Flow sample for {self._timeout:g} s; check that the headset "
                "is still streaming"
            )

    def _receive(self) -> None:
        try:
            for sample in self._receiver.samples():
                self._last_sample = time.monotonic()
                if put_newest(self._queue, (self._last_sample, sample)):
                    self.dropped += 1
        except BaseException as error:  # noqa: BLE001 - reported on the consumer thread
            self._error = error
        finally:
            self._ended = True
