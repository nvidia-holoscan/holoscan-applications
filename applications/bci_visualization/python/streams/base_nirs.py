"""
SPDX-FileCopyrightText: Copyright (c) 2026 Kernel.
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""

import abc
from collections.abc import Iterator
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


class ChannelInfo(NamedTuple):
    detector_module: NDArray[np.int_]
    detector_number: NDArray[np.int_]
    source_module: NDArray[np.int_]
    source_number: NDArray[np.int_]

    def __len__(self) -> int:
        return len(self.source_module)


class BaseNirsStream(abc.ABC):
    @abc.abstractmethod
    def start(self) -> None:
        pass

    @abc.abstractmethod
    def stop(self) -> None:
        pass

    @abc.abstractmethod
    def get_channels(self) -> ChannelInfo:
        pass

    @abc.abstractmethod
    def stream_nirs(self) -> Iterator[tuple[float, np.ndarray]]:
        """Yield (recording time in seconds, moments) pairs."""
