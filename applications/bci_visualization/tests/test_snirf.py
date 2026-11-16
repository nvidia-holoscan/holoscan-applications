# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SNIRF replay parsing on a generated recording."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from streams.snirf import NUM_MOMENTS, NUM_WAVELENGTHS, SNIRFStream
from synthetic import SAMPLE_PERIOD, write_synthetic_dataset


class SNIRFStreamTest(unittest.TestCase):
    def test_streams_timestamped_moments_per_unique_channel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = write_synthetic_dataset(Path(directory), samples=4)
            stream = SNIRFStream(data / "data.snirf")
            stream.start()
            try:
                channels = stream.get_channels()
                samples = list(stream.stream_nirs())
            finally:
                stream.stop()
        self.assertEqual(len(channels), 6)
        self.assertEqual(list(channels.source_module), [1, 1, 1, 2, 2, 2])
        self.assertEqual(list(channels.detector_number), [1, 2, 3, 1, 2, 3])
        self.assertEqual(len(samples), 4)
        for index, (timestamp, moments) in enumerate(samples):
            self.assertAlmostEqual(timestamp, index * SAMPLE_PERIOD)
            self.assertEqual(moments.shape, (NUM_MOMENTS, len(channels), NUM_WAVELENGTHS))
            self.assertFalse(np.isnan(moments).any())


if __name__ == "__main__":
    unittest.main()
