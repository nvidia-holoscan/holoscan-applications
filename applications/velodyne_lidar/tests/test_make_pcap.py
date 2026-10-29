# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SDK-free checks of the synthetic pcap fixture and the VLP-16 packet layout."""

import importlib.util
import math
import pathlib
import struct
import sys
import tempfile
import unittest

_PATH = pathlib.Path(__file__).resolve().parents[1] / "make_pcap.py"
_SPEC = importlib.util.spec_from_file_location("make_pcap", _PATH)
make_pcap = importlib.util.module_from_spec(_SPEC)
sys.modules["make_pcap"] = make_pcap
_SPEC.loader.exec_module(make_pcap)
_INSPECT_SPEC = importlib.util.spec_from_file_location(
    "vlp_inspect", _PATH.with_name("vlp_inspect.py")
)
vlp_inspect = importlib.util.module_from_spec(_INSPECT_SPEC)
_INSPECT_SPEC.loader.exec_module(vlp_inspect)


def read_records(path):
    data = path.read_bytes()
    magic, major, minor, _, _, _, link = struct.unpack_from("<IHHiIII", data)
    assert (magic, major, minor, link) == (0xA1B2C3D4, 2, 4, 1)
    offset = 24
    while offset < len(data):
        seconds, micros, captured, original = struct.unpack_from("<IIII", data, offset)
        offset += 16
        yield seconds * 1_000_000 + micros, data[offset : offset + captured], original
        offset += captured


class PacketTest(unittest.TestCase):
    def test_layout(self):
        packet = make_pcap.make_packet(5)
        self.assertEqual(len(packet), make_pcap.PACKET_BYTES)
        for block in range(12):
            self.assertEqual(packet[block * 100 : block * 100 + 2], b"\xff\xee")
        self.assertEqual(packet[1204:], b"\x37\x22")

    def test_points_lie_on_room_walls(self):
        packet = make_pcap.make_packet(0)
        for block in range(12):
            azimuth = struct.unpack_from("<H", packet, block * 100 + 2)[0] * 0.01
            for record in range(32):
                ticks = struct.unpack_from("<H", packet, block * 100 + 4 + record * 3)[0]
                yaw = math.radians(azimuth + (0.2 if record >= 16 else 0.0))
                pitch = math.radians(make_pcap.laser_pitch_degrees(record % 16))
                horizontal = ticks * 0.002 * math.cos(pitch)
                wall = max(abs(horizontal * math.sin(yaw)), abs(horizontal * math.cos(yaw)))
                self.assertAlmostEqual(wall, 10.0, delta=0.01)

    def test_pitch_table(self):
        self.assertEqual(
            [make_pcap.laser_pitch_degrees(i) for i in range(16)],
            [-15, 1, -13, 3, -11, 5, -9, 7, -7, 9, -5, 11, -3, 13, -1, 15],
        )


class PcapTest(unittest.TestCase):
    def test_inspect_skips_malformed_frames(self):
        ethernet = b"\0" * 12 + b"\x08\x00"
        ip = bytearray(28)
        ip[0], ip[9] = 0x4F, 17
        frames = [b"", b"\0" * 13, b"\0" * 12 + b"\x81\x00\0\0", ethernet + ip]
        for frame in frames:
            with self.subTest(frame=frame):
                self.assertIsNone(vlp_inspect.udp(1, frame))
        self.assertIsNone(vlp_inspect.udp(113, b"\0" * 15))

    def test_inspect_skips_snaplen_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "truncated.pcap"
            make_pcap.write_pcap(path, 2)
            data = bytearray(path.read_bytes())
            captured = struct.unpack_from("<I", data, 32)[0]
            struct.pack_into("<I", data, 36, captured + 1)
            path.write_bytes(data)
            self.assertEqual(len(list(vlp_inspect.records(path))), 3)

    def test_records_and_filter_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "fixture.pcap"
            make_pcap.write_pcap(path, 4)
            records = list(read_records(path))
        self.assertEqual(len(records), 6)
        lidar = [frame for _, frame, _ in records if len(frame) == 14 + 20 + 8 + 1206]
        self.assertEqual(len(lidar), 4)
        for index, frame in enumerate(lidar):
            self.assertEqual(struct.unpack_from("!H", frame, 36)[0], make_pcap.PORT)
            self.assertEqual(frame[42:], make_pcap.make_packet(index))
        times = [time for time, _, _ in records]
        self.assertEqual(times, sorted(times))
        self.assertTrue(all(len(frame) == original for _, frame, original in records))

    def test_creates_parent_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "data" / "velodyne" / "recording.pcap"
            make_pcap.write_pcap(path, 1)
            self.assertTrue(path.is_file())

    def test_inspect_reports_vlp16_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "fixture.pcap"
            make_pcap.write_pcap(path, 760)
            summary = vlp_inspect.inspect(path)
        self.assertEqual(summary["ports"][make_pcap.PORT], 760)
        self.assertEqual(dict(summary["models"]), {0x22: 760})
        self.assertAlmostEqual(summary["packets_per_s"], 1e6 / make_pcap.PACKET_PERIOD_US, delta=1)
        self.assertEqual(summary["layout"], "VLP-16")


if __name__ == "__main__":
    unittest.main()
