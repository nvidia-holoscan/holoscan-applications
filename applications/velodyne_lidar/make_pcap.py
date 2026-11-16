#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write a small synthetic VLP-16 pcap recording for replay tests (no sensor required).

The packets sweep a 20 m x 20 m square room at 10 Hz. Non-lidar traffic (an ARP frame and a
UDP datagram to another port) is interleaved to exercise the replay filter.
"""

import argparse
import math
import pathlib
import struct

PACKET_BYTES = 1206
PORT = 2368
PACKET_PERIOD_US = 1327  # ~754 packets/s, one 10 Hz revolution is ~76 packets


def laser_pitch_degrees(laser: int) -> float:
    return -15.0 + laser if laser % 2 == 0 else float(laser)


def make_packet(index: int, half_width: float = 10.0) -> bytes:
    """Return one VLP-16 strongest-return data packet."""
    out = bytearray()
    for block in range(12):
        azimuth = ((index * 12 + block) * 40) % 36000
        out += b"\xff\xee" + struct.pack("<H", azimuth)
        for record in range(32):
            yaw = math.radians(azimuth * 0.01 + (0.2 if record >= 16 else 0.0))
            pitch = math.radians(laser_pitch_degrees(record % 16))
            horizontal = half_width / max(abs(math.sin(yaw)), abs(math.cos(yaw)))
            distance = horizontal / math.cos(pitch)
            out += struct.pack("<HB", round(distance / 0.002), 16 * (record % 16) + 15)
    out += struct.pack("<I", (index * PACKET_PERIOD_US) % 3_600_000_000)
    out += bytes([0x37, 0x22])
    assert len(out) == PACKET_BYTES
    return bytes(out)


def udp_frame(payload: bytes, port: int) -> bytes:
    """Wrap a payload in Ethernet, IPv4, and UDP headers (checksums zero, as allowed)."""
    udp = struct.pack("!HHHH", 2368, port, 8 + len(payload), 0) + payload
    ip = struct.pack(
        "!BBHHHBBH4s4s",
        0x45,
        0,
        20 + len(udp),
        0,
        0,
        64,
        17,
        0,
        bytes([192, 168, 1, 201]),
        bytes([255, 255, 255, 255]),
    )
    ethernet = b"\xff" * 6 + b"\x60\x76\x88\x00\x00\x01" + b"\x08\x00"
    return ethernet + ip + udp


def write_pcap(path: pathlib.Path, packets: int, port: int = PORT) -> None:
    """Write \\p packets lidar frames plus two unrelated frames to a classic pcap file."""
    records = []
    for index in range(packets):
        records.append((index * PACKET_PERIOD_US, udp_frame(make_packet(index), port)))
        if index == 1:
            arp = b"\xff" * 6 + b"\x60\x76\x88\x00\x00\x01" + b"\x08\x06" + bytes(28)
            records.append((index * PACKET_PERIOD_US + 1, arp))
            records.append((index * PACKET_PERIOD_US + 2, udp_frame(b"\x00" * 512, 8308)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as out:
        out.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))
        for micros, frame in records:
            out.write(
                struct.pack(
                    "<IIII", micros // 1_000_000, micros % 1_000_000, len(frame), len(frame)
                )
            )
            out.write(frame)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--packets", type=int, default=760, help="default: ~1 s, 10 revolutions")
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    if args.packets < 1:
        parser.error("--packets must be positive")
    if args.output.exists():
        parser.error(f"{args.output} already exists")
    write_pcap(args.output, args.packets, args.port)


if __name__ == "__main__":
    main()
