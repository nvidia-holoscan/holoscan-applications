#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Report the Velodyne model and packet timing in classic pcap files (no dependencies).

Usage: python3 vlp_inspect.py FILE.pcap [...]

The model byte alone is not reliable (some VLP-16 recordings report 0x21, the HDL-32E code),
so the firing rate is also reported: distinct-azimuth blocks arrive about every 110.6 us on a
VLP-16 (~9040/s) and every 46 us on an HDL-32E (~21700/s), independent of rotation speed.
"""

import collections
import itertools
import statistics
import struct
import sys

_MAGIC = {
    b"\xd4\xc3\xb2\xa1": ("<", False),
    b"\x4d\x3c\xb2\xa1": ("<", True),
    b"\xa1\xb2\xc3\xd4": (">", False),
    b"\xa1\xb2\x3c\x4d": (">", True),
}
_LINK_ETHERNET, _LINK_LINUX_SLL = 1, 113
HDL32E_BLOCKS_PER_S = 15000  # threshold between the VLP-16 and HDL-32E firing rates


def records(path):
    """Yield (link type, timestamp in seconds, frame bytes) for each record."""
    with open(path, "rb") as f:
        head = f.read(24)
        if head[:4] not in _MAGIC:
            raise SystemExit(f"{path}: not a classic pcap (pcapng? use editcap -F pcap)")
        endian, nano = _MAGIC[head[:4]]
        link = struct.unpack(endian + "I", head[20:24])[0]
        while len(rec := f.read(16)) == 16:
            sec, frac, captured, _ = struct.unpack(endian + "IIII", rec)
            yield link, sec + frac * (1e-9 if nano else 1e-6), f.read(captured)


def udp(link, data):
    """Return (destination port, payload) for IPv4/UDP frames, else None."""
    off, ethertype = 0, 0x0800
    if link == _LINK_ETHERNET:
        off, ethertype = 14, struct.unpack(">H", data[12:14])[0]
        while ethertype in (0x8100, 0x88A8):
            ethertype = struct.unpack(">H", data[off + 2 : off + 4])[0]
            off += 4
    elif link == _LINK_LINUX_SLL:
        off, ethertype = 16, struct.unpack(">H", data[14:16])[0]
    if ethertype != 0x0800 or len(data) < off + 28 or data[off + 9] != 17:
        return None
    ihl = (data[off] & 0x0F) * 4
    port, length = struct.unpack(">HH", data[off + ihl + 2 : off + ihl + 6])
    return port, data[off + ihl + 8 : off + ihl + length]


def inspect(path):
    """Summarize the UDP ports, model bytes, return modes, and timing of a recording."""
    ports, models, modes = collections.Counter(), collections.Counter(), collections.Counter()
    times, steps = [], []
    for link, timestamp, data in records(path):
        if (parsed := udp(link, data)) is None:
            continue
        port, payload = parsed
        ports[port] += 1
        if len(payload) != 1206 or payload[0:2] != b"\xff\xee":
            continue
        modes[payload[1204]] += 1
        models[payload[1205]] += 1
        times.append(timestamp)
        azimuths = [struct.unpack_from("<H", payload, block * 100 + 2)[0] for block in range(12)]
        steps += [(b - a) % 36000 for a, b in itertools.pairwise(azimuths)]
    summary = {"ports": ports, "models": models, "modes": modes, "packets_per_s": 0.0}
    if len(times) > 1 and times[-1] > times[0]:
        moving = [step for step in steps if step != 0]
        summary["packets_per_s"] = (len(times) - 1) / (times[-1] - times[0])
        summary["azimuth_step_deg"] = statistics.median(moving) / 100 if moving else 0.0
        summary["blocks_per_s"] = summary["packets_per_s"] * 12 * len(moving) / len(steps)
        summary["layout"] = "HDL-32E" if summary["blocks_per_s"] > HDL32E_BLOCKS_PER_S else "VLP-16"
    return summary


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for path in sys.argv[1:]:
        s = inspect(path)
        print(f"== {path}")
        print("  UDP dst ports:", dict(s["ports"].most_common(4)))
        print(
            "  model bytes:",
            {hex(k): v for k, v in s["models"].items()},
            " return modes:",
            {hex(k): v for k, v in s["modes"].items()},
        )
        if "layout" in s:
            print(
                f"  data packets/s: {s['packets_per_s']:.0f}"
                f"  azimuth step: {s['azimuth_step_deg']:.2f} deg"
                f"  distinct blocks/s: {s['blocks_per_s']:.0f}"
            )
            print(f"  timing matches: {s['layout']} layout")


if __name__ == "__main__":
    main()
