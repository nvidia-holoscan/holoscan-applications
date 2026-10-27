# AprilTag Detection

This application detects [AprilTags](https://april.eecs.umich.edu/software/apriltag) and estimates
their 6-DoF poses. Its core is one reusable Holoscan SDK 5 operator, `AprilTagDetectorOp`: an image
in, the tags it contains out, on the CPU (AprilTag 3), the GPU (CUDA), or the PVA of Jetson Orin and
Thor (NVIDIA VPI). The application renders synthetic scenes, fans each frame out to a detector per
selected backend, checks every detection against the scene's ground truth, and can show the first
backend's detections on a HoloViz display.

## Backends

| Backend | Detection runs on | Notes |
| --- | --- | --- |
| `cpu` | CPU | The AprilTag 3 reference detector, with its default of one thread. |
| `cuda` | GPU | Every stage on the GPU; only the final detections are copied to the host. |
| `vpi` | PVA, or CPU | NVIDIA VPI 4's detector: on the PVA where there is one and the image suits it, otherwise on the CPU. x86_64 hosts have no PVA. |

Every backend accepts MONO8, RGB8, BGR8, RGBA8 and BGRA8 images and every AprilTag 3 family
(`tag36h11`, `tag25h9`, `tag16h5`, `tag36h10`, `tagStandard41h12`, `tagStandard52h13`,
`tagCircle21h7`, `tagCircle49h12`, `tagCustom48h12`). Poses are estimated on the host from the
corners, the same way for every backend. `--decimate`, `--refine-edges` and `--sharpening` tune
`cpu` and `cuda`; VPI ignores them. `--max-hamming 2` is refused for the 48-, 49- and 52-bit
families, whose decode tables would take gigabytes. The PVA takes image sides from 256 to 4096
pixels, except one past a multiple of 128.

| Where the image goes | Host image | CUDA device image |
| --- | --- | --- |
| `cpu`, `vpi` on the CPU | Read in place (color is converted to gray on the CPU). | Gray copied to the host. |
| `cuda` | Gray copied to the GPU. | Read in place (color is converted to gray on the GPU); no copy. |
| `vpi` on the PVA | Gray copied into a VPI image. | Gray copied into a VPI image on the GPU. |

## Requirements

- Linux on x86_64 or aarch64 with an NVIDIA GPU.
- Holoscan SDK 5.0 (Core and Sensor I/O; Visualization for the display; Testing for
  `holoscan test`), CMake 4.0+, a C++20 compiler and the CUDA toolkit.
- [AprilTag 3](https://github.com/AprilRobotics/apriltag) 3.4 or later
  ([BSD-2-Clause](https://github.com/AprilRobotics/apriltag/blob/master/LICENSE.md)): an installed
  copy, or release 3.4.5 downloaded at configure time and checked against its SHA-256.
- Optional: NVIDIA VPI 4 (`vpi4-dev`) for the `vpi` backend, as the [Dockerfile](Dockerfile) adds.

The Dockerfile installs VPI from NVIDIA's Jetson apt repository, a package source outside the SDK
image, accepting its signing key only with a pinned fingerprint and pinning the package versions;
see the repository's [threat model](../../SECURITY.md#threat-model).

## Build and run

From the repository root (add `--local` to build and run on the host, with `HOLOSCAN_SDK_ROOT` set):

```bash
export HOLOSCAN_CLI_BASE_IMAGE=nvcr.io/nvidia/clara-holoscan/holoscan:v5.0.0-cuda13
holoscan run apriltag_detection          # synthetic: 300 frames, displayed when possible
holoscan run apriltag_detection ci       # 30 frames, headless
```

Both modes run every compiled backend. `--run-args` adds options, for example
`--run-args='--backend cuda,vpi --family tagStandard41h12 --device-input true'`; `--help` lists them.

## Output

Each run prints a line per backend with its frame and tag counts, detection latency (mean, median,
95th percentile, maximum) and ground-truth errors, then `PASS`, `FAIL`, or `STOPPED` when an
interrupt or a closed window ends it early. It exits 0 on `PASS`, 2 for invalid options, else 1.

The operator emits one `AprilTagDetections` message per image
([schema](cpp/operators/apriltag_detections.fbs)): frame number, the image header's
`device_sequence`, image size, backend, family,
detection time, the tags dropped past its 64, whether the detector ran out of capacity before
decoding (tags may then be missing; it also logs a warning), and per tag its id, corrected bits, decision margin, center, corners (from the
tag's bottom-left, counter-clockwise as printed) and, with intrinsics, its pose. Poses place the tag
in the camera frame (x right, y down, z forward); the tag's x is right and y down as printed, z into
the tag. The display draws these axes as red, green and blue arrows (X and Y in the image plane
without a pose) and a dark blue label with the id and, with a pose, the distance.

## Use the operator in your graph

Connect your own camera operator's `ImageT` output. For poses, give the camera's intrinsics (pixel
centers at integer coordinates, as in OpenCV and ROS `CameraInfo`) and the tag's size: the side of
the square the detector finds, the black border for `tag36h11`.

```cpp
#include "operators/apriltag_detector_op.hpp"

auto detector = graph.op<tag_detection::AprilTagDetectorOp>(
    "apriltag", tag_detection::AprilTagDetectorConfig{
                    .backend = tag_detection::Backend::kCuda,
                    .options = {.family = "tag36h11"},
                    .intrinsics = tag_detection::CameraIntrinsics{910.5, 910.5, 639.5, 359.5},
                    .tag_size = 0.08});
graph.add_flow(camera->image, detector->image);              // Output<schema::ImageT>
graph.add_flow(detector->detections, consumer->detections);  // Output<AprilTagDetectionsT>
```

Build inside this tree, or copy the operator files into your project. The backends have no Holoscan
dependency (`tag_detection::create_backend`).

## Performance

Median time from receiving an image to having detections and poses, for 300 synthetic 1280x720
frames with four `tag36h11` tags, one backend at a time, in the application's image (SDK 5.0.0,
CUDA 13.2, VPI 4.0.5) on an RTX 5090 and a Ryzen 9 9950X3D. Every backend found all 1200 tags and
nothing else. Measured with:

```bash
holoscan run apriltag_detection --run-args='--backend cuda --count 300 --period-ms 1 --display false'
```

| Backend | Host RGB input | GPU RGB input (`--device-input true`) | Largest corner error |
| --- | --- | --- | --- |
| `cpu` (one thread) | 21.3 ms | 21.0 ms | 0.22 px |
| `cuda` | 5.3 ms | 4.8 ms | 0.22 px |
| `vpi` (VPI's CPU path on x86_64) | 5.7 ms | 5.4 ms | 0.40 px |

## Tests

`holoscan test apriltag_detection --no-xvfb` checks every backend against ground truth from host
and GPU memory (four families, among them `tagCircle21h7`, and two-bit correction), that frames
over a capacity report what they dropped, the CUDA backend against the CPU on tags cut by the
image edge, the operators
through the SDK's test harness, the application's runs, exit statuses and display graph, and the
overlay's axis drawing, which needs no SDK and also runs on hosted CI. Validated on x86_64 (RTX
5090, CUDA 13.2, VPI 4.0.5: `cpu`, `cuda` and `vpi` on the CPU) and on aarch64, a Jetson Thor
(JetPack 7, SDK 5.0.0, CUDA 13.0: `cpu`, `cuda` and `vpi` on the PVA). On the Thor, the `ci` mode
found all 120 tags with each backend, with median latencies at 1280x720 of 37.9 ms (`cpu`), 14.2 ms
(`cuda`) and 11.7 ms (`vpi` on the PVA).

## Licenses

The application is Apache 2.0. AprilTag 3, linked into the executable, is BSD-2-Clause; its notice
is installed as `share/apriltag_detection/licenses/apriltag-LICENSE.md`. VPI is under NVIDIA's own
terms, installed as `/opt/nvidia/vpi4/doc/VPI_EULA.txt`.
