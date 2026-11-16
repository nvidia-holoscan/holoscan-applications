# Holoscan Applications

Applications built with the [NVIDIA Holoscan SDK](https://github.com/nvidia-holoscan/holoscan-sdk).
Run them to explore accelerated sensor-processing and AI workflows, or adapt
them as starting points for your own projects.

This repository is part of the Holoscan ecosystem and is listed in the
[HoloHub registry](https://github.com/nvidia-holoscan/holohub). Applications are
developed and run from this independent repository, with their source,
application-specific operators, data preparation, and documentation kept
together.

## Choose an application

Browse [applications](applications/) to find a workflow that matches your needs.
Read the selected application's README and `metadata.json` for supported SDK
versions, platforms, hardware, dependencies, and data preparation. Requirements
vary by application.

## Getting started

Clone the repository and install its pinned tools, including the
[Holoscan CLI](https://github.com/nvidia-holoscan/holoscan-cli), from PyPI.
Use Python 3.12–3.14; this example uses Python 3.12:

```bash
git clone https://github.com/nvidia-holoscan/holoscan-applications.git
cd holoscan-applications
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
holoscan list
```

The CLI discovers applications in your checkout and provides build, run, and
test commands. Choose a name from `holoscan list` and inspect its modes:

```bash
holoscan modes <application>
```

Prepare any data the application's README requires before running it.

## Build and run

Run commands from the repository root. Replace `<application>` with a name
from `holoscan list`. Choose `<mode>` from `holoscan modes <application>`,
or omit it to use the default mode.

By default, the CLI builds and runs applications in a container. Install Docker
and the NVIDIA Container Toolkit, select a Holoscan SDK development image, then
preview and run:

```bash
export HOLOSCAN_CLI_BASE_IMAGE=nvcr.io/nvidia/clara-holoscan/holoscan:v5.0.0-cuda13
holoscan run <application> <mode> --dryrun --verbose
holoscan run <application> <mode>
```

`--dryrun --verbose` prints the selected image and the build and run commands
without executing them. The
[Holoscan SDK image](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/containers/holoscan)
above supports x86_64 and aarch64; use another image if the application's README
requires one. For reproducible builds, pin the image to an immutable digest
instead of a tag, as [SECURITY.md](SECURITY.md) assumes.

### Build on the host

To build and run without a container, install the application's development
dependencies and a compatible
[SDK installation](https://docs.nvidia.com/holoscan/sdk-user-guide/setup/sdk-installation),
then add `--local`:

```bash
holoscan run <application> <mode> --local
```

The CLI finds an SDK installed under `/usr`, such as the Debian packages, by
default. For an SDK installed elsewhere or a source build, set
`HOLOSCAN_SDK_ROOT=/path/to/holoscan/install`.

### Tests and troubleshooting

Where tests are configured, run `holoscan test <application>`, adding `--local`
on the host; `holoscan test` takes no mode. Follow the application's README for
any additional device, display, or data setup and for standalone builds. If a
command fails, `holoscan env-check` and `holoscan env-info` report the detected
environment. Use `holoscan <command> --help` for more options.

## Learn more

See [Holoscan Tutorials](https://github.com/nvidia-holoscan/holoscan-tutorials)
for development walkthroughs and [Holoscan Benchmarks](https://github.com/nvidia-holoscan/holoscan-benchmarks)
to measure performance.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for application layout, development setup,
testing, and pull request requirements.

## License

Code is licensed under [Apache 2.0](LICENSE). Separately downloaded models and
datasets have their own terms, linked from each application's documentation.
