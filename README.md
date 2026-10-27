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

Clone the repository and install its required Holoscan CLI release from
PyPI. Use Python 3.12–3.14; this example uses Python 3.12:

```bash
git clone https://github.com/nvidia-holoscan/holoscan-applications.git
cd holoscan-applications
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install "holoscan-cli==5.0.0"
holoscan list
```

The [Holoscan CLI](https://github.com/nvidia-holoscan/holoscan-cli) discovers
applications in your checkout and provides build, run, and test commands.
Choose a name from `holoscan list` and inspect its modes:

```bash
holoscan modes <application>
```

Prepare the selected application's SDK environment and data before running it.
For SDK setup, see the
[installation guide](https://docs.nvidia.com/holoscan/sdk-user-guide/setup/sdk-installation)
and the application's version-specific instructions.

## Build and run

Run commands from the repository root. Replace `<application>` with a name
from `holoscan list`. Choose `<mode>` from `holoscan modes <application>`,
or omit it to use the default mode.

For a local build, install the application's development dependencies and set
`HOLOSCAN_SDK_ROOT` to a compatible SDK installation:

```bash
export HOLOSCAN_SDK_ROOT=/path/to/holoscan/install
holoscan build <application> <mode> --local
holoscan run <application> <mode> --local
```

For container builds, install Docker and the NVIDIA Container Toolkit.
Set `HOLOSCAN_CLI_BASE_IMAGE` to a development image matching the application's
SDK version, architecture, and dependencies, then run:

```bash
holoscan build <application> <mode>
holoscan run <application> <mode> --no-docker-build
```

Follow the application's README for any additional device, display, or data
setup, standalone build instructions, and tests. Where tests are configured,
run `holoscan test <application> --local` in the required environment.
Use `holoscan <command> --help` for more options.

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
