# Holoscan Applications

A collection of applications built with the NVIDIA Holoscan SDK. Browse the
[applications](applications/) to find examples and starting points for your own
accelerated computing, streaming, and AI workflows.

Each application owns its source, custom operators, tests, data preparation, and
documentation. Applications can target different Holoscan SDK releases, including
older versions. Check each application's README and `metadata.json` for supported
SDK versions, hardware, dependencies, and data requirements.

## Getting started

Install the [Holoscan CLI](https://pypi.org/project/holoscan-cli/) from PyPI:

```bash
pip install holoscan-cli
```

From the repository root, discover available applications and their modes:

```bash
holoscan list
holoscan modes <application>
```

Replace `<application>` with an application name from `holoscan list`, and
`<mode>` in the examples below with a mode listed for that application. Follow
the application's README to prepare its environment and data before building.

## Build and run

For a local build, use a Holoscan SDK installation compatible with the selected
application:

```bash
export HOLOSCAN_SDK_ROOT=/path/to/holoscan/install
holoscan build <application> <mode> --local
holoscan run <application> <mode> --local
```

For a container build, set `HOLOSCAN_SDK_IMAGE` to a development image that
provides the application's supported SDK version and dependencies:

```bash
holoscan build <application> <mode> --base-img "$HOLOSCAN_SDK_IMAGE"
holoscan run <application> <mode> --no-docker-build
```

Applications may require additional setup for datasets, devices, displays, or
external services. These instructions belong in the individual application's
README. Run `holoscan <command> --help` for command options.

## Testing

Use the application's documented test configuration with the Holoscan CLI:

```bash
holoscan test <application> --local
```

Test requirements vary by application; some need a GPU, sample data, or attached
hardware. See the [CI guide](docs/ci.md) for repository CI and GPU runner setup.

## Contributing an application

Put its `metadata.json`, `CMakeLists.txt`, source, tests, and README under
`applications/<name>/`, then add `add_holoscan_application(<name>)` in alphabetical
order to `applications/CMakeLists.txt`. Keep language-specific build logic in
the corresponding application subdirectory, such as `cpp/` or `python/`.

Document supported SDK versions, platforms, dependencies, data preparation,
build and run modes, and tests in the application's README and metadata.
Keep custom operators inside the application, and explicitly document any
external operator dependencies.

Use the installed CLI to check changes:

```bash
holoscan lint
```

Code contributions use the Apache-2.0 [license](LICENSE). Sign commits with
`git commit -s` to certify the [Developer Certificate of Origin](https://developercertificate.org/).
