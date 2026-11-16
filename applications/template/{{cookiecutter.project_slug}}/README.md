# {{ cookiecutter.project_name }}

{{ cookiecutter.description }}

The application's graph emits a finite sequence of integers, doubles each value,
and verifies every result at the sink before printing a final verdict. Replace
the operators in `cpp/main.cpp` with your pipeline and keep a finite mode that
checks its own result.

## Requirements

- Linux on x86_64 or aarch64 with an NVIDIA GPU and a compatible driver.
- Holoscan SDK {{ cookiecutter.holoscan_version }} or later (C++).
- CMake 4.0 or later and a C++20 compiler.
- Python 3.12–3.14 and the CLI pinned in the repository's `requirements.txt`
  for repository workflows.

Add your application's hardware, dependencies, and data preparation here, and
link the license terms of any separately downloaded models or datasets.

## Build and run

Install the repository tools as described in Getting started in the repository
README. Run the commands below from the repository root.

### In a container

The CLI builds and runs in a container by default. Select the Holoscan SDK image:

```bash
export HOLOSCAN_CLI_BASE_IMAGE=nvcr.io/nvidia/clara-holoscan/holoscan:v{{ cookiecutter.holoscan_version }}-cuda13
holoscan run {{ cookiecutter.project_slug }}
holoscan run {{ cookiecutter.project_slug }} ci
```

Add `--dryrun --verbose` to preview the image and the build and run commands.

### On the host

Install the [requirements](#requirements) on the host and add `--local`:

```bash
holoscan run {{ cookiecutter.project_slug }} --local
```

The CLI finds an SDK installed under `/usr`, such as the Debian packages, by
default. For an SDK installed elsewhere or a source build, set
`HOLOSCAN_SDK_ROOT=/path/to/holoscan/install`.

### Modes and options

| Mode | Behavior |
| --- | --- |
| `default` | Processes 100 messages and verifies every result. |
| `ci` | Processes 10 messages without external data, a display, or devices. |

Each run prints `expected`, `received`, `mismatches`, and a `verdict`, and exits
nonzero on failure. `--run-args` appends options to the mode's command, for
example `--run-args='--period-ms 5'`; `--run-args='--help'` lists them all.

## Tests

The CLI supplies the repository's CTest driver; `holoscan test` takes no mode.
In a container, with `HOLOSCAN_CLI_BASE_IMAGE` selected:

```bash
holoscan test {{ cookiecutter.project_slug }} --no-xvfb
```

For host tests, add `--local`.

`tests/CMakeLists.txt` registers tests that run the built executable, so they
need the SDK. Register tests that need no SDK, GPU, or external data in the
top-level `CMakeLists.txt` so hosted CI runs them; see `docs/ci.md`.
