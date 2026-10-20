# {{ cookiecutter.project_name }}

{{ cookiecutter.description }}

The application's graph emits a finite sequence of integers, doubles each value,
and verifies every result at the sink before printing a final verdict. Replace
the operators in `cpp/main.cpp` with your pipeline and keep a finite mode that
checks its own result.

## Requirements

- Linux on x86_64 or aarch64.
- Holoscan SDK {{ cookiecutter.holoscan_version }} or later (C++).
- CMake 4.0 or later and a C++20 compiler.
- Python 3.12–3.14 with the CLI pinned in the repository's `requirements.txt`.

Add your application's hardware, dependencies, and data preparation here, and
link the license terms of any separately downloaded models or datasets.

## Build and run

From the repository root, set `HOLOSCAN_CLI_BASE_IMAGE` to a development image
matching the SDK version above, then run a mode:

```bash
holoscan run {{ cookiecutter.project_slug }}
holoscan run {{ cookiecutter.project_slug }} ci
```

To build and run on the host instead, set `HOLOSCAN_SDK_ROOT` to a compatible
SDK installation and add `--local`.

| Mode | Behavior |
| --- | --- |
| `default` | Processes 100 messages and verifies every result. |
| `ci` | Processes 10 messages without external data, a display, or devices. |

Each run prints `expected`, `received`, `mismatches`, and a `verdict`, and exits
nonzero on failure. `--run-args` appends options to the mode's command, for
example `--run-args='--period-ms 5'`. Run the executable with `--help` for all
options.

## Tests

```bash
holoscan test {{ cookiecutter.project_slug }} --no-xvfb
```

`tests/CMakeLists.txt` registers tests that run the built executable, so they
need the SDK. Register tests that need no SDK, GPU, or external data in the
top-level `CMakeLists.txt` so hosted CI runs them; see `docs/ci.md`.
