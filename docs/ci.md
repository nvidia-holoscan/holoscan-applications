# Continuous integration

Shared CI is maintained on `main`. Application branches add source, tests, and
metadata under `applications/`; they inherit the same repository workflow.
Adding an application requires no application names, paths, or special cases in
the workflow or CI driver.

The hosted `CI` workflow runs on pull requests, pushes to `main`, and manual
dispatch. It installs the pinned tools from `requirements.txt`, tests the shared
driver, discovers applications through `holoscan list --json`, validates CLI
lifecycle command construction, and runs `holoscan lint`.

Hosted runners have no Holoscan SDK or GPU. Lifecycle checks use `--dryrun`;
they do **not** compile applications, execute application tests or inference,
download datasets, or produce tested release artifacts. Driver unit tests,
metadata validation, and lint do execute. Real application testing requires a
compatible SDK and the application's documented hardware and data.

## Unified application interface

Use the [Holoscan CLI](https://github.com/nvidia-holoscan/holoscan-cli) for every
application, locally and in CI:

```text
holoscan build <application> [mode] [CLI options]
holoscan run <application> [mode] [CLI options]
holoscan test <application> [CLI options]
```

The pinned CLI has command-specific argument flags, not a literal `--args` flag:

| Command | Extra arguments |
| --- | --- |
| `build` | `--configure-args='-DOPTION=VALUE'` for CMake; repeat for more options |
| `run` | `--run-args='--flag value'` for application arguments; also accepts `--configure-args` |
| `test` | `--cmake-options='-DOPTION=VALUE;-DOTHER=VALUE'` and `--ctest-options='-DOPTION=VALUE'` |

Use semicolons between multiple CMake entries in `--cmake-options` with the
pinned CLI. `holoscan test` does not accept a build/run mode. Put test
registration and preparation in the application and document any test options
in its README. Run `holoscan <command> --help` for the complete option set.

## Automatic discovery and validation

`scripts/holoscan_ci.py` selects every `application` in the CLI's JSON discovery
output. Each declared language is checked separately (`C++` becomes `cpp`,
`Python` becomes `python`); applications without language metadata are checked
once without `--language`. Non-application projects are excluded. An empty
repository is valid and explicitly reports that no lifecycle checks ran.

Every application/language pair gets:

```bash
holoscan test <application> --local --no-xvfb --dryrun [--language <language>]
```

Applications with a `ci` mode additionally get build and run checks before test:

```bash
holoscan build <application> ci --local --dryrun [--language <language>]
holoscan run <application> ci --local --no-local-build --dryrun [--language <language>]
```

Applications without `ci` receive test command validation only; their build/run
commands are not checked. Discovery errors, invalid selectors, unsupported
languages, conflicting duplicate records, and failed CLI commands fail the job.
Application names must be unique for a given language: discovery records from
different source folders with the same selector are rejected as ambiguous.
The driver passes arguments directly to subprocesses without a shell and stops
at the first lifecycle failure.

## Application-owned CI mode

Define `application.modes.ci` in `metadata.json` to provide a finite, headless,
noninteractive build/run check. Keep application-specific CMake options and run
arguments in that mode. For example, inside an application's `modes` object:

```json
{
  "ci": {
    "description": "Process a bounded synthetic input without a display.",
    "build": {
      "cmake_options": ["-DEXAMPLE_ENABLE_VISUALIZATION=OFF"]
    },
    "run": {
      "command": "\"<holohub_app_bin>/example\" --synthetic true --count 10",
      "workdir": "holohub_app_bin"
    }
  }
}
```

Replace the illustrative option, executable, and arguments with the application's
own settings. This mode may require its supported SDK and GPU, but it must not
require external datasets, attached devices, a display, or external services.
Real-data, device, and longer integration tests belong in the application test
suite and remain accessible through `holoscan test <application>`.

Future GPU execution must use this same CLI interface and application-owned
configuration, with explicitly configured runner labels and compatible SDK
images. No application-specific GPU workflow or packaging job is provided by
the hosted contract checks.

## Developer commands

Install the repository tools:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --extra-index-url https://pypi.nvidia.com \
  -r requirements.txt
```

Run the same checks as hosted CI from the repository root:

```bash
python -m unittest discover -s tests -v
python scripts/holoscan_ci.py
holoscan lint
```

On a machine with the application's prerequisites installed, exercise its real
lifecycle by removing `--dryrun`:

```bash
holoscan build <application> ci --local
holoscan run <application> ci --local --no-local-build
holoscan test <application> --local --no-xvfb
```

These examples use the optional `ci` mode; otherwise select a documented mode
or omit it to use the default. Add the application's language and test options
when needed. SDK selection, data preparation, hardware requirements, and
application arguments are documented in each application's README.
