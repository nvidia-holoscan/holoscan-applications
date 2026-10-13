# Continuous integration

Shared CI is maintained on `main`. Application branches add source, tests, and
metadata under `applications/`; they inherit the same repository workflow.
Adding an application requires no application names, paths, or special cases in
the workflow or CI driver.

The `CI` workflow runs on pull requests, pushes to `main`, and manual dispatch.
Hosted jobs run the shared driver tests, CLI lifecycle contract checks, and
application tests without SDK dependencies on Python 3.12, 3.13, and 3.14. Lint
(including metadata, license markers, and internal references) runs once.
Separate jobs validate the workflow, build the repository tooling Dockerfile,
and check local and external documentation links, including section anchors.
New PR commits cancel older PR runs.

Hosted runners have no Holoscan SDK or GPU. Their lifecycle contract checks use
`--dryrun`; the application-owned CPU tests actually execute through CTest.
They do **not** compile C++/CUDA applications, exercise inference, download
datasets, or produce tested release artifacts. A separate SDK job performs
build/run/test execution on `main` when compatible runners and an SDK image
are configured. A skipped SDK job is not evidence of application correctness.

## Comparison with Holohub main

Reviewed on 2026-10-07 against Holohub commit
[`f9394a8`](https://github.com/nvidia-holoscan/holohub/tree/f9394a8de18c9ca4851ba6b270fe774d60965a2b).
Its public workflows check CLI compatibility, Docker builds, lint, metadata,
compliance, and links. Its
[CLI CTest job](https://github.com/nvidia-holoscan/holohub/blob/f9394a8de18c9ca4851ba6b270fe774d60965a2b/.github/workflows/check_cli.yml)
tests CLI integration; the
[Docker job](https://github.com/nvidia-holoscan/holohub/blob/f9394a8de18c9ca4851ba6b270fe774d60965a2b/.github/workflows/check_docker.yml)
builds tooling images. Neither establishes GPU application correctness. There
are application CTest scripts in Holohub, but the public GitHub workflows do not
invoke the all-application build/run script. This review does not establish the
coverage of any private or externally managed CI.

| Change here | What it does | Why |
| --- | --- | --- |
| Added hosted application tests | Runs CTest with all applications selected and both language scopes disabled; endoscopy's data-preparation regressions run with fixtures. | Existing CI only tested the driver and constructed application commands. |
| Moved CPU test registration | Registers SDK-free tests outside the C++/CUDA dependency scope, in the owning application. | SDK-free tests were unreachable without configuring the SDK and compiler. |
| Added Python 3.12–3.14 matrix | Tests the driver, installed CLI contract, and CPU tests across the application's documented Python versions. | Only Python 3.12 was checked; Holohub also checks multiple Python versions. |
| Added SDK execution job and `--execute` | On configured `main` runs, builds applications, runs finite `ci` modes, and executes their CTest suites on a GPU runner. | Dry runs cannot catch compiler, linker, runtime, or CUDA regressions. |
| Added zero-test failure and diagnostics | Fails executable test runs that discover no tests; uploads CTest output and SDK logs even after failure. | Empty green runs and lost diagnostics hide missing coverage and failures. |
| Added Docker build | Builds the actual repository Dockerfile on a small Python base; its existing CLI version check validates installation. | Dependency installation and Dockerfile regressions were unchecked. This does not validate an SDK image. |
| Added workflow and link validation | Runs pinned actionlint in the Docker CI job and Lychee for local and external Markdown links, including section anchors. | YAML syntax alone misses invalid Actions expressions; link checks catch missing files, sections, and unreachable URLs. Local lint does not require Docker. |
| Removed separate dry-run build command | `holoscan run ... ci` now includes the build phase; test command validation remains. | The extra build process repeated the same mode's build planning. |
| Removed repeated work from superseded PRs | Cancels obsolete PR runs; caches pre-commit environments; keeps lint outside the Python matrix. | Rechecking outdated commits and reinstalling unchanged hooks add cost without new coverage. |
| Enforced build execution | Overrides `HOLOSCAN_CLI_ALWAYS_BUILD` for the driver's child processes. | An inherited setting could otherwise bypass validation or run an old executable. |
| Excluded CUDA cache output | Ignores `.nv/` in Git and Docker contexts. | Generated GPU cache files are neither source nor image inputs. |
| Kept metadata/compliance inside lint | Retains the existing schema, SPDX identifier, and internal-artifact hooks. | Separate metadata/schema and copyright jobs would repeat overlapping pre-commit checks. No such standalone jobs existed here to delete. |

Holohub's standalone metadata workflow additionally gathers catalog metadata;
that catalog generation is distinct from schema validation. This repository's
SPDX hook now also validates copyright notices and years, and Codespell checks
spelling through the shared lint configuration.
CLI source-checkout and wrapper tests belong to the CLI/wrapper repository; this
repository tests the pinned installed CLI against its own applications.

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

`.github/scripts/holoscan_ci.py` selects every `application` in the CLI's JSON discovery
output. Each declared language is checked separately (`C++` becomes `cpp`,
`Python` becomes `python`); applications without language metadata are checked
once without `--language`. Non-application projects are excluded. An empty
repository is valid and explicitly reports that no lifecycle checks ran.

Every application/language pair gets:

```bash
holoscan test <application> --local --no-xvfb --dryrun [--language <language>]
```

Applications with a `ci` mode additionally get a run check, including its build,
before test:

```bash
holoscan run <application> ci --local --dryrun [--language <language>]
```

Applications without `ci` receive test command validation only; their build/run
commands are not checked. Discovery errors, invalid selectors, unsupported
languages, conflicting duplicate records, and failed CLI commands fail the job.
Application names must be unique for a given language: discovery records from
different source folders with the same selector are rejected as ambiguous.
The driver passes arguments directly to subprocesses without a shell and stops
at the first lifecycle failure. With `--execute`, it removes `--dryrun`, requires
at least one application, and passes `--ctest-options=--no-tests=error` to each
test command. The driver explicitly enables automatic builds, overriding
`HOLOSCAN_CLI_ALWAYS_BUILD` in its subprocesses. Run this option only with the
application's prerequisites present.

## Application tests on hosted runners

Hosted CI runs the normal CLI test entry point with C++ disabled:

```bash
holoscan test --local --no-xvfb --verbose \
  --cmake-options='-DBUILD_ALL=ON;-DHOLOHUB_BUILD_CPP=OFF;-DHOLOHUB_BUILD_PYTHON=OFF' \
  --ctest-options=--no-tests=error
```

Applications register SDK-free tests in their top-level `CMakeLists.txt`, guarded
by `BUILD_TESTING`, outside language subdirectories that load SDK dependencies.
Guard those subdirectories with `HOLOHUB_BUILD_CPP` and `HOLOHUB_BUILD_PYTHON`,
the same options the CLI's `--language` switch toggles; the repository defines
both at the top level with `ON` defaults.
These tests must use only the hosted test tools and local fixtures. Tests that
require the SDK, a built executable, a GPU, or external data stay under the
appropriate dependency scope. New applications follow this convention without
changes to the workflow. Hosted CI fails if the entire suite contains no tests;
this does not enforce a minimum test count for each individual application.

Endoscopy's data-preparation suite uses a fixture FFmpeg executable and local
archives. Its replay executable tests and CUDA/TensorRT tests remain in its C++
suite. CTest logs and XML results are retained for seven days.

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

## SDK and GPU execution

Configure these repository Actions variables to enable the SDK job:

| Variable | Value |
| --- | --- |
| `HOLOSCAN_CI_RUNNER_LABELS` | JSON array matching provisioned Linux GPU runners, for example `["self-hosted", "linux", "ARM64", "holoscan"]`. These are illustrative labels, not provisioned infrastructure. |
| `HOLOSCAN_CI_SDK_IMAGE` | Explicit compatible development image tag or digest. The image must provide all selected applications' SDK components, CUDA/TensorRT/compiler dependencies, Python, CMake 4+, and Ninja. |

Runners need Docker with NVIDIA Container Toolkit and access to the configured
image. Prefer ephemeral runners. The image must expose its SDK through
`HOLOSCAN_SDK_ROOT` or the CLI's standard `/opt/nvidia/holoscan` location, and
support application execution as the runner's numeric UID/GID. The container's
`HOME` is the mounted workspace, so compiler, CMake, and CUDA caches land in
ignored workspace directories instead of an unwritable default home. It must support
the applications' default test configurations as well as their `ci` modes:
`holoscan test` does not accept a mode or inherit the mode's CMake options.

The workflow layers the repository Dockerfile over that SDK image, then invokes
`python3 .github/scripts/holoscan_ci.py --execute` inside the container with GPU access.
Each application with a `ci` mode builds and runs its bounded smoke check, then
its normal test suite configures, builds, and runs separately. Those two build
configurations are intentionally distinct. Applications without `ci` run only
their test lifecycle. All discovered applications must be supported by the
configured image; applications targeting incompatible SDKs require separate
compatible execution environments before enabling this repository-wide job.

Execution is restricted to pushes to `main` and manual dispatch on `main`, after
hosted checks pass. Pull requests never execute on these GPU runners. Missing
configuration skips the job and produces an explicit hosted-job summary notice.
At review time this repository had no registered runners or SDK CI variables;
adding this workflow does not provision either. Build logs, test logs, and CTest
results are uploaded even after failure and retained for seven days.

The default endoscopy suite includes replay, malformed-engine, cache, and CUDA
postprocessor tests. Its real-model inference test still requires prepared data
and `ENDOSCOPY_DATA_DIR`, as documented in the application README. No automatic
dataset download, performance benchmark, release packaging, or hardware-device
integration is implied by the SDK job.

## Developer commands

`holoscan lint` runs the same pre-commit configuration locally and in CI:

- Ruff checks Python errors, import ordering, modernization, and Bugbear rules,
  then formats Python files. `pyproject.toml` extends Ruff's default rules,
  targets Python 3.12 (the oldest supported workflow version), and uses a
  100-column formatting target to match ClangFormat and cpplint.
- ClangFormat applies `.clang-format` to C, C++, and CUDA files; cpplint retains
  the existing C/C++ checks. CMake files use the upstream cmakelint hook.
- markdownlint-cli2 checks and fixes Markdown using `.markdownlint-cli2.yaml` and
  honors `.gitignore`. Its Node.js runtime is pinned and installed by pre-commit.
- Codespell reports common spelling mistakes in tracked text files, including
  source comments and documentation. It uses the same generated-file exclusions
  as the other hooks.
- File checks catch malformed JSON/YAML/TOML, merge conflicts, case collisions,
  broken symlinks, Python debug statements, private keys, and files over 1 MiB.
  Whitespace fixes preserve Markdown hard breaks and normalize line endings.
- Repository hooks check SPDX identifiers, copyright notices, internal staging
  references, and Holoscan metadata using `check_copyright.py`, `check_internal_artifacts.py`, and
  `validate_metadata.py` in `.github/scripts/`. Each script accepts filenames
  as arguments and can also be run directly from the repository root. Metadata
  validation uses the CLI and `jsonschema` from `requirements.txt`; the hooks
  require no SDK or GPU. Their regression tests run with the shared driver tests.

The SPDX hook requires an Apache-2.0 license identifier and a copyright notice
with a holder and a valid year or year range. It accepts SPDX copyright notices
and traditional `Copyright` headers. Reversed ranges and future years fail.
Changed files must include the current UTC year in at least one notice; older
third-party notices are preserved. Locally, this applies to staged, unstaged,
and untracked changes. CI compares against the pull request's base commit or
the previous push; the lint checkout fetches the required history. Set
`COPYRIGHT_BASE_REF` to a commit or ref to check committed changes locally.
An unavailable explicit base fails the check. Outside a Git checkout, or for
an initial commit or branch push, all supplied files need the current year.
Manual CI runs still validate all headers and ranges without requiring annual
changes to untouched files. The hook reports errors without rewriting notices.

Hook repositories use immutable commit revisions with release labels. To update
them, run `pre-commit autoupdate --freeze`, review the changes, and run
`holoscan lint`. Review the separately pinned Node.js runtime when updating
markdownlint-cli2. Formatters may update files; review those edits before staging.
Use `pre-commit install` to run the checks automatically when committing.

Install the repository tools:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r .github/requirements-ci.txt
```

The requirements pin the alpha CLI to a NVIDIA wheel URL and SHA-256 digest.
All other packages resolve from the default PyPI index; no extra index is needed.

Run the same checks as hosted CI from the repository root:

```bash
python -m unittest discover -s .github/scripts -v
python .github/scripts/holoscan_ci.py
holoscan test --local --no-xvfb \
  --cmake-options='-DBUILD_ALL=ON;-DHOLOHUB_BUILD_CPP=OFF;-DHOLOHUB_BUILD_PYTHON=OFF' \
  --ctest-options=--no-tests=error
holoscan lint
```

On a machine with every application's prerequisites installed, run the SDK job's
lifecycle directly:

```bash
python .github/scripts/holoscan_ci.py --execute
```

For a single application, use `holoscan run <application> ci --local` and
`holoscan test <application> --local --no-xvfb`, adding its documented language
and test options. SDK selection, data preparation, hardware requirements, and
application arguments are documented in each application's README. Workflow
validation and image builds require Docker; local lint and hosted CTest tools
are installed from `.github/requirements-ci.txt`.
