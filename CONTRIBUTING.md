# Contributing to Holoscan Applications

Contribute an application, improve an existing one, or help maintain the shared
tooling and documentation. Fork this repository, make your changes on a branch,
and open a pull request with signed-off commits.

## What belongs here

Applications are end-to-end pipelines that own their source, application-specific
operators, tests, and data preparation. Related contributions belong in:

- [Holoscan Tutorials](https://github.com/nvidia-holoscan/holoscan-tutorials)
  for development walkthroughs.
- [Holoscan Benchmarks](https://github.com/nvidia-holoscan/holoscan-benchmarks)
  for benchmarks and measurement tools.
- Independent project repositories for reusable operators and Holoscan Modules;
  see the [Create a Module tutorial](https://github.com/nvidia-holoscan/holoscan-tutorials/tree/main/tutorials/holoscan-modules/create-a-module).
- [Holoscan CLI](https://github.com/nvidia-holoscan/holoscan-cli) for CLI changes
  and the metadata schema.
- [Holoscan SDK issues](https://github.com/nvidia-holoscan/holoscan-sdk/issues)
  for SDK bugs and feature requests.

## Before you contribute an application

Holoscan Applications accepts applications from NVIDIA teams and external
contributors. The quality and maintenance bar is high, so you may prefer to
develop and maintain your application in your own repository instead.

Contributing an application here means you:

- Have CI in place before your first contribution. NVIDIA teams coordinate
  their application's SDK, GPU, and hardware needs with the maintainers before
  opening a pull request. External contributors show that CI already builds and
  tests the application, for example with passing runs in their own repository.
- Maintain the application after it merges. Name its maintainers in the pull
  request, and fix CI failures and SDK incompatibilities as they arise.

Applications that are no longer maintained may be deprecated and removed at
any time.

## Development setup

From the repository root, use Python 3.12–3.14 to install the pinned repository
tools. This example uses Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
holoscan lint
```

Run `holoscan lint` before submitting changes. It runs the same pre-commit hooks
as CI, including formatting, metadata, license, and internal-reference checks.
Formatters may update files; review their changes before committing. Optionally,
run `pre-commit install` to enable the hooks at commit time.

Application builds and tests also need the SDK, hardware, and dependencies
documented in the application's README. See the repository
[build and run instructions](README.md#build-and-run) and the
[CI guide](docs/ci.md#developer-commands) for additional test tools and commands.

## Adding or updating an application

Keep the application's source, custom operators, `metadata.json`, README, tests,
and data-preparation scripts together under `applications/<name>/`.
Use [Endoscopy Tool Tracking](applications/endoscopy_tool_tracking/) as an
example of the repository conventions.

- Describe supported SDK versions, platforms, hardware, dependencies, data
  preparation, build/run modes, and test commands in the application's README.
  Link the license terms for separately downloaded models and datasets.
- Define the application and its supported modes in `metadata.json`, keeping
  requirements consistent with the README. Check discovery with `holoscan list`
  and `holoscan modes <application>`.
- Register new applications with `add_holoscan_application(<name>)` in
  [applications/CMakeLists.txt](applications/CMakeLists.txt), in alphabetical
  order. Keep application-specific build logic in the application's directory.
- Register tests through CTest. Keep SDK-free tests outside language-specific
  dependency scopes so hosted CI can run them. See the
  [hosted test conventions](docs/ci.md#application-tests-on-hosted-runners).
- Provide a finite, headless, noninteractive `ci` mode when possible. It may
  require the SDK and GPU, but must not require external datasets, attached
  devices, a display, or external services. Follow the
  [CI mode contract](docs/ci.md#application-owned-ci-mode).

Shared CI discovers applications automatically; adding one does not require
application-specific changes to the workflow or CI driver.

## Testing and review

Open a draft pull request if you want early feedback. Before requesting review:

- Run `holoscan lint` and the tests relevant to your changes; required CI checks
  must pass.
- For application changes, report the SDK version, platform and hardware, exact
  test commands, and results in the pull request. Use
  `holoscan test <application> --local` and, when defined,
  `holoscan run <application> ci --local`, with the application's documented
  options. State any tests you could not run.
- Explain the change and obtain a maintainer's approval before merging.

Hosted CI runs SDK-free application tests and validates CLI command construction
with dry runs. It has no SDK or GPU and does not establish that an application
builds or runs with the SDK. A separately configured SDK/GPU job runs on `main`,
not on pull requests; a skipped job provides no runtime validation. See the
[CI guide](docs/ci.md) for coverage and local verification commands.

## Security-sensitive steps

If an application requires privileged containers, host device or kernel changes,
open network ports, or downloads without checksums, explain the risk in its
README before the relevant step and update the
[security threat model](SECURITY.md#threat-model).

Report vulnerabilities through [SECURITY.md](SECURITY.md#reporting-a-vulnerability)
instead of opening a public issue or pull request.

## Sign off your commits

Every contribution commit must include a `Signed-off-by` trailer. Add it with:

```bash
git commit -s -m "Describe your change"
```

The sign-off certifies the
[Developer Certificate of Origin](https://developercertificate.org/): you have
the right to submit the contribution under the repository's
[Apache 2.0 license](LICENSE).
