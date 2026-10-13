# Contributing to Holoscan Applications

Thank you for sharing an application built with the
[NVIDIA Holoscan SDK](https://github.com/nvidia-holoscan/holoscan-sdk)! To contribute,
fork this repository, add or update a directory under `applications/`, run
`holoscan lint`, and open a pull request with signed-off commits:

```bash
python3 -m venv .venv && . .venv/bin/activate
python -m pip install --extra-index-url https://pypi.nvidia.com -r requirements.txt
holoscan lint
git commit -s -m "Add my application"
```

The [README](README.md#contributing-an-application) describes how to lay out and
register an application. The [CI guide](docs/ci.md) describes the application
contract, including the `ci` mode and tests that run without an SDK. This guide covers
the rest of the process.

## What belongs here

This repository holds applications: end-to-end pipelines that each own their source,
custom operators, tests, and data preparation. Other contributions have a different
home:

- Tutorials go to [Holoscan Tutorials](https://github.com/nvidia-holoscan/holoscan-tutorials).
- Benchmarks and measurement tools go to
  [Holoscan Benchmarks](https://github.com/nvidia-holoscan/holoscan-benchmarks).
- Operators and Holoscan Modules meant for reuse across applications go to
  [HoloHub](https://github.com/nvidia-holoscan/holohub).
- Holoscan CLI changes and the metadata schema go to
  [Holoscan CLI](https://github.com/nvidia-holoscan/holoscan-cli).
- SDK bugs and feature requests go to the
  [Holoscan SDK](https://github.com/nvidia-holoscan/holoscan-sdk/issues).

## Review and acceptance

Open a draft pull request early if you want feedback while you work. When the
application is ready, mark the pull request ready for review. It is accepted once:

- `holoscan lint` and the CI workflow pass,
- a Holoscan maintainer approves it, and
- the pull request lists the SDK version, platform, and `holoscan test` or `ci` mode
  run you used to test it.

Hosted CI has no SDK or GPU, so it does not build or run your application. Your test
report is the evidence that the application works.

## Security-sensitive steps

Some applications ask readers to reconfigure their system. If yours runs privileged
containers, changes host devices or kernel settings, opens network ports, or downloads
files without checksums, say so in its README before the step. Then add the risk to
the threat model in [SECURITY.md](SECURITY.md).

## Signing your contribution

Every commit must carry a `Signed-off-by` trailer, added by `git commit -s`. Signing off
certifies the [Developer Certificate of Origin](https://developercertificate.org/): you
wrote the contribution, or otherwise have the right to submit it under the
[Apache-2.0 license](LICENSE). Pull requests with unsigned commits are not accepted.

To report a security vulnerability, follow [SECURITY.md](SECURITY.md) instead of opening a
public issue.
