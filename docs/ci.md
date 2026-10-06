# Continuous integration

The hosted `CI` workflow validates the repository with the pinned Holoscan CLI.
It checks that the CLI starts, application discovery succeeds, and every
committed pre-commit hook passes. These checks do not require a GPU or a Holoscan
SDK installation.

Application-specific build, run, test, data, hardware, and packaging steps
belong with the application that needs them. Keep those workflows parameterized:
do not commit runner labels, registry credentials, or an implicit SDK image.

## Developer commands

Install the repository tools, including the CLI and lint dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --extra-index-url https://pypi.nvidia.com \
  -r requirements.txt
```

Inspect the applications discovered from their metadata and registrations:

```bash
holoscan version
holoscan list
```

Run the repository lint hooks before submitting a change. Pre-commit creates its
hook environments automatically on the first run:

```bash
holoscan lint
```

Use each application's README for its supported SDK version, build modes, test
requirements, datasets, devices, and any additional CI setup.
