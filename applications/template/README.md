# Application template

Generates a self-contained C++ application for Holoscan SDK 5 that follows the
repository conventions in [CONTRIBUTING.md](../../CONTRIBUTING.md) and
[docs/ci.md](../../docs/ci.md):

- `metadata.json` with a `default` mode and a finite, headless `ci` mode;
- a top-level `CMakeLists.txt` for SDK-free tests and a C++ scope that the
  CLI's `--language` option can disable;
- a minimal SDK 5 graph in `cpp/main.cpp` that verifies its own result;
- CTest registration and a README to complete.

The template generates C++ applications only.

## Create an application

From the repository root, in the environment installed from
`requirements.txt`, preview the application and then create it:

```bash
holoscan create my_app --template applications/template --interactive false \
  --context description="One-sentence description" \
  --context full_name="Your Name" --context affiliation="Your Organization" --dryrun
holoscan create my_app --template applications/template --interactive false \
  --context description="One-sentence description" \
  --context full_name="Your Name" --context affiliation="Your Organization"
```

The application is created in `applications/my_app/`. Other `--context` keys
are `copyright_holder` (defaults to the affiliation), `version` (defaults to
`0.1.0`), and `holoscan_version` (the minimum SDK version, 5.0.0 or later).

Register the application in `applications/CMakeLists.txt` as
`add_holoscan_application(my_app)`, in alphabetical order. holoscan-cli 5.0.0's
`holoscan create` appends a registration line in another format to the end of
that file, which CMake rejects; replace that line.

Check discovery and metadata, then follow the generated README to build, run,
and test the application:

```bash
holoscan list
holoscan modes my_app
python .github/scripts/validate_metadata.py applications/my_app/metadata.json
```

## Complete the application

- Replace the example operators in `cpp/main.cpp`, keeping a finite `ci` mode
  that checks its own result.
- Update the authors, tags, platforms, ranking, and requirements in
  `metadata.json`, and complete the README.
- Add data preparation as an application script that runs before the
  application, never at runtime.
- Register SDK-free tests outside the language scope so hosted CI runs them.
- Run `holoscan lint` and the tests listed in [CONTRIBUTING.md](../../CONTRIBUTING.md).
