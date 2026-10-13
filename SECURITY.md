# Security Policy: Holoscan Applications

## Reporting a Vulnerability

If you discover a potential security vulnerability, **do not open a public issue,
pull request, or discussion**.

- **Web (preferred):** Report it through the
  [NVIDIA Vulnerability Disclosure Program](https://www.nvidia.com/en-us/security/).
- **Email:** Send details to [psirt@nvidia.com](mailto:psirt@nvidia.com). For
  encrypted communication, use the
  [NVIDIA public PGP key](https://www.nvidia.com/en-us/security/pgp-key).
- **GitHub:** Use this repository's **Security** tab and select
  **Report a vulnerability** to submit a private report.

Include the project name and affected version or branch, vulnerability type,
reproduction steps, proof-of-concept code when available, and an impact
assessment. Detailed reports help NVIDIA evaluate and address issues faster.

NVIDIA's Product Security Incident Response Team (PSIRT) will acknowledge the
report, validate and assess the issue, develop and test fixes, and publish a
security bulletin when appropriate.

## Supported Versions

Security fixes are applied to the current `main` branch. Application-specific
support depends on the Holoscan SDK versions declared in each application's
`metadata.json` and README.

## Security Architecture and Context

Holoscan Applications is a collection and build framework for accelerated
computing, streaming, and AI applications built with the NVIDIA Holoscan SDK.
The repository-level code discovers applications, registers CMake projects, and
uses the Holoscan CLI to build, run, test, and lint them. The current repository
registers the [Endoscopy Tool Tracking](applications/endoscopy_tool_tracking/README.md)
application, implemented in C++/CUDA with RGB24 replay, TensorRT inference,
optional visualization, and input recording.

This project operates at the application and build-tooling level. Its primary
security responsibility is to preserve the integrity of application source,
metadata, dependencies, container inputs, and CI automation. It does not provide
a shared authentication, authorization, network-security, or input-validation
layer for applications.

The principal security boundaries are:

- Application source, CMake logic, metadata, datasets, and configuration enter
  through repository contributions and application-specific setup.
- The Holoscan CLI, Holoscan SDK, Python package indexes, pre-commit hooks, GitHub
  Actions, and caller-provided container base image are external dependencies.
- Built applications may access GPUs, attached devices, displays, files,
  datasets, and external services with the permissions granted by the host or
  container runtime.
- CI executes repository tooling on GitHub-hosted runners. Local builds execute
  contributed build logic with the developer's privileges.

### Threat Model

1. **Malicious or vulnerable application contributions:** Source files, custom
   operators, CMake logic, and metadata added under `applications/` are executed
   during build, test, or runtime and could access host files, devices, GPU
   memory, credentials, or networks available to those processes.
2. **Build dependency substitution:** `Dockerfile` accepts a caller-provided
   `BASE_IMAGE`, and Python installation uses both the default package index and
   `pypi.nvidia.com`. An untrusted image, compromised index, or package-name
   collision could introduce code during container or CI builds.
3. **CI and lint supply-chain compromise:** GitHub Actions and remote pre-commit
   hooks execute third-party code selected by version tags. A compromised
   upstream release or mutable tag could affect CI runners and developer
   workstations.
4. **Unsafe application-specific inputs:** Endoscopy consumes ONNX models,
   TensorRT engine caches, and RGB24 video files. TensorRT parses ONNX when
   building an engine and deserializes cached engines before the application's
   compatibility checks. The cache key includes a non-cryptographic model
   fingerprint, GPU, TensorRT version, precision, and workspace settings;
   neither cache identity nor compatibility checks authenticate provenance.
5. **Sample downloads and video parsing:** Endoscopy's `prepare_data.py`
   downloads the NGC sample ZIP at version `20230222` over HTTPS or reads a local
   `--archive`. It extracts the ONNX model and H.264 video and invokes FFmpeg to
   convert the video to RGB24. Selected ZIP member paths, nonempty assets, and
   RGB frame sizes are checked, but the script does not verify a trusted
   cryptographic checksum or signature to authenticate the model or video.
6. **Resource exhaustion and unsafe device access:** Applications can perform
   GPU-intensive processing and interact with attached hardware. Unbounded
   inputs or incorrect device configuration could exhaust CPU, GPU, memory, or
   storage resources, or interfere with other workloads sharing the host.

### Critical Security Assumptions

- Maintainers review contributed application code, build logic, metadata, and
  dependency changes before execution or release.
- `BASE_IMAGE` is supplied from an approved registry and pinned to a trusted,
  preferably immutable digest.
- Python indexes, the Holoscan CLI and SDK, GitHub Actions, and pre-commit hook
  repositories are trusted and retrieved over authenticated TLS.
- The host OS, container runtime, GPU driver, Holoscan SDK, and hardware enforce
  process, device, and memory isolation appropriate to the deployment.
- Operators provide model, video, and cached engine artifacts from trusted
  sources and restrict write access to model/data and engine-cache locations.
  Endoscopy writes prepared assets, engine caches, and optional RGB recordings
  to operator-selected locations using the process's filesystem permissions;
  operators must also control access to preparation and recording destinations.
- Secrets are supplied through deployment-specific secret management and are
  not committed to the repository, embedded in metadata, or copied into build
  contexts.
- Operators grant applications only the filesystem, network, GPU, display, and
  device permissions required for their documented operation.

## Dependency and Distribution Security

New or upgraded third-party dependencies must have their license and
redistribution terms reviewed before integration. Dependencies distributed in
source, binary, or container form must follow the applicable NVIDIA Open Source
Review Board process and preserve required license notices and corresponding
source. Release builds should use approved, pinned artifacts and retain a
reviewable dependency inventory.
