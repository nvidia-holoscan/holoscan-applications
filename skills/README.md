# Agent skills

Agent skills for debugging Holoscan CLI commands and for developing,
packaging, and consuming Holoscan Modules. Each directory is a self-contained
skill: start with its `SKILL.md` and load only the references it names.

| Skill | Use it for |
| --- | --- |
| [`holoscan-app-debug-build-run`](holoscan-app-debug-build-run/SKILL.md) | Diagnosing one concrete `holoscan` command, here or in a standalone Module, that fails, hangs, regresses, or returns wrong output |
| [`holoscan-module-lifecycle`](holoscan-module-lifecycle/SKILL.md) | Developing and consuming external Holoscan Modules: scaffold, test, Python bindings, DEB/WHEEL packaging, clean-consumer proof, and use in this repository's applications |

The skills are verified against `holoscan-cli` 5.0.0, the version pinned in
[`requirements.txt`](../requirements.txt). The
`references/holoscan-cli-contract.md` file in each skill records the CLI
evidence they share.

## Validation and signing

Each skill's `evals/evals.json` holds its evaluation cases. After a skill
passes skill validation, its directory also carries the generated
`BENCHMARK.md`, `skill-card.md`, and `skill.oms.sig`. The signature covers the
skill's files, so do not edit or reformat a signed skill here. Change skill
content through validation, then copy the re-signed directory unchanged.
