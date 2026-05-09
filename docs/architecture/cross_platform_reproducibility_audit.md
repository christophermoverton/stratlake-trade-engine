# Cross-Platform Reproducibility Audit

Issue #360 establishes the Milestone 33 baseline for StratLake portability.
Windows is the only currently trusted development and validation environment.
Linux and macOS support is unverified until CI and runtime validation prove
otherwise.

This audit is evidence gathering, not a portability fix. It preserves the
existing architecture, the artifact-first design, and the M32 governance
source-of-truth boundaries. In particular, `promotion_gates.json` and
`promotion_gate_summary` remain canonical; governance observes, validates,
aggregates, and reports existing outcomes without replaying gates or
recomputing promotion decisions.

## Current Platform Baseline

StratLake has primarily been developed and locally validated on Windows. The
documentation and command examples reflect that history: README examples often
use PowerShell fences and Windows virtual-environment invocation syntax.
CONTRIBUTING.md includes both POSIX and Windows activation examples, but the
broader documentation set is not yet balanced across shells.

Pull-request CI currently runs on a hosted Ubuntu runner. That is useful signal,
but it is not the same as a declared cross-platform support matrix. There is no
Windows or macOS CI job proving installability, runtime behavior, path handling,
artifact determinism, or filesystem assumptions.

## Current Strengths

The repository already has several portability-friendly foundations:

* Most runtime code uses `pathlib.Path` rather than manual separator
  construction.
* JSON artifact helpers use deterministic formatting with sorted keys and a
  trailing newline.
* Existing docs path lint detects Windows absolute paths, Unix home paths, and
  file URI leakage across guarded release-facing docs and examples.
* CI runs docs path lint, deterministic rerun validation, Ruff, and pytest on
  Ubuntu.
* Many artifact references are serialized with POSIX-style separators through
  `Path.as_posix()`.

## Portability Risks

The audit found these actionable risk groups:

* Windows shell bias: README and many docs examples use PowerShell code fences
  and Windows virtual-environment command shapes.
* Environment path examples: `.env.example` contains workstation-style absolute
  path templates for both Windows and Unix home directories.
* Partial line-ending policy: `.gitattributes` normalizes CSV and JSON files,
  but not Python, Markdown, YAML, TOML, workflow, or shell-like text files.
* Packaging ambiguity: `pyproject.toml` lacks an explicit build-system table,
  so editable install and build behavior can vary by installer and platform.
* Artifact path persistence: some helpers resolve paths before reporting status
  or diagnostics, and example writers can fall back to absolute POSIX-rendered
  paths when a relative path cannot be derived.
* Case-sensitivity exposure: no dedicated audit currently checks for import,
  filename, or artifact reference casing that would fail on case-sensitive
  filesystems.

## CI And Release Risks

The current CI workflow validates one Linux environment. It does not prove:

* Windows CI parity with the locally trusted platform.
* macOS installability or runtime behavior.
* cross-shell command compatibility.
* package build reproducibility.
* line-ending stability for non-JSON and non-CSV text files.
* deterministic artifact behavior across OS-specific temporary roots.

Release automation should not claim cross-platform readiness until focused M33
jobs validate at least install, docs path lint, a deterministic artifact smoke
slice, and the relevant pytest slice on Windows, Linux, and macOS.

## Packaging And Installability Gaps

`pyproject.toml` declares project metadata, dependencies, and development
extras. It does not yet declare an explicit PEP 517 build-system. That gap is
not necessarily runtime-breaking today, but it is a release automation risk
because pip and build frontends can infer behavior differently across versions
and environments.

Follow-up work should add a minimal build-system, verify editable installs on
all target platforms, and eventually add package build validation once release
automation is in scope.

## Deterministic Artifact Risks

Artifact determinism is already a core design strength, but path portability
needs a wider audit:

* generated manifests and reports should prefer repository-relative paths when
  a referenced file is inside the repository or artifact root.
* serialized paths should use POSIX-style forward slashes.
* machine-local absolute roots should stay out of committed artifacts and
  release-facing reports.
* status and diagnostics that intentionally include local absolute paths should
  be separated from portable manifests and summaries.

This issue does not change artifact writers. It records the baseline so later
M33 work can adjust serialization contracts deliberately.

## Documentation Gaps

Documentation currently explains many workflows through PowerShell-first
examples. Documentation-only examples are not runtime bugs, but they can block
Linux and macOS users from reproducing workflows. The docs should eventually
provide POSIX shell equivalents for canonical setup, validation, and workflow
commands while keeping Windows commands available and clearly labeled.

The existing docs path lint is a good starting point. Later work should decide
whether `.env.example`, workflow files, and committed generated JSON/CSV outputs
belong in the same lint surface or a separate artifact-path lint.

## Windows-Specific Assumptions Found

The machine-readable summary in
`docs/architecture/cross_platform_reproducibility_audit.json` captures the
stable finding list. Highlights:

* README command examples are heavily PowerShell-oriented.
* CONTRIBUTING.md includes a Windows PowerShell virtual-environment activation
  example, though it also includes a POSIX activation example.
* `.env.example` includes absolute workstation path templates.
* docs and examples include Windows virtual-environment command shapes.
* `.gitattributes` only enforces LF endings for CSV and JSON.
* no Windows/macOS CI matrix currently validates platform behavior.

## Linux And macOS Likely Failure Points

Likely failure points include:

* copied PowerShell commands that do not run in POSIX shells.
* virtual-environment paths that assume the Windows `Scripts` layout instead
  of POSIX `bin`.
* missing build-system metadata during editable install or package build flows.
* case-sensitive filesystem behavior not represented by default Windows local
  validation.
* generated reports that persist local absolute roots instead of portable
  artifact-relative references.
* line-ending differences in Markdown, Python, YAML, TOML, and workflow files.

## Recommended Follow-Ups

Recommended M33 follow-up issues:

* Add a focused Windows, Linux, and macOS CI matrix.
* Add POSIX equivalents for canonical README and docs commands.
* Replace local absolute examples in `.env.example` with portable placeholders
  or broaden lint coverage to guard them intentionally.
* Extend `.gitattributes` text normalization beyond CSV and JSON.
* Add explicit build-system metadata and validate editable installs.
* Audit manifest and report writers for repository-relative, POSIX-style path
  serialization.
* Add case-sensitivity checks for filenames, imports, and artifact references.

## Out Of Scope

Issue #360 explicitly does not:

* fix every portability issue.
* add a Windows, Linux, or macOS CI matrix.
* add release automation.
* add dependency lockfiles.
* publish packages.
* change trading, alpha, ML, strategy, portfolio, or promotion-governance
  behavior.
* alter canonical M32 governance behavior.
