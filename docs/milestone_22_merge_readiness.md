# Milestone 22 Merge Readiness

This checklist defines the pre-merge and release-traceability workflow for
Milestone 22 verification hardening.

The goal is to make milestone validation explicit, deterministic, and auditable
through one reusable validation bundle.

## Scope Delivered

Milestone 22 issue M22.1 adds:

* milestone-branch validation workflow in
  `.github/workflows/milestone_validation.yml`
* docs/path lint guard for release-facing content
* deterministic rerun verification for selected canonical pipeline examples
* standardized validation bundle generation at
  `artifacts/qa/milestone_validation_bundle/`

## Guarded Docs/Examples Surfaces

Docs/path lint enforces no local absolute-path leakage across:

* `README.md`
* `docs/**/*.md`
* `docs/examples/**/*.md`
* `docs/examples/**/*.py`
* `examples/**/*.py`

Findings fail validation and are written to
`artifacts/qa/docs_path_lint.json` (or the bundle copy under
`artifacts/qa/milestone_validation_bundle/checks/docs_path_lint.json`).

## Deterministic Rerun Targets

Milestone rerun validation checks canonical examples:

* `docs/examples/pipelines/baseline_reference/pipeline.py`
* `docs/examples/pipelines/robustness_scenario_sweep/pipeline.py`
* `docs/examples/pipelines/declarative_builder/pipeline.py`

Each target is run twice in isolated output roots. Validation passes when the
normalized summary payloads remain identical across reruns.

## Milestone-Branch Validation Path

Run all commands from repository root with the virtual environment activated.

### 1. Docs/path lint

```powershell
python -m src.cli.run_docs_path_lint --output artifacts/qa/docs_path_lint.json
```

### 2. Deterministic rerun validation

```powershell
python -m src.cli.run_deterministic_rerun_validation --output artifacts/qa/deterministic_rerun.json
```

### 3. Milestone validation bundle

```powershell
python -m src.cli.run_milestone_validation --bundle-dir artifacts/qa/milestone_validation_bundle --include-full-pytest
```

This command runs:

* docs/path lint
* deterministic rerun validation
* `ruff` on milestone validation surfaces (`src/validation/`, new validation
  CLIs, and milestone pytest slice files)
* milestone validation pytest slice
* full pytest suite (when `--include-full-pytest` is set)

## Standardized Bundle Layout

`artifacts/qa/milestone_validation_bundle/` contains:

* `summary.json`
* `checks/docs_path_lint.json`
* `checks/deterministic_rerun.json`
* `checks/ruff.log`
* `checks/pytest_milestone_slice.log`
* `checks/pytest_milestone_slice_junit.xml`
* `checks/pytest_full.log` and `checks/pytest_full_junit.xml` when full pytest
  is enabled

The summary contract answers:

* what was validated
* which checks passed or failed
* where each report and log is stored
* whether milestone validation is merge-ready

## CI and Workflow Wiring

PR CI now includes:

* `Lint (ruff)`
* `Docs Path Lint`
* `Deterministic Rerun`
* `Test (pytest)`

Milestone branches add:

* `Milestone Validation Bundle` workflow job with downloadable bundle artifact

## Release-Traceability Expectations

Before merging a milestone branch into `main`, confirm:

* required CI checks are green
* docs/path lint finding count is zero
* deterministic rerun status is passed
* milestone validation bundle status is passed
* bundle `summary.json` and check reports are attached to the milestone PR

After merge, rerun the milestone bundle on `main` before tagging.
