# M34 Statistical Robustness Artifact Contract

Milestone 34 adds a statistical robustness evidence layer for research outputs. The first contract is intentionally small: it defines deterministic schemas, writer behavior, manifest conventions, and a Markdown skeleton that later diagnostics can populate without changing artifact names or downstream integration points.

## Purpose

The robustness bundle answers a different question than promotion governance. Governance reporting observes existing promotion outcomes, reason codes, and review metadata. Robustness validation records whether research evidence is statistically defensible enough to trust, including walk-forward efficiency, sample-size adequacy, sensitivity, multiple-testing metadata, and future temporal validation checks.

Issue 384 does not calculate those diagnostics. It creates the durable artifact interface that later M34 issues can extend.

## Canonical Bundle

Robustness reports are written under:

```text
artifacts/robustness/<report_id>/
```

Each bundle contains:

- `robustness_summary.json`
- `robustness_findings.json`
- `walk_forward_efficiency.csv`
- `sample_size_validation.json`
- `sensitivity_summary.csv`
- `multiple_testing_summary.json`
- `robustness_report.md`
- `manifest.json`

JSON files are emitted with sorted keys, two-space indentation, no non-finite floats, and LF line endings. CSV files use explicit column lists and LF line endings. Empty diagnostics are schema-valid: CSV files still contain headers and JSON files still contain their top-level arrays and field contracts.

## Finding Schema

Findings use stable fields:

- `check_id`
- `severity`
- `workflow_type`
- `run_id`
- `message`
- `details`

Severity values are normalized to:

- `info`
- `warning`
- `needs_review`
- `reject`
- `blocked`

`details` is reserved for structured evidence and is serialized through the same portable-path sanitizer as manifests.

## Summary Schema

`robustness_summary.json` rolls report evidence into deterministic counts:

- `report_id`
- `workflow_type_counts`
- `finding_count`
- `finding_count_by_severity`
- `highest_severity`
- `artifact_count`
- `source_run_count`
- `source_run_ids`
- `robustness_status_counts`
- `checks_present`
- `checks_missing`
- `generated_artifacts`

The summary is designed for CLI, notebook, API, and pipeline consumers that need a compact status view without parsing every artifact.

## Path Portability

Artifacts must not persist local absolute paths, file URIs, or machine-specific workspace roots. The writer uses the shared portable path helper from the artifact safety layer so references inside findings, upstream metadata, manifests, and writer config remain reproducible across machines.

`manifest.json` records portable artifact references, source artifact references, source run IDs, schema version, writer metadata, and the canonical generated artifact inventory.

## Walk-Forward Efficiency

Issue 385 adds the first concrete diagnostic: Walk-Forward Efficiency. WFE measures whether in-sample performance transfers to out-of-sample validation periods:

```text
WFE = Sharpe_OOS / Sharpe_IS
```

The default deterministic status bands are:

- `robust`: WFE is at least `0.75`
- `acceptable`: WFE is at least `0.50` and below `0.75`
- `weak`: WFE is at least `0.00` and below `0.50`
- `broken`: WFE is below `0.00`, or the in-sample Sharpe is negative
- `undefined`: WFE cannot be computed because an input is non-finite or the in-sample Sharpe is zero or near zero
- `missing`: required in-sample or out-of-sample Sharpe evidence is absent

Thresholds are represented by `WalkForwardEfficiencyThresholds` so later configuration work can tune the bands without changing artifact consumers. WFE rows are written to `walk_forward_efficiency.csv` using the Issue 384 column contract. Split dates, trade counts, threshold values, source run IDs, and edge-case reasons are included in the row `details` field and in matching robustness findings.

WFE findings use check IDs such as `walk_forward_efficiency.robust`, `walk_forward_efficiency.weak`, `walk_forward_efficiency.broken`, `walk_forward_efficiency.undefined`, and `walk_forward_efficiency.missing`. These findings are review evidence only. They do not automatically change promotion governance decisions in Issue 385; governance integration is deferred to later M34 work.

## Extension Points

Later M34 issues can populate the existing artifacts with real diagnostics:

- Additional walk-forward efficiency extraction sources in `walk_forward_efficiency.csv`
- Sample-size and trade-count checks in `sample_size_validation.json`
- Sensitivity or fragility rows in `sensitivity_summary.csv`
- Multiple-testing families and trial-count metadata in `multiple_testing_summary.json`
- Purged or embargoed validation findings through the shared finding schema
- Governance integration through upstream governance artifact references

The contract supports optional strategy, alpha, portfolio, campaign, governance, and generic upstream artifact references. Missing references are valid and degrade to empty manifest sections.

## Non-Goals

This contract does not implement sample-size validation logic, sensitivity reruns, multiple-testing haircuts, DSR, PBO, purged validation, dashboards, external services, or promotion governance decision changes. Those belong to later M34 issues and should consume this bundle rather than redefining it.
