# M34 Robustness Validation Examples

Milestone 34 adds deterministic robustness evidence as an artifact-first layer
over existing research outputs. The CI-safe example in this directory shows how
notebook, API, CLI, and pipeline workflows can assemble a robustness report
without external data, network access, or a separate execution path.

## Example Script

Run:

```bash
python docs/examples/robustness_report_example.py
```

The script writes:

```text
docs/examples/output/robustness_report_example/
```

It is deterministic and safe to rerun. The script clears and rewrites only that
example output directory.

## What It Demonstrates

The example uses synthetic fixture-style inputs to exercise the major M34
layers:

* Walk-Forward Efficiency rows and findings.
* Sample-size and trade-count sufficiency checks.
* Parameter sensitivity and fragility checks.
* Multiple-testing and trial-count metadata.
* Purged and embargoed temporal-validation split artifacts.
* Read-only governance context derived from the robustness report.

The synthetic values are intentionally small and inspectable. They demonstrate
artifact shape and integration behavior, not empirical strategy quality.

## Canonical Robustness Artifacts

The robustness report bundle contains:

* `robustness_summary.json`
* `robustness_findings.json`
* `walk_forward_efficiency.csv`
* `sample_size_validation.json`
* `sensitivity_summary.csv`
* `multiple_testing_summary.json`
* `robustness_report.md`
* `manifest.json`

These are the canonical Issue 384 bundle artifacts. JSON files use sorted keys
and avoid non-finite values. CSV files use stable column contracts and LF line
endings. Paths inside the bundle are repo-relative or artifact-relative.

## Temporal-Validation Companion Artifacts

The example also writes purged-split companion artifacts into the same output
directory:

* `purged_split_plan.json`
* `purged_split_summary.csv`
* `leakage_validation.json`

These artifacts describe ordered validation folds, train and validation
observation IDs, purge and embargo settings, leakage-validation checks, and
pass/review/block statuses. They are companion artifacts because the canonical
robustness bundle remains stable.

## Governance Context

After writing the report, the example loads governance-visible robustness
context from `robustness_summary.json` and companion artifacts. The printed
summary includes:

* `governance_available`
* `robustness_status`
* `highest_robustness_severity`
* `temporal_validation_status`
* sorted robustness reason codes
* the portable output directory

This mirrors the Issue 390 boundary: robustness evidence is visible to
governance reporting as review context, but it does not recompute or silently
change recorded promotion decisions.

## Synthetic Scope

The example does not run a strategy, train a model, fetch market data, or read
machine-specific files. Inputs are deterministic mappings defined in
`docs/examples/robustness_report_example.py`.

The example intentionally does not implement:

* Deflated Sharpe Ratio
* Probability of Backtest Overfitting
* Combinatorial Purged Cross-Validation
* statistical haircuts
* LLM black-swan or synthetic macro simulation

Those methods remain future extension points unless separately implemented,
documented, and tested.

## Validation

Focused example validation:

```bash
python docs/examples/robustness_report_example.py
python -m pytest tests/test_robustness_example.py
python -m pytest tests/test_docs_path_portability.py
```

Broader M34 validation is listed in
`docs/m34_release_validation_checklist.md`.
