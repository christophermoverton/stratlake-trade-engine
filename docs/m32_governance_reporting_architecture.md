# Milestone 32 Governance Reporting Architecture

## Overview

Milestone 32 adds a deterministic promotion governance observability layer under
`src/research/governance/`. It reads existing promotion artifacts, normalizes
them into comparable rows, validates consistency, and writes an audit bundle
under `artifacts/promotion_governance/<report_id>/`.

M32 is read-only with respect to promotion policy. It observes and audits
existing promotion outcomes. It does not make, replay, or recompute promotion
decisions.

## What M32 Does

* Loads registry, manifest, review, candidate-review, campaign, and scenario
  artifacts where present.
* Reads existing `promotion_gate_summary` payloads.
* Aggregates promotion status, severity, reason-code, workflow, campaign, and
  candidate summaries.
* Writes deterministic JSON, CSV, Markdown, and manifest outputs.
* Emits structured validation findings for consistency and path-safety issues.

## What M32 Does Not Do

M32 does not:

* evaluate promotion gates
* recompute `promotion_status`
* replay promotion policy
* perform sensitivity analysis
* add new statistical diagnostics
* create dashboard, database, or UI services
* write `promotion_decision.json`
* write `promotion_readiness.json`

## Canonical Promotion Artifacts

Promotion truth remains in the existing M31 path:

* `promotion_gates.json` is the canonical promotion policy artifact.
* `promotion_gate_summary` is the canonical summary embedded in manifests,
  registries, reviews, campaigns, and candidate-review context.

Governance reports read those artifacts and produce observability outputs. If a
canonical promotion summary is missing, governance can report that condition,
but it does not infer a replacement decision.

For run records where equivalent summaries are present in multiple places, M32
uses a deterministic read precedence:

1. manifest `promotion_gate_summary`
2. registry `promotion_gate_summary`
3. `promotion_gates.json` summary

Registry top-level `promotion_status` is treated as supporting evidence and is
validated against the selected canonical summary. Conflicts among equivalent
summary fields are reported in `consistency_validation.json`; governance does
not resolve them by replaying gates.

## Governance Package Layout

* `src/research/governance/models.py` defines immutable result and source
  dataclasses.
* `src/research/governance/normalization.py` owns canonical promotion and review
  status vocabularies plus legacy alias normalization.
* `src/research/governance/loader.py` discovers and loads governance-relevant
  artifacts.
* `src/research/governance/aggregator.py` creates row-level outcome records and
  deterministic summaries.
* `src/research/governance/validator.py` returns structured consistency
  findings.
* `src/research/governance/writer.py` writes the report bundle.
* `src/cli/run_promotion_governance_report.py` exposes the CLI entrypoint.

## Input Artifacts

The loader degrades gracefully when optional files are absent. It can read:

* registry JSONL rows
* run `manifest.json`
* `promotion_gates.json`
* embedded `promotion_gate_summary`
* review `review_summary.json`
* candidate-review `candidate_review_summary.json`
* campaign `summary.json`
* campaign `manifest.json`
* `scenario_catalog.json`
* `checkpoint.json`
* scenario `summary.json`
* scenario `manifest.json`

Campaign discovery checks likely roots such as `artifacts/research_campaigns/`
and campaign directories under the provided artifact root.

## Report Generation Flow

1. `load_governance_artifacts()` reads available sources into
   `GovernanceSourceRecord` rows.
2. `build_governance_outcome_rows()` converts records into deterministic
   row-level dictionaries.
3. Aggregators compute promotion, severity, reason-code, workflow, campaign,
   and candidate summaries.
4. `validate_governance_consistency()` returns JSON-safe findings.
5. `run_promotion_governance_report()` writes the bundle.

## Output Artifact Bundle

Every governance report writes:

* `promotion_governance_summary.json` - report-level counts, fractions, and
  top reason-code summaries.
* `promotion_outcome_matrix.csv` - one comparable row per governance source
  record.
* `reason_code_summary.csv` - reason-code counts sorted deterministically.
* `severity_summary.csv` - highest-severity and triggered-reason counts.
* `workflow_summary.csv` - row and status counts by workflow type.
* `consistency_validation.json` - canonical validation evidence.
* `promotion_governance_report.md` - short human-readable summary.
* `manifest.json` - report manifest with artifact inventory.

No candidate-specific or campaign-specific required artifact is added by M32.

## Outcome Matrix

`promotion_outcome_matrix.csv` is the main row-level audit table. Key columns:

* `run_id`
* `workflow_type`
* `promotion_status`
* `highest_severity`
* `review_status`
* `decision_reason_codes`
* `triggered_gate_names`
* `registry_path`
* `manifest_path`
* `campaign_id`
* `scenario_id`
* `campaign_status`
* `scenario_status`
* `candidate_id`
* `candidate_selection_run_id`
* `selected_candidate_id`
* `selected_run_id`
* `upstream_run_ids`
* `strategy_name`
* `portfolio_name`
* `alpha_model_name`
* `effective_n`
* `p_value`
* `hit_rate_p_value`
* `sharpe_stability_ratio`

List-like values such as reason codes and upstream run IDs are pipe-delimited
and sorted deterministically.

## Governance Summary

`promotion_governance_summary.json` includes:

* row counts
* promotion status counts
* highest severity counts
* reason-code counts
* workflow type counts
* campaign and scenario counts
* campaign and scenario status counts
* candidate-review, candidate-selection, and selected-candidate counts
* candidate status counts
* warning, review, reject, and block totals
* eligible, blocked, and review fractions

The summary is canonicalized and written with sorted JSON keys.

## Consistency Validation

Validation is structured and non-throwing by default. Findings include:

* `check_id`
* `severity`
* `run_id`
* `workflow_type`
* `message`
* `details`

`consistency_validation.json` is the canonical validation evidence artifact.
See `docs/m32_consistency_validation_design.md` for the finding categories.

## Campaign And Scenario Observability

Campaign records use `workflow_type=campaign`. Scenario records use
`workflow_type=campaign_scenario`.

Campaign/scenario metadata may include:

* `campaign_id`
* `scenario_id`
* `scenario_name`
* `scenario_status`
* `campaign_status`
* `checkpoint_status`
* `selected_run_id`
* `selected_candidate_id`
* `child_run_ids`
* scenario and campaign manifest/summary paths
* scenario catalog and checkpoint paths

M32 reads campaign-propagated promotion summaries. It does not recompute a
campaign or scenario outcome from raw metrics.

## Candidate-Selection Visibility

Candidate-review records use `workflow_type=candidate_review`. They preserve:

* `candidate_selection_run_id`
* `candidate_id`
* `selected_candidate_id`
* `selected_candidate_ids`
* `selected_run_id`
* `selected_run_ids`
* `portfolio_run_id`
* `strategy_run_id`
* `alpha_run_id`
* `upstream_run_ids`
* `candidate_promotion_status_counts`
* `promotion_context_present`
* candidate-review summary and manifest paths

These fields make candidate-selection context visible in the outcome matrix and
summary outputs without changing candidate-selection execution behavior.

## Required Vs Optional Evidence Paths

Candidate-review metadata distinguishes required and optional evidence:

* `artifact_evidence_paths` means required evidence. Missing required evidence
  may emit `candidate_review_stale_artifact_evidence_path`.
* `optional_artifact_evidence_paths` means best-effort evidence. Missing
  optional evidence does not emit stale-path warnings.

Validation details sanitize absolute paths so local machine paths are not
written into governance artifacts.

## Status Normalization

Canonical promotion statuses:

* `eligible`
* `warn`
* `needs_review`
* `rejected`
* `blocked`

Canonical review statuses:

* `candidate`
* `needs_review`
* `rejected`

Promotion-to-review mapping:

* `eligible -> candidate`
* `warn -> needs_review`
* `needs_review -> needs_review`
* `rejected -> rejected`
* `blocked -> rejected`

Supported aliases:

* `review -> needs_review`
* `needs work -> needs_review`
* `needs_work -> needs_review`
* `needs-work -> needs_review`

Alias normalization emits `legacy_status_normalized` with `info` severity.
Unknown statuses remain visible and produce validation findings.

## CLI Usage

Build a report from the default artifact root:

```bash
python -m src.cli.run_promotion_governance_report --artifact-root artifacts
```

Write under a custom output root:

```bash
python -m src.cli.run_promotion_governance_report \
  --artifact-root artifacts \
  --output-dir artifacts/promotion_governance/m32_example
```

Use strict validation:

```bash
python -m src.cli.run_promotion_governance_report \
  --artifact-root artifacts \
  --strict-validation
```

Strict validation still writes the complete governance artifact bundle first.
The CLI fails only after `consistency_validation.json` and the other artifacts
are available for debugging.

When `--output-dir` is omitted, reports write to the canonical M32 location
`artifacts/promotion_governance/<report_id>/` even if `--artifact-root` points
elsewhere.

## Python Usage

```python
from src.research.governance import run_promotion_governance_report

result = run_promotion_governance_report(
    artifact_root="artifacts",
    output_dir="artifacts/promotion_governance/m32_example",
)

print(result.report_id)
print(result.outcome_matrix_path)
print(result.validation["status"])
```

Lower-level helpers are available for notebooks and orchestrators:

```python
from src.research.governance import (
    build_governance_outcome_rows,
    load_governance_artifacts,
    validate_governance_consistency,
)

dataset = load_governance_artifacts(artifact_root="artifacts")
rows = build_governance_outcome_rows(dataset.records)
validation = validate_governance_consistency(dataset.records, rows)
```

## Notebook And Orchestrator Usage

Notebook and orchestrator workflows should call the Python API directly when
they need structured paths and validation payloads. CLI runs are suitable for
local release checks and automated artifact generation. Both paths use the same
writer and produce the same bundle.

## Determinism And Path Safety

M32 sorts records, reason codes, list-valued row fields, JSON keys, and
validation findings deterministically. Generated artifacts use relative or
sanitized paths. Absolute local paths and Windows drive prefixes are not
intended to appear in governance outputs.

## Testing And Validation

Governance coverage lives in:

* `tests/test_promotion_governance.py`
* `tests/test_promotion_governance_integration.py`

Recommended focused validation:

```bash
python -m pytest tests/test_promotion_governance.py tests/test_promotion_governance_integration.py
python -m pytest tests/test_docs_path_portability.py
```
