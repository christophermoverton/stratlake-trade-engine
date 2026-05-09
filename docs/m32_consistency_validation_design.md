# Milestone 32 Consistency Validation Design

## Purpose

M32 validation checks whether governance observability rows are consistent with
the canonical promotion artifacts they cite. It returns structured JSON-safe
findings instead of crashing whenever possible.

Validation is read-only. It does not modify promotion policy, replay gates, or
recompute promotion outcomes.

## Validation Output

`consistency_validation.json` contains:

* `status`: `pass` or `fail`
* `record_count`
* `finding_count`
* `counts_by_severity`
* `counts_by_check`
* `findings`

Each finding includes:

* `check_id`
* `severity`
* `run_id`
* `workflow_type`
* `message`
* `details`

Findings are sorted deterministically by severity, check ID, and run ID.

## Severity Semantics

* `info`: benign normalization or context. Info-only findings do not fail
  validation.
* `warning`: missing or stale optional operational evidence, incomplete context,
  or governance audit gaps that should be reviewed.
* `error`: conflicting canonical fields, unknown statuses, or conditions that
  make the cited governance outcome inconsistent.

Overall validation status is `fail` when any finding severity is not `info`.

## Promotion And Review Status Normalization

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

When a raw status differs from the normalized value, validation emits
`legacy_status_normalized` with `info` severity and details containing the field
name, raw status, and normalized status.

Unknown promotion or review statuses remain visible and produce validation
findings. Candidate as a promotion status is not silently mapped to `eligible`.

## Core Checks

Representative core findings:

* `missing_promotion_summary`: no canonical `promotion_gate_summary` was found.
* `unknown_promotion_status`: row promotion status is not canonical.
* `unknown_review_status`: review status is not canonical.
* `registry_promotion_status_mismatch`: registry status differs from
  `promotion_gate_summary.promotion_status` after normalization.
* `manifest_registry_promotion_summary_mismatch`: manifest
  `promotion_gate_summary` differs from registry `promotion_gate_summary`.
* `manifest_promotion_gates_summary_mismatch`: manifest
  `promotion_gate_summary` differs from the `promotion_gates.json` summary.
* `review_status_mismatch`: review status differs from the M31
  promotion-to-review mapping after normalization.
* `missing_or_stale_manifest_link`: a manifest path exists in metadata but does
  not resolve.
* `manifest_run_id_mismatch`: manifest identity does not match the governance
  source record.
* `non_relative_artifact_path`: generated governance path fields are not
  relative.

Equivalent promotion summary checks compare normalized `promotion_status`
values only. They intentionally do not require full payload equality, because
older or partial summaries may omit reason codes, severity counts, or gate
metadata while still preserving the canonical decision status.

## Candidate-Review Checks

Candidate-review validation covers promotion context and evidence paths:

* `candidate_review_context_mismatch`: candidate-review promotion context does
  not match the normalized row promotion status.
* `candidate_review_context_unknown_promotion_status`: portfolio promotion
  context contains an unknown promotion status.
* `candidate_review_manifest_run_id_mismatch`: candidate-review manifest and
  summary disagree on candidate-selection identity.
* `candidate_review_missing_promotion_context`: candidate-review summary has no
  `promotion_context`.
* `candidate_review_missing_selected_candidate_id`: selected run metadata exists
  but no selected candidate ID is present.
* `candidate_review_missing_upstream_run_reference`: selected run ID is absent
  from known governance records.
* `candidate_review_duplicate_upstream_run_ids`: upstream run IDs contain
  duplicates.
* `candidate_review_selected_candidate_id_mismatch`: summary and manifest
  disagree on selected candidate IDs.
* `candidate_review_selected_run_id_mismatch`: summary and manifest disagree on
  selected run IDs.
* `candidate_review_stale_artifact_evidence_path`: required candidate-review
  evidence path is missing.

Candidate-review evidence semantics:

* `artifact_evidence_paths` is required evidence.
* `optional_artifact_evidence_paths` is best-effort evidence.
* Missing optional evidence does not emit stale-path warnings.
* Validation details sanitize absolute paths to avoid leaking local machine
  paths.

## Campaign And Scenario Checks

Campaign/scenario validation detects propagation and artifact issues where data
is available:

* `campaign_highest_severity_mismatch`: campaign rollup highest severity does
  not match child scenario maximum severity.
* `campaign_id_mismatch`: campaign manifest identity differs from the
  governance campaign ID.
* `campaign_missing_scenario_reason_codes`: campaign rollup omits reason codes
  observed in child scenarios.
* `scenario_catalog_missing_scenario_dir`: scenario catalog references a
  missing scenario directory.
* `checkpoint_completed_scenario_missing_summary`: completed or reused scenario
  is missing `summary.json`.
* `checkpoint_completed_scenario_missing_manifest`: completed or reused scenario
  is missing `manifest.json`.
* `scenario_promotion_status_mismatch`: scenario summary promotion status
  differs from scenario manifest `promotion_gate_summary`.
* `scenario_id_mismatch`: scenario manifest identity differs from the
  governance scenario ID.
* `scenario_summary_missing_child_artifacts`: scenario summary references child
  artifact paths that are missing.

These checks compare propagated summaries. They do not rerun campaign stages or
promotion gates.

## Path Safety

Governance output path fields should be relative or sanitized. Validation flags
non-relative output path fields and sanitizes path details in findings. The
report bundle should not leak local temporary directories or Windows drive
prefixes.

## Strict Validation

Strict validation is a writer/CLI behavior:

1. The full governance report bundle is written.
2. `consistency_validation.json` is written as the canonical validation
   evidence artifact.
3. If validation status is not `pass`, strict mode raises and the CLI exits
   non-zero.

This order is intentional. It leaves debugging evidence on disk even when a
strict validation run fails.
