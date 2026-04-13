# Milestone Reporting Artifacts

## Overview

Milestone reporting now follows the same artifact-first conventions already used
by campaign, candidate-selection, portfolio, and unified-review outputs:

* one canonical `summary.json`
* one explicit `manifest.json`
* one optional-detail companion artifact, here `decision_log.json`
* canonical JSON serialization with sorted keys and LF newlines
* deterministic IDs and relative artifact references

The implementation lives in `src/research/reporting/milestone_artifacts.py`.

## Artifact Layout

One milestone report pack writes:

```text
<output_dir>/
    summary.json
    decision_log.json
    manifest.json
```

Default file names are fixed by contract:

* `summary.json` is the milestone report payload
* `decision_log.json` is the detailed decision record
* `manifest.json` inventories the pack and mirrors key counts

This mirrors the repo-wide pattern where `summary.json` is the main
machine-readable entrypoint and `manifest.json` is the file-level inventory.

## Naming Conventions

Milestone report identifiers should be deterministic and content-derived.

`build_milestone_report_id()` hashes the canonical JSON form of:

* `milestone_name`
* `title`
* `reporting_window`
* `scope`

The resulting identifier shape is:

```text
milestone_report_<12-char-sha256-prefix>
```

Artifact references stored inside report payloads should be:

* relative to the milestone artifact root
* expressed with forward slashes
* stable across machines and reruns

Absolute evidence paths are rejected by validation so committed examples do not
leak local workspace roots.

## `summary.json` Schema

`summary.json` uses:

* `run_type: "milestone_report"`
* `schema_version: 1`

Required fields:

* `milestone_id`
* `milestone_name`
* `title`
* `status`
* `summary`
* `decision_log_path`
* `decision_ids`
* `decision_counts_by_status`
* `decision_count`

Optional structured fields:

* `owner`
* `reporting_window`
* `scope`
* `key_findings`
* `recommendations`
* `open_questions`
* `related_artifacts`
* `metadata`

`decision_log_path` is always `decision_log.json` so summary-first consumers can
jump to the detailed audit trail without scanning the directory.

## `decision_log.json` Schema

`decision_log.json` uses:

* `run_type: "milestone_decision_log"`
* `schema_version: 1`

Top-level fields:

* `milestone_id`
* `milestone_name`
* `title`
* `decision_count`
* `decision_ids`
* `decisions`

Each decision entry includes:

* `decision_id`
* `title`
* `status`
* `summary`
* `rationale`

Optional decision fields:

* `impact`
* `owner`
* `category`
* `timestamp`
* `evidence_artifacts`
* `related_stage_names`
* `tags`
* `metadata`

Valid decision statuses:

* `accepted`
* `deferred`
* `rejected`
* `superseded`

## `manifest.json` Schema

`manifest.json` uses the same inventory shape already established by campaign
and review manifests:

* `artifact_files`
* `artifact_groups`
* `artifacts`
* `summary_path`
* `decision_log_path`

It also records:

* `milestone_id`
* `milestone_name`
* `status`
* `decision_ids`
* `decision_counts_by_status`
* deterministic `timestamp`

The current artifact groups are:

* `core`
* `report`
* `summary`
* `decision_log`

## Serialization Rules

Milestone artifacts use the existing canonical JSON utilities from
`src.research.registry`:

* `canonicalize_value()`
* `serialize_canonical_json()`

Serialization rules:

* mapping keys are sorted recursively
* paths normalize to POSIX form
* JSON is written with `indent=2`
* files are written with LF newlines
* schema payloads avoid NaN and infinite floats

These choices keep reruns byte-stable and align milestone packs with the rest
of the repo's deterministic artifact design.

## Validation Rules

`validate_milestone_report()` enforces:

* non-empty report identifiers and text fields
* report status is `draft` or `final`
* decision ids are unique
* decision status is from the supported set
* evidence artifact paths are relative and use forward slashes

Payload-level validators also confirm:

* `summary.json` has the expected `run_type`, `schema_version`, and
  `decision_log_path`
* `decision_log.json` has the expected `run_type`, `schema_version`, and
  `decision_count == len(decisions)`

## Usage

Typical usage:

```python
from src.research.reporting import (
    MilestoneDecisionEntry,
    MilestoneReport,
    build_milestone_report_id,
    write_milestone_report_artifacts,
)

report = MilestoneReport(
    milestone_id=build_milestone_report_id(
        milestone_name="Milestone 18",
        title="Milestone 18 Readiness Report",
    ),
    milestone_name="Milestone 18",
    title="Milestone 18 Readiness Report",
    status="final",
    summary="Milestone ready for review.",
)

decisions = [
    MilestoneDecisionEntry(
        decision_id="reuse_summary_contract",
        title="Reuse summary.json naming",
        status="accepted",
        summary="Keep summary-first loading semantics.",
        rationale="Matches existing campaign and review artifact packs.",
    )
]

write_milestone_report_artifacts(
    report=report,
    decisions=decisions,
    output_path="artifacts/milestone_reports/demo_report",
)
```

## Related Docs

* [milestone_16_campaign_workflow.md](milestone_16_campaign_workflow.md)
* [milestone_17_resume_workflow.md](milestone_17_resume_workflow.md)
* [strategy_comparison_cli.md](strategy_comparison_cli.md)
* [experiment_artifact_logging.md](experiment_artifact_logging.md)
