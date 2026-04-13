# Milestone Reporting Artifacts

## Overview

Milestone reporting now follows the same artifact-first conventions already used
by campaign, candidate-selection, portfolio, and unified-review outputs:

* one canonical `summary.json`
* one explicit `manifest.json`
* one optional-detail companion artifact, here `decision_log.json`
* one deterministic human-readable `report.md`
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
    report.md
```

Default file names are fixed by contract:

* `summary.json` is the milestone report payload
* `decision_log.json` is the detailed decision record
* `manifest.json` inventories the pack and mirrors key counts
* `report.md` is the human-readable milestone brief

Campaign-driven generation can now tune the pack without changing the base file
names:

* `summary.json`, `decision_log.json`, and `manifest.json` always persist when
  milestone generation is enabled
* `report.md` is optional and can be disabled through campaign config
* `decision_log.json.rendered` can include `markdown`, `text`, or both

This mirrors the repo-wide pattern where `summary.json` is the main
machine-readable entrypoint, `report.md` is the operator-facing summary, and
`manifest.json` is the file-level inventory.

When a campaign generates the pack automatically, the canonical location is:

```text
artifacts/research_campaigns/<campaign_run_id>/milestone_report/
```

Campaign stitched outputs then point at those files through
`output_paths.milestone_report_*` fields so operators can jump from the
campaign packet into the milestone packet without guessing paths.

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
* `schema_version: 2`

Required fields:

* `milestone_id`
* `milestone_name`
* `title`
* `status`
* `summary`
* `decision_log_path`
* `report_markdown_path`
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

`decision_log_path` is always `decision_log.json`.
`report_markdown_path` is `report.md` when markdown output is enabled and
`null` when the pack is configured to omit the standalone Markdown brief.

## `decision_log.json` Schema

`decision_log.json` uses:

* `run_type: "milestone_decision_log"`
* `schema_version: 2`

Top-level fields:

* `milestone_id`
* `milestone_name`
* `title`
* `decision_count`
* `decision_ids`
* `decisions`
* `rendered`

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
* `report_markdown_path`

It also records:

* `milestone_id`
* `milestone_name`
* `status`
* `decision_ids`
* `decision_counts_by_status`
* deterministic `timestamp`

The current artifact groups are:

* `core`
* `markdown`
* `report`
* `summary`
* `decision_log`

When enabled, `report.md` is generated deterministically from the milestone
summary metadata and decision log so it presents, in a stable order:

* campaign scope
* selections
* key findings and key metrics
* gate outcomes
* risks
* next steps
* open questions
* a decision snapshot
* related artifacts

Campaign config can independently disable any of those optional markdown
sections while leaving the canonical JSON artifacts intact.

## Campaign Integration

Milestone reporting can be produced in two ways:

* automatically at the end of `python -m src.cli.run_research_campaign`
* later from an existing campaign artifact directory through
  `python -m src.cli.generate_milestone_report`

Automatic generation uses the completed campaign `summary.json`,
`manifest.json`, and any attached review artifacts as the source of truth. The
milestone pack then writes back relative links to those upstream artifacts so
the review packet stays portable and auditable.

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
* `summary.json` references `report.md` only when Markdown output is enabled
* `decision_log.json` has the expected `run_type`, `schema_version`,
  `decision_count == len(decisions)`, and at least one supported rendered
  format

## Output Interpretation

Use the files this way:

* `summary.json`
  Read first for milestone status, high-level findings, decision counts, and
  linked artifact paths.
* `decision_log.json`
  Read second for auditable per-decision rationale, follow-up actions, and
  linked source artifacts.
* `report.md`
  Use for operator-facing review meetings, handoff notes, and quick narrative
  scans.
* `manifest.json`
  Use for deterministic inventory checks and automation entrypoints.

In practice, `summary.json` tells you what the review concluded,
`decision_log.json` tells you why, and `manifest.json` tells automation what
was actually persisted.

## Practical Review Workflow

One practical milestone review pass is:

1. open the campaign `summary.json`
2. jump to `output_paths.milestone_report_summary`
3. inspect `decision_counts_by_status`, `key_findings`, and
   `recommendations`
4. open `decision_log.json` for the accepted, deferred, or rejected items that
   need discussion
5. follow `related_artifacts` or per-decision `source_artifacts` back into the
   campaign, candidate-review, or research-review outputs when you need deeper
   evidence
6. use `report.md` as the meeting-ready brief after the machine-readable pack
   looks correct

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

* [milestone_18_milestone_review_workflow.md](milestone_18_milestone_review_workflow.md)
* [milestone_16_campaign_workflow.md](milestone_16_campaign_workflow.md)
* [milestone_17_resume_workflow.md](milestone_17_resume_workflow.md)
* [strategy_comparison_cli.md](strategy_comparison_cli.md)
* [experiment_artifact_logging.md](experiment_artifact_logging.md)
