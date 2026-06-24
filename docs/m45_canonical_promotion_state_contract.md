# M45 Canonical Promotion-State Contract

Issue #493 defines the engine-owned promotion-state contract for standalone
research reviews, research campaign containers, and existing strategy,
portfolio, and alpha artifacts that already carry promotion evidence. This is a
design and compatibility document only. It does not implement a writer, weaken
governance checks, create historical artifacts, or make notebooks responsible
for canonical engine evidence.

## Purpose

Promotion governance needs to distinguish four states that are currently too
easy to blur:

1. Missing or malformed promotion evidence.
2. Explicit engine-emitted non-decision because promotion policy was not
   configured.
3. Configured machine-evaluated promotion result.
4. Future human governance decision, if and only if a separately authorized
   workflow exists.

The important new state is the second one. A completed review or campaign with
no configured promotion policy must have a deterministic canonical artifact
that says no policy was configured and no promotion decision was made. Missing
evidence must remain meaningful and must not be normalized into that state.

## Current Behavior To Preserve

The current engine uses `promotion_gates.json` as the canonical configured
promotion artifact. `src/research/promotion.py` owns gate evaluation and writes
the artifact only when a promotion gate config is present. The configured
payload currently includes fields such as `configured`, `run_type`,
`evaluation_status`, `promotion_status`, gate counts, severity counts,
`decision_reason_codes`, `definitions`, and `results`.

`src/research/review.py` now writes canonical review-owned promotion state for
every successfully completed standalone review. Configured evaluation remains
compatible, while absent review policy produces explicit `not_reviewed` state.

`src/cli/run_research_campaign.py` now writes campaign-owned canonical state
during successful campaign finalization. It may also observe nested review
evidence as upstream context, but the campaign state preserves its own identity
and does not inherit the review result. A dedicated campaign promotion policy
surface still does not exist.

`src/research/governance/loader.py`,
`src/research/governance/aggregator.py`, and
`src/research/governance/validator.py` are observational. They load registry,
manifest, review, candidate-review, campaign, scenario, and promotion artifacts;
normalize statuses for reporting; classify canonical state; and emit distinct
missing, malformed, ownership, and consistency findings. Required missing M45
state does not borrow manifest summaries. Governance does not replay gates,
create promotion decisions, or mutate source evidence.

For the operational guide, see
[m45_canonical_promotion_state.md](m45_canonical_promotion_state.md).

The M31 and M32 architecture documents remain compatible with this contract:

* M31 established the centralized `promotion_gates.json` path and expanded
  configured machine outcomes to `eligible`, `warn`, `needs_review`,
  `rejected`, and `blocked`.
* M32 established governance as read-only observability over existing engine
  artifacts and explicitly rejected separate `promotion_decision.json` or
  `promotion_readiness.json` artifacts for that milestone.

M45 keeps configured gate behavior compatible and adds an explicit no-policy
case to the same engine-owned promotion-state surface.

## Canonical Artifact

The canonical artifact name remains:

```text
promotion_gates.json
```

For schema version 2 and later, the artifact represents a canonical
promotion-state document. The filename is retained for compatibility with M31
and M32 readers and to avoid creating a competing promotion source of truth.
The required top-level `artifact_type` distinguishes the v2 contract:

```json
{
  "schema_version": 2,
  "artifact_type": "promotion_state"
}
```

Older configured artifacts without `schema_version` are legacy configured gate
artifacts and are handled by the compatibility rules below.

## Location Conventions

| Object | Canonical location | Owner | Notes |
| --- | --- | --- | --- |
| Standalone research review | `<review_artifact_dir>/promotion_gates.json` beside `review_summary.json`, `leaderboard.csv`, and `manifest.json` | `src/research/review.py`, via the M45 promotion-state factory in later issues | Must be emitted for both configured review promotion gates and explicit no-policy review completion. |
| Single research campaign container | `<campaign_artifact_dir>/promotion_gates.json` beside campaign `summary.json`, `manifest.json`, `checkpoint.json`, and `campaign_config.json` | campaign artifact writer in `src/cli/run_research_campaign.py`, using the engine-owned M45 factory | Summarizes the campaign container's own promotion-state observation. It may reference selected review and selected run identities but must not collapse them into the campaign ID. |
| Scenario campaign container | `<orchestration_artifact_dir>/scenarios/<scenario_id>/promotion_gates.json` beside scenario `summary.json` and `manifest.json` | scenario campaign artifact writer, using the engine-owned M45 factory | Same semantics as a single campaign container, scoped to the scenario campaign ID and scenario ID. |
| Campaign orchestration container | `<orchestration_artifact_dir>/promotion_gates.json` beside orchestration `summary.json`, `manifest.json`, `scenario_catalog.json`, and scenario matrix artifacts | orchestration writer, using the engine-owned M45 factory | Rolls up observed scenario/container promotion states only when the later implementation issue defines deterministic rollup rules. It must preserve scenario identities. |
| Strategy run | Existing strategy run artifact directory `promotion_gates.json` beside `metrics.json`, `qa_summary.json`, and `manifest.json` | strategy artifact writer through `src/research/promotion.py` | Existing configured behavior remains compatible. If no promotion policy is configured, later issues may add a v2 no-policy artifact only through the engine writer, not through governance or notebooks. |
| Portfolio run | Existing portfolio run artifact directory `promotion_gates.json` beside portfolio metrics and manifest artifacts | portfolio artifact writer through `src/research/promotion.py` | Same compatibility and no-policy expectations as strategy. |
| Alpha evaluation run | Existing alpha evaluation artifact directory `promotion_gates.json` beside `alpha_metrics.json`, QA, config, and manifest artifacts | alpha artifact writer through `src/research/promotion.py` | Same compatibility and no-policy expectations as strategy. |

M45 requires new explicit no-policy promotion-state emission for completed
standalone research reviews and research campaign containers. Strategy,
portfolio, and alpha producers remain compatibility surfaces for this milestone.
Extending explicit no-policy emission to those producers requires a separately
scoped follow-on issue unless it is explicitly added to the M45 implementation
plan.

Embedded summaries remain allowed for fast discovery:

* `manifest.json` may include `promotion_gate_summary`.
* registry rows may include `promotion_status`, `review_status`,
  `review_metadata`, and `promotion_gate_summary`.
* campaign `summary.json`, campaign `manifest.json`, and scenario matrix rows
  may include propagated review promotion fields.

Those embedded summaries are copies or summaries of canonical engine evidence.
They must not be treated as independent artifact producers.

## Schema Version Strategy

Version 2 is the first explicit promotion-state contract.

* `schema_version` is required and must be integer `2` for newly emitted M45
  artifacts.
* Readers must reject unsupported future major versions unless an explicit
  migration is documented.
* Legacy artifacts without `schema_version` are treated as schema version 1
  configured promotion-gate artifacts only when they contain enough valid gate
  summary evidence, such as `promotion_status` plus configured gate counts or
  result details.
* Absence of a legacy artifact remains missing evidence. It is not evidence of
  a review, a non-decision, or no configured policy.
* A malformed or contradictory artifact remains invalid evidence. It must not be
  normalized into `not_reviewed`.

M45 implementation issues may add adapters that summarize legacy v1 configured
artifacts into v2-shaped reader models, but those adapters must label the source
as legacy and must not write replacement files unless an explicit migration
issue authorizes that behavior.

## Required Top-Level Fields

| Field | Type | Required | Semantics |
| --- | --- | --- | --- |
| `schema_version` | integer | yes | Contract version. New artifacts use `2`. |
| `artifact_type` | string | yes | Must be `promotion_state`. Distinguishes v2 canonical state from legacy gate-only payloads. |
| `run_type` | string | yes | Logical run family: `review`, `research_campaign`, `research_campaign_scenario`, `research_campaign_orchestration`, `strategy`, `portfolio`, `alpha_evaluation`, or a documented future engine-owned value. |
| `configured` | boolean | yes | Whether a promotion policy was configured for this object. `false` means no machine promotion evaluation was attempted. |
| `configuration_state` | string | yes | Configuration availability state. See vocabulary below. |
| `evaluation_status` | string | yes | Machine evaluation lifecycle or result. See vocabulary below. |
| `promotion_status` | string | yes | Conservative promotion outcome vocabulary. See vocabulary below. |
| `decision_authority` | string | yes | Boundary for who or what has decision authority. New engine artifacts use `engine` for machine gate evaluation or `none` for no-policy. |
| `human_decision` | object or null | yes | Reserved for a future authorized human decision workflow. Must be `null` until that workflow exists. |
| `decision_reason_codes` | array of strings | yes | Deterministic reason codes explaining the state. Empty only for a clean configured pass when no reason applies. |
| `gate_counts` | object | yes | Counts for total, passed, failed, missing, skipped, warning, review, rejected, and blocked gates. No-policy uses zero counts. |
| `gate_definitions` | array | yes | Canonical configured gate definitions. No-policy uses an empty array. |
| `gate_results` | array | yes | Canonical gate evaluation results. No-policy uses an empty array. |
| `provenance` | object | yes | Object identity and source artifact references. See provenance section. |
| `artifact_metadata` | object | yes | Deterministic metadata about the artifact writer and reproducibility. |

Configured gate artifacts may retain legacy fields such as `gate_count`,
`passed_gate_count`, `failed_gate_count`, `missing_gate_count`,
`highest_severity`, `severity_counts`, `status_on_pass`, `status_on_fail`,
`definitions`, and `results` for compatibility. New readers should prefer the
v2 names while preserving these legacy names until a migration removes them.

## Status Vocabulary

### `configuration_state`

Permitted values:

| Value | Meaning |
| --- | --- |
| `configured` | A promotion policy was configured and accepted for machine evaluation. Requires `configured: true`. |
| `not_configured` | No promotion policy was configured. Requires `configured: false`. |
| `invalid` | A policy or artifact was present but unreadable, malformed, unsupported, or internally contradictory. |
| `missing` | A canonical artifact was expected by the reader but absent. This is a reader/validator state, not a writer-emitted no-policy artifact state. |

Writers must emit only `configured` or `not_configured`. Readers and validators
may produce `invalid` or `missing` diagnostics.

### `evaluation_status`

Permitted values:

| Value | Meaning |
| --- | --- |
| `pass` | Configured machine evaluation ran and all required gates passed or were skipped by configured policy. |
| `fail` | Configured machine evaluation ran and at least one required gate failed or was missing. |
| `not_configured` | No promotion policy was configured, so no machine evaluation ran. |
| `invalid` | Evaluation evidence is unreadable, malformed, unsupported, or contradictory. Reader/validator state. |
| `missing` | Expected evaluation evidence is absent. Reader/validator state. |

`evaluation_status` remains `pass` or `fail` for configured v1-compatible
machine evaluations.

### `promotion_status`

Permitted values:

| Value | Meaning |
| --- | --- |
| `eligible` | Configured machine gates passed and the configured pass status resolves to eligible. This is a machine result, not human approval. |
| `warn` | Configured machine gates produced a warning-severity non-pass result. |
| `needs_review` | Configured machine gates produced a review-severity non-pass result. |
| `rejected` | Configured machine gates produced a rejection-severity non-pass result. |
| `blocked` | Configured machine gates produced a blocking non-pass result. |
| `not_reviewed` | No promotion policy was configured and no promotion decision was made. |
| `invalid` | Evidence is present but invalid or contradictory. Reader/validator state. |
| `missing` | Expected canonical evidence is absent. Reader/validator state. |

`not_reviewed` is deliberately conservative. It does not mean `eligible`,
`approved`, `promotable`, `production-ready`, `strategy-approved`,
`governance-ready`, or ready for deployment. It must never be counted as an
eligible or approved state.

Readers must not map missing evidence to `not_reviewed`. Only a valid canonical
artifact with `configured: false`, `configuration_state: not_configured`,
`evaluation_status: not_configured`, `promotion_status: not_reviewed`, and a
policy-not-configured reason code may produce the explicit non-decision state.

### `decision_authority` And `human_decision`

Permitted `decision_authority` values:

| Value | Meaning |
| --- | --- |
| `engine` | The engine made a machine evaluation from configured gates. This is not a human approval. |
| `none` | No promotion policy was configured and no decision authority was exercised. |
| `human` | Reserved for a future authorized human governance workflow. Must not appear in new artifacts until that workflow is implemented and documented. |
| `external` | Reserved for future explicitly integrated external authority. Must not appear without a documented contract. |

`human_decision` must be `null` for M45 no-policy and configured machine
artifacts. A future human decision object, if implemented, must include at
minimum actor identity, timestamp, rationale, decision value, decision authority
source, and links to the machine artifact it reviewed. This document does not
claim that workflow exists.

Machine-generated fields include `configured`, `configuration_state`,
`evaluation_status`, `promotion_status`, gate counts, gate definitions, gate
results, machine reason codes, provenance, and artifact metadata. Human
authoritative fields are limited to the reserved future `human_decision`
object. Notebooks, governance reports, and ad hoc repair scripts must not write
human-authoritative fields.

## Explicit No-Policy Shape

A completed standalone review with no configured promotion policy should emit a
canonical artifact shaped like:

```json
{
  "schema_version": 2,
  "artifact_type": "promotion_state",
  "run_type": "review",
  "configured": false,
  "configuration_state": "not_configured",
  "evaluation_status": "not_configured",
  "promotion_status": "not_reviewed",
  "decision_authority": "none",
  "human_decision": null,
  "decision_reason_codes": [
    "promotion_policy_not_configured"
  ],
  "gate_counts": {
    "total": 0,
    "passed": 0,
    "failed": 0,
    "missing": 0,
    "skipped": 0,
    "warning": 0,
    "review": 0,
    "rejected": 0,
    "blocked": 0
  },
  "gate_definitions": [],
  "gate_results": [],
  "provenance": {
    "object_id": "registry_review_123456789abc",
    "object_type": "review",
    "review_id": "registry_review_123456789abc",
    "campaign_id": null,
    "scenario_id": null,
    "selected_catalog_run_id": null,
    "selected_review_id": null,
    "source_artifacts": {
      "review_summary": "review_summary.json",
      "manifest": "manifest.json"
    }
  },
  "artifact_metadata": {
    "artifact_filename": "promotion_gates.json",
    "writer": "engine",
    "generated_by": "src.research.promotion",
    "deterministic": true
  }
}
```

The exact generated IDs and source artifact paths depend on the artifact
directory. Paths should be relative to the artifact directory when embedded in
the artifact.

## Configured Machine-Evaluation Shape

Configured v2 artifacts use the same top-level fields with:

* `configured: true`
* `configuration_state: configured`
* `evaluation_status: pass` or `fail`
* `promotion_status` resolved by the existing M31 severity-aware rules
* `decision_authority: engine`
* `human_decision: null`
* nonzero `gate_counts.total`
* populated `gate_definitions`
* populated `gate_results`

Configured artifacts must preserve existing M31 semantics:

* gate result statuses remain `pass`, `fail`, or `missing`
* missing metrics follow configured `missing_behavior`
* highest severity is derived deterministically from non-passing gates
* `eligible`, `warn`, `needs_review`, `rejected`, and `blocked` remain the
  configured machine outcome vocabulary
* legacy summary fields stay compatible unless a deliberate migration documents
  their removal

## Provenance And Identity Rules

`provenance` must preserve separate object identities. A campaign container ID
is not interchangeable with a selected catalog run ID.

Required provenance fields:

| Field | Meaning |
| --- | --- |
| `object_type` | The artifact owner: review, campaign, campaign_scenario, campaign_orchestration, strategy, portfolio, or alpha_evaluation. |
| `object_id` | The identifier of the object that owns this artifact. For campaigns, this is the campaign or orchestration ID, not a selected child run. |
| `run_type` | Mirrors the top-level `run_type` when useful for embedded readers. |
| `review_id` | Standalone review ID when the artifact belongs to or directly references a review. |
| `campaign_id` | Campaign or orchestration container ID when applicable. |
| `scenario_id` | Scenario ID when applicable. |
| `selected_review_id` | Review selected or referenced by a campaign, if any. |
| `selected_catalog_run_id` | Selected strategy, portfolio, alpha, or other catalog run ID, if any. |
| `selected_run_ids` | Structured map of selected child run IDs when a campaign has multiple selected objects. |
| `source_artifacts` | Relative paths to source artifacts used to construct this state, such as `review_summary.json`, `summary.json`, `manifest.json`, or a child `promotion_gates.json`. |

A campaign-level artifact may reference both a selected review and a selected
run, but it must preserve each object separately. For example, a campaign
`object_id` may be `research_campaign_abc`, `selected_review_id` may be
`registry_review_def`, and `selected_catalog_run_id` may be
`strategy_run_xyz`. Readers must not substitute one ID for another.

## Writer Ownership

Canonical promotion-state artifacts are engine artifacts. Permitted producers
are the engine promotion/review/campaign artifact writers introduced or updated
in Issues #494 through #499.

The following are not canonical producers:

* Notebook 14
* notebook-generated JSON
* notebook backfills
* ad hoc artifact repair scripts
* governance reports
* manual edits to historical artifact directories

Notebook workflows may read canonical engine artifacts and display them. They
must not create or repair canonical `promotion_gates.json` evidence.

Governance remains observational. It reports what engine artifacts state and
validates consistency. It does not create approval and does not manufacture
missing canonical evidence.

## Reader And Validator Expectations

Readers should use this precedence when multiple equivalent summaries are
available:

1. Valid v2 `promotion_gates.json`.
2. Embedded `promotion_gate_summary` that explicitly cites or mirrors a valid
   v2 artifact.
3. Legacy v1 configured summary from manifest, registry, or
   `promotion_gates.json`, under compatibility rules.

Validators must distinguish:

| Condition | Expected reader/validator treatment |
| --- | --- |
| Valid v2 no-policy artifact | `promotion_status: not_reviewed`, reason code `promotion_policy_not_configured`, no missing-evidence finding for that object. |
| Missing expected artifact | `missing_promotion_summary` or successor missing-evidence finding. Do not emit `not_reviewed`. |
| Malformed artifact | Invalid-evidence finding. Do not emit `not_reviewed`. |
| Contradictory artifact and embedded summary | Consistency mismatch finding. Do not resolve by choosing the more favorable status. |
| Legacy configured artifact | Preserve current configured promotion result and emit any legacy normalization info already supported. |
| Legacy absence of artifact | Missing evidence, unless no reader expectation exists for that artifact class before M45. |

Governance normalization should add `not_reviewed` as a canonical promotion
status only for explicit valid no-policy artifacts. Aggregates should count it
separately from eligible, blocked, rejected, warn, and needs-review totals.

## Backward Compatibility

Existing configured artifacts remain valid. In particular:

* Existing `promotion_gates.json` files without `schema_version` are treated as
  legacy configured machine-evaluation artifacts when their configured summary
  fields are valid.
* Existing `promotion_gate_summary` payloads in manifests and registry rows
  retain current precedence and mismatch validation behavior.
* Existing configured promotion-gate behavior must remain compatible unless a
  deliberate schema migration documents changed semantics.
* Existing missing `promotion_gates.json` files are not retroactively treated as
  proof that a review occurred, proof that no policy was configured, or proof
  that a non-decision state exists.
* Historical artifact directories must not be silently backfilled by notebooks
  or governance readers.

The transition path is additive. New writers emit v2 artifacts. Readers support
legacy configured artifacts and explicit v2 no-policy artifacts side by side.

Registry-backed review records are canonical-state-required. A registry entry
with `run_type: review` loads canonical state with `required=True` and review
identity validation. Missing canonical state for registry review records
produces `missing_canonical_promotion_state` without manifest fallback.

Custom configured evaluator compatibility statuses are accepted only from a
bounded vocabulary: `approved`, `manual_review`, `review_ready`, `needs_work`.
A configured compatibility value is valid only when:

1. it is in the bounded supported compatibility vocabulary;
2. it matches the correct pass/fail configuration field;
3. it is consistent with gate-result direction;
4. it does not override severity-driven canonical failure outcomes;
5. the artifact contains valid evaluator-shaped results and valid counts.

Unsupported custom configured status values are rejected at both the
serialization boundary and governance validation. Arbitrary unknown values
outside the bounded vocabulary remain validation errors regardless of whether
`promotion_status` matches `status_on_pass` or `status_on_fail`.

Artifact filename overrides for promotion state writers accept only a plain
basename (e.g. `promotion_gates.json`, `custom_state.json`). Path traversal
(`../`), nested paths (`nested/file.json`), absolute paths, and degenerate
values (`.`, `..`, empty string) are rejected with a domain error.

## Reason Codes

Initial reserved reason codes:

| Reason code | Required context |
| --- | --- |
| `promotion_policy_not_configured` | Required for explicit no-policy artifacts. |
| `gate_passed` | Existing configured gate pass result. |
| `gate_failed_threshold` | Existing configured gate threshold failure. |
| `gate_missing` | Existing configured required metric missing result. |
| `gate_missing_skipped` | Existing configured skipped missing metric result. |
| `severity_warn` | Existing configured warning severity. |
| `severity_review` | Existing configured review severity. |
| `severity_reject` | Existing configured reject severity. |
| `severity_block` | Existing configured block severity. |
| `promotion_evidence_missing` | Reader/validator diagnostic for absent expected evidence. |
| `promotion_evidence_invalid` | Reader/validator diagnostic for malformed or unreadable evidence. |
| `promotion_evidence_contradictory` | Reader/validator diagnostic for contradictory equivalent evidence. |

Configured gate reason codes are machine-generated. Future human reason codes
must live under the future `human_decision` object and must not overwrite
machine reason codes.

## Implementation Implications For Issues #494-#499

Issue #494 should introduce an engine-owned promotion-state model/factory that
can build both configured machine-evaluation artifacts and explicit no-policy
artifacts without changing governance semantics.

Issue #495 should update standalone research review artifact writing so a
completed review with no configured promotion policy emits valid v2
`promotion_gates.json` evidence with `promotion_status: not_reviewed`.

Issue #496 should update research campaign, scenario, and orchestration
container writers to emit campaign-level v2 promotion-state artifacts while
preserving campaign, selected review, and selected catalog run identities.

Issue #497 should update governance loader, normalization, aggregation, and
validator behavior so explicit v2 no-policy artifacts are recognized, missing
or invalid evidence remains distinct, and `not_reviewed` is counted separately.

Issue #498 should extend compatibility tests and documentation for existing
strategy, portfolio, alpha, review, and campaign configured artifacts, including
legacy v1 artifacts and embedded `promotion_gate_summary` precedence.

Issue #499 should perform repository-level verification of the completed M45
implementation. Verification should include native review and campaign artifact
inspection, governance consistency outcomes, compatibility coverage, and
regression evidence for the new contract.

Notebook 14 consumer alignment is tracked separately in
`christophermoverton/fintech-stratlake-notebook-workflows#134`. That downstream
work may read and display canonical engine evidence only. It must not generate,
backfill, repair, or otherwise act as a canonical promotion-state artifact
producer.

## Non-Goals

This contract does not:

* implement a factory or writer
* modify campaign or review execution behavior
* weaken governance validation
* add a human approval workflow
* claim any run is promotion-ready or approved
* alter Notebook 14
* fabricate historical artifacts
