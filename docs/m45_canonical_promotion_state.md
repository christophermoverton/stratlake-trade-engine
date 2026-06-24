# M45 Canonical Promotion State

## Purpose And Scope

Milestone 45 defines deterministic, engine-owned promotion-state evidence for:

* completed standalone research reviews
* completed research campaign containers

The canonical filename remains `promotion_gates.json`. For M45 artifacts,
`schema_version: 2` and `artifact_type: promotion_state` identify the canonical
state contract.

M45 does not introduce a new promotion policy model, a human approval workflow,
deployment or live-trading readiness, or automatic no-policy emission for
strategy, alpha, portfolio, or candidate-selection producers. Governance and
notebooks may read canonical evidence, but they must not create, repair,
backfill, or rewrite it.

The detailed field-level schema and compatibility rules remain in
[m45_canonical_promotion_state_contract.md](m45_canonical_promotion_state_contract.md).

## Ownership Model

### Review-Owned State

Standalone review evidence is written to:

```text
artifacts/reviews/<review_id>/promotion_gates.json
```

Its provenance identifies the review:

```json
{
  "object_type": "review",
  "object_id": "<review_id>",
  "review_id": "<review_id>",
  "run_type": "review"
}
```

### Campaign-Owned State

Campaign-container evidence is written to:

```text
artifacts/research_campaigns/<campaign_run_id>/promotion_gates.json
```

Its provenance identifies the campaign:

```json
{
  "object_type": "research_campaign",
  "object_id": "<campaign_run_id>",
  "campaign_run_id": "<campaign_run_id>",
  "run_type": "research_campaign"
}
```

A campaign state is not derived from or replaced by a nested review state. A
configured review result such as a valid evaluator-produced `approved` value
may coexist with a campaign `not_reviewed` state when no campaign-level policy
is configured.

## Canonical V2 Contract

The meaningful fields are:

* `schema_version` and `artifact_type`
* `run_type`
* `configured` and `configuration_state`
* `evaluation_status` and `promotion_status`
* `decision_authority` and `human_decision`
* `decision_reason_codes`
* `gate_counts`, `gate_definitions`, and `gate_results`
* `provenance`
* `artifact_metadata`

The explicit no-policy state includes:

```json
{
  "schema_version": 2,
  "artifact_type": "promotion_state",
  "configured": false,
  "configuration_state": "not_configured",
  "evaluation_status": "not_configured",
  "promotion_status": "not_reviewed",
  "decision_authority": "none",
  "human_decision": null,
  "decision_reason_codes": [
    "promotion_policy_not_configured"
  ]
}
```

The full artifact also contains all nine canonical zero-valued gate counts,
empty gate definitions and results, owner provenance, and deterministic writer
metadata.

## Configured And Unconfigured Semantics

| Condition | Evidence meaning | Governance treatment |
|---|---|---|
| Configured evaluation | Engine evaluated configured gates and preserved the evaluator result. | Preserve the raw result and validate consistency. |
| Explicit no-policy state | Engine completed the owner workflow without a configured promotion policy. | Preserve `not_reviewed`; treat as review-required. |
| Missing artifact | Required canonical evidence is absent. | Emit an integrity finding; do not synthesize `not_reviewed`. |
| Malformed or mismatched artifact | Evidence exists but violates schema, metadata, or ownership rules. | Emit precise integrity findings; do not fall back to summaries. |
| Future human decision | Separately authorized human evidence, if implemented later. | Not created or implied by M45. |

`not_reviewed` means promotion policy was not configured and no promotion
decision was made. It does not mean eligible, candidate, approved, promoted,
production-ready, deployment-ready, or approved by a human reviewer.

## Artifact Placement And References

A successful standalone review contains:

* `leaderboard.csv`
* `review_summary.json`
* `manifest.json`
* `promotion_gates.json`

A successful campaign container contains:

* `campaign_config.json`
* `checkpoint.json`
* `preflight_summary.json`
* `manifest.json`
* `summary.json`
* `promotion_gates.json`

Manifests inventory the canonical artifact exactly once. Summaries expose
compact discovery fields and paths, but a structurally valid canonical JSON
artifact is authoritative for its owner's promotion state.

## Campaign Finalization

Campaign completion is recorded only after the campaign-owned promotion state
and final campaign artifacts are written successfully. A failed finalization
does not write a successful completion marker. Where the normal failure marker
can be persisted, it records `failure_stage: campaign_finalization`, and the
original write exception remains visible to the caller.

A later clean rerun follows the normal checkpoint, retry, and reuse behavior.
M45 does not claim transactional atomicity across every filesystem write.

## Governance Behavior

Governance remains observational and read-only. It:

* discovers review and campaign canonical state
* validates schema, counts, metadata, filename, provenance run type, object
  type, and owner identity
* treats valid canonical evidence as authoritative
* requires canonical state for registry-backed review records; missing canonical
  state emits `missing_canonical_promotion_state` with no manifest fallback
* preserves legacy summary behavior for strategy, portfolio, and alpha records
  where M45 canonical state is not required
* does not borrow a manifest summary when required review or campaign state is
  missing
* preserves raw `not_reviewed` while mapping its review disposition to
  `needs_review`
* reports missing, malformed, identity-mismatched, and summary-inconsistent
  evidence separately
* does not replay gates or mutate source artifacts

Important integrity and consistency findings include:

* `missing_canonical_promotion_state`
* `promotion_state_schema_invalid`
* `promotion_state_provenance_run_type_mismatch`
* `promotion_state_object_type_mismatch`
* `promotion_state_owner_id_mismatch`
* `promotion_state_artifact_filename_mismatch`
* `campaign_promotion_state_summary_mismatch`
* `campaign_promotion_state_upstream_review_conflation`

These are governance findings, not promotion decisions.

## Legacy Compatibility

Legacy `promotion_gate_summary` and configured promotion-gate artifacts remain
supported where applicable. Valid custom evaluator statuses such as `approved`
or `manual_review` are accepted only when validated as genuine evaluator
outcomes — status consistency with evaluation direction, gate results, and
severity resolution is verified at both the serialization boundary and
governance validation. Arbitrary forged or caller-constructed status values are
rejected even when `promotion_status` matches `status_on_pass` or
`status_on_fail`. Artifact filename overrides are restricted to a plain
basename within the owner artifact directory; path traversal and absolute paths
are rejected.

M45 does not convert missing or legacy evidence into canonical no-policy state,
and it does not require strategy, portfolio, or alpha producers to emit new
no-policy artifacts.

## Commands

Run a standalone registry-backed review:

```powershell
python -m src.cli.compare_research --from-registry --output-path artifacts/reviews/m45_example
```

This reads existing research registries and writes a review pack, including
canonical review promotion state. Without review promotion gates, the state is
explicitly `not_reviewed`.

Run a campaign:

```powershell
python -m src.cli.run_research_campaign --config configs/research_campaign.yml
```

This executes the configured research campaign and writes campaign-owned
canonical state. The current config model has no dedicated campaign promotion
policy, so a successfully completed campaign normally emits explicit no-policy
state.

Run governance reporting:

```powershell
python -m src.cli.run_promotion_governance_report `
  --artifact-root artifacts `
  --output-dir artifacts/promotion_governance/m45_example
```

This reads existing evidence and writes a governance report bundle. It does not
create, repair, or backfill promotion-state artifacts.
