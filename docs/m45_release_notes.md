# M45 Release Notes - Canonical Promotion-State Contracts

Milestone title:
`M45 - Canonical Promotion-State Contracts and Governance Validation`

M45 branch:
`features/m45-7-promotion-state-merge-readiness-docs`

Target branch:
`main`

Issue range covered:
`#493` through `#500`

Latest patch release tag:
`v0.45.1`

Original M45 release tag:
`v0.45.0`

Latest package/build version:
`0.45.1`

Original M45 package/build version:
`0.45.0`

## Summary

M45 establishes deterministic, engine-owned canonical promotion-state evidence
for completed standalone research reviews and research campaign containers. The
canonical artifact uses `promotion_gates.json` with `schema_version: 2` and
`artifact_type: promotion_state`.

The milestone distinguishes three promotion-state categories:

* **Configured machine evaluation** — engine evaluates configured gates and
  preserves the evaluator result (`eligible`, `warn`, `needs_review`,
  `rejected`, `blocked`, or a bounded compatibility value).
* **Explicit no-policy state** — engine completed the owner workflow without a
  configured promotion policy (`promotion_status: not_reviewed`).
* **Missing or malformed evidence** — required canonical evidence is absent or
  invalid (governance reports integrity findings without synthesizing state).

## Architecture

`src.research.promotion` owns canonical construction and serialization.
Standalone review and campaign execution own artifact emission. Governance is
read-only: it classifies and validates existing evidence without replaying
policy or changing source artifacts.

Review state is review-owned. Campaign state is campaign-owned and cannot
inherit a nested review outcome. A configured review result such as `approved`
may coexist with a campaign `not_reviewed` state when no campaign-level policy
is configured.

## Key Behaviors

### No-Policy Semantics

`not_reviewed` means promotion policy was not configured and no promotion
decision was made. It does not mean eligible, candidate, approved, promoted,
production-ready, deployment-ready, or approved by a human reviewer.

### Registry-Backed Review Records

Registry-backed review records are canonical-state-required. A registry entry
with `run_type: review` loads canonical state with `required=True` and review
identity validation. Missing canonical state for registry review records
produces `missing_canonical_promotion_state` without manifest fallback.

### Bounded Compatibility Vocabulary

Custom configured evaluator statuses are accepted only from a bounded
vocabulary:

* `approved`
* `manual_review`
* `review_ready`
* `needs_work`

A configured compatibility value is valid only when:

1. it is in the bounded supported compatibility vocabulary;
2. it matches the correct pass/fail configuration field;
3. it is consistent with gate-result direction;
4. it does not override severity-driven canonical failure outcomes;
5. the artifact contains valid evaluator-shaped results and valid counts.

Unsupported custom configured status values are rejected at both the
serialization boundary and governance validation.

### Artifact Filename Security

Promotion artifact filename overrides are restricted to plain basenames within
the owner artifact directory. Path traversal (`../`), nested paths, absolute
paths, and degenerate values (`.`, `..`, empty string) are rejected at all
entry and consumption points.

### Governance Behavior

Governance remains observational and read-only. It:

* discovers review and campaign canonical state
* requires canonical state for registry-backed review records
* validates schema, counts, metadata, filename, provenance, and identity
* treats valid canonical evidence as authoritative
* preserves legacy summary behavior for non-review registry records
* reports missing, malformed, and identity-mismatched evidence separately
* does not replay gates or mutate source artifacts

### Campaign Finalization

Campaign completion is recorded only after the campaign-owned promotion state
and final campaign artifacts are written successfully. A failed finalization
records `failure_stage: campaign_finalization`.

## Issue Chain

* #493 — canonical contract and compatibility rules
* #494 — factories, serialization, validation, and configured compatibility
* #495 — standalone review emission
* #496 — campaign-container emission and finalization failure handling
* #497 — governance loading, normalization, aggregation, and validation
* #498 — semantic and integration coverage
* #499 — documentation and repository verification
* #500 — boundary hardening (registry review evidence, forged status
  prevention, filename containment, bounded compatibility vocabulary)

## Release Validation

Patch release `v0.45.1` workflow results:

* GitHub Release: [published successfully](https://github.com/christophermoverton/stratlake-trade-engine/releases/tag/v0.45.1)
* Release workflow run: passed for tag `v0.45.1`
* Package/build version: `0.45.1`
* Scope: portfolio CLI entrypoint fix and M11 portfolio workflow documentation
  readiness
* Release validation artifacts: uploaded by `.github/workflows/release.yml`
* Package build artifacts: uploaded by `.github/workflows/release.yml`
* Package publication: PyPI/TestPyPI remains out of scope for
  `.github/workflows/release.yml`

Release `v0.45.0` workflow results:

* GitHub Release: [published successfully](https://github.com/christophermoverton/stratlake-trade-engine/releases/tag/v0.45.0)
* TestPyPI: `0.45.0` published via workflow\_dispatch; tag-triggered publish
  cancelled (version already present on TestPyPI)
* Release pytest slice: passed
* Docs/path lint: passed
* Package build: `stratlake_trade_engine-0.45.0.tar.gz` and
  `stratlake_trade_engine-0.45.0-py3-none-any.whl` built successfully

Local verification (PR #500):

* Full repository suite: `2633 passed, 6 skipped, 348 warnings`
* Required M45 suite: `203 passed`
* `ruff check src tests`: passed
* `git diff --check`: passed

## Non-Goals

M45 does not:

* establish human approval or a human-decision workflow
* establish deployment or live-trading readiness
* add a configured campaign promotion policy
* broaden no-policy emission to strategy, portfolio, alpha, or
  candidate-selection producers
* alter notebook behavior
* add signing, encryption, or external attestations
* reinterpret `not_reviewed` as eligibility, approval, or promotion

## Documentation

* [m45_canonical_promotion_state.md](m45_canonical_promotion_state.md)
* [m45_canonical_promotion_state_contract.md](m45_canonical_promotion_state_contract.md)
* [m45_merge_readiness.md](m45_merge_readiness.md)
