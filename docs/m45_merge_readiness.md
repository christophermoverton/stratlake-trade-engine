# M45 Merge Readiness

## Milestone Scope

Milestone 45 establishes canonical promotion-state contracts for standalone
research reviews, research campaign containers, and read-only governance
consumption.

Issue chain:

* #493 - canonical contract and compatibility rules
* #494 - factories, serialization, validation, and configured compatibility
* #495 - standalone review emission
* #496 - campaign-container emission and finalization failure handling
* #497 - governance loading, normalization, aggregation, and validation
* #498 - semantic and integration coverage
* #499 - documentation and repository verification

## Final Architecture

`src.research.promotion` owns canonical construction and serialization.
Standalone review and campaign execution own artifact emission. Governance is
read-only: it classifies and validates existing evidence without replaying
policy or changing source artifacts.

Valid no-policy evidence remains `promotion_status: not_reviewed` and maps to a
review-required governance disposition. Missing and malformed evidence remain
integrity findings. Campaign state is campaign-owned and cannot inherit nested
review outcomes.

## Documentation Updated

* `README.md`
* `docs/m45_canonical_promotion_state.md`
* `docs/m45_canonical_promotion_state_contract.md`
* `docs/m31_release_notes.md`
* `docs/m32_release_notes.md`
* `docs/m32_governance_reporting_architecture.md`
* `docs/m32_consistency_validation_design.md`
* `docs/milestone_13_research_review_workflow.md`
* `docs/review_configuration.md`
* `docs/milestone_16_campaign_workflow.md`
* `docs/research_campaign_configuration.md`
* `docs/notebook_integration.md`

## Semantic Invariants

* `not_reviewed` is explicit unresolved/no-policy evidence.
* `not_reviewed` is not eligibility, approval, promotion, readiness, or human
  sign-off.
* Missing evidence is not normalized into `not_reviewed`.
* Malformed canonical evidence does not fall back to manifest or registry
  summaries.
* Registry-backed review records are canonical-state-required; missing canonical
  state produces an integrity finding without manifest fallback.
* Canonical configured compatibility statuses are accepted only from a bounded
  vocabulary (`approved`, `manual_review`, `review_ready`, `needs_work`) and
  only when consistent with evaluation direction and severity resolution.
* Artifact filename overrides are restricted to a plain basename within the
  owner artifact directory.
* Review and campaign identities remain separate.
* Governance does not write or repair source artifacts.
* Legacy configured evaluator statuses remain bounded compatibility values.

## Verification Commands

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe -m pytest tests\test_promotion_gates.py -q
.\.venv\Scripts\python.exe -m pytest tests\test_research_review.py -q
.\.venv\Scripts\python.exe -m pytest tests\test_cli_run_research_campaign.py -q
.\.venv\Scripts\python.exe -m pytest tests\test_promotion_governance.py -q
.\.venv\Scripts\python.exe -m pytest tests\test_promotion_governance_integration.py -q
.\.venv\Scripts\ruff.exe check src tests
git diff --check
```

The M45 semantic integration tests provide artifact-level smoke verification:
review and campaign no-policy emission, campaign/review anti-conflation,
governance classification, aggregate review-required treatment, deterministic
reruns, and source artifact immutability.

## Verification Results

Verification completed on the #499 branch:

* Full repository suite: `2582 passed, 6 skipped, 348 warnings`.
* Required M45 suite across promotion, review, campaign, and governance:
  `156 passed`.
* Broader M45 implementation regression suite: `192 passed, 7 warnings`.
* Documentation/path portability: `3 passed`.
* Refreshed Milestone 13 committed example validation: `1 passed`.
* `ruff check src tests`: passed.
* `git diff --check`: passed, with line-ending notices reported separately for
  two pre-existing CRLF documentation files.

The warnings are existing research-fixture diagnostics: low sample sizes,
degenerate or high-turnover signal behavior, consistency warnings in synthetic
pipeline fixtures, and matplotlib open-figure warnings. No test failures remain.

The initial full-suite pass identified one stale committed Milestone 13 example
artifact. Regenerating the deterministic example updated only its canonical v2
review promotion state and compact manifest summary. Its dedicated test and the
subsequent full repository suite passed.

## Change-Scope Audit

The cumulative M45 implementation is limited to promotion-state construction,
review and campaign emission, governance observation, tests, and documentation.
It does not add live-trading interfaces, notebook mutation, governance policy
replay, broad schema rewrites, or no-policy emission for strategy, portfolio,
alpha, or candidate-selection producers.

Issue #498 intentionally corrected one production behavior: an expected M45
review or campaign record with missing canonical state must not borrow a
manifest summary.

## Known Limitations

* M45 does not establish human approval.
* M45 does not establish deployment or live-trading readiness.
* M45 does not add a configured campaign promotion policy where no dedicated
  campaign policy surface exists.
* Valid `not_reviewed` is explicit unresolved/no-policy evidence.
* Governance is observational and read-only.
* Legacy artifact compatibility remains intentionally bounded.

## Release

Latest patch release tag: `v0.45.1`
Latest package/build version: `0.45.1`

Original M45 release tag: `v0.45.0`
Original M45 package/build version: `0.45.0`

GitHub Release published at:
https://github.com/christophermoverton/stratlake-trade-engine/releases/tag/v0.45.0

TestPyPI publication completed via workflow\_dispatch.

Release notes: [docs/m45_release_notes.md](m45_release_notes.md)

Patch release `v0.45.1` uses the tag-driven Release workflow. Pushing the
`v0.45.1` tag runs `.github/workflows/release.yml`, which publishes the GitHub
Release and uploads release-validation and package-build workflow artifacts.
PyPI/TestPyPI publication remains outside that Release workflow.

## Merge Recommendation

M45 is merged. Releases `v0.45.0` and patch `v0.45.1` confirm deterministic
research-artifact, governance, and portfolio workflow documentation readiness
only; they do not authorize live trading, deployment, or promotion.
