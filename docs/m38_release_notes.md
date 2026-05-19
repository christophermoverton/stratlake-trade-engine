# M38 Release Notes - Static Evidence Review Packs and Catalog Health Diagnostics

Milestone title: `M38 - Static Evidence Review Packs and Catalog Health Diagnostics`

M38 branch:
`feature/m38-static-evidence-review-packs-catalog-health`

Candidate milestone release tag:
`v0.38.0-static-evidence-review-packs-catalog-health`

## Milestone Principle

Evidence review should make canonical support, derived provenance, and health
warnings easy to inspect without turning review surfaces into sources of truth.

## Summary

M38 builds on the M37 canonicality and resolver-first guarantees by adding a
static evidence review-pack layer. Review packs organize selected-run evidence,
resolver summaries, one-hop lineage, diagnostics, generated inventories, and
human-readable reports under:

```text
artifacts/_derived/evidence_review/<review_id>/
```

The result is easier review and communication without authority drift:
canonical artifacts remain the source of truth, direct scan remains canonical
and default, and review packs stay derived, disposable, rebuildable,
non-authoritative, and write-back-forbidden.

## Issue Summary

| Issue | Outcome |
| --- | --- |
| `#420` | Added the `review_pack` derived classification, review-pack schemas, metadata helpers, and default `_derived/evidence_review` contract. |
| `#421` | Added `build_evidence_review_for_workflow(...)`, selected run/catalog filtering, resolver-backed source summaries, deterministic review IDs, and one-hop lineage reuse. |
| `#422` | Added `build_catalog_health_diagnostics(...)` with advisory PASS/WARN/FAIL/NA findings and aggregate summaries. |
| `#423` | Added `write_evidence_review_pack(...)`, deterministic pack emission, `report.md`, optional static `report.html`, `artifact_inventory.csv`, and `evidence_index.json`. |
| `#424` | Added `python -m src.cli.build_evidence_review` with `build` and `validate` commands, a CI-safe synthetic example, and usage docs. |
| `#425` | Added the focused deterministic M38 validation slice for model, diagnostics, writer, CLI, contracts, portability, and no-mutation guarantees. |
| `#427` | Fixed CI portability by ensuring derived review outputs preserve resolver summaries without embedding raw reopened source content. |

## Major Additions

- `review_pack` canonicality/load-source classification for derived review
  outputs
- `build_evidence_review_for_workflow(...)`
- `build_catalog_health_diagnostics(...)`
- `write_evidence_review_pack(...)`
- `validate_evidence_review_pack(...)`
- `python -m src.cli.build_evidence_review build`
- `python -m src.cli.build_evidence_review validate`
- deterministic static review-pack files beneath `_derived/evidence_review`
- deterministic Markdown `report.md`
- optional escaped, self-contained static `report.html`
- `artifact_inventory.csv`
- `evidence_index.json`
- CI-safe example script:
  `docs/examples/m38_static_evidence_review_pack_example.py`
- focused deterministic validation:
  `tests/test_m38_deterministic_validation.py`

## Usage Notes

Python remains the primary integration surface:

```python
from src.catalog import build_evidence_review_for_workflow, write_evidence_review_pack

model = build_evidence_review_for_workflow(
    "artifacts",
    selected_run_id="strategy_001",
)
result = write_evidence_review_pack(model)
```

CLI build:

```powershell
python -m src.cli.build_evidence_review build --artifacts-root artifacts --selected-run-id strategy_001
```

CLI validate:

```powershell
python -m src.cli.build_evidence_review validate --review-id review_abc123
```

CI-safe example:

```powershell
python docs/examples/m38_static_evidence_review_pack_example.py
```

## Preserved M37 Guarantees

- canonical artifacts remain the source of truth
- direct scan remains available, default, and canonical
- `_derived` remains disposable derived-output space
- resolver-first canonical reopening remains the decision-sensitive path
- derived outputs carry canonicality and load-source metadata
- review packs identify and defer to canonical source paths
- review packs can be deleted or rebuilt without changing canonical catalog
  identity

## Architecture Boundaries Preserved

M38 does not add:

- a second registry
- a second catalog
- a graph store
- a web server or dashboard
- a remote metadata service
- inferred lineage
- governance mutation
- promotion mutation
- live market data, credentials, network access, or external services

Diagnostics remain advisory review context. Static reports are review surfaces,
not dashboards or sources of truth.

## Validation Summary

Focused M38 validation covers deterministic review model generation,
diagnostics, writer output, CLI/API parity, contract validation, path
portability, no canonical mutation, deletion safety, direct/index/auto parity,
stale-index failure behavior, docs/path lint, and architecture guardrails.

Before release, record results for:

- focused Ruff slice
- focused M38 pytest slice
- CI-safe example smoke
- docs/path lint
- package build validation
- milestone validation bundle
- hosted GitHub Actions milestone validation

## Non-Goals

M38 does not implement:

- dashboard or server behavior
- graph UI or graph store
- remote metadata services
- governance or promotion decision changes
- inferred lineage
- live-data or credential-backed examples
- a replacement for canonical manifests, registries, markers, validation
  bundles, lineage artifacts, or governance artifacts

## Draft GitHub Release Notes

Title:
`M38 - Static Evidence Review Packs and Catalog Health Diagnostics`

Tag:
`v0.38.0-static-evidence-review-packs-catalog-health`

Branch:
`feature/m38-static-evidence-review-packs-catalog-health`

Summary:
M38 adds derived static evidence review packs and advisory catalog-health
diagnostics so selected-run evidence is easier to inspect, validate, and share
without turning review surfaces into sources of truth.

Highlights:

- Added resolver-backed evidence review model APIs.
- Added advisory catalog-health diagnostics.
- Added deterministic static pack writer with manifest, validation payload,
  evidence index, artifact inventory, selected lineage exports, and `report.md`.
- Added optional escaped static `report.html`.
- Added build/validate CLI:
  `python -m src.cli.build_evidence_review`.
- Added CI-safe example and focused deterministic validation slice.
- Preserved M37 canonicality, load-source, direct-scan, `_derived`, and
  resolver-first guarantees.

Validation:

- Focused Ruff: `<record result>`
- Focused M38 pytest: `<record result>`
- Example smoke: `<record result>`
- Docs/path lint: `<record result>`
- Package build: `<record result>`
- Milestone validation: `<record result>`
- Hosted GitHub Actions: `<record result>`

Known boundaries:
Review packs are derived, disposable, rebuildable, non-authoritative, and
write-back-forbidden. They do not replace canonical artifacts and do not add a
dashboard, graph store, remote service, second registry, inferred lineage, or
governance/promotion mutation.

## Further Reading

- `docs/m38_static_evidence_review_pack_contracts.md`
- `docs/m38_release_validation_checklist.md`
- `docs/examples/m38_static_evidence_review_pack_example.py`
