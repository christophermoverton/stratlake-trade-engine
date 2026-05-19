# M38.1 - Static Evidence Review Pack Contracts

Milestone principle: evidence review should make canonical support, derived
provenance, and health warnings easy to inspect without turning review surfaces
into sources of truth.

M38.1 defines the contract foundation for static evidence review packs before
any builder, writer, diagnostics engine, CLI, or renderer exists. A review pack
is a derived artifact family rooted at:

```text
artifacts/_derived/evidence_review/<review_id>/
```

It is disposable, rebuildable, non-authoritative, and forbidden from writing
back to canonical artifacts. Canonical manifests, registries, markers,
summaries, validation bundles, lineage artifacts, and governance artifacts
remain authoritative.

## Required Files

Every completed review pack is expected to expose:

```text
manifest.json
review_request.json
review_summary.json
catalog_health_diagnostics.json
validation.json
selected_record.json
related_records.json
resolver_resolution.json
evidence_index.json
artifact_inventory.csv
report.md
```

Optional deterministic outputs are:

```text
selected_lineage.openlineage.json
selected_lineage.prov.json
report.html
```

The first contract layer added in M38.1 defines JSON Schemas for:

- `review_pack_manifest.v1`
- `review_request.v1`
- `review_summary.v1`
- `catalog_health_diagnostics.v1`
- `resolver_resolution.v1`
- `evidence_index.v1`
- `review_pack_validation.v1`

Later M38 issues may add builder-side population rules, richer diagnostics, pack
writing, CLI behavior, and rendering details, but those concerns are
intentionally outside this issue.

`build_evidence_review_for_workflow(...)` is the Python-first integration
surface for notebook and pipeline callers that need the pure selected-run model
before any pack writing or report rendering exists. It reuses workflow loading,
selected-run one-hop lineage export behavior, and resolver-first canonical
reopening while returning JSON-safe data only.

`build_catalog_health_diagnostics(...)` adds advisory review context on top of
that model. Diagnostics are still derived, non-authoritative evidence: they
summarize selected-record resolution, metadata integrity, path hygiene, explicit
lineage coverage, and available governance or release-validation evidence, but
they do not mutate artifacts or change governance or promotion decisions.

`write_evidence_review_pack(...)` is the static writer layer. It consumes an
existing review model and writes a deterministic pack under
`artifacts/_derived/evidence_review/<review_id>/`. The writer emits the required
JSON payloads, `artifact_inventory.csv`, `evidence_index.json`, and a static
`report.md`; `report.html` is available only when callers opt in with
`include_html=True`.

The generated Markdown report includes selected-record, resolver, canonical
source, diagnostics, load-source, lineage, related-record, generated-evidence,
validation, and authority-boundary sections. Near the top it carries the visible
banner that the pack is derived, disposable, rebuildable, non-authoritative, and
write-back-forbidden while canonical artifacts remain the source of truth.

`manifest.json` is a pack manifest only. It records the generated-file
inventory, required/optional outputs, and deterministic digests for generated
files; it is not a second registry. `artifact_inventory.csv` and
`evidence_index.json` support review navigation only and never replace canonical
manifests, registries, markers, lineage exports, or governance evidence.

## Usage

Python remains the primary integration surface:

```python
from src.catalog import build_evidence_review_for_workflow, write_evidence_review_pack

model = build_evidence_review_for_workflow(
    "artifacts",
    selected_run_id="strategy_001",
)
result = write_evidence_review_pack(model)
```

Build from the CLI:

```powershell
python -m src.cli.build_evidence_review build --artifacts-root artifacts --selected-run-id strategy_001
```

Validate an existing pack:

```powershell
python -m src.cli.build_evidence_review validate --review-id review_abc123
```

The build command defaults to direct canonical scans and also accepts
`--index-mode index` or `--index-mode auto` with `--index-path` when callers
want the optional derived index path. Subject selection accepts
`--selected-run-id` and `--selected-catalog-id`; `--review-id` can pin an
explicit pack ID, otherwise the builder computes one deterministically from the
request. `--include-html` adds the optional static self-contained `report.html`.

The CI-safe synthetic example is:

```powershell
python docs/examples/m38_static_evidence_review_pack_example.py
```

It writes only under
`docs/examples/output/m38_static_evidence_review_pack_example/` and requires no
network, credentials, live market data, or external services.

Diagnostic statuses are:

- `PASS`: the conservative check succeeded
- `WARN`: advisory review concern
- `FAIL`: review evidence is incomplete or unsafe for inspection
- `NA`: the check does not apply to the current subject

Current diagnostic categories include selection, resolver, canonicality,
load-source metadata, path portability, derived namespace, lineage, catalog
validation, governance, and release validation. A `FAIL` finding means the
review context is weak; it is not itself a governance or promotion outcome.

## Canonicality And Load Source

Review-pack machine-readable outputs must carry:

- Canonicality Envelope v1 with `derived_class: review_pack`
- `load_source.v1` with `loaded_from: review_pack`

Those fields state that review packs remain rebuildable, non-authoritative, and
write-back-forbidden while deferring to canonical artifacts under `artifacts/`.
Serialized metadata uses repository-relative POSIX paths only.

Review packs complement the M37 resolver-first model. Any consequential review
consumer should reopen canonical manifests and registries through resolver APIs
before relying on the selected record or related evidence. A review pack may
surface `resolver_resolution.json`, but the pack itself never becomes the
authority that the resolver replaces.

## Boundary

Review packs may organize evidence for inspection. They must never:

- replace canonical manifests, registries, markers, summaries, validation
  bundles, lineage artifacts, or governance artifacts
- mutate canonical artifacts
- become a second registry or second catalog
- infer lineage edges
- introduce a graph store, remote metadata service, dashboard, or web server
- change governance or promotion behavior

Direct scan remains canonical and default. The `_derived` namespace remains
excluded from canonical catalog construction, and deleting or rebuilding review
packs must not change canonical identity.

## Focused M38 Validation

Run the focused review-pack validation slice before merging M38 evidence-review
changes:

```powershell
ruff check src\catalog\review_pack.py src\cli\build_evidence_review.py src\catalog\__init__.py tests\test_m38_deterministic_validation.py tests\test_evidence_review_cli.py tests\test_evidence_review_pack_writer.py tests\test_evidence_review_builder.py tests\test_catalog_health_diagnostics.py tests\test_review_pack_contracts.py tests\test_m37_architecture_guardrails.py docs\examples\m38_static_evidence_review_pack_example.py
```

```powershell
pytest tests\test_m38_deterministic_validation.py tests\test_evidence_review_cli.py tests\test_evidence_review_pack_writer.py tests\test_evidence_review_builder.py tests\test_catalog_health_diagnostics.py tests\test_review_pack_contracts.py tests\test_canonical_resolver.py tests\test_catalog_lineage_export.py tests\test_catalog_derived_index.py tests\test_derived_namespace_and_load_source.py tests\test_m37_architecture_guardrails.py tests\test_portable_paths.py tests\test_docs_path_portability.py -q
```

Smoke the CI-safe example:

```powershell
python docs/examples/m38_static_evidence_review_pack_example.py
```

For milestone-level confidence, run the milestone validation bundle:

```powershell
python -m src.cli.run_milestone_validation --bundle-dir artifacts/qa/m38_validation_bundle --include-full-pytest
```

This slice protects deterministic model generation, pack writing, CLI/API
parity, contract validation, portability, no canonical mutation, review-pack
deletion safety, direct/index/auto identity parity, stale-index failure behavior,
and the architecture guardrails that keep review-pack utilities out of
decision-authority paths.

Release readiness for M38 is tracked in:

- `docs/m38_release_notes.md`
- `docs/m38_release_validation_checklist.md`
