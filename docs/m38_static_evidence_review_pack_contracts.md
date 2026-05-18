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
