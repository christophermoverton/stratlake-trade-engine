# M37 Artifact-First Canonicality Contracts

Milestone principle: every derived surface must identify, defer to, and be
invalidated by its canonical source.

## Canonicality Envelope v1

Newly generated M37 derived evidence surfaces carry a deterministic top-level
`canonicality` object:

```json
{
  "canonicality": {
    "schema_version": "canonicality.v1",
    "authority_kind": "artifact_tree",
    "authority_root": "artifacts",
    "authority_paths": [],
    "authority_fingerprint": "<sha256>",
    "derived_class": "sqlite_read_model|lineage_export|evidence_view|workflow_view",
    "rebuildable": true,
    "non_authoritative": true,
    "write_back_forbidden": true,
    "stale_if_source_changes": true,
    "resolver_hint": "reopen canonical manifests/registries before decision-sensitive use"
  }
}
```

The envelope is deterministic: sorted keys, stable ordering, POSIX-style
repository-relative paths, no absolute paths, no URI-like paths, no parent
traversal, and no machine-local metadata.

## Authority Boundary

Canonical artifacts remain the source of truth:

- artifact trees
- manifests
- registries
- markers
- summaries
- validation bundles
- governance artifacts

Derived indexes are disposable read models. OpenLineage-style and PROV-style
lineage exports are local JSON views. Evidence/explorer/workflow helper outputs
are local read views. All derived outputs are non-authoritative, rebuildable,
forbidden from write-back, and stale if the canonical source changes.

Decision-sensitive consumers should reopen canonical manifests and registries
before making release, governance, promotion, or other authoritative decisions.

## Derived Namespace And Load Source

New M37 derived files should live under `artifacts/_derived/`, such as:

- `artifacts/_derived/catalog_index/catalog_index.sqlite`
- `artifacts/_derived/lineage/`
- `artifacts/_derived/evidence/`

This namespace is for disposable, rebuildable read models only. Direct scans
remain canonical and ignore `_derived` as a canonical artifact family. Explicit
legacy M36 paths remain supported when supplied by callers.

Derived JSON surfaces now also expose `load_source` metadata. It states whether
a view came from a direct scan, a validated derived index, a lineage export, or
an evidence view, and records `requested_mode`, `resolved_mode`,
`index_validated`, `canonical_source`, and the decision-sensitive resolver hint
where those fields apply. `load_source.index_path` is portable
repository-relative metadata; absolute, URI-like, `file://`, and
parent-traversal paths are rejected.

## Backward Compatibility

Existing M36 derived artifacts without envelopes remain readable. Where a reader
can expose compatibility status, no-envelope payloads are labeled
`legacy_no_envelope`; envelope-bearing payloads are labeled `canonicality_v1`.
Direct artifact scanning remains unchanged and remains the canonical catalog load
path.

## Resolver-First Access

Catalog queries, derived indexes, exports, and evidence views are safe for
discovery, search, filtering, and display. Decision-sensitive consumers should
call resolver APIs such as `resolve_canonical_record(...)` first. The resolver
reopens the record's declared canonical registries, manifests, markers,
summaries, and source files, validates their portable repository-relative paths,
requires reopened files to remain under `artifacts_root`, and computes a
deterministic fingerprint over the reopened sources.

Resolver results are read-only and report `resolved`, `partial`, or `unresolved`
status. Missing, non-portable, or repo-relative files outside `artifacts_root`
fail safely with warnings. The resolver does not make indexes, exports, or views
canonical; it is the explicit bridge back to canonical artifacts.

## Architecture Guardrails

Derived indexes may support discovery, query acceleration, validation, local
lineage export, evidence views, and notebook/workflow read helpers. They must
not become decision authority for artifact writers, registry writers, promotion
or governance decisions, release readiness, milestone validation decisions, or
canonical catalog construction.

Direct scans remain canonical and are the default load mode. Derived outputs are
disposable and rebuildable: creating, deleting, or rebuilding `_derived`
artifacts must not change canonical catalog identity or mutate canonical source
files. Consequential consumers must cross back to canonical artifacts through
resolver-first APIs before acting.

Architecture tests enforce these boundaries by rejecting forbidden direct-module
imports and public-facade imports from `src.catalog` in decision-authority
modules, and by proving `_derived` namespace exclusion plus derived-index
disposability against fixture trees.

## Updated Derived Surfaces

- derived SQLite index metadata: `derived_class: sqlite_read_model`
- OpenLineage-style root payloads: `derived_class: lineage_export`
- PROV-style root payloads: `derived_class: lineage_export`
- shared workflow lineage outputs inherit `derived_class: lineage_export`
- shared workflow evidence views: `derived_class: evidence_view`

`workflow_view` is reserved for a future wrapper-level derived payload if one is
explicitly introduced.

## Non-Goals

- no second registry
- no graph store
- no remote metadata service
- no dashboard or web server
- no inferred lineage edges
- no mutation of source artifacts
- no change to promotion or governance decisions
- no rewrite of the M36 catalog/index/export architecture
