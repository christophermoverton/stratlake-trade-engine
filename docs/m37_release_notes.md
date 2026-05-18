# M37 Release Notes - Artifact-First Evidence Hardening and Canonicality Contracts

Milestone title: `M37 - Artifact-First Evidence Hardening and Canonicality Contracts`

M37 branch:
`feature/m37-artifact-first-canonicality-contracts`

Candidate milestone release tag:
`v0.37.0-artifact-first-canonicality-contracts`

## Milestone Principle

Every derived surface must identify, defer to, and be invalidated by its
canonical source.

## Summary

M37 follows the M36 evidence interoperability work by making the source-of-truth
boundary explicit and testable. It adds:

- Canonicality Envelope v1 on newly generated derived outputs
- `load_source.v1` metadata for direct, index-backed, and derived-view loads
- `artifacts/_derived/` as the default namespace for new disposable outputs
- resolver-first canonical access APIs with artifact-root-bounded source reopening
- architecture guardrails against derived-source-of-truth drift
- deterministic combined-stack validation across the new contracts

The result is a stronger artifact-first evidence model: derived read models stay
useful for discovery and display, while canonical artifacts remain authoritative.

## Issue Summary

| Issue | Outcome |
| --- | --- |
| `#412` | Added Canonicality Envelope v1 to identify canonical authority from derived outputs. |
| `#413` | Added `load_source.v1` signaling and the default `artifacts/_derived/` namespace. |
| `#414` | Added resolver-first canonical access APIs and artifact-root-bounded reopening. |
| `#415` | Added architecture guardrails for derived read-model boundaries, including public-facade import checks. |
| `#416` | Added deterministic validation across the combined M37 stack. |
| `#419` | Triaged Milestone Validation failure and updated CLI query tests for M37 `load_source`-wrapped outputs. |

## What Changed

- Newly generated derived indexes, lineage exports, and evidence views carry
  deterministic canonicality metadata.
- Catalog query and summary JSON surfaces now expose `load_source` metadata next
  to the records or summary they return.
- New derived outputs default beneath `artifacts/_derived/`; direct scans remain
  canonical and ignore that namespace as an artifact family.
- Resolver APIs reopen declared canonical source files from artifact-backed
  records, reject non-portable or outside-artifacts paths, and compute stable
  fingerprints over reopened sources.
- Architecture tests prevent derived read models from leaking into writer,
  promotion, governance-decision, release-decision, or canonical-construction
  paths.

## Backward Compatibility

- Existing M36 derived outputs without canonicality envelopes remain readable.
- No-envelope payloads surface `legacy_no_envelope` compatibility status where
  readers expose canonicality status.
- Explicit M36 paths remain supported when supplied by callers.
- Direct scan remains available, canonical, and the default load mode.

## Architecture Boundaries Preserved

- canonical artifacts remain the source of truth
- derived indexes remain disposable, rebuildable read models
- lineage exports remain local JSON views
- evidence, explorer, and workflow outputs remain non-authoritative
- dataset and feature lineage remain optional and explicit
- resolver-first APIs are the bridge for consequential decisions
- no graph store, dashboard, backend service, inferred lineage system, remote
  metadata service, second registry, or second source of truth is introduced

## Validation Summary

M37 validation now covers canonicality envelopes, load-source metadata, resolver
behavior, architecture boundaries, deterministic direct/index/auto parity,
derived-output disposability, legacy compatibility, CLI output regressions,
docs/path lint, package build validation, and full milestone validation.

## Non-Goals

M37 does not add:

- promotion or governance decision changes
- new artifact writer behavior
- a graph store, dashboard, web server, or remote metadata service
- inferred lineage or a second registry
- formal W3C PROV conformance
- a production metadata backend

## Further Reading

- `docs/m37_artifact_first_canonicality_contracts.md`
- `docs/m37_release_validation_checklist.md`
- `docs/catalog_indexer.md`
- `docs/catalog_lineage.md`
