# M36 Release Notes - Scalable Evidence Interoperability and Release Hardening

Milestone title: `M36 - Scalable Evidence Interoperability and Release Hardening`

M36 branch:
`feature/m36-scalable-evidence-interoperability-release-hardening`

Candidate milestone release tag:
`v0.36.0-scalable-evidence-interoperability-release-hardening`

## Milestone Principle

Scalable evidence systems should make research artifacts faster to find, easier
to export, and safer to release without weakening deterministic artifact
provenance.

## Issue #402 Scope

Issue #402 is a release and process hardening pass. It prepares M36 for later
catalog interoperability work by clarifying version policy, milestone branch
naming, and milestone-validation workflow coverage.

This issue does not add catalog features, derived indexes, lineage export,
dataset or feature fingerprints, artifact contract migrations, promotion policy
changes, or governance decision changes.

## Version Policy

Package version and milestone release tags are intentionally separate.

The package version in `pyproject.toml` is Python distribution metadata. It is
used by editable installs, package metadata checks, wheel/sdist build
validation, and installed import metadata. For M36 it remains `0.1.0` because
Issue #402 does not change package distribution semantics or publish the package
to PyPI/TestPyPI.

Milestone release tags identify repository snapshots and release-validation
evidence. M36 milestone release tags should use the `v0.36.0-<slug>` form, with
`v0.36.0-scalable-evidence-interoperability-release-hardening` as the candidate
tag. These tags drive `.github/workflows/release.yml`, create GitHub Releases,
and attach release-validation evidence. They do not imply that the Python
package version has changed.

Future milestones should update milestone release notes, validation checklists,
and candidate tag names for the milestone. They should preserve
`pyproject.toml` unless the milestone intentionally changes package metadata,
install behavior, or distribution compatibility.

## Branch Naming

M36 uses:

```text
feature/m36-scalable-evidence-interoperability-release-hardening
```

Future milestone branches should use:

```text
feature/m<NUMBER>-<short-kebab-description>
```

The milestone validation workflow covers this `feature/m*` convention while
preserving legacy `milestone/**` and `m22/**` branch support.

## Validation Workflow Coherence

`.github/workflows/milestone_validation.yml` keeps manual dispatch support and
continues to run the existing milestone validation bundle job. Its branch
trigger coverage now includes:

- `milestone/**`
- `m22/**`
- `feature/m*`

The pull request job guard also accepts `feature/m` source branches, so pull
requests from
`feature/m36-scalable-evidence-interoperability-release-hardening` into `main`
can run milestone validation.

## GitHub Actions Supply-Chain Hardening

Issue #403 pins every external GitHub Action reference in `.github/workflows/`
to a full commit SHA. This reduces exposure to mutable tag movement while
keeping the human-readable upstream tag context in workflow comments.

Current workflow action inventory:

| Action | Classification | Pinned SHA | Upstream tag represented |
| --- | --- | --- | --- |
| `actions/checkout` | GitHub-maintained | `34e114876b0b11c390a56381ad16ebd13914f8d5` | `v4` |
| `actions/setup-python` | GitHub-maintained | `a26af69be951a213d495a4c3e4e4022e16d87065` | `v5` |
| `actions/upload-artifact` | GitHub-maintained | `ea165f8d65b6e75b540449e92b4886f43607fa02` | `v4` |
| `softprops/action-gh-release` | Third-party | `3bb12739c298aeb8a4eeaf626c5b8d85266b0e65` | `v2` |

There are no local reusable actions in the current workflow set. There are no
intentionally unpinned external action references in the current workflow set.

To refresh a pinned action safely:

1. Review the upstream release/tag page and the exact commit represented by the
   desired tag.
2. Replace the full SHA and update the nearby tag-to-SHA comment in every
   affected workflow.
3. Re-run workflow YAML parsing, workflow pinning tests, focused workflow tests,
   docs/path lint, package build validation, and full pytest when practical.
4. Re-check that workflow names, job names, matrices, install commands, release
   artifacts, and milestone trigger coverage remain unchanged.

SHA pinning is part of M36 release hardening. It strengthens workflow provenance
without changing package publication scope, release tag semantics, catalog
behavior, artifact contracts, or governance decisions.

## Catalog Scale Baselines

Issue #404 adds deterministic scale baselines before any optional derived
metadata index work. The new synthetic test fixture builds a compact temporary
artifact history spanning strategy, alpha, portfolio, campaign/scenario,
robustness, governance, milestone-validation, release-validation, sparse, and
registry-only cases.

The required scale tests measure the existing direct-scan workflow only:

- catalog indexing over the synthetic artifact root
- evidence query filters
- lineage edge extraction
- explorer JSON, Markdown, and table rendering
- notebook/API helper views and renderers

Deterministic assertions cover record counts, family counts, query counts,
lineage edge counts, ordering, path portability, and source immutability. A
single intentionally broad elapsed-time ceiling protects against accidental
pathological regressions while avoiding brittle micro-benchmark behavior; exact
timings are environment-dependent and are not treated as deterministic output.

Issue #404 does not add a derived metadata index, database, cache, graph store,
search backend, or alternate canonical registry. Later M36 optimization work
should compare against these baselines rather than guessing at scale behavior.

## Optional Derived Metadata Index

Issue #405 adds an optional local SQLite metadata index for faster evidence
discovery while preserving canonical direct scans as the default behavior. The
index is a disposable read model built from `build_catalog()`: it can be
deleted, ignored, or rebuilt without changing source artifacts.

The index metadata records its derived status, schema version, source artifact
root, record count, evidence-family counts, source fingerprint, builder version,
and `canonical_source: artifacts`. Record payloads keep the existing
`CatalogRecord` shape and portable repository-relative paths.

Supported loading modes are:

- `direct`: canonical artifact scan; default
- `index`: require a valid derived index
- `auto`: use a valid index when present, fall back to direct scan only when the
  index file is absent

Index-backed records are fed through the existing in-memory query functions, so
supported query filters must return the same results as direct scan. Missing
indexes are safe; stale, mismatched, corrupt, or incompatible indexes fail with
rebuild guidance rather than being silently trusted.

Example commands:

```powershell
python -m src.cli.catalog_index build --artifacts-root artifacts --output artifacts/catalog_index/catalog_index.sqlite
python -m src.cli.catalog_index validate --index artifacts/catalog_index/catalog_index.sqlite --artifacts-root artifacts
python -m src.cli.query_catalog --artifacts-root artifacts --index artifacts/catalog_index/catalog_index.sqlite --index-mode auto
```

Issue #405 does not add a remote metadata service, production search backend,
graph store, artifact-writer dependency, second registry, or second source of
truth.

## Standards-Based Lineage Export

Issue #406 adds deterministic local JSON exports over the existing explicit
catalog lineage model. The exporter builds one normalized graph from
`CatalogRecord` objects and `LineageEdge` objects, then renders:

- OpenLineage-style JSON for run/dataset-oriented interchange
- PROV-style JSON for provenance-oriented interchange

Catalog records export as record/entity nodes. Existing
`manifest_declares_artifact` edges also expose the already-declared artifact as
an artifact/entity node so emitted relations remain closed. Every exported
relationship preserves the original StratLake `edge_type`, source ids, target
ids, and relationship metadata instead of pretending the external style is a
perfect semantic match.

The `prov` format is intentionally PROV-style local JSON, not a formal W3C PROV
conformance implementation. It renders relations with a conservative
`wasDerivedFrom`-style mapping, while the original StratLake `edge_type`
remains the authoritative semantic label. Strict PROV conformance belongs in a
separately scoped future issue.

Exports may cover the full graph or a selected run's direct one-hop
neighborhood. They are deterministic, portable, local JSON only, and derived
from explicit lineage already present in the catalog layer. Direct scan remains
canonical; Issue #405 index-backed loading may accelerate record loading, but
it does not become a lineage source of truth.

The selected-run neighborhood is intentionally non-recursive: it includes the
selected catalog record, directly connected source/target records, and directly
connected emitted artifact nodes only. It does not expand recursively across
multi-hop graph paths.

The portability validator currently rejects URI-like strings as well as absolute
local paths. That strictness is deliberate for local artifact exports: it helps
prevent accidental path leakage, and external URL metadata is not yet a
supported lineage-export field. A future URL allowlist, if ever needed, should
be handled separately with explicit tests.

Example commands:

```powershell
python -m src.cli.export_catalog_lineage --artifacts-root artifacts --format openlineage --output artifacts/lineage/openlineage.json
python -m src.cli.export_catalog_lineage --artifacts-root artifacts --format prov --selected-run-id strategy_000 --output artifacts/lineage/strategy_000_prov.json
```

Issue #406 does not add a graph database, remote lineage backend, inferred
unsupported relationships, artifact mutation, or a second registry.

## Dataset And Feature Lineage Fingerprints

Issue #407 adds optional deterministic dataset and feature lineage metadata for
artifacts that already carry explicit provenance. The contract is intentionally
small and backward-compatible:

- `dataset_lineage`: logical dataset id, role, portable dataset path, contract
  version, schema/partition/source fingerprints, row and symbol counts,
  timeframe, and date bounds
- `feature_lineage`: feature group names, column count, feature-column/schema
  fingerprints, contract version, and build-config fingerprint

Fingerprints use stable SHA-256 hashing over sorted JSON payloads, sorted column
lists, sorted schema entries, and POSIX-style portable paths. They exclude
wall-clock timestamps and local absolute paths.

Existing feature metadata summaries now emit these blocks where dataset paths
are explicit. Catalog records preserve explicit `dataset_lineage` and
`feature_lineage` blocks found in manifests, summaries, or registry records;
derived indexes serialize them automatically; lineage exports include them as
record facets without creating new inferred edges.

Issue #407 does not add a remote data catalog, migration framework, live-data
system, inferred lineage layer, replacement feature contract, or second source
of truth. Artifacts that do not contain the new optional blocks remain readable.

## CLI And API Ergonomics

Issue #408 adds shared workflow helpers for the common M36 user journeys:

- `load_catalog_for_workflow(...)`
- `build_lineage_export_for_workflow(...)`
- `build_evidence_view_for_workflow(...)`

These helpers compose the existing direct-scan/index loader, lineage extractor,
lineage exporter, and evidence explorer instead of duplicating behavior between
CLI commands, notebooks, or pipeline wrappers. Direct scan remains the default.
`index` mode requires a valid derived index, while `auto` uses a valid index when
present and falls back only when the index file is absent.

The CLI surface is now aligned around the same path and index options where they
apply:

```powershell
python -m src.cli.catalog_index build --artifacts-root artifacts --output artifacts/catalog_index/catalog_index.sqlite
python -m src.cli.query_catalog --artifacts-root artifacts --index artifacts/catalog_index/catalog_index.sqlite --index-mode auto --format json
python -m src.cli.export_catalog_lineage --artifacts-root artifacts --index artifacts/catalog_index/catalog_index.sqlite --index-mode auto --format openlineage --output artifacts/lineage/openlineage.json
python -m src.cli.explore_catalog_evidence --artifacts-root artifacts --index artifacts/catalog_index/catalog_index.sqlite --index-mode auto --run-id strategy_000 --format json
```

Notebook and wrapper callers should prefer the shared helpers when they need to
load records, build evidence views, or export lineage JSON. The helpers are thin
composition surfaces, not a second implementation path.

## Deterministic Validation

Issue #409 adds a focused end-to-end validation layer for the combined M36
evidence stack. It proves:

- direct scan, derived-index, auto-mode, and shared-helper record parity
- derived-index deletion/rebuild disposability without canonical artifact mutation
- deterministic OpenLineage-style and PROV-style export structure
- preservation of dataset/feature lineage metadata across direct records,
  index-loaded records, exports, and helper surfaces
- CLI/API parity for build, query, export, and evidence exploration workflows
- portable, read-only outputs with no absolute machine paths or URI leakage
- continued release/workflow hardening assumptions from Issues #402 and #403

The validation suite keeps timings out of deterministic assertions and relies on
the existing synthetic catalog tree rather than live data or external services.

## Release Notes Semantics

Human milestone release notes live in milestone docs such as this file. The
Release workflow renders deterministic GitHub Release body text from the pushed
tag name and validation steps. Both are release evidence, but neither changes
catalog records, artifact contracts, package publication scope, promotion
outcomes, or governance decisions.

## Further Reading

- `docs/m36_release_validation_checklist.md`
- `docs/m35_release_notes.md`
- `docs/m35_release_validation_checklist.md`
- `.github/workflows/milestone_validation.yml`
- `.github/workflows/release.yml`
- `pyproject.toml`
