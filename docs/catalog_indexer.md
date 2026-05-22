# Catalog Indexer — M29 Read-Only Artifact Scanner

## Overview

`src/catalog/indexer.py` implements a deterministic, read-only offline scanner
that builds normalized in-memory catalog records from existing StratLake
provenance sources.

It is the first concrete layer of the M29 Unified Research Catalog. It observes
existing outputs — registries, manifests, marker files, metrics, and QA
summaries — and produces normalized `CatalogRecord` and `ArtifactRecord`
instances without modifying anything.

---

## Read-Only Guarantee

The indexer **never** writes, modifies, deletes, moves, locks, or appends to
any source artifact, registry, manifest, or marker file.

It reads files using standard Python `open`/`json.loads`. No write calls are
made anywhere in `src/catalog/`.

---

## Known Artifact Roots

The indexer scans the following directory families under `artifacts/`:

| Family | Run Type |
| --- | --- |
| `strategies/<run_id>/` | `strategy` |
| `alpha/<run_id>/` | `alpha_evaluation` |
| `portfolios/<run_id>/` | `portfolio` |
| `comparisons/<run_id>/` | `comparison` |
| `pipelines/<run_id>/` | `pipeline` |
| `qa/<bundle_id>/` | `qa` |
| `reviews/<review_id>/` | `review` |
| `candidate_selection/<run_id>/` | `candidate_selection` |
| `regime_stress_tests/<run_id>/` | `regime_stress_test` |
| `robustness/<report_id>/` | `robustness_bundle` |
| `promotion_governance/<report_id>/` | `governance_bundle` |
| `milestone_validation/<bundle_id>/` | `milestone_validation_bundle` |
| `release_validation/<release_id>/` | `release_validation_artifact` |
| `corporate_actions/<run_id>/` | `corporate_action_event_dataset` |
| `benchmark_pack_*/<run_id>/` | `benchmark_pack` |

A directory is treated as an artifact root if it contains at least one of:

```
manifest.json      metrics.json       alpha_metrics.json
metrics_readiness.json                summary.json
qa_summary.json    _SUCCESS.json      _FAILED.json
_RUNNING.json      checkpoint.json    scenario_catalog.json
decision_log.json
robustness_summary.json              promotion_governance_summary.json
consistency_validation.json          release_validation.json
release_validation_summary.json
```

---

## Registry Paths

The following JSONL registry files are read (read-only):

| Path | Run Type |
| --- | --- |
| `artifacts/strategies/registry.jsonl` | `strategy` |
| `artifacts/alpha/registry.jsonl` | `alpha_evaluation` |
| `artifacts/portfolios/registry.jsonl` | `portfolio` |
| `artifacts/registry/portfolios.jsonl` | `portfolio_template` |

Registry entries are indexed by `run_id` and used to enrich matching artifact
roots with metadata such as `strategy_name`, `timeframe`, `start_ts`, `end_ts`,
`review`, and `promotion_status`.

**Note:** `artifacts/registry/portfolios.jsonl` is a portfolio template registry
(versioned templates), not a per-run artifact directory index.

**`portfolio_template` records are metadata/template records** derived from
`artifacts/registry/portfolios.jsonl`. They represent versioned portfolio
template definitions, not completed research experiment runs. They should not
be treated as completed backtest, alpha, portfolio, or campaign outputs by
downstream query defaults. Their `status` will typically be `registry_only`
because template entries have no corresponding per-run artifact directory.

**Note:** Registry entries without a matching on-disk artifact root produce a
`CatalogRecord` with `status="registry_only"` and a
`registry_entry_no_artifact_root` validation warning.

---

## Marker File Precedence

Run lifecycle status is determined by marker files using the following
precedence (most authoritative first):

| Marker | Status | Priority |
| --- | --- | --- |
| `_FAILED.json` | `failed` | Highest |
| `_SUCCESS.json` | `completed` | Medium |
| `_RUNNING.json` | `running` | Lowest |
| (none) | `unknown` | — |

**Rationale:** failure markers must not be hidden by stale success or running
files left from an interrupted run.

If multiple markers exist simultaneously (e.g., after a crash during cleanup),
`_FAILED.json` always wins.

---

## Deterministic Output Contract

## Canonicality Envelope v1

M37 adds a deterministic `canonicality` envelope to newly generated derived
catalog surfaces. Canonical artifacts remain the source of truth: registries,
manifests, markers, summaries, validation bundles, and governance artifacts are
authoritative; the SQLite index is a disposable read model only.

New derived index metadata includes:

- `schema_version: canonicality.v1`
- the canonical artifact-tree root and portable source paths
- repository-relative path validation that rejects absolute, URI-like, and
  parent-traversal authority paths
- a deterministic source fingerprint
- `derived_class: sqlite_read_model`
- `rebuildable: true`
- `non_authoritative: true`
- `write_back_forbidden: true`
- `stale_if_source_changes: true`
- resolver guidance to reopen canonical manifests/registries before
  decision-sensitive use

Legacy M36 indexes without the envelope remain readable and are surfaced with a
`legacy_no_envelope` compatibility status rather than being promoted to
authority.

New M37 index builds default to
`artifacts/_derived/catalog_index/catalog_index.sqlite`. Explicit legacy M36
paths such as `artifacts/catalog_index/catalog_index.sqlite` are still accepted,
but `_derived` is reserved for disposable read models and is never scanned as a
canonical artifact family. Load-source metadata distinguishes direct scans from
validated index reads and shows how `auto` mode resolved. Its `index_path` field
is portable repository-relative metadata and rejects absolute, URI-like,
`file://`, and parent-traversal paths.

Index-backed records are suitable for discovery and filtering. Before
decision-sensitive use, call `resolve_canonical_record(...)` to reopen the
declared canonical registry, manifest, marker, and source files. Resolver
results are deterministic, read-only, and fail safely when source files are
missing, non-portable, or outside the resolved artifacts root.

M37 architecture guardrails keep that boundary executable. Derived indexes may
accelerate reads and validation, but they are forbidden as authority inputs for
writers, registries, promotion/governance decisions, release readiness, or
canonical catalog construction. Tests verify `_derived` is excluded from direct
scans and that deleting or rebuilding derived indexes preserves canonical
catalog identity and source files.

### `catalog_id`

```
catalog_id = sha256(run_id + "|" + artifact_root)[:16]
```

Stable for a given `(run_id, artifact_root)` pair across all calls.

### `artifact_id`

```
artifact_id = sha256(catalog_id + "|" + relative_path)[:16]
```

### Sort Order

`CatalogRecord` list is sorted by `(run_type, run_id or "", artifact_root)`.

`ArtifactRecord` list is sorted by `relative_path`.

Given the same artifact tree, output order and IDs are identical across
repeated calls.

---

## Validation Status

Each `CatalogRecord` includes a `CatalogValidationStatus` with:

| Field | Description |
| --- | --- |
| `catalog_status` | Overall status: `completed`, `failed`, `running`, `indexed`, `discovered`, `registry_only`, `error` |
| `marker_status` | `present`, `failed`, `running`, `missing` |
| `manifest_status` | `present` or `missing` |
| `artifact_status` | `ok` or `incomplete` (if declared artifacts are missing) |
| `qa_status` | Value from `qa_summary.json` if present |
| `validation_warnings` | List of warning codes |
| `validation_errors` | List of error codes |

## M35 Evidence Record Families

M35 extends the M29 model with optional evidence fields on `CatalogRecord`.
These fields are populated only from existing artifacts and remain absent or
false when evidence is missing:

| Field | Source |
| --- | --- |
| `record_family` | One of `robustness_bundle`, `governance_bundle`, `milestone_validation_bundle`, or `release_validation_artifact` for evidence roots. |
| `robustness_status` | `robustness_summary.json`. |
| `wfe_status` | `walk_forward_efficiency.csv`. |
| `sample_size_status`, `trade_count_status` | `sample_size_validation.json`. |
| `sensitivity_status`, `fragility_status` | `sensitivity_summary.csv`. |
| `multiple_testing_status` | `multiple_testing_summary.json`. |
| `temporal_validation_status` | `leakage_validation.json` and temporal-validation findings. |
| `governance_status`, `promotion_review_status` | Existing governance or validation bundle summaries. |
| `validation_readiness_present` | Milestone validation bundle summary presence. |
| `release_validation_present` | Release-validation artifact presence. |

The extension is read-only. It does not create a registry, database, persistent
cache, search backend, policy simulation layer, or governance enforcement path.
See
[`docs/m35_evidence_catalog_foundation.md`](m35_evidence_catalog_foundation.md)
for the source-of-truth mapping and missing-evidence semantics.

## M40 Dividend Event Evidence

M40 adds read-only discovery for local corporate-actions dividend import
artifacts under `artifacts/corporate_actions/<run_id>/`. These records are
discovered by direct scan; there is no separate corporate-actions registry and
no write-back path from the catalog.

Dividend records use:

| Field | Value |
| --- | --- |
| `record_family` | `corporate_action_event_dataset` |
| `run_type` | `corporate_action_event_dataset` |
| `artifact_type` | `corporate_action_event_dataset` |
| `evidence_type` | `dividend_events` |
| `source_domain` | `corporate_actions` |
| `event_domain` | `dividends` |
| `schema_version` | `corporate_actions.dividends.v1` |
| `canonicality` | `canonical_import_artifact` |

The catalog metadata points back to the canonical curated dataset root and the
import artifact bundle, including `manifest.json`, `summary.json`,
`qa_summary.json`, `schema_contract.json`, and `source_provenance.json`.
Catalog records are discovery views only. The curated dividend event dataset
and import artifact bundle remain the source of truth.

Dividend evidence remains separate from OHLCV bars, adjusted prices, strategy
outputs, alpha outputs, portfolio outputs, promotion decisions, and backtest
returns. Direct scan remains available and canonical; derived catalog indexes
remain disposable, rebuildable, and non-authoritative.

See [Corporate Actions Dividend Evidence](corporate_actions_dividend_evidence.md)
and [Corporate Actions Event Contracts](corporate_actions_event_contracts.md).

### Warning Codes

| Code | Meaning |
| --- | --- |
| `artifact_root_no_registry_entry` | Artifact directory found but no matching registry entry |
| `registry_entry_no_artifact_root` | Registry entry found but artifact directory does not exist |
| `manifest_missing` | No `manifest.json` found in the artifact root |
| `manifest_artifact_missing:<rel>` | A file declared in the manifest is absent on disk |
| `undeclared_artifact:<rel>` | A file found on disk was not declared in the manifest |
| `failed_marker_present` | `_FAILED.json` exists |
| `running_marker_present` | `_RUNNING.json` exists |

---

## Limitations and Non-Goals

- **No persistent query backend.** Query APIs and CLIs operate over in-memory
  catalog records derived from existing artifacts.
- **No persistent lineage graph.** Lineage APIs derive in-memory edges; they do
  not create a graph database, graph cache, or canonical lineage store.
- **No persistent cache.** There is no on-disk export of catalog results. If a
  cache is added in a future issue, it must be a derived, non-canonical artifact.
- **No checksum computation.** `checksum_optional` is always `None` in this
  implementation.
- **No notebook, dashboard, or database backend.**
- **No new registry writer or execution wrapper.**
- **`artifacts/registry/portfolios.jsonl`** is indexed for metadata enrichment
  only. Portfolio template entries have no per-run artifact directory and produce
  a `registry_only` record if no artifact root is found.
- Deeply nested artifact structures (e.g., multi-level campaign hierarchies) may
  require recursive discovery in a future issue.
- Malformed JSON in metrics or summary files is silently skipped with a debug
  log entry.

For the M35 evidence catalog release overview, see
[`docs/m35_evidence_catalog_foundation.md`](m35_evidence_catalog_foundation.md)
and [`docs/m35_release_notes.md`](m35_release_notes.md).

---

## Public API

```python
from src.catalog import build_catalog, build_artifact_records, CatalogRecord, ArtifactRecord

# Scan all artifact roots
records: list[CatalogRecord] = build_catalog("artifacts", repo_root=".")

# Build artifact-level records for one catalog entry
artifact_records: list[ArtifactRecord] = build_artifact_records(records[0], repo_root=Path("."))

# Serialize to dict
d = records[0].to_dict()
```

### `build_catalog(artifacts_root, *, repo_root)`

Scans all known artifact families and returns a sorted, deterministic list of
`CatalogRecord` instances.

### `build_catalog_record(artifact_root, *, repo_root, registry_index)`

Builds a single `CatalogRecord` for a given artifact root directory.

### `build_artifact_records(record, *, repo_root)`

Builds `ArtifactRecord` entries for all files under the record's artifact root.

### `load_json_file(path)`

Returns the parsed JSON dict for a file path, or `None` on any failure.
