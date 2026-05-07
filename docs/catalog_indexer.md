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
| `benchmark_pack_*/<run_id>/` | `benchmark_pack` |

A directory is treated as an artifact root if it contains at least one of:

```
manifest.json      metrics.json       alpha_metrics.json
metrics_readiness.json                summary.json
qa_summary.json    _SUCCESS.json      _FAILED.json
_RUNNING.json      checkpoint.json    scenario_catalog.json
decision_log.json
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

- **No query CLI.** Use Python directly; a catalog query CLI is deferred.
- **No lineage graph.** Parent/child relationship extraction is deferred.
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
