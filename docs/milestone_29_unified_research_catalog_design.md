# Milestone 29 Unified Research Catalog Design

## Purpose

This document specifies the design of the M29 Unified Research Catalog — a
read-only index layer that aggregates and normalizes provenance metadata from
existing StratLake artifact roots, registries, manifests, QA summaries, marker
files, and validation bundles into a queryable in-memory catalog structure.

M29 does not replace any existing registry, manifest, or execution surface.
It reads existing outputs and presents them through a unified schema without
modifying the source artifacts.

This is a design and specification document. Implementation details (indexer,
query engine, lineage graph, catalog CLI) are deferred to follow-on issues.

---

## M28 Architectural Principle

> One execution system, multiple entry points, no duplicated workflow logic.

M29 extends this principle to provenance surfacing:

> One unified catalog view, multiple provenance sources, no duplicated registry
> behavior.

The catalog layer is a consumer of existing outputs, not a producer of new
execution state. It observes registries, artifact roots, manifests, and marker
files as source-of-truth inputs and produces a normalized in-memory view on
demand.

If a future issue adds a persisted cache or export of catalog results, that
file must be treated as a derived convenience artifact. It must not become the
canonical source of run identity, lifecycle status, lineage, or artifact
inventory.

---

## Why This Is Not Another Registry

Prior milestones already established experiment registries, portfolio template
registries, alpha model catalogs, and artifact manifests. M29 must not create a
parallel registry that competes with or duplicates those surfaces.

Key distinctions:

| Property | Existing Registries | M29 Catalog |
| --- | --- | --- |
| Written by workflows | Yes — appended at run time | No — read-only observer |
| Source of truth for run identity | Yes | No |
| Canonically deduplicates runs | Yes (registry locking) | No — deduplication is by source |
| Persisted as primary artifact | Yes (`registry.jsonl`, `manifest.json`) | No — computed on demand |
| Defines execution metadata | Yes | No — reads existing fields |
| Adds new workflow logic | Yes | No |

The catalog does not write registry entries, generate run IDs, or add execution
hooks. If a registry entry exists, the catalog reflects it. If it does not
exist, the catalog infers from available artifact structure and marks missing
metadata as null rather than guessing.

---

## Existing Provenance Surfaces

The following surfaces are the catalog's input sources. Each is already produced
by existing StratLake workflows and must be treated as source-of-truth inputs.

### Registry Surfaces

| Surface | Path Pattern | Written By |
| --- | --- | --- |
| Strategy experiment registry | `artifacts/strategies/registry.jsonl` | `src/research/registry.py` via `upsert_registry_entry`; individual run artifacts live under `artifacts/strategies/<run_id>/` |
| Portfolio template registry | `artifacts/registry/portfolios.jsonl` | `src/portfolio/registry.py` via `register_portfolio_template` |
| Alpha evaluation registry | `artifacts/alpha/registry.jsonl` | `src/research/alpha_eval/registry.py` via `upsert_registry_entry`; individual alpha eval artifacts live under `artifacts/alpha/<run_id>/` |
| Alpha model catalog (in-memory) | `src/research/alpha/catalog.py` | Loaded at import time from `configs/alphas.yml` |

### Manifest and Inventory Files

| File | Location Pattern | Description |
| --- | --- | --- |
| `manifest.json` | Per run artifact root | Declares expected and produced artifacts with relative paths |
| `inventory.json` | Benchmark pack and campaign roots | Batch artifact inventory |
| `batch_plan.json` | Benchmark pack roots | Deterministic batch specification |
| `checkpoint.json` | Campaign and benchmark pack roots | Checkpoint/resume state |
| `scenario_catalog.json` | Market simulation and campaign roots | Scenario identity and configuration records |

### Metrics and Summary Files

| File | Location Pattern | Description |
| --- | --- | --- |
| `metrics.json` | Strategy artifact roots | Core performance metrics |
| `alpha_metrics.json` | Alpha artifact roots | Alpha model evaluation metrics |
| `aggregate_metrics.json` | Portfolio and campaign roots | Aggregated multi-run metrics |
| `pipeline_metrics.json` | Pipeline artifact roots | Pipeline execution metrics |
| `summary.json` | Most artifact roots | Human-readable summary for the run |
| `training_summary.json` | Alpha training roots | Training diagnostics |
| `qa_summary.json` | Strategy and portfolio roots | QA gate status, diagnostics, and signal checks |
| `signal_diagnostics.json` | Strategy roots | Signal-level diagnostics |
| `benchmark_matrix_summary` | Benchmark pack roots | Cross-scenario benchmark comparison |
| `decision_log.json` | Promotion and review roots | Promotion gate decision records |

### Promotion and Review Files

| File | Location Pattern | Description |
| --- | --- | --- |
| `promotion_status.json` (or embedded in registry) | Strategy/regime artifact roots | Promotion gate pass/fail summary |
| `review_status` (registry fields) | `registry.jsonl` entries | `status`, `promotion_status`, `decision_reason` fields |
| Regime review pack artifacts | `artifacts/reviews/<review_id>/` | Evidence index, decision log, promotion summary |
| Candidate selection artifacts | `artifacts/candidate_selection/<run_id>/` | Candidate selection summary, criteria, leaderboard |

### Marker Files

| File | Description |
| --- | --- |
| `_RUNNING.json` | Written at run start by `mark_run_started`; contains `run_type` and timestamp |
| `_SUCCESS.json` | Written on clean completion by `mark_run_completed`; contains status metadata |
| `_FAILED.json` | Written on failure by `mark_run_failed`; contains error metadata |

These markers are defined in `src/artifacts/safety.py` and serve as the
authoritative lifecycle state indicators for any artifact root.

### QA and Validation Artifacts

| Surface | Path Pattern | Description |
| --- | --- | --- |
| Milestone validation bundle | `artifacts/qa/` | Docs lint, deterministic rerun, and cross-layer validation reports |
| Cross-layer validation report | `artifacts/qa/<bundle_id>/checks/cross_layer_validation_report.json` | Normalized artifact comparison across CLI, API, notebook, and orchestrator layers |
| Deterministic rerun report | `artifacts/qa/<bundle_id>/checks/deterministic_rerun.json` | Two-run comparison for summary hash equality |
| Docs path lint report | `artifacts/qa/<bundle_id>/checks/docs_path_lint.json` | Absolute path guard check |

### Artifact Directory Structure

The top-level `artifacts/` directory contains well-known run type subdirectories:

```text
artifacts/
  strategies/         # Strategy and walk-forward experiment outputs
  alpha/              # Alpha model training and evaluation outputs
  alpha_comparisons/  # Cross-alpha comparison outputs
  portfolios/         # Portfolio walk-forward outputs
  comparisons/        # Strategy comparison leaderboard outputs
  pipelines/          # Pipeline run outputs
  feature_runs/       # Feature build outputs
  candidate_selection/
  regime_stress_tests/
  registry/           # Shared portfolio template registry
  reviews/            # Regime review pack outputs
  qa/                 # Validation and QA bundle outputs
  benchmark_pack_*/   # Benchmark pack run outputs
```

### Notebook and Pipeline Output Surfaces

Pipeline runs produce `manifest.json`, `metrics.json`, `summary.json`, and
optionally `checkpoint.json` under `artifacts/pipelines/<pipeline_run_id>/`.

Research campaigns produce multi-stage artifact roots under the configured
`campaign_artifacts_root`, with checkpoint state, scenario orchestration logs,
milestone reports, and scenario-level sub-roots.

Market simulation runs produce `scenario_catalog.json`, per-scenario artifact
roots, simulation summaries, and optionally policy failure leaderboards.

Notebook-style runs through `src.execution` produce the same artifact contracts
as CLI runs; they share the same output paths and manifests.

---

## Prior Milestone Capabilities Already Implemented

The following capabilities are already in the repository. M29 must not
reimplement or duplicate them.

| Capability | Source |
| --- | --- |
| Strategy experiment registry append and deduplication | `src/research/registry.py` — `upsert_registry_entry` with file locking |
| Registry locking and collision prevention | `src/research/registry.py` — `_registry_lock` context manager |
| Portfolio registry integration | `src/portfolio/registry.py` — `register_portfolio_template` |
| Alpha evaluation registry support | `src/research/alpha/registry.py`, `src/research/alpha/catalog.py` |
| Deterministic run IDs | `src/research/registry.py` — `stable_timestamp_from_run_id` |
| Artifact manifests | Per-run `manifest.json` produced by all major workflow surfaces |
| QA summaries | `src/research/strategy_qa.py`, `src/portfolio/walk_forward.py` — `generate_strategy_qa_summary`, `generate_portfolio_qa_summary` |
| Signal diagnostics | `src/research/signal_diagnostics.py` |
| Cross-artifact consistency validation | `src/research/consistency.py` |
| Deterministic rerun validation | `src/validation/deterministic_rerun.py` |
| Cross-layer validation | `src/validation/cross_layer.py` |
| Promotion and review metadata | `src/research/registry.py` — `build_review_metadata`, `src/research/promotion.py` |
| Campaign and scenario metadata | `src/execution/orchestration.py`, `src/research/benchmark_pack.py` |
| Checkpoint and resume | `src/research/campaign_checkpoint.py`, `src/research/benchmark_pack.py` |
| Milestone validation bundles | `src/validation/milestone_bundle.py` |
| M28 marker files and artifact safety | `src/artifacts/safety.py` — `_RUNNING.json`, `_SUCCESS.json`, `_FAILED.json` |
| Artifact collision and idempotency policy | `src/artifacts/safety.py` — `ensure_output_root_available` |
| ExecutionResult notebook inspection API | `src/execution/result.py` |
| Regime benchmark and promotion surfaces | `src/research/regime_benchmark_pack.py`, `src/research/regime_promotion_gates.py` |
| Regime review packs | `src/research/regime_review_pack.py` |
| Market simulation scenario catalogs | `src/research/market_simulation/` |

---

## Source-of-Truth Inputs

The catalog reads from these sources only. It does not write to them.

1. **Marker files** (`_RUNNING.json`, `_SUCCESS.json`, `_FAILED.json`) — authoritative lifecycle state per artifact root.
2. **Registry files** (`registry.jsonl`, `portfolios.jsonl`) — canonical run identity, review metadata, and promotion status.
3. **Manifest files** (`manifest.json`) — declared artifact inventory with relative paths.
4. **Metrics and summary files** (`metrics.json`, `alpha_metrics.json`, `summary.json`, `qa_summary.json`, etc.) — evaluation and diagnostics data.
5. **Checkpoint and scenario files** (`checkpoint.json`, `scenario_catalog.json`) — campaign and simulation metadata.
6. **Artifact directory structure** — fallback discovery when explicit metadata is absent; used to infer artifact root type from directory name conventions only.

---

## Unified Catalog Boundary

The catalog boundary is defined as follows:

**In scope:**
- All artifact roots under `artifacts/` that contain a recognized run type marker, registry entry, or manifest.
- All entries in known registry files (`registry.jsonl`, `portfolios.jsonl`).
- Pipeline, campaign, benchmark pack, and validation bundle outputs.
- Alpha, strategy, portfolio, comparison, regime, and market simulation artifact roots.

**Out of scope:**
- Feature dataset builds (`artifacts/features/`, `artifacts/feature_runs/`) — these are input surfaces, not research experiment outputs.
- Raw market data under `data/`.
- Config files under `configs/` — these define inputs, not outputs.
- In-memory alpha model registry (`src/research/alpha/registry.py`) — this is a runtime class registry, not a persisted artifact.
- Temporary files, `.tmp` partial writes, and `.lock` files.

---

## Proposed Catalog Record Schema

A catalog record represents one logical research run or experiment, aggregated
from its available provenance sources. All fields are optional unless marked
required.

```json
{
  "catalog_id":              "<str: deterministic hash of run_id + artifact_root>",
  "run_id":                  "<str: from registry or manifest — required if available>",
  "run_type":                "<str: strategy | portfolio | alpha_evaluation | pipeline | campaign | benchmark_pack | comparison | regime_benchmark | regime_review | regime_candidate_selection | market_simulation | milestone_validation | portfolio_template | unknown>",
  "status":                  "<str: completed | failed | running | incomplete | unknown>",
  "artifact_root":           "<str: relative path from repo root — required>",
  "source_registry_path":    "<str: relative path to registry.jsonl entry source | null>",
  "source_manifest_path":    "<str: relative path to manifest.json | null>",
  "source_marker_path":      "<str: relative path to _SUCCESS.json/_FAILED.json/_RUNNING.json | null>",
  "created_at":              "<str: ISO-8601 timestamp from marker or registry | null>",
  "timeframe":               "<str: e.g. '2022-01-01/2023-12-31' | null>",
  "start_ts":                "<str: ISO-8601 start date | null>",
  "end_ts":                  "<str: ISO-8601 end date | null>",
  "strategy_name":           "<str | null>",
  "portfolio_name":          "<str | null>",
  "allocator_name":          "<str | null>",
  "alpha_model_name":        "<str | null>",
  "regime_method":           "<str | null>",
  "campaign_id":             "<str | null>",
  "scenario_id":             "<str | null>",
  "metrics_summary":         "<dict: key scalar performance metrics | null>",
  "qa_status":               "<str: passed | failed | warning | not_run | null>",
  "review_status":           "<str: candidate | promoted | rejected | needs_review | null>",
  "promotion_status":        "<str: promoted | not_promoted | pending | null>",
  "tags":                    "<list[str]: arbitrary string labels | []>",
  "source_files":            "<list[str]: relative paths of all source files used to build this record>",
  "metadata":                "<dict: pass-through of extra fields from source artifacts | {}>"
}
```

**Optionality notes:**
- `run_id` is required when a registry entry or manifest exists; otherwise null.
- `strategy_name`, `portfolio_name`, `allocator_name`, `alpha_model_name`, `regime_method` are populated only for matching `run_type` values.
- `campaign_id`, `scenario_id` are populated for campaign children and scenario runs.
- `metrics_summary` contains only scalar-valued keys from `metrics.json` or `alpha_metrics.json`; complex nested objects are not inlined.
- `tags` may be populated from registry `metadata.tags`, config labels, or inferred run type.
- `catalog_id` is computed deterministically from `run_id` (if present) plus `artifact_root`.

---

## Proposed Artifact Record Schema

An artifact record represents one file declared in or discovered under an
artifact root. Artifact records link to their parent catalog record via
`catalog_id`.

```json
{
  "artifact_id":           "<str: deterministic hash of catalog_id + relative_path>",
  "catalog_id":            "<str: parent catalog record ID>",
  "run_id":                "<str | null>",
  "artifact_type":         "<str: manifest | metrics | alpha_metrics | summary | qa_summary | signal_diagnostics | checkpoint | scenario_catalog | registry | marker_running | marker_success | marker_failed | decision_log | comparison_csv | comparison_json | plot | report | unknown>",
  "path":                  "<str: relative path from repo root>",
  "relative_path":         "<str: path relative to artifact_root>",
  "filename":              "<str>",
  "extension":             "<str: e.g. '.json', '.csv', '.png'>",
  "declared_in_manifest":  "<bool: true if path appears in manifest.json | false>",
  "exists":                "<bool: whether the file exists on disk at index time>",
  "size_bytes":            "<int | null>",
  "modified_time":         "<str: ISO-8601 mtime | null>",
  "checksum_optional":     "<str: sha256 hex digest — only populated if explicitly requested | null>",
  "schema_hint":           "<str: known schema name for this file type | null>",
  "metadata":              "<dict | {}>"
}
```

**Notes:**
- `artifact_type` is inferred from filename and extension conventions defined in the catalog layer.
- `declared_in_manifest` indicates whether the file is listed in the run's `manifest.json`; files discovered by directory scan that are not in the manifest are flagged.
- `checksum_optional` is not computed by default to keep indexing fast; it is only populated when explicitly requested by the caller.
- Marker files (`_RUNNING.json`, `_SUCCESS.json`, `_FAILED.json`) are represented as artifact records with `artifact_type` = `marker_running`, `marker_success`, or `marker_failed` respectively.

---

## Proposed Lineage Edge Schema

A lineage edge represents a directional relationship between two catalog records.
Edges are derived from existing metadata fields such as `component_run_ids`,
`campaign_id`, `scenario_id`, `comparison_id`, and manifest declarations.

```json
{
  "edge_id":              "<str: deterministic hash of source_catalog_id + target_catalog_id + edge_type>",
  "source_catalog_id":    "<str: upstream catalog record>",
  "target_catalog_id":    "<str: downstream catalog record>",
  "source_run_id":        "<str | null>",
  "target_run_id":        "<str | null>",
  "edge_type":            "<str: see edge type table below>",
  "relationship_source":  "<str: field or file path that declared this relationship>",
  "relationship_path":    "<str: relative path to source file | null>",
  "metadata":             "<dict | {}>"
}
```

**Edge types:**

| Edge Type | Description | Derived From |
| --- | --- | --- |
| `portfolio_component` | Portfolio run depends on strategy component run | `component_run_ids` in portfolio manifest or summary |
| `campaign_child` | Campaign stage or scenario is a child of a campaign run | `campaign_id` field in stage artifact roots |
| `scenario_child` | Scenario run is a child of a benchmark pack or simulation run | `scenario_id` or batch plan entries |
| `comparison_member` | Strategy or alpha run is a member of a comparison leaderboard | `comparison_id` or comparison result JSON |
| `benchmark_member` | Run is a member of a benchmark pack batch | Benchmark pack inventory or checkpoint |
| `manifest_declares_artifact` | Manifest declares a relationship to an artifact file | `manifest.json` inventory entries |
| `validation_references_run` | Validation report references a specific run | Deterministic rerun or cross-layer validation targets |
| `notebook_references_artifact` | Notebook output references an artifact root | Notebook output metadata (if present) |
| `pipeline_wraps_execution` | Pipeline run wraps an underlying strategy or alpha run | Pipeline manifest or stage metadata |

**Notes:**
- Edges are derived only from existing metadata. No new relationship fields are introduced.
- An edge is only created when both the `source_catalog_id` and `target_catalog_id` resolve to known catalog records.
- Orphan references (edges where one side cannot be resolved) are recorded as validation warnings, not errors.

---

## Proposed Validation / Status Schema

Each catalog record carries a validation status summary derived from all
available provenance sources.

```json
{
  "catalog_status":      "<str: valid | warning | error | unknown>",
  "marker_status":       "<str: completed | failed | running | incomplete | missing>",
  "manifest_status":     "<str: present_complete | present_incomplete | missing>",
  "artifact_status":     "<str: all_present | some_missing | all_missing | not_checked>",
  "qa_status":           "<str: passed | failed | warning | not_run | null>",
  "validation_errors":   "<list[str]: blocking inconsistencies>",
  "validation_warnings": "<list[str]: non-blocking anomalies>"
}
```

**Status derivation rules:**

| Condition | Effect |
| --- | --- |
| `_SUCCESS.json` present | `marker_status = completed` |
| `_FAILED.json` present | `marker_status = failed` |
| `_RUNNING.json` present, no success/failed | `marker_status = running` |
| No marker files present | `marker_status = missing` |
| `manifest.json` present; all declared artifacts exist | `manifest_status = present_complete` |
| `manifest.json` present; some declared artifacts missing | `manifest_status = present_incomplete` |
| `manifest.json` absent | `manifest_status = missing` |
| Registry entry exists without artifact directory | `validation_warnings += ["registry_entry_no_artifact_root"]` |
| Artifact directory exists without registry entry | `validation_warnings += ["artifact_root_no_registry_entry"]` |
| Artifact declared in manifest but not on disk | `validation_warnings += ["manifest_artifact_missing: <path>"]` |
| Artifact on disk but not declared in manifest | `validation_warnings += ["undeclared_artifact: <path>"]` |
| `_RUNNING.json` present and run appears stale (no updates for threshold period) | `validation_warnings += ["stale_running_marker"]` |
| `marker_status = failed` with no `_FAILED.json` explanation | `validation_errors += ["failed_run_no_explanation"]` |
| `catalog_status = error` | One or more `validation_errors` present |
| `catalog_status = warning` | No errors but one or more `validation_warnings` |
| `catalog_status = valid` | No errors, no warnings, `marker_status = completed` or `manifest_status = present_complete` |
| `catalog_status = unknown` | Insufficient information to determine status |

---

## Source Precedence Rules

When multiple provenance sources provide conflicting or overlapping values for
the same catalog field, the following precedence order applies. Higher-numbered
sources override lower-numbered sources for the same field.

1. **Directory structure inference** — lowest precedence; used only when all
   explicit metadata is absent. Provides a fallback `run_type` guess from
   directory name conventions (e.g., `artifacts/strategies/` → `run_type = strategy`).
   Must not be used to infer `run_id`, dates, metrics, or review status.

2. **Metrics and summary files** — `metrics.json`, `alpha_metrics.json`,
   `summary.json` provide `metrics_summary` and derived evaluation fields
   (`start_ts`, `end_ts`, `timeframe`, `strategy_name`, `alpha_model_name`).

3. **QA files** — `qa_summary.json` provides `qa_status`. Overrides any
   `qa_status` inferred from metrics or summary files.

4. **Manifest files** — `manifest.json` provides the canonical artifact
   inventory and `run_id` when present. Overrides directory-scan artifact lists.

5. **Registry entries** — `registry.jsonl` and `portfolios.jsonl` entries
   provide canonical `run_id`, `run_type`, `created_at`, `review_status`,
   `promotion_status`, and `decision_reason`. Registry values override all
   inferred values for the same fields.

6. **M28 marker files** — `_SUCCESS.json`, `_FAILED.json`, `_RUNNING.json`
   provide authoritative `status` and lifecycle timestamps. Marker status
   overrides all inferred lifecycle status from other sources.

**Missing metadata:**
Fields that cannot be resolved from any source are represented as `null` in
the catalog record. They are never filled in with guesses. Missing required
fields (e.g., a run with no `run_id` and no manifest) are recorded as
`validation_warnings`.

---

## Conflict and Missing Metadata Handling

### Conflicting run_id values

If a registry entry and a manifest disagree on `run_id`, the registry entry
takes precedence (rule 5 above). The conflict is recorded as a
`validation_warning`.

### Missing registry entry for an artifact root

A catalog record is still created. `source_registry_path` is set to null.
`run_id` is populated from the manifest if available, otherwise null.
A `validation_warning` of `artifact_root_no_registry_entry` is recorded.

### Missing manifest for a registered run

A catalog record is created from the registry entry. `source_manifest_path`
is set to null. `manifest_status = missing`. A `validation_warning` of
`registered_run_no_manifest` is recorded.

### Duplicate registry entries

The catalog de-duplicates by `run_id` using the last-seen registry entry
semantics that match existing `upsert_registry_entry` behavior. No new
deduplication logic is added.

### Missing metrics files

`metrics_summary` is set to null. No inference is attempted from other
artifact files.

### Partially written artifact roots

If `_RUNNING.json` is present without a corresponding `_SUCCESS.json` or
`_FAILED.json`, the run is treated as `status = running` and
`catalog_status = warning`. The catalog does not attempt to determine whether
the run is still active.

---

## Anti-Duplication Rules

The following rules prevent M29 from duplicating prior milestone behavior:

1. **No new registry writer.** The catalog only reads `registry.jsonl` and
   `portfolios.jsonl`; it never appends to them.

2. **No new run ID generation.** Catalog IDs are deterministic hashes computed
   from existing `run_id` and `artifact_root` values. They are not registry
   run IDs.

3. **No new manifest writer.** The catalog reads `manifest.json` files; it
   does not produce them.

4. **No new execution wrappers.** The catalog does not call workflow functions
   or execution surfaces.

5. **No new QA logic.** QA status is read from `qa_summary.json`; the catalog
   does not re-evaluate QA gates.

6. **No new promotion logic.** Promotion status is read from registry entries
   and `decision_log.json`; the catalog does not re-evaluate promotion gates.

7. **No new lineage writers.** Lineage edges are derived from existing metadata
   fields; no new relationship files are written to artifact roots.

8. **No replacement for strategy registry.** The catalog is an aggregation view
   over `registry.jsonl`; it does not replace or modify registry behavior.

9. **No duplication of `ExecutionResult`.** The catalog does not reimplement
   notebook inspection helpers. It provides a different interface for offline
   bulk catalog queries.

---

## Non-Goals

The following are explicitly out of scope for M29:

- `src/catalog/indexer.py` — implementation deferred to follow-on issue.
- `src/catalog/lineage.py` — implementation deferred to follow-on issue.
- `src/catalog/query.py` — implementation deferred to follow-on issue.
- `src/catalog/validation.py` — implementation deferred to follow-on issue.
- CLI commands for catalog queries or lineage exploration.
- Notebook examples using the catalog.
- Dashboard, web UI, or database backend.
- New registry writer or new execution wrapper.
- New model logic, strategy logic, or workflow logic.
- Live data ingestion or external service integration.
- Scheduler or orchestration dependency.
- Any modification to existing `registry.jsonl` format.
- Any modification to existing `manifest.json` format.
- Any modification to `_RUNNING.json`, `_SUCCESS.json`, or `_FAILED.json` format.

---

## Follow-On Implementation Issues

The following issues are the expected next steps after this design is accepted.

| Issue | Title | Scope |
| --- | --- | --- |
| M29.1 | Catalog Indexer — Core Schema + JSONL Sources | Implement `src/catalog/indexer.py` reading `registry.jsonl`, `portfolios.jsonl`, `manifest.json`, marker files, and producing in-memory `CatalogRecord` objects per the schema defined here. |
| M29.2 | Catalog Indexer — Metrics and QA Sources | Extend indexer to read `metrics.json`, `alpha_metrics.json`, `summary.json`, `qa_summary.json`, and populate `metrics_summary` and `qa_status` fields. |
| M29.3 | Artifact Record Builder | Implement artifact record construction from manifest inventories and directory scans; populate `declared_in_manifest` and `exists` fields. |
| M29.4 | Lineage Edge Builder | Derive lineage edges from `component_run_ids`, `campaign_id`, `scenario_id`, comparison JSON, and benchmark pack inventories. |
| M29.5 | Catalog Validation Layer | Implement the validation/status schema derivation rules and produce `catalog_status`, `validation_errors`, and `validation_warnings` per record. |
| M29.6 | Catalog Query Surface | Implement a read-only query API over the in-memory catalog (`filter_by_run_type`, `filter_by_status`, `find_by_run_id`, `find_by_campaign_id`, `get_lineage_edges`). |
| M29.7 | Catalog CLI Entrypoint | Add `src/cli/run_catalog_index.py` and `src/execution/catalog.py` for CLI and notebook-friendly catalog access consistent with M23/M28 patterns. |
| M29.8 | Catalog Integration Tests | Tests confirming catalog records match expected schema for strategy, portfolio, alpha, campaign, and benchmark pack artifact roots in `artifacts/`. |

---

## Acceptance Checklist

- [ ] `docs/milestone_29_unified_research_catalog_design.md` is committed.
- [ ] Document contains all required sections listed in the issue spec.
- [ ] Document uses relative repository paths only; no absolute local paths.
- [ ] No claims of implemented indexer, query, or lineage code in this document.
- [ ] No new registry design proposed; existing `registry.jsonl` behavior is preserved.
- [ ] No new execution wrappers proposed.
- [ ] Proposed catalog record schema covers all required fields.
- [ ] Proposed artifact record schema covers all required fields.
- [ ] Proposed lineage edge schema includes all required edge types.
- [ ] Proposed validation/status schema accounts for all marker file states.
- [ ] Source precedence rules are deterministic and consistent.
- [ ] Anti-duplication rules explicitly reference existing implementations.
- [ ] Follow-on implementation issues are enumerated.
- [ ] `pytest -q` passes without errors.
- [ ] `python -m compileall src` completes without errors.
