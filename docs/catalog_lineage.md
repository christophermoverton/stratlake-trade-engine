# Catalog Lineage

`src/catalog/lineage.py` adds a read-only extraction layer over M29 catalog
records. It consumes in-memory `CatalogRecord` objects and, when `repo_root` is
provided, reads existing JSON artifacts needed to inspect manifests,
checkpoints, scenario catalogs, and validation summaries.

It does not write, modify, delete, repair, lock, register, or execute anything.
There is no lineage database, cache, export, CLI, dashboard, or notebook surface.

## Public API

```python
from src.catalog import build_catalog, build_lineage_edges

records = build_catalog("artifacts", repo_root=".")
edges = build_lineage_edges(records, repo_root=".")
```

Helper lookups are also available:

```python
from src.catalog import build_catalog_lookup, build_run_lookup
```

## Edge Direction

| Edge type | Direction | Derived from |
| --- | --- | --- |
| `portfolio_component` | component strategy/alpha run -> portfolio run | `component_run_ids`, `components[*].run_id` |
| `comparison_member` | member run -> comparison run | `member_run_ids`, `comparison_members`, `run_ids`, `inputs` |
| `benchmark_member` | child/member run -> benchmark pack run | `member_run_ids`, `child_run_ids`, `scenario_run_ids`, `run_ids` |
| `campaign_child` | campaign parent -> campaign child | `parent_run_id`, `parent_catalog_id`, resolvable `campaign_id` |
| `scenario_child` | scenario parent -> scenario child | `parent_run_id`, `parent_catalog_id`, resolvable `campaign_id` |
| `manifest_declares_artifact` | run catalog record -> declared artifact | `manifest.json` entries with `declared_in_manifest=True` |
| `validation_references_run` | referenced run -> validation record | `referenced_run_ids`, `validation_target_run_ids`, `run_ids` |
| `pipeline_wraps_execution` | pipeline run -> wrapped execution run | `wrapped_run_id`, `child_run_id`, `stage_run_ids` |

`manifest_declares_artifact` uses `target_catalog_id=None` because artifacts are
represented by `ArtifactRecord`, not `CatalogRecord`. Artifact identity is stored
in edge metadata: `artifact_id`, `artifact_path`, `artifact_type`, and
`relative_path`.

## Determinism

Edge IDs are SHA-256 hashes over:

```text
edge_type|source_catalog_id|target_catalog_id|source_run_id|target_run_id|relationship_source|relationship_path
```

The first 16 hex characters are used. Output edges are deduplicated by
`edge_id` and sorted by:

```text
edge_type, source_run_id or "", target_run_id or "", relationship_source, relationship_path or ""
```

Given the same catalog records and artifact state, edge IDs and ordering are
stable.

## Unresolved References

Unresolved references are skipped deterministically. The lineage layer does not
guess parent records, fabricate orphan edges, or infer relationships from path
similarity. If a referenced run ID or catalog ID cannot be found in the supplied
catalog records, no edge is emitted.

## Limitations

- The layer only derives relationships from explicit metadata fields or existing
  JSON artifacts.
- It does not persist, query, visualize, or export lineage graphs.
- It does not introduce new registry, manifest, marker, or checkpoint schemas.
- It does not execute strategies, portfolios, campaigns, benchmark packs,
  validations, notebooks, or pipeline wrappers.
- Scenario hierarchy is emitted only where explicit parent metadata is present
  or a resolvable campaign identifier points at a cataloged parent run.
