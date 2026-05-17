# Catalog Lineage

`src/catalog/lineage.py` adds a read-only extraction layer over M29 catalog
records. It consumes in-memory `CatalogRecord` objects and, when `repo_root` is
provided, reads existing JSON artifacts needed to inspect manifests,
checkpoints, scenario catalogs, and validation summaries.

It does not write, modify, delete, repair, lock, register, or execute anything.
There is no lineage database, cache, export, CLI, dashboard, or notebook surface.

M37 lineage exports add Canonicality Envelope v1 to newly generated root
payloads. OpenLineage-style and PROV-style exports are local JSON views over
canonical artifacts, never canonical artifacts themselves. Decision-sensitive
consumers should reopen the canonical manifests and registries before relying on
an exported view. Legacy M36 export payloads without the envelope remain valid
and are reported as `legacy_no_envelope`. Workflow lineage helpers delegate to
the same exporter and therefore keep `derived_class: lineage_export`.
New lineage outputs should be written under `artifacts/_derived/lineage/` when a
file is requested. Their `load_source` metadata records that they are derived
lineage exports and, for workflow-generated exports, whether input records came
from direct scan or a validated derived index.
Lineage exports remain display/interchange views only; decision-sensitive
consumers should resolve the underlying catalog records back to canonical source
files under `artifacts_root` with the resolver APIs before relying on them.

For the full M35 evidence catalog overview and release-readiness docs, see
[`docs/m35_evidence_catalog_foundation.md`](m35_evidence_catalog_foundation.md)
and [`docs/m35_release_notes.md`](m35_release_notes.md).

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
| `campaign_child` | campaign parent -> campaign child | `campaign_parent_run_id`, `parent_run_id`, `campaign_parent_catalog_id`, `parent_catalog_id`, resolvable campaign-parent `campaign_id` |
| `scenario_child` | scenario parent -> scenario child | `scenario_parent_run_id`, `parent_scenario_run_id`, `scenario_parent_catalog_id`, `parent_scenario_catalog_id` |
| `manifest_declares_artifact` | run catalog record -> declared artifact | `manifest.json` entries with `declared_in_manifest=True` |
| `validation_references_run` | referenced run -> validation record | `referenced_run_ids`, `validation_target_run_ids`, `run_ids` |
| `pipeline_wraps_execution` | pipeline run -> wrapped execution run | `wrapped_run_id`, `child_run_id`, `stage_run_ids` |
| `run_to_robustness_evidence` | source run -> robustness evidence bundle | explicit robustness `source_run_ids`, source-run references, or source-artifact references |
| `run_to_governance_evidence` | source run -> governance evidence bundle | explicit governance source run metadata or `promotion_outcome_matrix.csv` rows |
| `run_to_validation_bundle` | source run -> milestone validation bundle | explicit validation source/target run metadata |
| `run_to_release_validation` | source run -> release-validation artifact | explicit release-validation source run metadata |
| `validation_bundle_to_release_validation` | validation bundle -> release-validation artifact | explicit release metadata naming validation bundle run IDs |
| `campaign_to_evidence_bundle` | campaign record -> evidence bundle | explicit campaign run/catalog/campaign ID references |
| `scenario_to_evidence_bundle` | scenario record -> evidence bundle | explicit scenario run/catalog/scenario ID references |

`campaign_child` and `scenario_child` are intentionally distinct. Generic
campaign parent metadata such as `parent_run_id`, `parent_catalog_id`, or a
resolvable `campaign_id` does not produce scenario lineage. `scenario_child`
requires explicit scenario-parent metadata, and `scenario_id` alone is not
treated as a parent reference.

`portfolio_template` records are metadata-only and are intentionally excluded
from executed `portfolio_component` lineage. Template-component lineage, if
needed later, should use a distinct edge type such as
`portfolio_template_component`.

`manifest_declares_artifact` uses `target_catalog_id=None` because artifacts are
represented by `ArtifactRecord`, not `CatalogRecord`. Artifact identity is stored
in edge metadata: `artifact_id`, `artifact_path`, `artifact_type`, and
`relative_path`.

## M35 Evidence Lineage

M35 evidence lineage is derived only from existing catalog records and source
artifacts already indexed by M29/M35:

- robustness summaries such as `robustness_summary.json`
- governance bundles such as `promotion_governance_summary.json`,
  `consistency_validation.json`, `manifest.json`, and
  `promotion_outcome_matrix.csv`
- milestone validation bundle `summary.json` and manifest metadata
- release-validation JSON artifacts

Supported source metadata fields include `source_run_ids`, `source_run_id`,
`source_run_references`, `referenced_run_ids`, `validation_target_run_ids`,
`validated_run_ids`, `run_ids`, `source_artifacts`, `source_artifact_refs`,
`source_artifact_references`, `upstream_artifacts`, campaign/scenario reference
fields, and release-validation bundle fields such as
`validation_bundle_run_id` or `validation_bundle_run_ids`.

Source-artifact references may resolve to a catalog record when the referenced
path is inside that record's artifact root. Paths are normalized as portable
POSIX-style strings in edge metadata.

Unsupported inferred edges are intentionally omitted. The lineage layer does not
link records from name similarity, aggregate counts, status values, report IDs,
or release IDs alone. For example, a `robustness_report_demo` record is not
linked to `demo` unless a supported source field or artifact path explicitly
references that run.

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

This behavior also applies to M35 evidence references. Missing source run IDs,
source artifact paths that do not resolve to a supplied catalog record, and
release-validation references to absent validation bundles do not fail lineage
extraction.

## Limitations

- The layer only derives relationships from explicit metadata fields or existing
  JSON artifacts.
- It does not persist, query, visualize, or export lineage graphs.
- It does not add a graph database, graph cache, or canonical lineage store.
- It does not introduce new registry, manifest, marker, or checkpoint schemas.
- It does not execute strategies, portfolios, campaigns, benchmark packs,
  validations, notebooks, or pipeline wrappers.
- Scenario hierarchy is emitted only where explicit scenario-parent metadata is
  present.
- Governance evidence remains read-only review context; lineage edges do not
  replay promotion gates, enforce governance policy, or mutate promotion
  decisions.

For a CLI-first local renderer over catalog records and evidence lineage, see
[`docs/catalog_evidence_explorer.md`](catalog_evidence_explorer.md).
