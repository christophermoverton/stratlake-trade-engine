# M35 Evidence Catalog Foundation

Milestone 35 extends the M29 read-only catalog model so evidence bundles created
by later milestones are discoverable beside strategy, alpha, portfolio, QA, and
review artifacts. This is an extension of `src.catalog`; it is not a second
catalog, registry, database, cache, or policy engine.

## Record Families

Evidence-oriented artifact roots use `CatalogRecord.record_family` to identify
the review surface while preserving the existing `run_type`, query, validation,
and artifact-record behavior.

| Record family | Source artifacts | Meaning |
| --- | --- | --- |
| `robustness_bundle` | `robustness_summary.json` and companion M34 artifacts | Statistical robustness evidence for review. |
| `governance_bundle` | `promotion_governance_summary.json`, `consistency_validation.json`, `manifest.json` | M32 governance observability output. |
| `milestone_validation_bundle` | milestone `summary.json` with `run_type: milestone_validation_bundle` | Deterministic milestone validation readiness evidence. |
| `release_validation_artifact` | `release_validation.json` or `release_validation_summary.json` | Release-validation metadata when such artifacts exist. |

The indexer scans only known artifact families and known indicator filenames. It
does not recursively search arbitrary repository paths for evidence.

## Field Mapping

Evidence fields are populated only when the source artifact has the information
or a deterministic companion file exposes it.

| Catalog field | Source of truth |
| --- | --- |
| `robustness_status` | `robustness_summary.json` `robustness_status`, falling back to explicit robustness status counts. |
| `wfe_status` | `walk_forward_efficiency.csv` `status` rows. |
| `sample_size_status` | `sample_size_validation.json` check statuses. |
| `trade_count_status` | trade-related checks in `sample_size_validation.json`. |
| `sensitivity_status` and `fragility_status` | `sensitivity_summary.csv` `status` rows. |
| `multiple_testing_status` | `multiple_testing_summary.json` family statuses. |
| `temporal_validation_status` | `leakage_validation.json` `overall_status` and temporal-validation findings. |
| `governance_status` | governance or validation summary status already written by the source bundle. |
| `promotion_review_status` | a single explicit review-status count in governance summary output, when unambiguous. |
| `validation_readiness_present` | presence of a milestone validation bundle summary. |
| `release_validation_present` | presence of a release-validation artifact. |

Missing or sparse evidence remains `None` for status fields and `False` for
presence booleans. The catalog does not infer unsupported statuses from absent
files, and it does not treat missing robustness or governance evidence as a
promotion decision.

## Path Portability

Indexed catalog paths remain repository-relative POSIX-style strings wherever
the source is under `repo_root`. The evidence extension does not emit local
absolute paths, Windows-only separators, or `file://` links for source-file and
artifact records derived from repository artifacts.

## Read-Only Boundary

The M35 foundation only reads existing artifacts:

* it does not write canonical registry entries;
* it does not mutate source manifests, markers, governance reports, or
  robustness reports;
* it does not replay promotion gates or enforce promotion policy;
* it does not introduce a persistent database, persistent cache, search backend,
  live monitor, or dashboard service.

Discovery facets, lineage edges, explorer rendering, notebook ergonomics, and
release documentation are intentionally left to Issues #395 through #400.
