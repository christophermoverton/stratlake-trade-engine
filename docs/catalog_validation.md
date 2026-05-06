# Catalog Validation

## Purpose

`src/catalog/validation.py` adds deterministic, read-only consistency checks for
the M29 unified research catalog. It validates in-memory `CatalogRecord` objects
against artifact roots, `ArtifactRecord` inventory, marker files, manifests,
source metadata paths, and common JSON artifacts.

The validator is a consumer of existing provenance. It does not create a new
registry, cache, database, report file, or execution surface.

## Read-Only Guarantee

Catalog validation only reads files and returns `CatalogValidationReport`
objects in memory. It does not write, repair, delete, move, lock, append to, or
export artifacts, registries, manifests, markers, caches, or validation reports.

Validation reuses the existing indexer inventory through
`build_artifact_records()` and does not duplicate canonical artifact discovery.

## Python API

```python
from src.catalog import build_catalog, validate_catalog, validate_record

records = build_catalog("artifacts", repo_root=".")
report = validate_catalog(records, repo_root=".")

for issue in report.issues:
    print(issue.severity, issue.code, issue.run_id, issue.path)

one_record_issues = validate_record(records[0], repo_root=".")
```

`CatalogValidationIssue.to_dict()` and `CatalogValidationReport.to_dict()`
return deterministic JSON-ready dictionaries.

## Issue Codes

Supported validation issue codes:

| Code | Severity | Meaning |
| --- | --- | --- |
| `artifact_root_missing` | error | Non-registry-only record points at a missing artifact root |
| `artifact_path_missing` | error | Artifact record references a missing file unexpectedly |
| `manifest_missing` | warning | Artifact root has no `manifest.json` |
| `manifest_artifact_missing` | warning | Manifest declares an artifact that is absent on disk |
| `undeclared_artifact` | warning | Non-internal file exists but is not declared in `manifest.json` |
| `marker_missing` | warning | Artifact root has no `_FAILED`, `_SUCCESS`, or `_RUNNING` marker |
| `multiple_markers` | warning | More than one lifecycle marker is present |
| `failed_marker_present` | warning | `_FAILED.json` exists and keeps highest marker precedence |
| `running_marker_present` | warning | `_RUNNING.json` exists |
| `registry_only_record` | warning | Record is registry-derived and does not require an artifact root |
| `unknown_status` | warning | `record.status == "unknown"` |
| `unknown_run_type` | warning | `record.run_type == "unknown"` |
| `corrupt_json` | error | Common JSON artifact exists but cannot be parsed |
| `source_file_missing` | warning | A `source_files` path no longer exists |
| `source_manifest_missing` | warning | `source_manifest_path` no longer exists |
| `source_marker_missing` | warning | `source_marker_path` no longer exists |

The validator reserves the `info` severity for future non-actionable
observations. By default, `validate_catalog(..., include_info=False)` omits info
issues from reports.

## Marker Rules

Validation preserves the M28 marker precedence already used by the indexer:

```text
_FAILED.json > _SUCCESS.json > _RUNNING.json
```

The validator does not reinterpret marker lifecycle state. It only reports
observable consistency signals:

- no marker on a normal artifact root produces `marker_missing`
- multiple markers produce `multiple_markers`
- `_FAILED.json` produces `failed_marker_present`
- `_RUNNING.json` produces `running_marker_present`

## Manifest and Artifact Rules

Artifact inventory checks are based on `build_artifact_records(record,
repo_root=...)`.

If an artifact record is declared by the manifest but missing on disk, validation
emits `manifest_artifact_missing`.

If an artifact exists on disk but is not declared in the manifest, validation
emits `undeclared_artifact` unless the file is an expected internal metadata
file:

```text
manifest.json
_SUCCESS.json
_FAILED.json
_RUNNING.json
metrics.json
alpha_metrics.json
summary.json
qa_summary.json
signal_diagnostics.json
checkpoint.json
scenario_catalog.json
decision_log.json
```

These files are excluded because current artifact roots commonly use them as
catalog/indexer metadata surfaces even when older manifests do not list them.

## Registry-Only Behavior

`status == "registry_only"` records represent registry metadata without a
corresponding per-run artifact root. This is expected for some M29 records,
especially portfolio template registry entries.

Validation emits `registry_only_record` but does not emit
`artifact_root_missing`, `manifest_missing`, or `marker_missing` for these
records. Source paths that are explicitly present on the record are still
validated.

## Malformed JSON Handling

Validation directly parses common JSON files when they exist so it can
distinguish missing files from malformed files. Malformed files produce
`corrupt_json` and do not crash validation.

Checked JSON filenames:

```text
manifest.json
metrics.json
alpha_metrics.json
summary.json
qa_summary.json
_SUCCESS.json
_FAILED.json
_RUNNING.json
checkpoint.json
scenario_catalog.json
decision_log.json
```

## Deterministic Reports

Issue ordering is stable by catalog ID, run ID, path, code, and severity.
Summary dictionaries sort keys deterministically and include:

- `by_severity`
- `by_code`
- `by_run_type`
- `by_status`
- `records_with_errors`
- `records_with_warnings`

## Limitations and Non-Goals

Catalog validation does not:

- execute strategies, portfolios, campaigns, notebooks, pipelines, or validators
- repair artifacts, manifests, markers, or registries
- persist validation reports
- create a cache, database, dashboard, or notebook example
- replace existing registry, manifest, marker, or indexer semantics
- compute checksums
- recursively infer new artifact-root schemas beyond the indexer's inventory
