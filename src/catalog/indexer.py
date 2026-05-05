"""Read-only artifact catalog indexer for M29.

This module scans existing StratLake artifact roots and produces normalized
in-memory CatalogRecord and ArtifactRecord instances. It never writes, modifies,
deletes, moves, or appends to any source artifact, registry, manifest, or marker.

Marker-file status precedence (most authoritative first):
    _FAILED.json  -> failed
    _SUCCESS.json -> completed
    _RUNNING.json -> running
    none          -> unknown

Deterministic catalog_id and artifact_id:
    catalog_id  = sha256(run_id + artifact_root)[:16]
    artifact_id = sha256(catalog_id + relative_path)[:16]

Output ordering:
    CatalogRecord: sorted by (run_type, run_id or "", artifact_root)
    ArtifactRecord: sorted by relative_path
"""

from __future__ import annotations

import hashlib
import json
import logging
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.catalog.models import ArtifactRecord, CatalogRecord, CatalogValidationStatus

_LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SUCCESS_MARKER = "_SUCCESS.json"
_FAILED_MARKER = "_FAILED.json"
_RUNNING_MARKER = "_RUNNING.json"

# Marker precedence: higher index = lower priority
_MARKER_PRECEDENCE: list[tuple[str, str]] = [
    (_FAILED_MARKER, "failed"),
    (_SUCCESS_MARKER, "completed"),
    (_RUNNING_MARKER, "running"),
]

# Files that indicate a directory is an artifact root
_ARTIFACT_ROOT_INDICATORS: frozenset[str] = frozenset(
    {
        "manifest.json",
        "metrics.json",
        "alpha_metrics.json",
        "summary.json",
        "qa_summary.json",
        "_SUCCESS.json",
        "_FAILED.json",
        "_RUNNING.json",
        "checkpoint.json",
        "scenario_catalog.json",
        "decision_log.json",
    }
)

# Well-known artifact family directories (relative to artifacts_root)
_KNOWN_FAMILIES: tuple[str, ...] = (
    "strategies",
    "alpha",
    "portfolios",
    "comparisons",
    "pipelines",
    "qa",
    "reviews",
    "candidate_selection",
    "regime_stress_tests",
)

# Registry paths relative to artifacts_root
_REGISTRY_SPECS: list[dict[str, str]] = [
    {"family": "strategies", "path": "strategies/registry.jsonl", "run_type": "strategy"},
    {"family": "alpha", "path": "alpha/registry.jsonl", "run_type": "alpha_evaluation"},
    {"family": "portfolios", "path": "portfolios/registry.jsonl", "run_type": "portfolio"},
    {
        "family": "portfolio_template",
        "path": "registry/portfolios.jsonl",
        "run_type": "portfolio_template",
    },
]

# Known metrics scalar keys worth lifting into metrics_summary
_SCALAR_METRIC_KEYS: frozenset[str] = frozenset(
    {
        "sharpe_ratio",
        "sortino_ratio",
        "cagr",
        "max_drawdown",
        "annual_return",
        "annual_volatility",
        "calmar_ratio",
        "mean_ic",
        "ic_ir",
        "mean_rank_ic",
        "rank_ic_ir",
        "n_periods",
        "win_rate",
        "hit_rate",
        "total_return",
        "beta",
        "alpha",
        "information_ratio",
    }
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_catalog(
    artifacts_root: str | Path = "artifacts",
    *,
    repo_root: str | Path | None = None,
) -> list[CatalogRecord]:
    """Scan artifact roots and return a sorted, deterministic list of CatalogRecords.

    Parameters
    ----------
    artifacts_root:
        Path to the top-level artifacts directory. Defaults to ``"artifacts"``
        relative to the current working directory.
    repo_root:
        Repository root for computing relative paths. Defaults to the parent
        of *artifacts_root*.

    Returns
    -------
    list[CatalogRecord]
        Sorted by (run_type, run_id or "", artifact_root). The list is
        deterministic for a given artifact tree.
    """
    resolved_root = Path(artifacts_root).resolve()
    resolved_repo = Path(repo_root).resolve() if repo_root is not None else resolved_root.parent

    # Load registry entries indexed by run_id
    registry_index = _load_registry_index(resolved_root)

    # Discover all artifact roots
    artifact_roots = discover_artifact_roots(resolved_root)

    seen_roots: set[Path] = set()
    records: list[CatalogRecord] = []

    for root in artifact_roots:
        if root in seen_roots:
            continue
        seen_roots.add(root)
        try:
            record = build_catalog_record(root, repo_root=resolved_repo, registry_index=registry_index)
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("Skipping artifact root %s: %s", root, exc)
            continue
        records.append(record)

    # Records for registry entries that have no artifact root on disk
    for run_id, entry in registry_index.items():
        artifact_dir = _resolve_artifact_dir(entry, resolved_root)
        if artifact_dir is None or artifact_dir not in seen_roots:
            if artifact_dir is None or not artifact_dir.exists():
                record = _build_registry_only_record(entry, resolved_repo, resolved_root)
                records.append(record)

    records.sort(key=lambda r: (r.run_type, r.run_id or "", r.artifact_root))
    return records


def discover_artifact_roots(artifacts_root: str | Path) -> list[Path]:
    """Return a sorted list of directories that look like artifact roots.

    Scans known family directories plus any benchmark_pack_* glob. Falls back
    to scanning direct children of each family directory.
    """
    resolved = Path(artifacts_root).resolve()
    candidates: list[Path] = []

    if not resolved.exists():
        return []

    # Known families
    family_dirs: list[Path] = []
    for family in _KNOWN_FAMILIES:
        d = resolved / family
        if d.is_dir():
            family_dirs.append(d)

    # benchmark_pack_* directories
    for d in sorted(resolved.glob("benchmark_pack_*")):
        if d.is_dir():
            family_dirs.append(d)

    for family_dir in family_dirs:
        for child in sorted(family_dir.iterdir()):
            if not child.is_dir():
                continue
            if _is_artifact_root(child):
                candidates.append(child)
        # Also check the family dir itself (e.g., benchmark packs)
        if _is_artifact_root(family_dir):
            candidates.append(family_dir)

    # Deduplicate while preserving order
    seen: set[Path] = set()
    result: list[Path] = []
    for p in candidates:
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result


def build_catalog_record(
    artifact_root: Path,
    *,
    repo_root: Path,
    registry_index: dict[str, dict[str, Any]] | None = None,
) -> CatalogRecord:
    """Build a single CatalogRecord for *artifact_root*.

    This function is read-only. It does not modify any files.
    """
    if registry_index is None:
        registry_index = {}

    rel_root = _relative_posix(artifact_root, repo_root)
    run_id = _infer_run_id(artifact_root, registry_index)
    run_type = _infer_run_type(artifact_root, run_id, registry_index)

    # Marker status
    marker_file, status, marker_path = _read_marker_status(artifact_root)
    source_marker_path = _relative_posix(Path(marker_path), repo_root) if marker_path else None

    # Manifest
    manifest_data, manifest_path = _load_manifest(artifact_root)
    source_manifest_path = _relative_posix(manifest_path, repo_root) if manifest_path else None

    # Metrics / summary files
    combined_meta: dict[str, Any] = {}
    metrics_summary: dict[str, Any] = {}
    for fname in ("metrics.json", "alpha_metrics.json", "summary.json", "signal_diagnostics.json"):
        data = load_json_file(artifact_root / fname)
        if data is not None:
            combined_meta[fname] = data
            _extract_scalars(data, metrics_summary)

    qa_data = load_json_file(artifact_root / "qa_summary.json")
    qa_status: str | None = None
    if qa_data is not None:
        combined_meta["qa_summary.json"] = qa_data
        qa_status = _str_or_none(qa_data.get("status") or qa_data.get("qa_status"))

    # Registry enrichment
    registry_entry = registry_index.get(run_id) if run_id else None
    source_registry_path: str | None = None
    if registry_entry is not None:
        source_registry_path = _str_or_none(registry_entry.get("_registry_path"))

    # Field extraction (registry wins over summary wins over manifest).
    # Use safe helper dicts to avoid Python operator-precedence issues with
    # "if manifest_data else None" applying to the entire "or" chain.
    registry = registry_entry or {}
    manifest = manifest_data or {}
    summary = combined_meta.get("summary.json") or {}

    strategy_name = _str_or_none(
        registry.get("strategy_name")
        or summary.get("strategy_name")
        or manifest.get("strategy_name")
    )
    portfolio_name = _str_or_none(
        registry.get("portfolio_name")
        or summary.get("portfolio_name")
        or manifest.get("portfolio_name")
    )
    allocator_name = _str_or_none(
        registry.get("allocator_name")
        or summary.get("allocator_name")
        or manifest.get("allocator_name")
    )
    alpha_model_name = _str_or_none(
        registry.get("alpha_name")
        or registry.get("alpha_model_name")
        or summary.get("alpha_name")
        or summary.get("alpha_model_name")
        or manifest.get("alpha_name")
        or manifest.get("alpha_model_name")
    )
    timeframe = _str_or_none(
        registry.get("timeframe")
        or summary.get("timeframe")
        or manifest.get("timeframe")
    )
    start_ts = _str_or_none(
        registry.get("start_ts")
        or summary.get("start_ts")
        or manifest.get("start_ts")
    )
    end_ts = _str_or_none(
        registry.get("end_ts")
        or summary.get("end_ts")
        or manifest.get("end_ts")
    )
    regime_method = _str_or_none(
        registry.get("regime_method")
        or summary.get("regime_method")
        or manifest.get("regime_method")
    )
    campaign_id = _str_or_none(
        registry.get("campaign_id")
        or summary.get("campaign_id")
        or manifest.get("campaign_id")
    )
    scenario_id = _str_or_none(
        registry.get("scenario_id")
        or summary.get("scenario_id")
        or manifest.get("scenario_id")
    )
    # review_status and promotion_status: registry entries may store these as
    # top-level "review_status" / "promotion_status" fields (portfolio schema),
    # inside a "review_metadata" mapping (strategy schema), or inside a legacy
    # "review" mapping.  Check all variants in precedence order.
    _review_meta = registry.get("review_metadata")
    _review_legacy = registry.get("review")
    review_status = _str_or_none(
        registry.get("review_status")
        or (_review_meta.get("status") if isinstance(_review_meta, dict) else None)
        or (_review_legacy.get("status") if isinstance(_review_legacy, dict) else None)
    )
    promotion_status = _str_or_none(
        registry.get("promotion_status")
        or (_review_meta.get("promotion_status") if isinstance(_review_meta, dict) else None)
        or (_review_legacy.get("promotion_status") if isinstance(_review_legacy, dict) else None)
    )
    created_at = _str_or_none(
        registry.get("timestamp")
        or manifest.get("created_at")
        or summary.get("created_at")
    )
    if created_at is None and marker_path:
        marker_data = load_json_file(Path(marker_path))
        if marker_data:
            created_at = _str_or_none(marker_data.get("recorded_at_utc"))

    catalog_id = _make_catalog_id(run_id, rel_root)

    # Collect source files
    source_files: list[str] = []
    if source_registry_path:
        source_files.append(source_registry_path)
    if source_manifest_path:
        source_files.append(source_manifest_path)
    if source_marker_path:
        source_files.append(source_marker_path)
    for fname in combined_meta:
        p = artifact_root / fname
        if p.exists():
            source_files.append(_relative_posix(p, repo_root))
    if qa_data is not None:
        p = artifact_root / "qa_summary.json"
        if p.exists():
            source_files.append(_relative_posix(p, repo_root))

    # Validation status
    validation = _build_validation_status(
        artifact_root=artifact_root,
        manifest_data=manifest_data,
        marker_file=marker_file,
        status=status,
        registry_entry=registry_entry,
        qa_status=qa_status,
    )

    return CatalogRecord(
        catalog_id=catalog_id,
        run_id=run_id,
        run_type=run_type,
        status=status,
        artifact_root=rel_root,
        source_registry_path=source_registry_path,
        source_manifest_path=source_manifest_path,
        source_marker_path=source_marker_path,
        created_at=created_at,
        timeframe=timeframe,
        start_ts=start_ts,
        end_ts=end_ts,
        strategy_name=strategy_name,
        portfolio_name=portfolio_name,
        allocator_name=allocator_name,
        alpha_model_name=alpha_model_name,
        regime_method=regime_method,
        campaign_id=campaign_id,
        scenario_id=scenario_id,
        metrics_summary=metrics_summary if metrics_summary else None,
        qa_status=qa_status,
        review_status=review_status,
        promotion_status=promotion_status,
        tags=[],
        source_files=sorted(set(source_files)),
        metadata=combined_meta,
        validation=validation,
    )


def build_artifact_records(
    record: CatalogRecord,
    *,
    repo_root: Path,
) -> list[ArtifactRecord]:
    """Build ArtifactRecord entries for all files under *record.artifact_root*.

    Read-only. Does not modify any files.
    """
    artifact_root = repo_root / record.artifact_root
    if not artifact_root.exists():
        return []

    # Parse manifest for declared paths
    manifest_path = repo_root / record.source_manifest_path if record.source_manifest_path else None
    manifest_data = load_json_file(manifest_path) if manifest_path and manifest_path.exists() else None
    declared_relative: set[str] = set()
    if manifest_data is not None:
        declared_relative = _extract_manifest_paths(manifest_data)

    # Scan actual files
    actual_files: list[Path] = []
    for p in sorted(artifact_root.rglob("*")):
        if p.is_file():
            actual_files.append(p)

    actual_relative: dict[str, Path] = {}
    for p in actual_files:
        try:
            rel = p.relative_to(artifact_root).as_posix()
        except ValueError:
            rel = p.as_posix()
        actual_relative[rel] = p

    artifact_records: list[ArtifactRecord] = []

    # Declared but possibly missing
    for rel in sorted(declared_relative):
        abs_path = artifact_root / rel
        exists = abs_path.exists()
        if rel in actual_relative:
            actual_relative.pop(rel)  # will not add as undeclared
        artifact_records.append(
            _make_artifact_record(
                catalog_id=record.catalog_id,
                run_id=record.run_id,
                artifact_root=artifact_root,
                repo_root=repo_root,
                abs_path=abs_path,
                relative_path=rel,
                declared_in_manifest=True,
                exists=exists,
            )
        )

    # Undeclared (discovered but not in manifest)
    for rel, abs_path in sorted(actual_relative.items()):
        artifact_records.append(
            _make_artifact_record(
                catalog_id=record.catalog_id,
                run_id=record.run_id,
                artifact_root=artifact_root,
                repo_root=repo_root,
                abs_path=abs_path,
                relative_path=rel,
                declared_in_manifest=False,
                exists=True,
            )
        )

    artifact_records.sort(key=lambda r: r.relative_path)
    return artifact_records


def load_json_file(path: Path) -> dict[str, Any] | None:
    """Read a JSON file and return its contents or None on any failure."""
    if not path or not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
        if isinstance(data, dict):
            return data
        return None
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        _LOG.debug("Could not read %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_artifact_root(directory: Path) -> bool:
    """Return True if *directory* contains at least one artifact root indicator."""
    return any((directory / name).exists() for name in _ARTIFACT_ROOT_INDICATORS)


def _load_registry_index(artifacts_root: Path) -> dict[str, dict[str, Any]]:
    """Load all known registry files and index entries by run_id."""
    index: dict[str, dict[str, Any]] = {}
    for spec in _REGISTRY_SPECS:
        registry_path = artifacts_root / spec["path"]
        if not registry_path.exists():
            continue
        try:
            entries = _read_jsonl(registry_path)
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("Could not read registry %s: %s", registry_path, exc)
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            run_id = _str_or_none(entry.get("run_id"))
            if run_id:
                enriched = dict(entry)
                enriched["_registry_path"] = _posix_str(registry_path)
                enriched.setdefault("run_type", spec["run_type"])
                index[run_id] = enriched
    return index


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file and return a list of decoded objects."""
    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                _LOG.warning("Skipping malformed JSON on line %d of %s: %s", lineno, path, exc)
                continue
            if isinstance(obj, dict):
                entries.append(obj)
    return entries


def _read_marker_status(artifact_root: Path) -> tuple[str | None, str, str | None]:
    """Return (marker_filename, status_string, marker_abs_path) using precedence rules."""
    for marker_file, status in _MARKER_PRECEDENCE:
        p = artifact_root / marker_file
        if p.exists():
            return marker_file, status, p.as_posix()
    return None, "unknown", None


def _load_manifest(artifact_root: Path) -> tuple[dict[str, Any] | None, Path | None]:
    p = artifact_root / "manifest.json"
    data = load_json_file(p)
    if data is not None:
        return data, p
    return None, None


def _infer_run_id(artifact_root: Path, registry_index: dict[str, dict[str, Any]]) -> str | None:
    # Prefer manifest run_id
    manifest_data = load_json_file(artifact_root / "manifest.json")
    if manifest_data:
        rid = _str_or_none(manifest_data.get("run_id"))
        if rid:
            return rid

    # Check marker files for run_id
    for marker_file, _ in _MARKER_PRECEDENCE:
        marker_data = load_json_file(artifact_root / marker_file)
        if marker_data:
            rid = _str_or_none(marker_data.get("run_id") or marker_data.get("metadata", {}).get("run_id"))
            if rid:
                return rid

    # Try directory name as run_id if it matches a registry entry
    dir_name = artifact_root.name
    if dir_name in registry_index:
        return dir_name

    # Fallback: use directory name as run_id
    return dir_name if dir_name else None


def _infer_run_type(
    artifact_root: Path,
    run_id: str | None,
    registry_index: dict[str, dict[str, Any]],
) -> str:
    if run_id and run_id in registry_index:
        return _str_or_none(registry_index[run_id].get("run_type")) or "unknown"

    parent_name = artifact_root.parent.name
    _FAMILY_TO_RUN_TYPE: dict[str, str] = {
        "strategies": "strategy",
        "alpha": "alpha_evaluation",
        "portfolios": "portfolio",
        "comparisons": "comparison",
        "pipelines": "pipeline",
        "qa": "qa",
        "reviews": "review",
        "candidate_selection": "candidate_selection",
        "regime_stress_tests": "regime_stress_test",
        "registry": "portfolio_template",
    }
    if parent_name in _FAMILY_TO_RUN_TYPE:
        return _FAMILY_TO_RUN_TYPE[parent_name]

    if artifact_root.name.startswith("benchmark_pack_"):
        return "benchmark_pack"

    parent = artifact_root.parent
    if parent.name.startswith("benchmark_pack_"):
        return "benchmark_pack"

    return "unknown"


def _resolve_artifact_dir(entry: dict[str, Any], artifacts_root: Path) -> Path | None:
    """Try to resolve the on-disk artifact directory for a registry entry."""
    artifact_dir = _str_or_none(entry.get("artifact_dir") or entry.get("artifact_root"))
    if artifact_dir:
        p = Path(artifact_dir)
        if not p.is_absolute():
            p = artifacts_root.parent / p
        return p.resolve()
    # Infer from run_type + run_id
    run_id = _str_or_none(entry.get("run_id"))
    run_type = _str_or_none(entry.get("run_type")) or "strategy"
    _RUN_TYPE_TO_FAMILY = {
        "strategy": "strategies",
        "alpha_evaluation": "alpha",
        "portfolio": "portfolios",
    }
    family = _RUN_TYPE_TO_FAMILY.get(run_type)
    if family and run_id:
        return (artifacts_root / family / run_id).resolve()
    return None


def _build_registry_only_record(
    entry: dict[str, Any],
    repo_root: Path,
    artifacts_root: Path,
) -> CatalogRecord:
    """Build a CatalogRecord for a registry entry whose artifact root does not exist."""
    run_id = _str_or_none(entry.get("run_id"))
    run_type = _str_or_none(entry.get("run_type")) or "unknown"
    registry_path = _str_or_none(entry.get("_registry_path"))
    if registry_path:
        try:
            registry_path = _relative_posix(Path(registry_path), repo_root)
        except ValueError:
            pass

    artifact_dir = _resolve_artifact_dir(entry, artifacts_root)
    rel_root = _relative_posix(artifact_dir, repo_root) if artifact_dir else ""
    catalog_id = _make_catalog_id(run_id, rel_root)

    metrics_summary: dict[str, Any] = {}
    _extract_scalars(entry.get("metrics_summary") or {}, metrics_summary)

    review = entry.get("review") if isinstance(entry.get("review"), dict) else {}
    review_status = _str_or_none(review.get("status"))
    promotion_status = _str_or_none(review.get("promotion_status"))

    validation = CatalogValidationStatus(
        catalog_status="registry_only",
        marker_status="missing",
        manifest_status="missing",
        artifact_status="missing",
        qa_status=None,
        validation_warnings=["registry_entry_no_artifact_root"],
    )

    return CatalogRecord(
        catalog_id=catalog_id,
        run_id=run_id,
        run_type=run_type,
        status="registry_only",
        artifact_root=rel_root,
        source_registry_path=registry_path,
        source_manifest_path=None,
        source_marker_path=None,
        created_at=_str_or_none(entry.get("timestamp")),
        timeframe=_str_or_none(entry.get("timeframe")),
        start_ts=_str_or_none(entry.get("start_ts")),
        end_ts=_str_or_none(entry.get("end_ts")),
        strategy_name=_str_or_none(entry.get("strategy_name")),
        portfolio_name=_str_or_none(entry.get("portfolio_name")),
        allocator_name=_str_or_none(entry.get("allocator_name")),
        alpha_model_name=_str_or_none(entry.get("alpha_name")),
        regime_method=_str_or_none(entry.get("regime_method")),
        campaign_id=_str_or_none(entry.get("campaign_id")),
        scenario_id=_str_or_none(entry.get("scenario_id")),
        metrics_summary=metrics_summary if metrics_summary else None,
        qa_status=None,
        review_status=review_status,
        promotion_status=promotion_status,
        tags=[],
        source_files=[registry_path] if registry_path else [],
        metadata={},
        validation=validation,
    )


def _build_validation_status(
    *,
    artifact_root: Path,
    manifest_data: dict[str, Any] | None,
    marker_file: str | None,
    status: str,
    registry_entry: dict[str, Any] | None,
    qa_status: str | None,
) -> CatalogValidationStatus:
    errors: list[str] = []
    warnings: list[str] = []

    # Marker status
    if marker_file is None:
        marker_status = "missing"
    elif marker_file == _FAILED_MARKER:
        marker_status = "failed"
        warnings.append("failed_marker_present")
    elif marker_file == _RUNNING_MARKER:
        marker_status = "running"
        warnings.append("running_marker_present")
    else:
        marker_status = "present"

    # Manifest status
    if manifest_data is None:
        manifest_status = "missing"
        warnings.append("manifest_missing")
    else:
        manifest_status = "present"

    # Registry status
    if registry_entry is None:
        warnings.append("artifact_root_no_registry_entry")

    # Artifact status – check declared vs actual
    artifact_status = "ok"
    if manifest_data is not None:
        declared = _extract_manifest_paths(manifest_data)
        missing_declared = [
            rel for rel in declared if not (artifact_root / rel).exists()
        ]
        undeclared = [
            p.relative_to(artifact_root).as_posix()
            for p in sorted(artifact_root.rglob("*"))
            if p.is_file() and p.relative_to(artifact_root).as_posix() not in declared
        ]
        for rel in missing_declared:
            warnings.append(f"manifest_artifact_missing:{rel}")
        for rel in undeclared:
            warnings.append(f"undeclared_artifact:{rel}")
        if missing_declared:
            artifact_status = "incomplete"

    # Catalog status
    if errors:
        catalog_status = "error"
    elif marker_file == _FAILED_MARKER:
        catalog_status = "failed"
    elif marker_file == _RUNNING_MARKER:
        catalog_status = "running"
    elif marker_file == _SUCCESS_MARKER:
        catalog_status = "completed"
    elif registry_entry is not None:
        catalog_status = "indexed"
    else:
        catalog_status = "discovered"

    return CatalogValidationStatus(
        catalog_status=catalog_status,
        marker_status=marker_status,
        manifest_status=manifest_status,
        artifact_status=artifact_status,
        qa_status=qa_status,
        validation_errors=errors,
        validation_warnings=warnings,
    )


def _make_artifact_record(
    *,
    catalog_id: str,
    run_id: str | None,
    artifact_root: Path,
    repo_root: Path,
    abs_path: Path,
    relative_path: str,
    declared_in_manifest: bool,
    exists: bool,
) -> ArtifactRecord:
    artifact_id = _make_artifact_id(catalog_id, relative_path)
    filename = abs_path.name
    extension = abs_path.suffix.lstrip(".")
    artifact_type = _infer_artifact_type(filename, extension)
    schema_hint = _infer_schema_hint(filename)

    size_bytes: int | None = None
    modified_time: str | None = None
    if exists and abs_path.is_file():
        try:
            stat = abs_path.stat()
            size_bytes = stat.st_size
            modified_time = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
        except OSError:
            pass

    metadata: dict[str, Any] = {}
    if not declared_in_manifest:
        metadata["undeclared"] = True

    return ArtifactRecord(
        artifact_id=artifact_id,
        catalog_id=catalog_id,
        run_id=run_id,
        artifact_type=artifact_type,
        path=_posix_str(abs_path),
        relative_path=relative_path,
        filename=filename,
        extension=extension,
        declared_in_manifest=declared_in_manifest,
        exists=exists,
        size_bytes=size_bytes,
        modified_time=modified_time,
        checksum_optional=None,
        schema_hint=schema_hint,
        metadata=metadata,
    )


def _make_catalog_id(run_id: str | None, artifact_root: str) -> str:
    raw = f"{run_id or ''}|{artifact_root}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _make_artifact_id(catalog_id: str, relative_path: str) -> str:
    raw = f"{catalog_id}|{relative_path}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _extract_manifest_paths(manifest_data: dict[str, Any]) -> set[str]:
    """Extract relative file paths declared in a manifest."""
    paths: set[str] = set()
    for key in ("artifacts", "files", "outputs", "inputs"):
        val = manifest_data.get(key)
        if isinstance(val, list):
            for item in val:
                if isinstance(item, str):
                    paths.add(item.lstrip("/"))
                elif isinstance(item, dict):
                    p = _str_or_none(item.get("path") or item.get("relative_path") or item.get("file"))
                    if p:
                        paths.add(p.lstrip("/"))
        elif isinstance(val, dict):
            for item_key, item_val in val.items():
                if isinstance(item_val, str):
                    paths.add(item_val.lstrip("/"))
                elif isinstance(item_val, dict):
                    p = _str_or_none(item_val.get("path") or item_val.get("relative_path"))
                    if p:
                        paths.add(p.lstrip("/"))
    return paths


def _extract_scalars(source: dict[str, Any], target: dict[str, Any]) -> None:
    """Lift known scalar metric keys from *source* into *target*."""
    if not isinstance(source, dict):
        return
    for key in _SCALAR_METRIC_KEYS:
        if key in source and isinstance(source[key], (int, float)) and key not in target:
            target[key] = source[key]


def _infer_artifact_type(filename: str, extension: str) -> str:
    _FILE_TYPES: dict[str, str] = {
        "manifest.json": "manifest",
        "metrics.json": "metrics",
        "alpha_metrics.json": "metrics",
        "summary.json": "summary",
        "qa_summary.json": "qa_summary",
        "signal_diagnostics.json": "signal_diagnostics",
        "checkpoint.json": "checkpoint",
        "scenario_catalog.json": "scenario_catalog",
        "decision_log.json": "decision_log",
        "_SUCCESS.json": "marker",
        "_FAILED.json": "marker",
        "_RUNNING.json": "marker",
        "registry.jsonl": "registry",
    }
    if filename in _FILE_TYPES:
        return _FILE_TYPES[filename]
    ext_types: dict[str, str] = {
        "csv": "data",
        "parquet": "data",
        "json": "json",
        "jsonl": "jsonl",
        "pkl": "pickle",
        "png": "plot",
        "html": "report",
        "txt": "text",
        "md": "doc",
        "log": "log",
        "yaml": "config",
        "yml": "config",
    }
    return ext_types.get(extension.lower(), "file")


def _infer_schema_hint(filename: str) -> str | None:
    _HINTS: dict[str, str] = {
        "manifest.json": "pipeline_manifest",
        "metrics.json": "pipeline_metrics",
        "_SUCCESS.json": "artifact_marker",
        "_FAILED.json": "artifact_marker",
        "_RUNNING.json": "artifact_marker",
        "scenario_catalog.json": "scenario_catalog",
    }
    return _HINTS.get(filename)


def _str_or_none(val: Any) -> str | None:
    if val is None:
        return None
    s = str(val).strip()
    return s if s else None


def _relative_posix(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def _posix_str(path: Path) -> str:
    return path.as_posix()
