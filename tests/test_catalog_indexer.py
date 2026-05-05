"""Tests for src/catalog/indexer.py — M29 read-only catalog indexer.

All tests use temporary directories with synthetic artifacts. No local generated
artifacts are used.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from src.catalog.indexer import (
    build_artifact_records,
    build_catalog,
    build_catalog_record,
    discover_artifact_roots,
    load_json_file,
)
from src.catalog.models import ArtifactRecord, CatalogRecord, CatalogValidationStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r) for r in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_strategy_root(
    artifacts_root: Path,
    run_id: str,
    *,
    with_registry: bool = True,
    marker: str = "_SUCCESS.json",
    with_manifest: bool = True,
    manifest_artifacts: list[str] | None = None,
    extra_files: list[str] | None = None,
    strategy_name: str = "TestStrategy",
) -> Path:
    """Create a synthetic strategy artifact root."""
    run_dir = artifacts_root / "strategies" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    write_json(run_dir / marker, {"status": "completed", "recorded_at_utc": "2025-01-01T00:00:00Z"})
    write_json(run_dir / "metrics.json", {"sharpe_ratio": 1.5, "cagr": 0.12})
    write_json(run_dir / "summary.json", {"strategy_name": strategy_name, "timeframe": "1d"})

    if with_manifest:
        declared = manifest_artifacts or ["metrics.json", "summary.json", marker]
        write_json(
            run_dir / "manifest.json",
            {
                "run_id": run_id,
                "artifacts": declared,
                "strategy_name": strategy_name,
            },
        )

    for fname in extra_files or []:
        (run_dir / fname).parent.mkdir(parents=True, exist_ok=True)
        (run_dir / fname).write_text("data", encoding="utf-8")

    if with_registry:
        registry_path = artifacts_root / "strategies" / "registry.jsonl"
        entry = {
            "run_id": run_id,
            "run_type": "strategy",
            "strategy_name": strategy_name,
            "timeframe": "1d",
            "start_ts": "2020-01-01",
            "end_ts": "2024-01-01",
            "artifact_dir": run_dir.as_posix(),
        }
        existing = []
        if registry_path.exists():
            existing = [json.loads(l) for l in registry_path.read_text().splitlines() if l.strip()]
        existing.append(entry)
        write_jsonl(registry_path, existing)

    return run_dir


# ---------------------------------------------------------------------------
# Test 1: Strategy artifact root with registry + manifest + success marker
# ---------------------------------------------------------------------------


def test_strategy_root_with_registry_manifest_success(tmp_path):
    """Catalog record created for a completed strategy run."""
    run_id = "strat_20250101_abc123"
    make_strategy_root(tmp_path, run_id, marker="_SUCCESS.json")

    records = build_catalog(tmp_path, repo_root=tmp_path)

    assert len(records) >= 1
    strat_records = [r for r in records if r.run_id == run_id]
    assert len(strat_records) == 1
    r = strat_records[0]

    assert r.status == "completed"
    assert r.source_manifest_path is not None
    assert r.source_marker_path is not None
    assert r.source_registry_path is not None
    assert r.validation.marker_status == "present"
    assert r.validation.manifest_status == "present"

    # Artifact records
    artifact_root_abs = tmp_path / r.artifact_root
    ar_list = build_artifact_records(r, repo_root=tmp_path)
    assert len(ar_list) > 0
    relpaths = [a.relative_path for a in ar_list]
    assert "metrics.json" in relpaths or any("metrics" in p for p in relpaths)


# ---------------------------------------------------------------------------
# Test 2: Artifact root without registry entry
# ---------------------------------------------------------------------------


def test_artifact_root_without_registry_entry(tmp_path):
    """Record is created when no registry entry exists; warning is included."""
    run_id = "orphan_20250101_xyz999"
    run_dir = tmp_path / "strategies" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "_SUCCESS.json", {"status": "completed"})
    write_json(run_dir / "metrics.json", {"sharpe_ratio": 0.9})

    records = build_catalog(tmp_path, repo_root=tmp_path)
    orphan_records = [r for r in records if r.run_id == run_id or run_id in r.artifact_root]
    assert len(orphan_records) >= 1
    r = orphan_records[0]
    assert "artifact_root_no_registry_entry" in r.validation.validation_warnings


# ---------------------------------------------------------------------------
# Test 3: Registry entry without artifact root
# ---------------------------------------------------------------------------


def test_registry_entry_without_artifact_root(tmp_path):
    """Registry-only records are included with deterministic behavior and warning."""
    run_id = "ghost_20250101_ghost1"
    # Write registry entry but do NOT create the artifact directory
    registry_path = tmp_path / "strategies" / "registry.jsonl"
    entry = {
        "run_id": run_id,
        "run_type": "strategy",
        "strategy_name": "GhostStrategy",
        "artifact_dir": (tmp_path / "strategies" / run_id).as_posix(),
    }
    write_jsonl(registry_path, [entry])

    records = build_catalog(tmp_path, repo_root=tmp_path)
    ghost_records = [r for r in records if r.run_id == run_id]
    assert len(ghost_records) == 1
    r = ghost_records[0]
    assert r.status == "registry_only"
    assert "registry_entry_no_artifact_root" in r.validation.validation_warnings


# ---------------------------------------------------------------------------
# Test 4: Failed run marker
# ---------------------------------------------------------------------------


def test_failed_run_marker(tmp_path):
    run_id = "strat_20250101_fail01"
    make_strategy_root(tmp_path, run_id, marker="_FAILED.json", with_manifest=False)

    records = build_catalog(tmp_path, repo_root=tmp_path)
    r = next((x for x in records if x.run_id == run_id), None)
    assert r is not None
    assert r.status == "failed"
    assert "failed_marker_present" in r.validation.validation_warnings


# ---------------------------------------------------------------------------
# Test 5: Running run marker
# ---------------------------------------------------------------------------


def test_running_run_marker(tmp_path):
    run_id = "strat_20250101_run001"
    make_strategy_root(tmp_path, run_id, marker="_RUNNING.json", with_manifest=False)

    records = build_catalog(tmp_path, repo_root=tmp_path)
    r = next((x for x in records if x.run_id == run_id), None)
    assert r is not None
    assert r.status == "running"
    assert "running_marker_present" in r.validation.validation_warnings


# ---------------------------------------------------------------------------
# Test 6: Missing declared artifact
# ---------------------------------------------------------------------------


def test_missing_declared_artifact(tmp_path):
    """ArtifactRecord(exists=False) is created for manifest-declared missing files."""
    run_id = "strat_20250101_mis001"
    # Declare a file in manifest that does not exist on disk
    make_strategy_root(
        tmp_path,
        run_id,
        marker="_SUCCESS.json",
        manifest_artifacts=["metrics.json", "does_not_exist.csv"],
    )

    records = build_catalog(tmp_path, repo_root=tmp_path)
    r = next((x for x in records if x.run_id == run_id), None)
    assert r is not None

    ar_list = build_artifact_records(r, repo_root=tmp_path)
    missing = [a for a in ar_list if a.relative_path == "does_not_exist.csv"]
    assert len(missing) == 1
    assert missing[0].exists is False
    assert missing[0].declared_in_manifest is True

    # Validation warning
    assert any("manifest_artifact_missing" in w for w in r.validation.validation_warnings)


# ---------------------------------------------------------------------------
# Test 7: Undeclared artifact
# ---------------------------------------------------------------------------


def test_undeclared_artifact(tmp_path):
    """Discovered files not declared in the manifest are flagged."""
    run_id = "strat_20250101_und001"
    # Only declare metrics.json in manifest but also create an extra file
    make_strategy_root(
        tmp_path,
        run_id,
        marker="_SUCCESS.json",
        manifest_artifacts=["metrics.json"],
        extra_files=["undeclared_output.csv"],
    )

    records = build_catalog(tmp_path, repo_root=tmp_path)
    r = next((x for x in records if x.run_id == run_id), None)
    assert r is not None

    ar_list = build_artifact_records(r, repo_root=tmp_path)
    undeclared = [a for a in ar_list if not a.declared_in_manifest]
    undeclared_names = [a.filename for a in undeclared]
    assert "undeclared_output.csv" in undeclared_names

    # Should have undeclared_artifact warning
    assert any("undeclared_artifact" in w for w in r.validation.validation_warnings)


# ---------------------------------------------------------------------------
# Test 8: Deterministic ordering
# ---------------------------------------------------------------------------


def test_deterministic_ordering(tmp_path):
    """Repeated calls return identical record order and catalog IDs."""
    for i in range(3):
        make_strategy_root(tmp_path, f"strat_20250101_det{i:03d}", with_manifest=True)

    records_1 = build_catalog(tmp_path, repo_root=tmp_path)
    records_2 = build_catalog(tmp_path, repo_root=tmp_path)

    assert len(records_1) == len(records_2)
    for r1, r2 in zip(records_1, records_2):
        assert r1.catalog_id == r2.catalog_id
        assert r1.run_id == r2.run_id
        assert r1.artifact_root == r2.artifact_root
        assert r1.run_type == r2.run_type


# ---------------------------------------------------------------------------
# Test 9: Read-only behavior
# ---------------------------------------------------------------------------


def test_read_only_behavior(tmp_path):
    """build_catalog must not modify any source files."""
    run_id = "strat_20250101_ro0001"
    run_dir = make_strategy_root(tmp_path, run_id, marker="_SUCCESS.json")

    # Collect (path, mtime, size) snapshot before
    def snapshot(root: Path) -> dict[str, tuple[float, int]]:
        result = {}
        for p in sorted(root.rglob("*")):
            if p.is_file():
                s = p.stat()
                result[p.as_posix()] = (s.st_mtime, s.st_size)
        return result

    before = snapshot(tmp_path)
    build_catalog(tmp_path, repo_root=tmp_path)
    after = snapshot(tmp_path)

    assert before == after, "build_catalog modified source files (mtime or size changed)"


# ---------------------------------------------------------------------------
# Test: load_json_file
# ---------------------------------------------------------------------------


def test_load_json_file_returns_none_for_missing(tmp_path):
    result = load_json_file(tmp_path / "nonexistent.json")
    assert result is None


def test_load_json_file_returns_dict(tmp_path):
    p = tmp_path / "data.json"
    write_json(p, {"key": "value"})
    result = load_json_file(p)
    assert result == {"key": "value"}


def test_load_json_file_returns_none_for_invalid_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{invalid json}", encoding="utf-8")
    result = load_json_file(p)
    assert result is None


# ---------------------------------------------------------------------------
# Test: discover_artifact_roots skips missing families gracefully
# ---------------------------------------------------------------------------


def test_discover_artifact_roots_empty(tmp_path):
    roots = discover_artifact_roots(tmp_path / "nonexistent_artifacts")
    assert roots == []


def test_discover_artifact_roots_finds_strategy_dirs(tmp_path):
    run_dir = tmp_path / "strategies" / "run_001"
    run_dir.mkdir(parents=True)
    write_json(run_dir / "_SUCCESS.json", {"status": "completed"})

    roots = discover_artifact_roots(tmp_path)
    assert run_dir.resolve() in [r.resolve() for r in roots]


# ---------------------------------------------------------------------------
# Test: Marker precedence — failed beats success and running
# ---------------------------------------------------------------------------


def test_marker_precedence_failed_wins(tmp_path):
    """_FAILED.json takes precedence over _SUCCESS.json."""
    run_dir = tmp_path / "strategies" / "strat_marker_test"
    run_dir.mkdir(parents=True)
    write_json(run_dir / "_FAILED.json", {"status": "failed"})
    write_json(run_dir / "_SUCCESS.json", {"status": "completed"})

    records = build_catalog(tmp_path, repo_root=tmp_path)
    r = next((x for x in records if "strat_marker_test" in x.artifact_root), None)
    assert r is not None
    assert r.status == "failed"


def test_marker_precedence_success_beats_running(tmp_path):
    """_SUCCESS.json takes precedence over _RUNNING.json."""
    run_dir = tmp_path / "strategies" / "strat_marker_sr"
    run_dir.mkdir(parents=True)
    write_json(run_dir / "_SUCCESS.json", {"status": "completed"})
    write_json(run_dir / "_RUNNING.json", {"status": "running"})

    records = build_catalog(tmp_path, repo_root=tmp_path)
    r = next((x for x in records if "strat_marker_sr" in x.artifact_root), None)
    assert r is not None
    assert r.status == "completed"


# ---------------------------------------------------------------------------
# Test: CatalogRecord / ArtifactRecord to_dict serializable
# ---------------------------------------------------------------------------


def test_catalog_record_to_dict_is_serializable(tmp_path):
    run_id = "strat_20250101_ser001"
    make_strategy_root(tmp_path, run_id, marker="_SUCCESS.json")

    records = build_catalog(tmp_path, repo_root=tmp_path)
    r = next((x for x in records if x.run_id == run_id), None)
    assert r is not None

    d = r.to_dict()
    # Should be JSON-serializable (no Path objects etc.)
    serialized = json.dumps(d)
    assert run_id in serialized


def test_artifact_record_to_dict_is_serializable(tmp_path):
    run_id = "strat_20250101_ser002"
    make_strategy_root(tmp_path, run_id, marker="_SUCCESS.json")

    records = build_catalog(tmp_path, repo_root=tmp_path)
    r = next((x for x in records if x.run_id == run_id), None)
    assert r is not None

    ar_list = build_artifact_records(r, repo_root=tmp_path)
    assert len(ar_list) > 0
    d = ar_list[0].to_dict()
    json.dumps(d)  # must not raise


# ---------------------------------------------------------------------------
# Test: Multiple artifact families (alpha, portfolios)
# ---------------------------------------------------------------------------


def test_alpha_artifact_family(tmp_path):
    """Alpha artifact roots are indexed under run_type=alpha_evaluation."""
    run_id = "alpha_eval_20250101_a01"
    run_dir = tmp_path / "alpha" / run_id
    run_dir.mkdir(parents=True)
    write_json(run_dir / "_SUCCESS.json", {"status": "completed"})
    write_json(run_dir / "alpha_metrics.json", {"mean_ic": 0.05, "ic_ir": 0.8})

    records = build_catalog(tmp_path, repo_root=tmp_path)
    r = next((x for x in records if run_id in x.artifact_root), None)
    assert r is not None
    assert r.run_type == "alpha_evaluation"
    assert r.status == "completed"


def test_portfolio_artifact_family(tmp_path):
    """Portfolio artifact roots are indexed under run_type=portfolio."""
    run_id = "port_20250101_p01"
    run_dir = tmp_path / "portfolios" / run_id
    run_dir.mkdir(parents=True)
    write_json(run_dir / "_SUCCESS.json", {"status": "completed"})
    write_json(run_dir / "summary.json", {"portfolio_name": "TestPortfolio"})

    records = build_catalog(tmp_path, repo_root=tmp_path)
    r = next((x for x in records if run_id in x.artifact_root), None)
    assert r is not None
    assert r.run_type == "portfolio"


# ---------------------------------------------------------------------------
# Test: benchmark_pack_* glob
# ---------------------------------------------------------------------------


def test_benchmark_pack_family(tmp_path):
    """benchmark_pack_* directories are discovered."""
    run_dir = tmp_path / "benchmark_pack_v1" / "run_bp_001"
    run_dir.mkdir(parents=True)
    write_json(run_dir / "_SUCCESS.json", {"status": "completed"})
    write_json(run_dir / "summary.json", {"batch": "v1"})

    records = build_catalog(tmp_path, repo_root=tmp_path)
    bp_records = [r for r in records if "benchmark_pack_v1" in r.artifact_root]
    assert len(bp_records) >= 1


# ---------------------------------------------------------------------------
# Fix 2 — Regression: registry metadata populates without manifest (Test A)
# ---------------------------------------------------------------------------


def test_registry_metadata_populates_without_manifest(tmp_path):
    """Registry fields must be populated even when manifest.json is absent."""
    run_id = "strategy_test_nm_001"
    run_dir = tmp_path / "strategies" / run_id
    run_dir.mkdir(parents=True)
    write_json(run_dir / "_SUCCESS.json", {"status": "completed"})
    write_json(run_dir / "metrics.json", {"sharpe_ratio": 1.2})
    # No manifest.json

    registry_path = tmp_path / "strategies" / "registry.jsonl"
    entry = {
        "run_id": run_id,
        "run_type": "strategy",
        "strategy_name": "momentum_v1",
        "timeframe": "1D",
        "start_ts": "2024-01-01",
        "end_ts": "2024-12-31",
        "review_status": "candidate",
        "promotion_status": "eligible",
        "artifact_dir": run_dir.as_posix(),
    }
    write_jsonl(registry_path, [entry])

    records = build_catalog(tmp_path, repo_root=tmp_path)
    r = next((x for x in records if x.run_id == run_id), None)
    assert r is not None, "No catalog record produced"

    assert r.strategy_name == "momentum_v1", f"Expected 'momentum_v1', got {r.strategy_name!r}"
    assert r.timeframe == "1D", f"Expected '1D', got {r.timeframe!r}"
    assert r.start_ts == "2024-01-01"
    assert r.end_ts == "2024-12-31"
    assert r.review_status == "candidate", f"Expected 'candidate', got {r.review_status!r}"
    assert r.promotion_status == "eligible", f"Expected 'eligible', got {r.promotion_status!r}"
    assert r.source_manifest_path is None
    assert r.validation.manifest_status == "missing"


# ---------------------------------------------------------------------------
# Fix 2 — Regression: summary metadata populates without manifest (Test B)
# ---------------------------------------------------------------------------


def test_summary_metadata_populates_without_manifest(tmp_path):
    """summary.json fields must populate when neither registry nor manifest exists."""
    run_id = "strat_nm_summary_001"
    run_dir = tmp_path / "strategies" / run_id
    run_dir.mkdir(parents=True)
    write_json(run_dir / "_SUCCESS.json", {"status": "completed"})
    write_json(
        run_dir / "summary.json",
        {"strategy_name": "mean_reversion_v1", "timeframe": "1D"},
    )
    # No registry.jsonl, no manifest.json

    records = build_catalog(tmp_path, repo_root=tmp_path)
    r = next((x for x in records if x.run_id == run_id), None)
    assert r is not None, "No catalog record produced"

    assert r.strategy_name == "mean_reversion_v1", f"Expected 'mean_reversion_v1', got {r.strategy_name!r}"
    assert r.timeframe == "1D", f"Expected '1D', got {r.timeframe!r}"
    assert r.source_registry_path is None
    assert r.source_manifest_path is None


# ---------------------------------------------------------------------------
# Fix 2 — Regression: registry wins over summary and manifest (Test C)
# ---------------------------------------------------------------------------


def test_registry_wins_over_summary_and_manifest(tmp_path):
    """Registry strategy_name takes precedence over summary.json and manifest.json."""
    run_id = "strat_precedence_001"
    run_dir = tmp_path / "strategies" / run_id
    run_dir.mkdir(parents=True)
    write_json(run_dir / "_SUCCESS.json", {"status": "completed"})
    write_json(run_dir / "summary.json", {"strategy_name": "summary_strategy"})
    write_json(
        run_dir / "manifest.json",
        {"run_id": run_id, "strategy_name": "manifest_strategy"},
    )

    registry_path = tmp_path / "strategies" / "registry.jsonl"
    entry = {
        "run_id": run_id,
        "run_type": "strategy",
        "strategy_name": "registry_strategy",
        "artifact_dir": run_dir.as_posix(),
    }
    write_jsonl(registry_path, [entry])

    records = build_catalog(tmp_path, repo_root=tmp_path)
    r = next((x for x in records if x.run_id == run_id), None)
    assert r is not None

    assert r.strategy_name == "registry_strategy", (
        f"Expected 'registry_strategy' (registry wins), got {r.strategy_name!r}"
    )


# ---------------------------------------------------------------------------
# Fix 2 — Regression: review_status from review_metadata nested schema
# ---------------------------------------------------------------------------


def test_review_status_from_review_metadata(tmp_path):
    """review_status is populated from review_metadata.status (strategy registry schema)."""
    run_id = "strat_review_meta_001"
    run_dir = tmp_path / "strategies" / run_id
    run_dir.mkdir(parents=True)
    write_json(run_dir / "_SUCCESS.json", {"status": "completed"})

    registry_path = tmp_path / "strategies" / "registry.jsonl"
    entry = {
        "run_id": run_id,
        "run_type": "strategy",
        "strategy_name": "rv_strategy",
        "artifact_dir": run_dir.as_posix(),
        "review_metadata": {
            "status": "promoted",
            "promotion_status": "passed",
        },
    }
    write_jsonl(registry_path, [entry])

    records = build_catalog(tmp_path, repo_root=tmp_path)
    r = next((x for x in records if x.run_id == run_id), None)
    assert r is not None

    assert r.review_status == "promoted", f"Expected 'promoted', got {r.review_status!r}"
    assert r.promotion_status == "passed", f"Expected 'passed', got {r.promotion_status!r}"


# ---------------------------------------------------------------------------
# Fix 2 — Regression: review_status from top-level field (portfolio schema)
# ---------------------------------------------------------------------------


def test_review_status_from_top_level_field(tmp_path):
    """review_status populated from top-level field (portfolio registry schema)."""
    run_id = "port_review_tl_001"
    run_dir = tmp_path / "portfolios" / run_id
    run_dir.mkdir(parents=True)
    write_json(run_dir / "_SUCCESS.json", {"status": "completed"})

    registry_path = tmp_path / "portfolios" / "registry.jsonl"
    entry = {
        "run_id": run_id,
        "run_type": "portfolio",
        "portfolio_name": "test_port",
        "artifact_dir": run_dir.as_posix(),
        "review_status": "needs_review",
        "promotion_status": "pending",
    }
    write_jsonl(registry_path, [entry])

    records = build_catalog(tmp_path, repo_root=tmp_path)
    r = next((x for x in records if x.run_id == run_id), None)
    assert r is not None

    assert r.review_status == "needs_review", f"Expected 'needs_review', got {r.review_status!r}"
    assert r.promotion_status == "pending", f"Expected 'pending', got {r.promotion_status!r}"
