from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from src.catalog import (
    build_catalog,
    build_derived_index,
    load_catalog_records,
    records_to_dicts,
    records_to_rows,
    resolve_canonical_record,
    resolve_canonical_record_by_id,
)
from tests.catalog_scale_fixtures import build_catalog_scale_tree, snapshot_tree


def test_resolves_direct_scan_record_to_canonical_sources(tmp_path: Path) -> None:
    build_catalog_scale_tree(tmp_path)
    record = next(record for record in build_catalog(tmp_path, repo_root=tmp_path) if record.run_id == "strategy_000")

    resolved = resolve_canonical_record(record, artifacts_root=tmp_path, repo_root=tmp_path)

    assert resolved.resolution_status == "resolved"
    assert resolved.canonicality_status == "not_applicable"
    assert {source.kind for source in resolved.resolved_sources} >= {"registry", "manifest", "marker"}
    assert resolved.source_fingerprint is not None


def test_resolves_index_loaded_record_and_serialized_record(tmp_path: Path) -> None:
    build_catalog_scale_tree(tmp_path)
    index_path = tmp_path / "_derived" / "catalog_index" / "catalog_index.sqlite"
    build_derived_index(tmp_path, index_path, repo_root=tmp_path)
    indexed = next(
        record
        for record in load_catalog_records(tmp_path, repo_root=tmp_path, index_path=index_path, mode="index")
        if record.run_id == "strategy_000"
    )
    as_dict = records_to_dicts([indexed])[0]

    from_index = resolve_canonical_record(indexed, artifacts_root=tmp_path, repo_root=tmp_path)
    from_dict = resolve_canonical_record(as_dict, artifacts_root=tmp_path, repo_root=tmp_path)

    assert from_index.resolution_status == "resolved"
    assert from_dict.resolution_status == "resolved"
    assert from_index.source_fingerprint == from_dict.source_fingerprint
    assert from_dict.canonicality_status == "legacy_no_envelope"


def test_row_representation_is_accepted_and_unresolved_without_source_paths(tmp_path: Path) -> None:
    build_catalog_scale_tree(tmp_path)
    record = next(record for record in build_catalog(tmp_path, repo_root=tmp_path) if record.run_id == "strategy_000")
    row = records_to_rows([record])[0]

    resolved = resolve_canonical_record(row, artifacts_root=tmp_path, repo_root=tmp_path)

    assert resolved.record.run_id == "strategy_000"
    assert resolved.resolution_status == "unresolved"
    assert resolved.source_paths == []


def test_missing_and_non_portable_sources_fail_safely(tmp_path: Path) -> None:
    build_catalog_scale_tree(tmp_path)
    record = next(record for record in build_catalog(tmp_path, repo_root=tmp_path) if record.run_id == "strategy_000")
    (tmp_path / record.source_manifest_path).unlink()
    with_bad_path = replace(record, source_files=[*record.source_files, "../outside.json"])

    resolved = resolve_canonical_record(with_bad_path, artifacts_root=tmp_path, repo_root=tmp_path)

    assert resolved.resolution_status == "partial"
    assert record.source_manifest_path in resolved.missing_sources
    assert "../outside.json" in resolved.missing_sources
    assert f"missing_source:{record.source_manifest_path}" in resolved.warnings
    assert "non_portable_source_path:../outside.json" in resolved.warnings


def test_resolver_is_read_only_deterministic_and_does_not_require_index(tmp_path: Path) -> None:
    build_catalog_scale_tree(tmp_path)
    before = snapshot_tree(tmp_path)
    first = resolve_canonical_record_by_id(
        artifacts_root=tmp_path,
        repo_root=tmp_path,
        run_id="strategy_000",
    )
    second = resolve_canonical_record_by_id(
        artifacts_root=tmp_path,
        repo_root=tmp_path,
        run_id="strategy_000",
    )

    assert first.source_fingerprint == second.source_fingerprint
    assert snapshot_tree(tmp_path) == before
