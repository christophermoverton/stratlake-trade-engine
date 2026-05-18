from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil
import sqlite3

from src.catalog import (
    build_canonicality_envelope,
    build_catalog,
    build_derived_index,
    build_evidence_view_for_workflow,
    build_lineage_export_for_workflow,
    load_catalog_records_with_source,
    resolve_canonical_record,
    validate_derived_index,
)
from tests.catalog_scale_fixtures import build_catalog_scale_tree, snapshot_tree
from tests.test_m37_architecture_guardrails import _forbidden_derived_imports_from_source


def test_m37_metadata_outputs_are_deterministic(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    first_index_path = artifacts_root / "_derived" / "catalog_index" / "first.sqlite"
    second_index_path = artifacts_root / "_derived" / "catalog_index" / "second.sqlite"
    build_catalog_scale_tree(artifacts_root)

    first_metadata = build_derived_index(artifacts_root, first_index_path, repo_root=tmp_path)
    first_load = load_catalog_records_with_source(
        artifacts_root,
        repo_root=tmp_path,
        index_path=first_index_path,
        mode="auto",
    )
    first_lineage = build_lineage_export_for_workflow(
        "artifacts",
        repo_root=tmp_path,
        index_path=first_index_path,
        index_mode="auto",
        selected_run_id="strategy_000",
    )
    first_evidence = build_evidence_view_for_workflow(
        "artifacts",
        repo_root=tmp_path,
        index_path=first_index_path,
        index_mode="auto",
        selected_run_id="strategy_000",
    )
    first_envelope = build_canonicality_envelope(
        derived_class="lineage_export",
        authority_paths=["artifacts/zeta/manifest.json", "artifacts/alpha/manifest.json"],
        fingerprint_payload={"b": 2, "a": 1},
    )

    second_metadata = build_derived_index(artifacts_root, second_index_path, repo_root=tmp_path)
    second_load = load_catalog_records_with_source(
        artifacts_root,
        repo_root=tmp_path,
        index_path=second_index_path,
        mode="auto",
    )
    second_lineage = build_lineage_export_for_workflow(
        "artifacts",
        repo_root=tmp_path,
        index_path=second_index_path,
        index_mode="auto",
        selected_run_id="strategy_000",
    )
    second_evidence = build_evidence_view_for_workflow(
        "artifacts",
        repo_root=tmp_path,
        index_path=second_index_path,
        index_mode="auto",
        selected_run_id="strategy_000",
    )
    second_envelope = build_canonicality_envelope(
        derived_class="lineage_export",
        authority_paths=["artifacts/alpha/manifest.json", "artifacts/zeta/manifest.json"],
        fingerprint_payload={"a": 1, "b": 2},
    )

    assert second_metadata == first_metadata
    assert second_load.load_source | {"index_path": first_load.load_source["index_path"]} == first_load.load_source
    assert _normalize_index_path(second_lineage, replacement=first_lineage["load_source"]["index_path"]) == first_lineage
    assert _normalize_index_path(second_evidence, replacement=first_evidence["load_source"]["index_path"]) == first_evidence
    assert second_envelope == first_envelope


def test_m37_metadata_paths_are_portable(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    index_path = artifacts_root / "_derived" / "catalog_index" / "catalog_index.sqlite"
    build_catalog_scale_tree(artifacts_root)
    metadata = build_derived_index(artifacts_root, index_path, repo_root=tmp_path)
    load_result = load_catalog_records_with_source(
        artifacts_root,
        repo_root=tmp_path,
        index_path=index_path,
        mode="auto",
    )
    lineage = build_lineage_export_for_workflow(
        "artifacts",
        repo_root=tmp_path,
        index_path=index_path,
        index_mode="auto",
        selected_run_id="strategy_000",
    )
    evidence = build_evidence_view_for_workflow(
        "artifacts",
        repo_root=tmp_path,
        index_path=index_path,
        index_mode="auto",
        selected_run_id="strategy_000",
    )
    record = next(record for record in build_catalog(artifacts_root, repo_root=tmp_path) if record.run_id == "strategy_000")
    resolved = resolve_canonical_record(record, artifacts_root=artifacts_root, repo_root=tmp_path)

    _assert_portable(
        metadata,
        load_result.load_source,
        lineage,
        evidence,
        _resolver_path_payload(resolved),
        tmp_path=tmp_path,
    )


def test_direct_index_auto_record_identity_parity(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    index_path = artifacts_root / "_derived" / "catalog_index" / "catalog_index.sqlite"
    build_catalog_scale_tree(artifacts_root)
    build_derived_index(artifacts_root, index_path, repo_root=tmp_path)

    direct = load_catalog_records_with_source(artifacts_root, repo_root=tmp_path, mode="direct")
    indexed = load_catalog_records_with_source(
        artifacts_root,
        repo_root=tmp_path,
        index_path=index_path,
        mode="index",
    )
    auto_index = load_catalog_records_with_source(
        artifacts_root,
        repo_root=tmp_path,
        index_path=index_path,
        mode="auto",
    )
    auto_direct = load_catalog_records_with_source(
        artifacts_root,
        repo_root=tmp_path,
        index_path=artifacts_root / "_derived" / "catalog_index" / "missing.sqlite",
        mode="auto",
    )

    expected = _identity_rows(direct.records)
    assert _identity_rows(indexed.records) == expected
    assert _identity_rows(auto_index.records) == expected
    assert _identity_rows(auto_direct.records) == expected
    assert direct.load_source["resolved_mode"] == "direct"
    assert auto_index.load_source["resolved_mode"] == "index"
    assert auto_direct.load_source["resolved_mode"] == "direct"


def test_derived_outputs_are_disposable_without_changing_canonical_identity(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    index_path = artifacts_root / "_derived" / "catalog_index" / "catalog_index.sqlite"
    build_catalog_scale_tree(artifacts_root)
    baseline_records = _identity_rows(build_catalog(artifacts_root, repo_root=tmp_path))
    baseline_snapshot = snapshot_tree(artifacts_root)

    build_derived_index(artifacts_root, index_path, repo_root=tmp_path)
    derived_only = artifacts_root / "_derived" / "evidence" / "view"
    derived_only.mkdir(parents=True)
    (derived_only / "manifest.json").write_text('{"run_id":"derived_only"}\n', encoding="utf-8")
    with_derived = _identity_rows(build_catalog(artifacts_root, repo_root=tmp_path))
    shutil.rmtree(artifacts_root / "_derived")
    after_delete = _identity_rows(build_catalog(artifacts_root, repo_root=tmp_path))

    assert with_derived == baseline_records
    assert after_delete == baseline_records
    assert snapshot_tree(artifacts_root) == baseline_snapshot


def test_resolver_fingerprints_are_stable_for_direct_and_index_records(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    index_path = artifacts_root / "_derived" / "catalog_index" / "catalog_index.sqlite"
    build_catalog_scale_tree(artifacts_root)
    direct_record = next(
        record
        for record in build_catalog(artifacts_root, repo_root=tmp_path)
        if record.run_id == "strategy_000"
    )
    build_derived_index(artifacts_root, index_path, repo_root=tmp_path)
    indexed_record = next(
        record
        for record in load_catalog_records_with_source(
            artifacts_root,
            repo_root=tmp_path,
            index_path=index_path,
            mode="index",
        ).records
        if record.run_id == "strategy_000"
    )

    direct_first = resolve_canonical_record(direct_record, artifacts_root=artifacts_root, repo_root=tmp_path)
    direct_second = resolve_canonical_record(direct_record, artifacts_root=artifacts_root, repo_root=tmp_path)
    indexed = resolve_canonical_record(indexed_record, artifacts_root=artifacts_root, repo_root=tmp_path)
    unsafe = resolve_canonical_record(
        replace(direct_record, source_files=[*direct_record.source_files, "../outside.json"]),
        artifacts_root=artifacts_root,
        repo_root=tmp_path,
    )

    assert direct_first.source_fingerprint == direct_second.source_fingerprint == indexed.source_fingerprint
    assert direct_first.resolution_status == indexed.resolution_status == "resolved"
    assert unsafe.resolution_status == "partial"
    assert "../outside.json" in unsafe.missing_sources
    assert "non_portable_source_path:../outside.json" in unsafe.warnings


def test_legacy_no_envelope_compatibility_remains_readable(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    index_path = artifacts_root / "_derived" / "catalog_index" / "catalog_index.sqlite"
    build_catalog_scale_tree(artifacts_root)
    build_derived_index(artifacts_root, index_path, repo_root=tmp_path)

    with sqlite3.connect(index_path) as connection:
        connection.execute("DELETE FROM metadata WHERE key = 'canonicality'")
        connection.commit()

    validation = validate_derived_index(index_path, artifacts_root=artifacts_root, repo_root=tmp_path)

    assert validation.metadata["canonicality_status"] == "legacy_no_envelope"
    assert len(validation.records) == 48


def test_architecture_guardrail_detection_remains_integrated() -> None:
    assert _forbidden_derived_imports_from_source("import src.catalog.derived_index") == [
        ("src.catalog.derived_index", None)
    ]
    assert _forbidden_derived_imports_from_source("from src.catalog import build_derived_index") == [
        ("src.catalog", "build_derived_index")
    ]
    assert _forbidden_derived_imports_from_source(
        "from src.catalog import resolve_canonical_record"
    ) == []


def _identity_rows(records: list) -> list[tuple[str, str | None, str, str, str | None, str | None]]:
    return [
        (
            record.catalog_id,
            record.run_id,
            record.run_type,
            record.artifact_root,
            record.source_manifest_path,
            record.source_marker_path,
        )
        for record in records
    ]


def _assert_portable(*payloads: object, tmp_path: Path) -> None:
    serialized = "".join(json.dumps(payload, sort_keys=True) for payload in payloads)
    assert str(tmp_path) not in serialized
    assert "file://" not in serialized
    assert "\\" not in serialized
    assert "C:/" not in serialized
    assert "../" not in serialized
    assert '": "/' not in serialized


def _normalize_index_path(payload: dict, *, replacement: str) -> dict:
    normalized = json.loads(json.dumps(payload, sort_keys=True))
    normalized["load_source"]["index_path"] = replacement
    return normalized


def _resolver_path_payload(resolved) -> dict[str, object]:
    return {
        "source_paths": resolved.source_paths,
        "resolved_source_paths": [source.path for source in resolved.resolved_sources],
        "missing_sources": resolved.missing_sources,
        "warnings": resolved.warnings,
    }
