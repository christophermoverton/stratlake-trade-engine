from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from src.catalog import (
    EvidenceReviewError,
    build_catalog,
    build_derived_index,
    build_evidence_review_for_workflow,
)
from src.catalog.resolver import CanonicalRecordResolution
from tests.catalog_scale_fixtures import build_catalog_scale_tree, snapshot_tree


def test_builds_selected_run_review_model_from_direct_scan(tmp_path: Path) -> None:
    build_catalog_scale_tree(tmp_path)

    model = build_evidence_review_for_workflow(
        ".",
        repo_root=tmp_path,
        selected_run_id="strategy_000",
    )

    assert model["schema_version"] == "review_pack.v1"
    assert model["selected_record"]["run_id"] == "strategy_000"
    assert model["review_root"].startswith("artifacts/_derived/evidence_review/review_")
    assert model["load_source_summary"] == {
        "index_path": None,
        "index_validated": False,
        "loaded_from": "direct_scan",
        "requested_mode": "direct",
        "resolved_mode": "direct",
    }
    assert model["resolver_resolution"]["resolution_status"] == "resolved"
    assert model["lineage_summary"]["formats"] == ["openlineage", "prov"]
    assert "portfolio_000" in model["lineage_summary"]["related_run_ids"]
    assert model["canonicality"]["derived_class"] == "review_pack"
    assert model["load_source"]["loaded_from"] == "review_pack"
    assert model["catalog_health_diagnostics"]["schema_version"] == "catalog_health_diagnostics.v1"


def test_builds_selected_catalog_id_review_model(tmp_path: Path) -> None:
    build_catalog_scale_tree(tmp_path)
    selected = next(record for record in build_catalog(tmp_path, repo_root=tmp_path) if record.run_id == "strategy_000")

    model = build_evidence_review_for_workflow(
        ".",
        repo_root=tmp_path,
        selected_catalog_id=selected.catalog_id,
    )

    assert model["selected_record"]["catalog_id"] == selected.catalog_id
    assert model["selected_record"]["run_id"] == "strategy_000"


def test_selection_errors_are_explicit_and_deterministic(tmp_path: Path) -> None:
    build_catalog_scale_tree(tmp_path)

    with pytest.raises(EvidenceReviewError, match="required"):
        build_evidence_review_for_workflow(".", repo_root=tmp_path)
    with pytest.raises(EvidenceReviewError, match="not found"):
        build_evidence_review_for_workflow(".", repo_root=tmp_path, selected_run_id="missing")


def test_direct_index_and_auto_review_models_preserve_identity(tmp_path: Path) -> None:
    build_catalog_scale_tree(tmp_path)
    index_path = tmp_path / "_derived" / "catalog_index" / "catalog_index.sqlite"
    build_derived_index(tmp_path, index_path, repo_root=tmp_path)

    direct = build_evidence_review_for_workflow(".", repo_root=tmp_path, selected_run_id="strategy_000")
    indexed = build_evidence_review_for_workflow(
        ".",
        repo_root=tmp_path,
        index_path=index_path,
        index_mode="index",
        selected_run_id="strategy_000",
    )
    auto = build_evidence_review_for_workflow(
        ".",
        repo_root=tmp_path,
        index_path=index_path,
        index_mode="auto",
        selected_run_id="strategy_000",
    )

    assert direct["selected_record"] == indexed["selected_record"] == auto["selected_record"]
    assert direct["related_records"] == indexed["related_records"] == auto["related_records"]
    assert direct["lineage_summary"] == indexed["lineage_summary"] == auto["lineage_summary"]
    assert indexed["load_source_summary"]["resolved_mode"] == "index"
    assert auto["load_source_summary"]["resolved_mode"] == "index"


def test_review_id_is_deterministic_from_normalized_request_inputs(tmp_path: Path) -> None:
    build_catalog_scale_tree(tmp_path)

    first = build_evidence_review_for_workflow(".", repo_root=tmp_path, selected_run_id="strategy_000")
    second = build_evidence_review_for_workflow(".", repo_root=tmp_path, selected_run_id="strategy_000")
    different = build_evidence_review_for_workflow(
        ".",
        repo_root=tmp_path,
        selected_run_id="strategy_000",
        resolve_related=True,
    )

    assert first["review_id"] == second["review_id"]
    assert first == second
    assert first["review_id"] != different["review_id"]


def test_resolver_partial_and_missing_source_warnings_are_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_catalog_scale_tree(tmp_path)
    record = next(record for record in build_catalog(tmp_path, repo_root=tmp_path) if record.run_id == "strategy_000")
    missing_path = "strategies/strategy_000/missing.json"

    from src.catalog.resolver import resolve_canonical_record as real_resolve_canonical_record

    def with_missing_source(*args, **kwargs):
        return real_resolve_canonical_record(
            replace(record, source_files=[*record.source_files, missing_path]),
            **kwargs,
        )

    monkeypatch.setattr("src.catalog.review_pack.resolve_canonical_record", with_missing_source)

    model = build_evidence_review_for_workflow(
        ".",
        repo_root=tmp_path,
        selected_run_id="strategy_000",
    )

    assert model["resolver_resolution"]["resolution_status"] == "partial"
    assert missing_path in model["resolver_resolution"]["missing_sources"]
    assert f"missing_source:{missing_path}" in model["warning_summary"]["warnings"]


def test_resolver_unresolved_status_is_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_catalog_scale_tree(tmp_path)
    record = next(record for record in build_catalog(tmp_path, repo_root=tmp_path) if record.run_id == "strategy_000")

    def unresolved(*args, **kwargs) -> CanonicalRecordResolution:
        return CanonicalRecordResolution(
            record=record,
            source_paths=[],
            resolved_sources=[],
            missing_sources=[],
            source_fingerprint=None,
            resolution_status="unresolved",
            canonicality_status="not_applicable",
            load_source={},
            warnings=["no_declared_sources"],
        )

    monkeypatch.setattr("src.catalog.review_pack.resolve_canonical_record", unresolved)

    model = build_evidence_review_for_workflow(
        ".",
        repo_root=tmp_path,
        selected_run_id="strategy_000",
    )

    assert model["resolver_resolution"]["resolution_status"] == "unresolved"
    assert model["canonical_sources"] == []


def test_resolve_related_is_opt_in(tmp_path: Path) -> None:
    build_catalog_scale_tree(tmp_path)

    default = build_evidence_review_for_workflow(".", repo_root=tmp_path, selected_run_id="strategy_000")
    expanded = build_evidence_review_for_workflow(
        ".",
        repo_root=tmp_path,
        selected_run_id="strategy_000",
        resolve_related=True,
    )

    assert default["related_resolver_resolutions"] == []
    assert expanded["related_resolver_resolutions"]


def test_builder_is_json_safe_portable_and_side_effect_free(tmp_path: Path) -> None:
    build_catalog_scale_tree(tmp_path)
    before = snapshot_tree(tmp_path)

    model = build_evidence_review_for_workflow(".", repo_root=tmp_path, selected_run_id="strategy_000")
    serialized = json.dumps(model, sort_keys=True)

    assert "\\" not in serialized
    assert "file://" not in serialized
    assert str(tmp_path) not in serialized
    assert not (tmp_path / "artifacts" / "_derived" / "evidence_review").exists()
    assert snapshot_tree(tmp_path) == before


def test_explicit_review_id_is_used_without_writing_outputs(tmp_path: Path) -> None:
    build_catalog_scale_tree(tmp_path)

    model = build_evidence_review_for_workflow(
        ".",
        repo_root=tmp_path,
        selected_run_id="strategy_000",
        review_id="manual_review",
        lineage_format="prov",
    )

    assert model["review_id"] == "manual_review"
    assert model["review_root"] == "artifacts/_derived/evidence_review/manual_review"
    assert model["lineage_summary"]["formats"] == ["prov"]
    assert set(model["selected_lineage"]) == {"prov"}
