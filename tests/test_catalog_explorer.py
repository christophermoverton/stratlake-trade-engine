from __future__ import annotations

import json
from pathlib import Path

from src.catalog.explorer import (
    build_evidence_explorer_view,
    render_evidence_json,
    render_evidence_markdown,
    render_evidence_table,
)
from src.catalog.models import CatalogRecord, CatalogValidationStatus
from src.catalog.query import CatalogQuery


def _validation() -> CatalogValidationStatus:
    return CatalogValidationStatus(
        catalog_status="valid",
        marker_status="present",
        manifest_status="missing",
        artifact_status="ok",
        qa_status=None,
        validation_errors=[],
        validation_warnings=[],
    )


def _record(
    run_id: str,
    run_type: str,
    *,
    record_family: str | None = None,
    metadata: dict | None = None,
    artifact_root: str | None = None,
    robustness_status: str | None = None,
    wfe_status: str | None = None,
    governance_status: str | None = None,
    promotion_review_status: str | None = None,
    validation_readiness_present: bool = False,
    release_validation_present: bool = False,
) -> CatalogRecord:
    return CatalogRecord(
        catalog_id=f"catalog_{run_id}",
        run_id=run_id,
        run_type=run_type,
        status="completed",
        artifact_root=artifact_root or f"artifacts/{run_type}/{run_id}",
        source_registry_path=None,
        source_manifest_path=None,
        source_marker_path=None,
        created_at=None,
        timeframe=None,
        start_ts=None,
        end_ts=None,
        strategy_name=None,
        portfolio_name=None,
        allocator_name=None,
        alpha_model_name=None,
        regime_method=None,
        campaign_id=None,
        scenario_id=None,
        metrics_summary=None,
        qa_status=None,
        review_status="candidate",
        promotion_status="pending",
        record_family=record_family,
        robustness_status=robustness_status,
        wfe_status=wfe_status,
        governance_status=governance_status,
        promotion_review_status=promotion_review_status,
        validation_readiness_present=validation_readiness_present,
        release_validation_present=release_validation_present,
        tags=[],
        source_files=[],
        metadata=metadata or {},
        validation=_validation(),
    )


def _records() -> list[CatalogRecord]:
    strategy = _record("strategy_a", "strategy", artifact_root="strategies/strategy_a")
    robustness = _record(
        "robustness_a",
        "robustness_bundle",
        record_family="robustness_bundle",
        metadata={"source_run_ids": ["strategy_a"]},
        artifact_root="robustness/robustness_a",
        robustness_status="needs_review",
        wfe_status="weak",
    )
    governance = _record(
        "governance_a",
        "governance_bundle",
        record_family="governance_bundle",
        metadata={"source_run_ids": ["strategy_a"]},
        artifact_root="promotion_governance/governance_a",
        governance_status="pass",
        promotion_review_status="needs_review",
    )
    validation = _record(
        "validation_a",
        "milestone_validation_bundle",
        record_family="milestone_validation_bundle",
        metadata={"source_run_ids": ["strategy_a"]},
        artifact_root="qa/validation_a",
        validation_readiness_present=True,
    )
    release = _record(
        "release_a",
        "release_validation_artifact",
        record_family="release_validation_artifact",
        metadata={"validation_bundle_run_id": "validation_a"},
        artifact_root="release_validation/release_a",
        release_validation_present=True,
    )
    return [release, validation, governance, robustness, strategy]


def test_markdown_rendering_is_deterministic() -> None:
    view = build_evidence_explorer_view(_records(), query=CatalogQuery(record_family="robustness_bundle"))

    first = render_evidence_markdown(view)
    second = render_evidence_markdown(view)

    assert first == second
    assert "# M35 Catalog Evidence Explorer" in first
    assert "| robustness_a | robustness_bundle | completed | robustness_bundle | robustness/robustness_a" in first
    assert "run_to_robustness_evidence" in first


def test_json_rendering_is_deterministic_and_sorted() -> None:
    view = build_evidence_explorer_view(_records(), query=CatalogQuery(record_family="governance_bundle"))

    first = render_evidence_json(view)
    second = render_evidence_json(view)
    payload = json.loads(first)

    assert first == second
    assert list(payload) == sorted(payload)
    assert payload["catalog_records"][0]["record_family"] == "governance_bundle"
    assert payload["evidence_status"][0]["governance_status"] == "pass"


def test_table_rendering_is_stable() -> None:
    view = build_evidence_explorer_view(_records(), query=CatalogQuery(release_validation_present=True))

    table = render_evidence_table(view)

    assert table.splitlines()[0].startswith("section\trun_id\trun_type")
    assert "evidence\trelease_a" in table
    assert "release_validation_artifact" in table


def test_empty_result_set_renders_gracefully() -> None:
    view = build_evidence_explorer_view(_records(), query=CatalogQuery(record_family="missing"))

    markdown = render_evidence_markdown(view)
    payload = json.loads(render_evidence_json(view))

    assert payload["total_matching_records"] == 0
    assert "No matching records." in markdown
    assert "No evidence lineage found." in markdown


def test_sparse_records_with_missing_optional_fields_render_gracefully() -> None:
    sparse = _record("sparse", "strategy", robustness_status=None, governance_status=None)
    view = build_evidence_explorer_view([sparse], include_lineage=True)

    markdown = render_evidence_markdown(view)

    assert "sparse" in markdown
    assert "None" not in markdown


def test_selected_run_expands_related_evidence_lineage() -> None:
    view = build_evidence_explorer_view(_records(), selected_run_id="strategy_a")

    run_ids = {row["run_id"] for row in view["catalog_records"]}
    edge_types = {row["edge_type"] for row in view["lineage_edges"]}

    assert {"strategy_a", "robustness_a", "governance_a", "validation_a"}.issubset(run_ids)
    assert "run_to_robustness_evidence" in edge_types
    assert "run_to_governance_evidence" in edge_types
    assert "run_to_validation_bundle" in edge_types


def test_filtered_result_set_renders_only_matching_records() -> None:
    view = build_evidence_explorer_view(_records(), query=CatalogQuery(governance_status="pass"))

    assert [row["run_id"] for row in view["catalog_records"]] == ["governance_a"]


def test_rendered_paths_are_portable() -> None:
    view = build_evidence_explorer_view(_records(), selected_run_id="strategy_a")
    rendered = render_evidence_json(view)

    assert str(Path.cwd()) not in rendered
    assert "\\" not in rendered
    assert "file://" not in rendered
