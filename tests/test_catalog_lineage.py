from __future__ import annotations

import json
from pathlib import Path

from src.catalog.indexer import build_catalog_record
from src.catalog.lineage import build_lineage_edges
from src.catalog.models import CatalogRecord, CatalogValidationStatus


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
    metadata: dict | None = None,
    campaign_id: str | None = None,
    scenario_id: str | None = None,
    artifact_root: str | None = None,
    source_manifest_path: str | None = None,
    source_files: list[str] | None = None,
) -> CatalogRecord:
    return CatalogRecord(
        catalog_id=f"catalog_{run_id}",
        run_id=run_id,
        run_type=run_type,
        status="completed",
        artifact_root=artifact_root or f"artifacts/{run_type}/{run_id}",
        source_registry_path=None,
        source_manifest_path=source_manifest_path,
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
        campaign_id=campaign_id,
        scenario_id=scenario_id,
        metrics_summary=None,
        qa_status=None,
        review_status=None,
        promotion_status=None,
        tags=[],
        source_files=source_files or [],
        metadata=metadata or {},
        validation=_validation(),
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_portfolio_component_lineage_direction_and_count() -> None:
    strat_a = _record("strategy_a", "strategy")
    strat_b = _record("strategy_b", "strategy")
    portfolio = _record(
        "portfolio_1",
        "portfolio",
        metadata={"component_run_ids": ["strategy_b", "strategy_a"]},
    )

    edges = build_lineage_edges([portfolio, strat_b, strat_a])
    component_edges = [edge for edge in edges if edge.edge_type == "portfolio_component"]

    assert len(component_edges) == 2
    assert {(edge.source_run_id, edge.target_run_id) for edge in component_edges} == {
        ("strategy_a", "portfolio_1"),
        ("strategy_b", "portfolio_1"),
    }


def test_comparison_member_lineage_from_member_run_ids() -> None:
    strat_a = _record("strategy_a", "strategy")
    strat_b = _record("strategy_b", "strategy")
    comparison = _record(
        "comparison_1",
        "comparison",
        metadata={"member_run_ids": ["strategy_a", "strategy_b"]},
    )

    edges = build_lineage_edges([comparison, strat_a, strat_b])
    comparison_edges = [edge for edge in edges if edge.edge_type == "comparison_member"]

    assert {(edge.source_run_id, edge.target_run_id) for edge in comparison_edges} == {
        ("strategy_a", "comparison_1"),
        ("strategy_b", "comparison_1"),
    }


def test_benchmark_member_lineage_from_child_run_ids() -> None:
    child_a = _record("campaign_child_a", "campaign")
    child_b = _record("campaign_child_b", "campaign")
    benchmark = _record(
        "benchmark_1",
        "benchmark_pack",
        metadata={"child_run_ids": ["campaign_child_b", "campaign_child_a"]},
    )

    edges = build_lineage_edges([benchmark, child_a, child_b])
    benchmark_edges = [edge for edge in edges if edge.edge_type == "benchmark_member"]

    assert {(edge.source_run_id, edge.target_run_id) for edge in benchmark_edges} == {
        ("campaign_child_a", "benchmark_1"),
        ("campaign_child_b", "benchmark_1"),
    }


def test_campaign_child_lineage_from_parent_run_id() -> None:
    parent = _record("campaign_parent", "campaign")
    child = _record(
        "campaign_child",
        "campaign",
        metadata={"parent_run_id": "campaign_parent", "campaign_id": "campaign_parent"},
    )

    edges = build_lineage_edges([child, parent])
    campaign_edges = [edge for edge in edges if edge.edge_type == "campaign_child"]

    assert len(campaign_edges) == 1
    assert campaign_edges[0].source_run_id == "campaign_parent"
    assert campaign_edges[0].target_run_id == "campaign_child"


def test_campaign_parent_metadata_does_not_imply_scenario_edge() -> None:
    parent = _record("campaign_parent", "campaign")
    child = _record(
        "campaign_child",
        "campaign",
        metadata={"parent_run_id": "campaign_parent", "campaign_id": "campaign_parent"},
    )

    edges = build_lineage_edges([child, parent])

    assert len([edge for edge in edges if edge.edge_type == "campaign_child"]) == 1
    assert [edge for edge in edges if edge.edge_type == "scenario_child"] == []


def test_explicit_scenario_parent_emits_only_scenario_child() -> None:
    parent = _record("scenario_parent", "campaign")
    child = _record(
        "scenario_child",
        "campaign",
        scenario_id="scenario_a",
        metadata={"scenario_parent_run_id": "scenario_parent"},
    )

    edges = build_lineage_edges([child, parent])
    scenario_edges = [edge for edge in edges if edge.edge_type == "scenario_child"]

    assert len(scenario_edges) == 1
    assert scenario_edges[0].source_run_id == "scenario_parent"
    assert scenario_edges[0].target_run_id == "scenario_child"
    assert [edge for edge in edges if edge.edge_type == "campaign_child"] == []


def test_scenario_child_lineage_from_explicit_parent_catalog_id() -> None:
    parent = _record("scenario_parent", "campaign")
    child = _record(
        "scenario_child",
        "campaign",
        scenario_id="scenario_a",
        metadata={"scenario_parent_catalog_id": parent.catalog_id},
    )

    edges = build_lineage_edges([child, parent])
    scenario_edges = [edge for edge in edges if edge.edge_type == "scenario_child"]

    assert len(scenario_edges) == 1
    assert scenario_edges[0].source_run_id == "scenario_parent"
    assert scenario_edges[0].target_run_id == "scenario_child"


def test_portfolio_template_does_not_emit_portfolio_component() -> None:
    strat = _record("strategy_a", "strategy")
    template = _record(
        "portfolio_template_1",
        "portfolio_template",
        metadata={"component_run_ids": ["strategy_a"]},
    )

    edges = build_lineage_edges([template, strat])

    assert [edge for edge in edges if edge.edge_type == "portfolio_component"] == []


def test_synthetic_records_include_explicit_validation_state() -> None:
    record = _record("strategy_a", "strategy")

    assert record.validation.catalog_status == "valid"
    assert record.validation.marker_status == "present"


def test_validation_and_pipeline_edges_use_declared_directions() -> None:
    strategy = _record("strategy_a", "strategy")
    validation = _record(
        "validation_1",
        "milestone_validation",
        metadata={"validation_target_run_ids": ["strategy_a"]},
    )
    pipeline = _record(
        "pipeline_1",
        "pipeline",
        metadata={"wrapped_run_id": "strategy_a"},
    )

    edges = build_lineage_edges([validation, pipeline, strategy])
    validation_edges = [edge for edge in edges if edge.edge_type == "validation_references_run"]
    pipeline_edges = [edge for edge in edges if edge.edge_type == "pipeline_wraps_execution"]

    assert [(edge.source_run_id, edge.target_run_id) for edge in validation_edges] == [
        ("strategy_a", "validation_1")
    ]
    assert [(edge.source_run_id, edge.target_run_id) for edge in pipeline_edges] == [
        ("pipeline_1", "strategy_a")
    ]


def test_manifest_declares_artifact_lineage(tmp_path: Path) -> None:
    run_root = tmp_path / "strategies" / "strategy_manifest"
    _write_json(
        run_root / "manifest.json",
        {"run_id": "strategy_manifest", "artifacts": ["metrics.json", "reports/report.json"]},
    )
    _write_json(run_root / "metrics.json", {"sharpe_ratio": 1.2})
    _write_json(run_root / "reports" / "report.json", {"status": "ok"})

    record = build_catalog_record(run_root, repo_root=tmp_path, registry_index={})
    edges = build_lineage_edges([record], repo_root=tmp_path)
    manifest_edges = [edge for edge in edges if edge.edge_type == "manifest_declares_artifact"]

    assert len(manifest_edges) == 2
    metadata = [edge.metadata for edge in manifest_edges]
    assert {item["relative_path"] for item in metadata} == {"metrics.json", "reports/report.json"}
    assert all(item["artifact_id"] for item in metadata)
    assert all(item["artifact_path"] for item in metadata)
    assert {item["artifact_type"] for item in metadata} == {"json", "metrics"}


def test_unresolved_references_are_skipped_and_deterministic() -> None:
    portfolio = _record(
        "portfolio_1",
        "portfolio",
        metadata={"component_run_ids": ["missing_run"]},
    )

    first = build_lineage_edges([portfolio])
    second = build_lineage_edges([portfolio])

    assert first == []
    assert second == []


def test_deterministic_edge_ordering() -> None:
    records = [
        _record("strategy_b", "strategy"),
        _record("portfolio_1", "portfolio", metadata={"component_run_ids": ["strategy_b", "strategy_a"]}),
        _record("strategy_a", "strategy"),
        _record("comparison_1", "comparison", metadata={"member_run_ids": ["strategy_b", "strategy_a"]}),
    ]

    first = build_lineage_edges(records)
    second = build_lineage_edges(list(reversed(records)))

    assert [edge.edge_id for edge in first] == [edge.edge_id for edge in second]
    assert [edge.to_dict() for edge in first] == [edge.to_dict() for edge in second]


def test_duplicate_references_are_deduplicated() -> None:
    strat = _record("strategy_a", "strategy")
    portfolio = _record(
        "portfolio_1",
        "portfolio",
        metadata={"component_run_ids": ["strategy_a", "strategy_a"]},
    )

    edges = build_lineage_edges([portfolio, strat])
    component_edges = [edge for edge in edges if edge.edge_type == "portfolio_component"]

    assert len(component_edges) == 1
    assert component_edges[0].source_run_id == "strategy_a"


def test_read_only_behavior_for_source_files(tmp_path: Path) -> None:
    run_root = tmp_path / "strategies" / "strategy_ro"
    _write_json(run_root / "manifest.json", {"run_id": "strategy_ro", "artifacts": ["metrics.json"]})
    _write_json(run_root / "metrics.json", {"sharpe_ratio": 1.2})
    record = build_catalog_record(run_root, repo_root=tmp_path, registry_index={})

    before = {
        path.as_posix(): (path.stat().st_mtime_ns, path.stat().st_size)
        for path in sorted(tmp_path.rglob("*"))
        if path.is_file()
    }
    build_lineage_edges([record], repo_root=tmp_path)
    after = {
        path.as_posix(): (path.stat().st_mtime_ns, path.stat().st_size)
        for path in sorted(tmp_path.rglob("*"))
        if path.is_file()
    }

    assert before == after
