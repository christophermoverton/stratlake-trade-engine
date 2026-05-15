from __future__ import annotations

import json
import runpy
from pathlib import Path

from src.catalog import (
    CatalogQuery,
    build_evidence_explorer_view,
    build_notebook_evidence_view,
    evidence_for_run,
    evidence_lineage_rows,
    find_governance_evidence,
    find_release_evidence,
    find_robustness_evidence,
    find_validation_evidence,
    query_catalog,
    render_evidence_markdown,
    render_notebook_json,
    render_notebook_markdown,
    render_notebook_table,
    summarize_evidence_for_run,
)
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
    return [
        _record("strategy_a", "strategy", artifact_root="strategies/strategy_a"),
        _record(
            "robustness_a",
            "robustness_bundle",
            record_family="robustness_bundle",
            metadata={"source_run_ids": ["strategy_a"]},
            artifact_root="robustness/robustness_a",
            robustness_status="needs_review",
            wfe_status="weak",
        ),
        _record(
            "governance_a",
            "governance_bundle",
            record_family="governance_bundle",
            metadata={"source_run_ids": ["strategy_a"]},
            artifact_root="promotion_governance/governance_a",
            governance_status="pass",
            promotion_review_status="needs_review",
        ),
        _record(
            "validation_a",
            "milestone_validation_bundle",
            record_family="milestone_validation_bundle",
            metadata={"source_run_ids": ["strategy_a"]},
            artifact_root="qa/validation_a",
            validation_readiness_present=True,
        ),
        _record(
            "release_a",
            "release_validation_artifact",
            record_family="release_validation_artifact",
            metadata={"validation_bundle_run_id": "validation_a"},
            artifact_root="release_validation/release_a",
            release_validation_present=True,
        ),
    ]


def test_public_imports_are_stable() -> None:
    assert callable(find_robustness_evidence)
    assert callable(evidence_for_run)
    assert callable(render_notebook_markdown)


def test_helpers_return_deterministic_serializable_rows() -> None:
    records = _records()

    first = find_robustness_evidence(records, robustness_status="needs_review", wfe_status="weak")
    second = find_robustness_evidence(list(reversed(records)), robustness_status="needs_review", wfe_status="weak")

    assert first == second
    assert [row["run_id"] for row in first] == ["robustness_a"]
    json.dumps(first, sort_keys=True)


def test_governance_validation_and_release_helpers() -> None:
    records = _records()

    assert [row["run_id"] for row in find_governance_evidence(records, governance_status="pass")] == [
        "governance_a"
    ]
    assert [row["run_id"] for row in find_validation_evidence(records)] == ["validation_a"]
    assert [row["run_id"] for row in find_release_evidence(records)] == ["release_a"]


def test_empty_and_sparse_records_are_safe() -> None:
    sparse = [_record("sparse", "strategy", robustness_status=None)]

    assert find_robustness_evidence([]) == []
    assert evidence_lineage_rows(sparse) == []
    summary = summarize_evidence_for_run(sparse, "sparse")
    assert summary["related_evidence_count"] == 0
    assert render_notebook_markdown(sparse)


def test_evidence_for_run_includes_related_evidence_and_lineage() -> None:
    view = evidence_for_run(_records(), "strategy_a")

    run_ids = {row["run_id"] for row in view["records"]}
    edge_types = {row["edge_type"] for row in view["lineage_edges"]}

    assert {"strategy_a", "robustness_a", "governance_a", "validation_a"}.issubset(run_ids)
    assert "run_to_robustness_evidence" in edge_types
    assert "run_to_governance_evidence" in edge_types


def test_evidence_lineage_rows_filter_by_run_id() -> None:
    rows = evidence_lineage_rows(_records(), run_id="strategy_a")

    assert [row["edge_type"] for row in rows] == [
        "run_to_governance_evidence",
        "run_to_robustness_evidence",
        "run_to_validation_bundle",
    ]


def test_helper_query_results_match_direct_catalog_query() -> None:
    records = _records()
    helper_rows = find_robustness_evidence(records, robustness_status="needs_review")
    direct_records = query_catalog(
        records,
        CatalogQuery(record_family="robustness_bundle", robustness_status="needs_review"),
    )

    assert [row["run_id"] for row in helper_rows] == [record.run_id for record in direct_records]


def test_notebook_view_and_renderers_match_explorer_renderer() -> None:
    records = _records()
    query = CatalogQuery(record_family="robustness_bundle")

    helper_view = build_notebook_evidence_view(records, query=query)
    direct_view = build_evidence_explorer_view(records, query=query)

    assert helper_view == direct_view
    assert render_notebook_markdown(records, query=query) == render_evidence_markdown(direct_view)
    assert json.loads(render_notebook_json(records, query=query)) == direct_view
    assert "robustness_a" in render_notebook_table(records, query=query)


def test_helper_outputs_use_portable_paths() -> None:
    payload = json.dumps(evidence_for_run(_records(), "strategy_a"), sort_keys=True)

    assert str(Path.cwd()) not in payload
    assert "\\" not in payload
    assert "file://" not in payload


def test_catalog_evidence_notebook_workflow_example_runs() -> None:
    namespace = runpy.run_path("docs/examples/catalog_evidence_notebook_workflow.py")
    result = namespace["run_catalog_evidence_notebook_workflow"]()

    assert [row["run_id"] for row in result["robustness_rows"]] == ["robustness_a"]
    assert [row["run_id"] for row in result["governance_rows"]] == ["governance_a"]
    assert result["selected_run_view"]["run_id"] == "strategy_a"
    assert any(row["edge_type"] == "run_to_robustness_evidence" for row in result["lineage_rows"])
