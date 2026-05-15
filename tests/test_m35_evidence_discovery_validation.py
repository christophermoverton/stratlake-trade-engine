from __future__ import annotations

import json
import runpy
from pathlib import Path
from typing import Any

from src.catalog import (
    CatalogQuery,
    build_catalog,
    build_evidence_explorer_view,
    build_lineage_edges,
    build_notebook_evidence_view,
    evidence_for_run,
    evidence_lineage_rows,
    find_governance_evidence,
    find_release_evidence,
    find_robustness_evidence,
    find_validation_evidence,
    query_catalog,
    render_evidence_json,
    render_evidence_markdown,
    render_evidence_table,
    render_notebook_json,
    render_notebook_markdown,
    render_notebook_table,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _write_csv(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _catalog_dicts(root: Path) -> list[dict[str, Any]]:
    return [record.to_dict() for record in build_catalog(root, repo_root=root)]


def _json_dump(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _assert_portable(payload: Any, root: Path) -> None:
    serialized = _json_dump(payload) if not isinstance(payload, str) else payload
    assert str(root) not in serialized
    assert "file://" not in serialized
    assert "\\" not in serialized


def _build_m35_artifacts(root: Path) -> None:
    _write_json(
        root / "strategies" / "strategy_a" / "_SUCCESS.json",
        {"recorded_at_utc": "2025-01-01T00:00:00Z", "status": "completed"},
    )
    _write_json(
        root / "strategies" / "strategy_a" / "summary.json",
        {"run_id": "strategy_a", "run_type": "strategy", "strategy_name": "deterministic_strategy"},
    )

    _write_json(
        root / "robustness" / "robustness_a" / "robustness_summary.json",
        {
            "checks_present": ["walk_forward_efficiency", "sample_size"],
            "finding_count": 1,
            "report_id": "robustness_a",
            "robustness_status": "needs_review",
            "source_run_ids": ["strategy_a"],
        },
    )
    _write_csv(root / "robustness" / "robustness_a" / "walk_forward_efficiency.csv", "run_id,status\nstrategy_a,weak\n")
    _write_json(
        root / "robustness" / "robustness_a" / "sample_size_validation.json",
        {
            "checks": [
                {"check_id": "sample_size.minimum_total_samples", "status": "pass"},
                {"check_id": "sample_size.minimum_total_trades", "status": "warning"},
            ]
        },
    )
    _write_csv(root / "robustness" / "robustness_a" / "sensitivity_summary.csv", "run_id,status\nstrategy_a,fragile\n")
    _write_json(
        root / "robustness" / "robustness_a" / "multiple_testing_summary.json",
        {"families": [{"family_id": "m35", "status": "high_risk"}]},
    )
    _write_json(root / "robustness" / "robustness_a" / "leakage_validation.json", {"overall_status": "blocked"})

    _write_json(
        root / "promotion_governance" / "governance_a" / "promotion_governance_summary.json",
        {"review_status_counts": {"needs_review": 1}, "row_count": 1},
    )
    _write_json(root / "promotion_governance" / "governance_a" / "consistency_validation.json", {"status": "pass"})
    _write_json(root / "promotion_governance" / "governance_a" / "manifest.json", {"run_type": "promotion_governance"})
    _write_csv(
        root / "promotion_governance" / "governance_a" / "promotion_outcome_matrix.csv",
        "workflow_type,run_id,review_status\nstrategy,strategy_a,needs_review\n",
    )

    _write_json(
        root / "qa" / "validation_a" / "summary.json",
        {
            "checks": {"catalog": {"status": "passed"}},
            "run_id": "validation_a",
            "run_type": "milestone_validation_bundle",
            "source_run_ids": ["strategy_a"],
            "status": "passed",
        },
    )
    _write_json(root / "qa" / "validation_a" / "_SUCCESS.json", {"status": "completed"})

    _write_json(
        root / "release_validation" / "release_a" / "release_validation.json",
        {
            "finding_count": 0,
            "release_id": "m35",
            "status": "pass",
            "validation_bundle_run_id": "validation_a",
        },
    )

    _write_json(root / "robustness" / "sparse_robustness" / "robustness_summary.json", {"report_id": "sparse_robustness"})
    _write_json(
        root / "robustness" / "orphan_robustness" / "robustness_summary.json",
        {"report_id": "orphan_robustness", "source_run_ids": ["missing_run"]},
    )
    _write_json(
        root / "promotion_governance" / "aggregate_governance" / "promotion_governance_summary.json",
        {"review_status_counts": {"pass": 3}, "row_count": 3},
    )
    _write_json(root / "promotion_governance" / "aggregate_governance" / "consistency_validation.json", {"status": "pass"})


def test_m35_evidence_discovery_chain_is_deterministic(tmp_path: Path) -> None:
    _build_m35_artifacts(tmp_path)

    records_a = build_catalog(tmp_path, repo_root=tmp_path)
    records_b = build_catalog(tmp_path, repo_root=tmp_path)
    query = CatalogQuery(robustness_status="needs_review", wfe_status="weak")

    view_a = build_evidence_explorer_view(records_a, selected_run_id="strategy_a", repo_root=tmp_path)
    view_b = build_evidence_explorer_view(records_b, selected_run_id="strategy_a", repo_root=tmp_path)

    assert [record.to_dict() for record in records_a] == [record.to_dict() for record in records_b]
    assert [record.to_dict() for record in query_catalog(records_a, query)] == [
        record.to_dict() for record in query_catalog(records_b, query)
    ]
    assert [edge.to_dict() for edge in build_lineage_edges(records_a, repo_root=tmp_path)] == [
        edge.to_dict() for edge in build_lineage_edges(records_b, repo_root=tmp_path)
    ]
    assert render_evidence_json(view_a) == render_evidence_json(view_b)
    assert render_evidence_markdown(view_a) == render_evidence_markdown(view_b)
    assert render_evidence_table(view_a) == render_evidence_table(view_b)
    assert build_notebook_evidence_view(records_a, run_id="strategy_a", repo_root=tmp_path) == build_notebook_evidence_view(
        records_b, run_id="strategy_a", repo_root=tmp_path
    )
    assert evidence_for_run(records_a, "strategy_a", repo_root=tmp_path) == evidence_for_run(
        records_b, "strategy_a", repo_root=tmp_path
    )


def test_m35_discovery_tools_do_not_mutate_source_artifacts(tmp_path: Path) -> None:
    _build_m35_artifacts(tmp_path)
    before = _snapshot(tmp_path)

    records = build_catalog(tmp_path, repo_root=tmp_path)
    query_catalog(records, CatalogQuery(governance_status="pass"))
    build_lineage_edges(records, repo_root=tmp_path)
    view = build_evidence_explorer_view(records, selected_run_id="strategy_a", repo_root=tmp_path)
    render_evidence_markdown(view)
    render_evidence_json(view)
    render_evidence_table(view)
    find_robustness_evidence(records, robustness_status="needs_review")
    find_governance_evidence(records, governance_status="pass")
    find_validation_evidence(records)
    find_release_evidence(records)
    evidence_lineage_rows(records, run_id="strategy_a", repo_root=tmp_path)
    evidence_for_run(records, "strategy_a", repo_root=tmp_path)
    render_notebook_markdown(records, run_id="strategy_a", repo_root=tmp_path)
    render_notebook_json(records, run_id="strategy_a", repo_root=tmp_path)
    render_notebook_table(records, run_id="strategy_a", repo_root=tmp_path)

    assert before == _snapshot(tmp_path)


def test_m35_outputs_do_not_contain_absolute_or_file_uri_paths(tmp_path: Path) -> None:
    _build_m35_artifacts(tmp_path)
    records = build_catalog(tmp_path, repo_root=tmp_path)
    view = build_evidence_explorer_view(records, selected_run_id="strategy_a", repo_root=tmp_path)

    payloads: list[Any] = [
        [record.to_dict() for record in records],
        [edge.to_dict() for edge in build_lineage_edges(records, repo_root=tmp_path)],
        render_evidence_json(view),
        render_evidence_markdown(view),
        render_evidence_table(view),
        build_notebook_evidence_view(records, run_id="strategy_a", repo_root=tmp_path),
        evidence_for_run(records, "strategy_a", repo_root=tmp_path),
        render_notebook_json(records, run_id="strategy_a", repo_root=tmp_path),
        render_notebook_markdown(records, run_id="strategy_a", repo_root=tmp_path),
        render_notebook_table(records, run_id="strategy_a", repo_root=tmp_path),
    ]

    for payload in payloads:
        _assert_portable(payload, tmp_path)
    for record in records:
        assert not Path(record.artifact_root).is_absolute()
        assert all(not Path(path).is_absolute() for path in record.source_files)


def test_m35_empty_and_sparse_artifact_roots_are_safe(tmp_path: Path) -> None:
    assert _catalog_dicts(tmp_path) == []
    empty_view = build_evidence_explorer_view([], include_lineage=True)
    assert "No matching records." in render_evidence_markdown(empty_view)
    assert evidence_lineage_rows([]) == []
    assert find_robustness_evidence([]) == []

    _write_json(tmp_path / "robustness" / "sparse" / "robustness_summary.json", {"report_id": "sparse"})
    records = build_catalog(tmp_path, repo_root=tmp_path)
    assert [record.record_family for record in records] == ["robustness_bundle"]
    assert find_robustness_evidence(records)
    assert render_notebook_markdown(records)
    _assert_portable(build_notebook_evidence_view(records, repo_root=tmp_path), tmp_path)


def test_m35_governance_boundary_is_read_only(tmp_path: Path) -> None:
    _build_m35_artifacts(tmp_path)
    governance_files = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in sorted((tmp_path / "promotion_governance").rglob("*"))
        if path.is_file()
    }

    records = build_catalog(tmp_path, repo_root=tmp_path)
    governance = next(record for record in records if record.run_id == "governance_a")
    assert governance.governance_status == "pass"
    assert governance.promotion_review_status == "needs_review"
    assert governance.promotion_status is None

    build_lineage_edges(records, repo_root=tmp_path)
    build_evidence_explorer_view(records, query=CatalogQuery(record_family="governance_bundle"), repo_root=tmp_path)
    find_governance_evidence(records, governance_status="pass")

    assert governance_files == {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in sorted((tmp_path / "promotion_governance").rglob("*"))
        if path.is_file()
    }


def test_m35_lineage_does_not_invent_edges_for_orphans(tmp_path: Path) -> None:
    _build_m35_artifacts(tmp_path)
    records = build_catalog(tmp_path, repo_root=tmp_path)
    edges = build_lineage_edges(records, repo_root=tmp_path)
    edge_pairs = {(edge.source_run_id, edge.target_run_id, edge.edge_type) for edge in edges}

    assert ("missing_run", "orphan_robustness", "run_to_robustness_evidence") not in edge_pairs
    assert all(edge.target_run_id != "aggregate_governance" for edge in edges)
    assert ("strategy_a", "robustness_a", "run_to_robustness_evidence") in edge_pairs
    assert ("strategy_a", "governance_a", "run_to_governance_evidence") in edge_pairs


def test_m35_example_workflow_is_ci_safe_and_deterministic() -> None:
    namespace = runpy.run_path("docs/examples/catalog_evidence_notebook_workflow.py")
    first = namespace["run_catalog_evidence_notebook_workflow"]()
    second = namespace["run_catalog_evidence_notebook_workflow"]()

    assert first == second
    json.dumps(first, sort_keys=True)
    assert [row["run_id"] for row in first["robustness_rows"]] == ["robustness_a"]
    assert [row["run_id"] for row in first["governance_rows"]] == ["governance_a"]
    assert "strategy_a" in _json_dump(first)
    assert "file://" not in _json_dump(first)
    assert "\\" not in _json_dump(first)

