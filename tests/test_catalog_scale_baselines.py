from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from time import perf_counter

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
from tests.catalog_scale_fixtures import SCALE_CONFIG, build_catalog_scale_tree, snapshot_tree


EXPECTED_RECORD_COUNT = 48
EXPECTED_FAMILY_COUNTS = {
    "governance_bundle": 4,
    "milestone_validation_bundle": 3,
    "release_validation_artifact": 2,
    "robustness_bundle": 6,
}


def test_catalog_scale_baseline_workflow_is_deterministic_and_read_only(tmp_path: Path) -> None:
    build_catalog_scale_tree(tmp_path)
    before = snapshot_tree(tmp_path)

    started = perf_counter()
    records = build_catalog(tmp_path, repo_root=tmp_path)
    edges = build_lineage_edges(records, repo_root=tmp_path)
    selected_view = build_evidence_explorer_view(records, selected_run_id="strategy_000", repo_root=tmp_path)
    full_view = build_evidence_explorer_view(records, repo_root=tmp_path)
    elapsed_seconds = perf_counter() - started

    records_again = build_catalog(tmp_path, repo_root=tmp_path)
    edges_again = build_lineage_edges(records_again, repo_root=tmp_path)

    assert len(records) == EXPECTED_RECORD_COUNT
    assert [record.to_dict() for record in records] == [record.to_dict() for record in records_again]
    assert [edge.to_dict() for edge in edges] == [edge.to_dict() for edge in edges_again]
    assert Counter(record.record_family for record in records if record.record_family) == EXPECTED_FAMILY_COUNTS
    assert elapsed_seconds < 15
    assert before == snapshot_tree(tmp_path)

    assert [record.run_id for record in records[:3]] == ["alpha_000", "alpha_001", "alpha_002"]
    assert full_view["total_matching_records"] == EXPECTED_RECORD_COUNT
    assert selected_view["total_matching_records"] == 5
    assert {row["run_id"] for row in selected_view["catalog_records"]} == {
        "governance_000",
        "robustness_000",
        "strategy_000",
        "validation_000",
        "portfolio_000",
    }


def test_catalog_scale_queries_lineage_and_sparse_evidence_are_stable(tmp_path: Path) -> None:
    build_catalog_scale_tree(tmp_path)
    records = build_catalog(tmp_path, repo_root=tmp_path)
    edges = build_lineage_edges(records, repo_root=tmp_path)

    assert [record.run_id for record in query_catalog(records, CatalogQuery(run_types=("strategy",)))] == [
        *(f"strategy_{index:03d}" for index in range(12)),
        "strategy_registry_only",
    ]
    assert [
        record.run_id
        for record in query_catalog(
            records,
            CatalogQuery(record_family="robustness_bundle", robustness_status="needs_review"),
        )
    ] == ["robustness_000", "robustness_002"]
    assert [record.run_id for record in query_catalog(records, CatalogQuery(release_validation_present=True))] == [
        "release_000",
        "release_001",
    ]

    edge_counts = Counter(edge.edge_type for edge in edges)
    assert edge_counts["portfolio_component"] == 12
    assert edge_counts["scenario_child"] == 3
    assert edge_counts["run_to_robustness_evidence"] == 4
    assert edge_counts["run_to_governance_evidence"] == 3
    assert edge_counts["run_to_validation_bundle"] == 3
    assert edge_counts["validation_bundle_to_release_validation"] == 2
    assert all(edge.target_run_id != "robustness_orphan" for edge in edges)

    assert [row["run_id"] for row in find_robustness_evidence(records, robustness_status="needs_review")] == [
        "robustness_000",
        "robustness_002",
    ]
    assert len(find_governance_evidence(records, governance_status="pass")) == 4
    assert len(find_validation_evidence(records)) == 3
    assert len(find_release_evidence(records)) == 2
    assert [row["edge_type"] for row in evidence_lineage_rows(records, run_id="strategy_000", repo_root=tmp_path)] == [
        "run_to_governance_evidence",
        "run_to_robustness_evidence",
        "run_to_validation_bundle",
    ]


def test_catalog_scale_renderers_and_notebook_helpers_keep_portable_paths(tmp_path: Path) -> None:
    build_catalog_scale_tree(tmp_path)
    records = build_catalog(tmp_path, repo_root=tmp_path)
    view = build_evidence_explorer_view(records, selected_run_id="strategy_000", repo_root=tmp_path)

    payloads = [
        render_evidence_json(view),
        render_evidence_markdown(view),
        render_evidence_table(view),
        build_notebook_evidence_view(records, run_id="strategy_000", repo_root=tmp_path),
        evidence_for_run(records, "strategy_000", repo_root=tmp_path),
        render_notebook_json(records, run_id="strategy_000", repo_root=tmp_path),
        render_notebook_markdown(records, run_id="strategy_000", repo_root=tmp_path),
        render_notebook_table(records, run_id="strategy_000", repo_root=tmp_path),
    ]

    serialized_payloads = [payload if isinstance(payload, str) else json.dumps(payload, sort_keys=True) for payload in payloads]
    assert render_evidence_json(view) == render_evidence_json(view)
    assert all(str(tmp_path) not in payload for payload in serialized_payloads)
    assert all("file://" not in payload for payload in serialized_payloads)
    assert all("\\" not in payload for payload in serialized_payloads)


def test_catalog_scale_summary_keeps_deterministic_counts_separate_from_timing(tmp_path: Path) -> None:
    build_catalog_scale_tree(tmp_path)
    records = build_catalog(tmp_path, repo_root=tmp_path)
    edges = build_lineage_edges(records, repo_root=tmp_path)
    summary = {
        "schema_version": 1,
        "fixture_scale": SCALE_CONFIG,
        "deterministic_counts": {
            "record_count": len(records),
            "record_family_counts": dict(sorted(Counter(record.record_family for record in records if record.record_family).items())),
            "release_query_count": len(query_catalog(records, CatalogQuery(release_validation_present=True))),
            "lineage_edge_count": len(edges),
            "selected_explorer_record_count": build_evidence_explorer_view(
                records,
                selected_run_id="strategy_000",
                repo_root=tmp_path,
            )["total_matching_records"],
        },
        "timing": {"measured_locally": True},
    }

    assert summary["deterministic_counts"] == {
        "record_count": 48,
        "record_family_counts": EXPECTED_FAMILY_COUNTS,
        "release_query_count": 2,
        "lineage_edge_count": 100,
        "selected_explorer_record_count": 5,
    }
    assert summary["timing"] == {"measured_locally": True}
