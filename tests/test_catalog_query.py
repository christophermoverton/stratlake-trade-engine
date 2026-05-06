from __future__ import annotations

from src.catalog.models import CatalogRecord, CatalogValidationStatus
from src.catalog.query import (
    CatalogQuery,
    filter_catalog_records,
    get_downstream_records,
    get_upstream_records,
    query_catalog,
    records_to_dicts,
    records_to_rows,
    related_records,
    summarize_catalog,
)


def _validation() -> CatalogValidationStatus:
    return CatalogValidationStatus(
        catalog_status="valid",
        marker_status="present",
        manifest_status="present",
        artifact_status="ok",
        qa_status=None,
        validation_errors=[],
        validation_warnings=[],
    )


def _record(
    run_id: str,
    run_type: str,
    *,
    status: str = "completed",
    strategy_name: str | None = None,
    portfolio_name: str | None = None,
    allocator_name: str | None = None,
    alpha_model_name: str | None = None,
    regime_method: str | None = None,
    campaign_id: str | None = None,
    scenario_id: str | None = None,
    metrics_summary: dict | None = None,
    start_ts: str | None = None,
    end_ts: str | None = None,
    metadata: dict | None = None,
) -> CatalogRecord:
    return CatalogRecord(
        catalog_id=f"catalog_{run_id}",
        run_id=run_id,
        run_type=run_type,
        status=status,
        artifact_root=f"artifacts/{run_type}/{run_id}",
        source_registry_path=None,
        source_manifest_path=None,
        source_marker_path=None,
        created_at=None,
        timeframe="1D",
        start_ts=start_ts,
        end_ts=end_ts,
        strategy_name=strategy_name,
        portfolio_name=portfolio_name,
        allocator_name=allocator_name,
        alpha_model_name=alpha_model_name,
        regime_method=regime_method,
        campaign_id=campaign_id,
        scenario_id=scenario_id,
        metrics_summary=metrics_summary,
        qa_status=None,
        review_status="candidate",
        promotion_status="pending",
        tags=[],
        source_files=[],
        metadata=metadata or {},
        validation=_validation(),
    )


def test_default_excludes_portfolio_template() -> None:
    strategy = _record("strategy_1", "strategy")
    portfolio = _record("portfolio_1", "portfolio")
    template = _record("template_1", "portfolio_template", status="registry_only")

    default_records = query_catalog([template, portfolio, strategy])
    included_records = filter_catalog_records([template, portfolio, strategy], include_templates=True)

    assert [record.run_id for record in default_records] == ["portfolio_1", "strategy_1"]
    assert [record.run_id for record in included_records] == ["portfolio_1", "template_1", "strategy_1"]


def test_filter_by_run_type_and_status() -> None:
    records = [
        _record("strategy_completed", "strategy", status="completed"),
        _record("strategy_failed", "strategy", status="failed"),
        _record("portfolio_completed", "portfolio", status="completed"),
    ]

    result = query_catalog(records, CatalogQuery(run_types=("strategy",), statuses=("completed",)))

    assert [record.run_id for record in result] == ["strategy_completed"]


def test_filter_by_names() -> None:
    records = [
        _record("strategy_1", "strategy", strategy_name="momentum_v1"),
        _record("portfolio_1", "portfolio", portfolio_name="risk_parity", allocator_name="hrp"),
        _record("alpha_1", "alpha_evaluation", alpha_model_name="alpha_fast"),
        _record("regime_1", "regime_stress_test", regime_method="hmm"),
    ]

    assert [r.run_id for r in filter_catalog_records(records, strategy_name="momentum_v1")] == ["strategy_1"]
    assert [r.run_id for r in filter_catalog_records(records, portfolio_name="risk_parity")] == ["portfolio_1"]
    assert [r.run_id for r in filter_catalog_records(records, allocator_name="hrp")] == ["portfolio_1"]
    assert [r.run_id for r in filter_catalog_records(records, alpha_model_name="alpha_fast")] == ["alpha_1"]
    assert [r.run_id for r in filter_catalog_records(records, regime_method="hmm")] == ["regime_1"]


def test_metric_threshold_filtering() -> None:
    records = [
        _record("high", "strategy", metrics_summary={"sharpe_ratio": 1.5, "max_drawdown": -0.2}),
        _record("low", "strategy", metrics_summary={"sharpe_ratio": 0.5, "max_drawdown": -0.05}),
        _record("missing", "strategy", metrics_summary=None),
    ]

    assert [r.run_id for r in filter_catalog_records(records, min_metric=("sharpe_ratio", 1.0))] == ["high"]
    assert [r.run_id for r in filter_catalog_records(records, max_metric=("max_drawdown", -0.1))] == ["high"]
    assert [r.run_id for r in filter_catalog_records(records, metric_equals=("sharpe_ratio", 0.5))] == ["low"]


def test_time_filtering_uses_lexicographic_iso_like_bounds() -> None:
    records = [
        _record("early", "strategy", start_ts="2023-01-01", end_ts="2023-12-31"),
        _record("inside", "strategy", start_ts="2024-01-01", end_ts="2024-12-31"),
        _record("late", "strategy", start_ts="2025-01-01", end_ts="2025-12-31"),
        _record("missing", "strategy"),
    ]

    result = filter_catalog_records(records, start_ts="2024-01-01", end_ts="2024-12-31")

    assert [record.run_id for record in result] == ["inside"]


def test_include_unknown_false_excludes_unknown_run_type_or_status() -> None:
    records = [
        _record("known", "strategy", status="completed"),
        _record("unknown_type", "unknown", status="completed"),
        _record("unknown_status", "strategy", status="unknown"),
    ]

    result = filter_catalog_records(records, include_unknown=False)

    assert [record.run_id for record in result] == ["known"]


def test_summary_counts() -> None:
    records = [
        _record("strategy_1", "strategy", status="completed"),
        _record("strategy_2", "strategy", status="failed"),
        _record("portfolio_1", "portfolio", status="completed"),
    ]

    summary = summarize_catalog(records)

    assert summary["total_count"] == 3
    assert summary["by_run_type"] == {"portfolio": 1, "strategy": 2}
    assert summary["by_status"] == {"completed": 2, "failed": 1}
    assert summary["by_run_type_status"] == {
        "portfolio:completed": 1,
        "strategy:completed": 1,
        "strategy:failed": 1,
    }


def test_upstream_downstream_related_records() -> None:
    strategy = _record("strategy_1", "strategy")
    portfolio = _record("portfolio_1", "portfolio", metadata={"component_run_ids": ["strategy_1"]})
    unrelated = _record("strategy_2", "strategy")
    records = [portfolio, unrelated, strategy]

    assert [r.run_id for r in get_upstream_records(portfolio, records)] == ["strategy_1"]
    assert [r.run_id for r in get_downstream_records(strategy, records)] == ["portfolio_1"]
    assert [r.run_id for r in related_records(strategy, records, direction="both")] == ["portfolio_1"]


def test_related_records_can_filter_edge_types() -> None:
    strategy = _record("strategy_1", "strategy")
    portfolio = _record("portfolio_1", "portfolio", metadata={"component_run_ids": ["strategy_1"]})

    result = get_downstream_records(strategy, [strategy, portfolio], edge_types=("comparison_member",))

    assert result == []


def test_deterministic_ordering_and_serialization() -> None:
    records = [
        _record("strategy_b", "strategy"),
        _record("portfolio_a", "portfolio"),
        _record("strategy_a", "strategy"),
    ]

    first = query_catalog(records)
    second = query_catalog(list(reversed(records)))

    assert [record.run_id for record in first] == [record.run_id for record in second]
    assert [row["run_id"] for row in records_to_rows(records)] == ["portfolio_a", "strategy_a", "strategy_b"]
    assert [row["run_id"] for row in records_to_dicts(records)] == ["portfolio_a", "strategy_a", "strategy_b"]
