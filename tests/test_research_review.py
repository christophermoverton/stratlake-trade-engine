from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cli.compare_research import parse_args, run_cli
from src.config.review import load_review_config
from src.research.review import (
    ResearchReviewError,
    ResearchReviewResult,
    build_research_review_id,
    compare_research_runs,
)


def _write_registry(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _without_output_path(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(payload))
    review_config = normalized.get("review_config")
    if isinstance(review_config, dict):
        output = review_config.get("output")
        if isinstance(output, dict):
            output.pop("path", None)
    return normalized


def _strategy_entry(
    *,
    run_id: str,
    strategy_name: str,
    timestamp: str,
    sharpe_ratio: object,
    total_return: object,
    timeframe: str = "1D",
    dataset: str = "features_daily",
    evaluation_mode: str = "single",
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "run_type": "strategy",
        "timestamp": timestamp,
        "strategy_name": strategy_name,
        "dataset": dataset,
        "timeframe": timeframe,
        "evaluation_mode": evaluation_mode,
        "artifact_path": f"artifacts/strategies/{run_id}",
        "promotion_status": "eligible",
        "promotion_gate_summary": {
            "promotion_status": "eligible",
            "passed_gate_count": 2,
            "gate_count": 2,
        },
        "metrics_summary": {
            "sharpe_ratio": sharpe_ratio,
            "total_return": total_return,
        },
    }


def _strategy_entry_with_review(
    *,
    run_id: str,
    strategy_name: str,
    timestamp: str,
    review_status: str,
    decision_reason: str,
) -> dict[str, object]:
    payload = _strategy_entry(
        run_id=run_id,
        strategy_name=strategy_name,
        timestamp=timestamp,
        sharpe_ratio=1.1,
        total_return=0.09,
    )
    payload["review_status"] = review_status
    payload["review_metadata"] = {
        "decision_reason": decision_reason,
        "decision_source": "manual_review",
        "promotion_gate_summary": payload["promotion_gate_summary"],
        "promotion_status": payload["promotion_status"],
        "reviewed_at": "2026-03-19T00:10:00Z",
        "schema_version": 1,
        "status": review_status,
    }
    return payload


def _portfolio_entry(
    *,
    run_id: str,
    portfolio_name: str,
    timestamp: str,
    sharpe_ratio: object,
    total_return: object,
    timeframe: str = "1D",
    split_count: int | None = None,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "run_type": "portfolio",
        "timestamp": timestamp,
        "portfolio_name": portfolio_name,
        "timeframe": timeframe,
        "artifact_path": f"artifacts/portfolios/{run_id}",
        "split_count": split_count,
        "promotion_status": "blocked",
        "promotion_gate_summary": {
            "promotion_status": "blocked",
            "passed_gate_count": 1,
            "gate_count": 2,
        },
        "metrics_summary": {
            "sharpe_ratio": sharpe_ratio,
            "total_return": total_return,
        },
    }


def _alpha_entry(
    *,
    run_id: str,
    alpha_name: str,
    timestamp: str,
    ic_ir: object,
    mean_ic: object,
    timeframe: str = "1D",
    dataset: str = "features_daily",
    mapping_name: str | None = None,
    sleeve_sharpe_ratio: object | None = None,
    sleeve_total_return: object | None = None,
) -> dict[str, object]:
    config: dict[str, object] = {}
    if mapping_name is not None:
        config["signal_mapping"] = {
            "policy": "top_bottom_quantile",
            "quantile": 0.2,
            "metadata": {"name": mapping_name},
        }
    manifest: dict[str, object] = {}
    if sleeve_sharpe_ratio is not None or sleeve_total_return is not None:
        manifest["sleeve"] = {
            "enabled": True,
            "metric_summary": {
                "sharpe_ratio": sleeve_sharpe_ratio,
                "total_return": sleeve_total_return,
            },
        }
    return {
        "run_id": run_id,
        "run_type": "alpha_evaluation",
        "timestamp": timestamp,
        "alpha_name": alpha_name,
        "dataset": dataset,
        "timeframe": timeframe,
        "evaluation_horizon": 1,
        "artifact_path": f"artifacts/alpha/{run_id}",
        "promotion_status": "eligible",
        "promotion_gate_summary": {
            "promotion_status": "eligible",
            "passed_gate_count": 1,
            "gate_count": 1,
        },
        "config": config,
        "manifest": manifest,
        "metrics_summary": {
            "ic_ir": ic_ir,
            "mean_ic": mean_ic,
            "mean_rank_ic": 0.1,
            "rank_ic_ir": 0.2,
            "n_periods": 5,
        },
    }


def test_compare_research_runs_builds_unified_registry_review(tmp_path: Path) -> None:
    strategy_root = tmp_path / "artifacts" / "strategies"
    portfolio_root = tmp_path / "artifacts" / "portfolios"
    alpha_root = tmp_path / "artifacts" / "alpha"
    _write_registry(
        strategy_root / "registry.jsonl",
        [
            _strategy_entry(
                run_id="strategy-old",
                strategy_name="momentum_v1",
                timestamp="2026-03-19T00:00:00Z",
                sharpe_ratio=1.0,
                total_return=0.08,
            ),
            _strategy_entry(
                run_id="strategy-new",
                strategy_name="momentum_v1",
                timestamp="2026-03-19T00:05:00Z",
                sharpe_ratio=1.3,
                total_return=0.10,
            ),
            _strategy_entry(
                run_id="strategy-beta",
                strategy_name="mean_reversion_v1",
                timestamp="2026-03-19T00:02:00Z",
                sharpe_ratio=0.9,
                total_return=0.06,
            ),
        ],
    )
    _write_registry(
        portfolio_root / "registry.jsonl",
        [
            _portfolio_entry(
                run_id="portfolio-core",
                portfolio_name="core_portfolio",
                timestamp="2026-03-19T00:01:00Z",
                sharpe_ratio=1.5,
                total_return=0.14,
            ),
            _portfolio_entry(
                run_id="portfolio-alt",
                portfolio_name="alt_portfolio",
                timestamp="2026-03-19T00:02:00Z",
                sharpe_ratio=1.1,
                total_return=0.11,
                split_count=3,
            ),
        ],
    )
    _write_registry(
        alpha_root / "registry.jsonl",
        [
            _alpha_entry(
                run_id="alpha-a",
                alpha_name="alpha_one",
                timestamp="2026-03-19T00:00:00Z",
                ic_ir=0.8,
                mean_ic=0.03,
            ),
            _alpha_entry(
                run_id="alpha-b",
                alpha_name="alpha_two",
                timestamp="2026-03-19T00:01:00Z",
                ic_ir=0.6,
                mean_ic=0.02,
            ),
        ],
    )

    result = compare_research_runs(
        strategy_artifacts_root=strategy_root,
        portfolio_artifacts_root=portfolio_root,
        alpha_artifacts_root=alpha_root,
        output_path=tmp_path,
    )

    assert result.csv_path == tmp_path / "leaderboard.csv"
    assert result.json_path == tmp_path / "review_summary.json"
    assert result.manifest_path == tmp_path / "manifest.json"
    assert result.promotion_gate_path is None
    assert result.plot_paths == {
        "alpha_evaluation_metric_comparison": tmp_path / "plots" / "alpha_evaluation" / "metric_comparison_ic_ir.png",
        "portfolio_metric_comparison": tmp_path / "plots" / "portfolio" / "metric_comparison_sharpe_ratio.png",
        "strategy_metric_comparison": tmp_path / "plots" / "strategy" / "metric_comparison_sharpe_ratio.png",
    }
    assert [entry.run_type for entry in result.entries] == [
        "alpha_evaluation",
        "alpha_evaluation",
        "strategy",
        "strategy",
        "portfolio",
        "portfolio",
    ]
    assert [entry.entity_name for entry in result.entries if entry.run_type == "strategy"] == [
        "momentum_v1",
        "mean_reversion_v1",
    ]
    assert [entry.run_id for entry in result.entries if entry.run_type == "strategy"] == [
        "strategy-new",
        "strategy-beta",
    ]
    assert result.entries[0].selected_metric_name == "ic_ir"
    assert result.entries[2].selected_metric_name == "sharpe_ratio"
    assert result.entries[-1].evaluation_mode == "walk_forward"

    frame = pd.read_csv(result.csv_path)
    assert list(frame.columns) == [
        "run_type",
        "rank_within_type",
        "entity_name",
        "run_id",
        "selected_metric_name",
        "selected_metric_value",
        "secondary_metric_name",
        "secondary_metric_value",
        "timeframe",
        "evaluation_mode",
        "promotion_status",
        "passed_gate_count",
        "gate_count",
        "mapping_name",
        "sleeve_metric_name",
        "sleeve_metric_value",
        "sleeve_secondary_metric_name",
        "sleeve_secondary_metric_value",
        "linked_portfolio_count",
        "linked_portfolio_names",
        "linked_portfolio_metric_name",
        "linked_portfolio_metric_value",
        "artifact_path",
    ]
    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert payload["counts_by_run_type"] == {
        "alpha_evaluation": 2,
        "strategy": 2,
        "portfolio": 2,
    }
    assert payload["entry_count"] == 6
    assert payload["review_config"]["ranking"]["strategy_primary_metric"] == "sharpe_ratio"
    assert payload["plot_paths"] == {
        "alpha_evaluation_metric_comparison": "plots/alpha_evaluation/metric_comparison_ic_ir.png",
        "portfolio_metric_comparison": "plots/portfolio/metric_comparison_sharpe_ratio.png",
        "strategy_metric_comparison": "plots/strategy/metric_comparison_sharpe_ratio.png",
    }
    assert payload["skipped_plots"] == {}
    assert frame["promotion_status"].tolist() == [
        "eligible",
        "eligible",
        "eligible",
        "eligible",
        "blocked",
        "blocked",
    ]

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["artifact_files"] == [
        "leaderboard.csv",
        "manifest.json",
        "plots/alpha_evaluation/metric_comparison_ic_ir.png",
        "plots/portfolio/metric_comparison_sharpe_ratio.png",
        "plots/strategy/metric_comparison_sharpe_ratio.png",
        "review_summary.json",
    ]
    assert manifest["artifact_groups"]["plots"] == [
        "plots/alpha_evaluation/metric_comparison_ic_ir.png",
        "plots/portfolio/metric_comparison_sharpe_ratio.png",
        "plots/strategy/metric_comparison_sharpe_ratio.png",
    ]
    assert manifest["artifact_groups"]["review"] == [
        "leaderboard.csv",
        "manifest.json",
        "review_summary.json",
    ]
    assert manifest["leaderboard_path"] == "leaderboard.csv"
    assert manifest["review_summary_path"] == "review_summary.json"
    assert manifest["artifacts"]["leaderboard.csv"]["rows"] == 6
    assert manifest["artifacts"]["leaderboard.csv"]["columns"] == list(frame.columns)
    assert manifest["counts_by_run_type"] == payload["counts_by_run_type"]
    assert manifest["review_config"]["output"]["emit_plots"] is True


def test_compare_research_runs_exposes_alpha_sleeve_context_without_changing_alpha_ranking(tmp_path: Path) -> None:
    alpha_root = tmp_path / "artifacts" / "alpha"
    _write_registry(
        alpha_root / "registry.jsonl",
        [
            _alpha_entry(
                run_id="alpha-best-forecast",
                alpha_name="alpha_best_forecast",
                timestamp="2026-03-19T00:00:00Z",
                ic_ir=0.9,
                mean_ic=0.04,
                mapping_name="top_q20",
                sleeve_sharpe_ratio=0.8,
                sleeve_total_return=0.05,
            ),
            _alpha_entry(
                run_id="alpha-better-sleeve",
                alpha_name="alpha_better_sleeve",
                timestamp="2026-03-19T00:01:00Z",
                ic_ir=0.7,
                mean_ic=0.03,
                mapping_name="top_q20",
                sleeve_sharpe_ratio=1.6,
                sleeve_total_return=0.12,
            ),
        ],
    )

    result = compare_research_runs(
        run_types=["alpha_evaluation"],
        alpha_artifacts_root=alpha_root,
        output_path=tmp_path,
    )

    assert [entry.run_id for entry in result.entries] == ["alpha-best-forecast", "alpha-better-sleeve"]
    assert result.entries[0].selected_metric_name == "ic_ir"
    assert result.entries[0].selected_metric_value == pytest.approx(0.9)
    assert result.entries[0].mapping_name == "top_q20"
    assert result.entries[0].sleeve_metric_name == "sharpe_ratio"
    assert result.entries[0].sleeve_metric_value == pytest.approx(0.8)
    assert result.entries[0].sleeve_secondary_metric_name == "total_return"
    assert result.entries[0].sleeve_secondary_metric_value == pytest.approx(0.05)
    assert result.entries[0].linked_portfolio_count is None

    payload = _read_json(result.json_path)
    assert payload["entries"][0]["selected_metric_name"] == "ic_ir"
    assert payload["entries"][0]["sleeve_metric_name"] == "sharpe_ratio"
    assert payload["entries"][0]["sleeve_metric_value"] == pytest.approx(0.8)


def test_compare_research_runs_links_alpha_rows_to_selected_portfolio_context(tmp_path: Path) -> None:
    portfolio_root = tmp_path / "artifacts" / "portfolios"
    alpha_root = tmp_path / "artifacts" / "alpha"
    _write_registry(
        alpha_root / "registry.jsonl",
        [
            _alpha_entry(
                run_id="alpha-linked",
                alpha_name="alpha_linked",
                timestamp="2026-03-19T00:00:00Z",
                ic_ir=0.8,
                mean_ic=0.03,
                sleeve_sharpe_ratio=1.1,
                sleeve_total_return=0.09,
            ),
            _alpha_entry(
                run_id="alpha-unlinked",
                alpha_name="alpha_unlinked",
                timestamp="2026-03-19T00:01:00Z",
                ic_ir=0.6,
                mean_ic=0.02,
            ),
        ],
    )
    _write_registry(
        portfolio_root / "registry.jsonl",
        [
            {
                **_portfolio_entry(
                    run_id="portfolio-using-alpha",
                    portfolio_name="alpha_portfolio",
                    timestamp="2026-03-19T00:02:00Z",
                    sharpe_ratio=1.5,
                    total_return=0.14,
                ),
                "components": [
                    {
                        "strategy_name": "alpha_sleeve_v1",
                        "run_id": "alpha-linked",
                        "artifact_type": "alpha_sleeve",
                    },
                    {
                        "strategy_name": "beta_v1",
                        "run_id": "strategy-beta",
                        "artifact_type": "strategy",
                    },
                ],
            },
            {
                **_portfolio_entry(
                    run_id="portfolio-no-alpha",
                    portfolio_name="plain_portfolio",
                    timestamp="2026-03-19T00:03:00Z",
                    sharpe_ratio=1.2,
                    total_return=0.10,
                ),
                "components": [
                    {
                        "strategy_name": "beta_v1",
                        "run_id": "strategy-beta",
                        "artifact_type": "strategy",
                    }
                ],
            },
        ],
    )

    result = compare_research_runs(
        run_types=["alpha_evaluation", "portfolio"],
        portfolio_artifacts_root=portfolio_root,
        alpha_artifacts_root=alpha_root,
        output_path=tmp_path,
    )

    alpha_entry = next(entry for entry in result.entries if entry.run_id == "alpha-linked")
    assert alpha_entry.linked_portfolio_count == 1
    assert alpha_entry.linked_portfolio_names == "alpha_portfolio"
    assert alpha_entry.linked_portfolio_metric_name == "sharpe_ratio"
    assert alpha_entry.linked_portfolio_metric_value == pytest.approx(1.5)

    unlinked_alpha_entry = next(entry for entry in result.entries if entry.run_id == "alpha-unlinked")
    assert unlinked_alpha_entry.linked_portfolio_count is None
    assert unlinked_alpha_entry.linked_portfolio_names is None

    frame = pd.read_csv(result.csv_path)
    linked_row = frame.loc[frame["run_id"] == "alpha-linked"].iloc[0]
    assert linked_row["sleeve_metric_name"] == "sharpe_ratio"
    assert linked_row["linked_portfolio_names"] == "alpha_portfolio"
    assert linked_row["linked_portfolio_metric_name"] == "sharpe_ratio"


def test_compare_research_runs_applies_filters_and_top_k_per_type(tmp_path: Path) -> None:
    strategy_root = tmp_path / "artifacts" / "strategies"
    portfolio_root = tmp_path / "artifacts" / "portfolios"
    alpha_root = tmp_path / "artifacts" / "alpha"
    _write_registry(
        strategy_root / "registry.jsonl",
        [
            _strategy_entry(
                run_id="strategy-daily",
                strategy_name="momentum_v1",
                timestamp="2026-03-19T00:05:00Z",
                sharpe_ratio=1.3,
                total_return=0.10,
                timeframe="1D",
            ),
            _strategy_entry(
                run_id="strategy-minute",
                strategy_name="intraday_v1",
                timestamp="2026-03-19T00:06:00Z",
                sharpe_ratio=1.6,
                total_return=0.04,
                timeframe="1Min",
                dataset="features_minute",
            ),
        ],
    )
    _write_registry(
        portfolio_root / "registry.jsonl",
        [
            _portfolio_entry(
                run_id="portfolio-minute",
                portfolio_name="intraday_portfolio",
                timestamp="2026-03-19T00:03:00Z",
                sharpe_ratio=1.2,
                total_return=0.08,
                timeframe="1Min",
            ),
        ],
    )
    _write_registry(
        alpha_root / "registry.jsonl",
        [
            _alpha_entry(
                run_id="alpha-minute",
                alpha_name="alpha_intraday",
                timestamp="2026-03-19T00:00:00Z",
                ic_ir=0.9,
                mean_ic=0.04,
                timeframe="1Min",
                dataset="features_minute",
            ),
            _alpha_entry(
                run_id="alpha-daily",
                alpha_name="alpha_daily",
                timestamp="2026-03-19T00:01:00Z",
                ic_ir=0.5,
                mean_ic=0.02,
                timeframe="1D",
            ),
        ],
    )

    result = compare_research_runs(
        run_types=["alpha_evaluation", "strategy"],
        timeframe="1Min",
        dataset="features_minute",
        top_k_per_type=1,
        strategy_artifacts_root=strategy_root,
        portfolio_artifacts_root=portfolio_root,
        alpha_artifacts_root=alpha_root,
        output_path=tmp_path / "filtered.csv",
    )

    assert result.csv_path == tmp_path / "filtered.csv"
    assert result.json_path == tmp_path / "review_summary.json"
    assert result.manifest_path == tmp_path / "manifest.json"
    assert [entry.run_type for entry in result.entries] == ["alpha_evaluation", "strategy"]
    assert [entry.entity_name for entry in result.entries] == ["alpha_intraday", "intraday_v1"]

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["artifact_files"] == [
        "filtered.csv",
        "manifest.json",
        "review_summary.json",
    ]
    assert manifest["leaderboard_path"] == "filtered.csv"
    assert result.plot_paths == {}
    assert result.skipped_plots == {
        "alpha_evaluation_metric_comparison": "Skipped alpha_evaluation metric comparison because at least 2 rows are required.",
        "strategy_metric_comparison": "Skipped strategy metric comparison because at least 2 rows are required.",
    }


def test_compare_research_runs_writes_stable_artifacts_and_review_id(tmp_path: Path) -> None:
    strategy_root = tmp_path / "artifacts" / "strategies"
    portfolio_root = tmp_path / "artifacts" / "portfolios"
    alpha_root = tmp_path / "artifacts" / "alpha"
    _write_registry(
        strategy_root / "registry.jsonl",
        [
            _strategy_entry(
                run_id="strategy-one",
                strategy_name="momentum_v1",
                timestamp="2026-03-19T00:00:00Z",
                sharpe_ratio=1.2,
                total_return=0.09,
            )
        ],
    )
    _write_registry(
        portfolio_root / "registry.jsonl",
        [
            _portfolio_entry(
                run_id="portfolio-one",
                portfolio_name="core_portfolio",
                timestamp="2026-03-19T00:00:00Z",
                sharpe_ratio=1.1,
                total_return=0.07,
            )
        ],
    )
    _write_registry(
        alpha_root / "registry.jsonl",
        [
            _alpha_entry(
                run_id="alpha-one",
                alpha_name="alpha_one",
                timestamp="2026-03-19T00:00:00Z",
                ic_ir=0.8,
                mean_ic=0.03,
            )
        ],
    )

    first = compare_research_runs(
        strategy_artifacts_root=strategy_root,
        portfolio_artifacts_root=portfolio_root,
        alpha_artifacts_root=alpha_root,
    )
    second = compare_research_runs(
        strategy_artifacts_root=strategy_root,
        portfolio_artifacts_root=portfolio_root,
        alpha_artifacts_root=alpha_root,
    )

    assert first.review_id == second.review_id
    assert first.csv_path == second.csv_path
    assert first.json_path == second.json_path
    assert first.manifest_path == second.manifest_path
    assert first.promotion_gate_path == second.promotion_gate_path
    assert first.csv_path.read_bytes() == second.csv_path.read_bytes()
    assert first.json_path.read_bytes() == second.json_path.read_bytes()
    assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()
    assert first.review_id == build_research_review_id(
        filters=first.filters,
        entries=first.entries,
        review_config=first.review_config,
    )


def test_compare_research_runs_rejects_empty_match_set(tmp_path: Path) -> None:
    with pytest.raises(ResearchReviewError, match="No research runs matched"):
        compare_research_runs(
            strategy_artifacts_root=tmp_path / "artifacts" / "strategies",
            portfolio_artifacts_root=tmp_path / "artifacts" / "portfolios",
            alpha_artifacts_root=tmp_path / "artifacts" / "alpha",
        )


def test_compare_research_runs_prefers_review_status_and_falls_back_to_legacy_promotion_status(tmp_path: Path) -> None:
    strategy_root = tmp_path / "artifacts" / "strategies"
    _write_registry(
        strategy_root / "registry.jsonl",
        [
            _strategy_entry_with_review(
                run_id="strategy-promoted",
                strategy_name="momentum_v1",
                timestamp="2026-03-19T00:05:00Z",
                review_status="promoted",
                decision_reason="committee approved",
            ),
            _strategy_entry(
                run_id="strategy-legacy",
                strategy_name="mean_reversion_v1",
                timestamp="2026-03-19T00:04:00Z",
                sharpe_ratio=0.9,
                total_return=0.06,
            ),
        ],
    )

    result = compare_research_runs(
        run_types=["strategy"],
        strategy_artifacts_root=strategy_root,
        portfolio_artifacts_root=tmp_path / "artifacts" / "portfolios",
        alpha_artifacts_root=tmp_path / "artifacts" / "alpha",
        output_path=tmp_path,
    )

    assert [entry.promotion_status for entry in result.entries] == ["promoted", "eligible"]


def test_compare_research_runs_writes_review_level_promotion_artifact_when_configured(tmp_path: Path) -> None:
    strategy_root = tmp_path / "artifacts" / "strategies"
    _write_registry(
        strategy_root / "registry.jsonl",
        [
            _strategy_entry(
                run_id="strategy-one",
                strategy_name="momentum_v1",
                timestamp="2026-03-19T00:05:00Z",
                sharpe_ratio=1.3,
                total_return=0.10,
            ),
            _strategy_entry(
                run_id="strategy-two",
                strategy_name="mean_reversion_v1",
                timestamp="2026-03-19T00:04:00Z",
                sharpe_ratio=1.1,
                total_return=0.08,
            ),
        ],
    )

    result = compare_research_runs(
        run_types=["strategy"],
        strategy_artifacts_root=strategy_root,
        portfolio_artifacts_root=tmp_path / "artifacts" / "portfolios",
        alpha_artifacts_root=tmp_path / "artifacts" / "alpha",
        output_path=tmp_path,
        promotion_gate_config={
            "status_on_pass": "review_ready",
            "status_on_fail": "needs_work",
            "gates": [
                {
                    "gate_id": "minimum_review_rows",
                    "source": "metrics",
                    "metric_path": "entry_count",
                    "comparator": "gte",
                    "threshold": 2,
                }
            ],
        },
    )

    assert result.promotion_gate_path == tmp_path / "promotion_gates.json"
    promotion_payload = json.loads(result.promotion_gate_path.read_text(encoding="utf-8"))
    assert promotion_payload["promotion_status"] == "review_ready"
    assert promotion_payload["gate_count"] == 1
    assert promotion_payload["results"][0]["actual_value"] == pytest.approx(2.0)

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["artifact_files"] == [
        "leaderboard.csv",
        "manifest.json",
        "plots/strategy/metric_comparison_sharpe_ratio.png",
        "promotion_gates.json",
        "review_summary.json",
    ]
    assert manifest["promotion_gate_summary"] == {
        "artifact_filename": "promotion_gates.json",
        "configured": True,
        "evaluation_status": "pass",
        "failed_gate_count": 0,
        "gate_count": 1,
        "missing_gate_count": 0,
        "passed_gate_count": 1,
        "promotion_status": "review_ready",
        "status_on_fail": "needs_work",
        "status_on_pass": "review_ready",
    }
    assert manifest["plot_paths"] == {
        "strategy_metric_comparison": "plots/strategy/metric_comparison_sharpe_ratio.png",
    }


def test_compare_research_runs_applies_review_config_precedence_and_persists_effective_config(tmp_path: Path) -> None:
    strategy_root = tmp_path / "artifacts" / "strategies"
    _write_registry(
        strategy_root / "registry.jsonl",
        [
            _strategy_entry(
                run_id="strategy-a",
                strategy_name="momentum_v1",
                timestamp="2026-03-19T00:05:00Z",
                sharpe_ratio=1.0,
                total_return=0.20,
            ),
            _strategy_entry(
                run_id="strategy-b",
                strategy_name="mean_reversion_v1",
                timestamp="2026-03-19T00:04:00Z",
                sharpe_ratio=1.4,
                total_return=0.08,
            ),
        ],
    )

    result = compare_research_runs(
        run_types=["strategy"],
        strategy_artifacts_root=strategy_root,
        portfolio_artifacts_root=tmp_path / "artifacts" / "portfolios",
        alpha_artifacts_root=tmp_path / "artifacts" / "alpha",
        output_path=tmp_path,
        review_config={
            "filters": {"top_k_per_type": 1},
            "ranking": {"strategy_primary_metric": "total_return"},
            "output": {"emit_plots": False},
        },
        strategy_metric="sharpe_ratio",
    )

    assert [entry.run_id for entry in result.entries] == ["strategy-b"]
    assert result.review_config["filters"]["top_k_per_type"] == 1
    assert result.review_config["ranking"]["strategy_primary_metric"] == "sharpe_ratio"
    assert result.review_config["output"]["emit_plots"] is False
    assert result.plot_paths == {}
    assert result.skipped_plots == {
        "strategy_metric_comparison": "Skipped plot generation because review_config.output.emit_plots is false."
    }

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["review_config"] == result.review_config


def test_compare_research_runs_handles_missing_metrics_and_nested_review_metadata_counts(tmp_path: Path) -> None:
    strategy_root = tmp_path / "artifacts" / "strategies"
    nested_review = _strategy_entry(
        run_id="strategy-nested",
        strategy_name="carry_v1",
        timestamp="2026-03-19T00:06:00Z",
        sharpe_ratio=None,
        total_return=0.12,
    )
    nested_review.pop("promotion_gate_summary")
    nested_review.pop("promotion_status")
    nested_review["review_metadata"] = {
        "decision_reason": "committee deferred to candidate queue",
        "decision_source": "manual_review",
        "promotion_gate_summary": {
            "promotion_status": "eligible",
            "passed_gate_count": 1,
            "gate_count": 3,
        },
        "promotion_status": "eligible",
        "reviewed_at": "2026-03-19T00:11:00Z",
        "schema_version": 1,
        "status": "candidate",
    }
    _write_registry(
        strategy_root / "registry.jsonl",
        [
            _strategy_entry(
                run_id="strategy-strong",
                strategy_name="momentum_v1",
                timestamp="2026-03-19T00:05:00Z",
                sharpe_ratio=1.4,
                total_return=0.08,
            ),
            nested_review,
            _strategy_entry(
                run_id="strategy-missing",
                strategy_name="value_v1",
                timestamp="2026-03-19T00:04:00Z",
                sharpe_ratio=None,
                total_return=0.02,
            ),
        ],
    )

    result = compare_research_runs(
        run_types=["strategy"],
        strategy_artifacts_root=strategy_root,
        portfolio_artifacts_root=tmp_path / "artifacts" / "portfolios",
        alpha_artifacts_root=tmp_path / "artifacts" / "alpha",
        output_path=tmp_path,
    )

    assert [entry.run_id for entry in result.entries] == [
        "strategy-strong",
        "strategy-nested",
        "strategy-missing",
    ]
    assert [entry.selected_metric_value for entry in result.entries] == [1.4, None, None]
    assert [entry.secondary_metric_value for entry in result.entries] == [0.08, 0.12, 0.02]
    assert result.entries[1].promotion_status == "candidate"
    assert result.entries[1].passed_gate_count == 1
    assert result.entries[1].gate_count == 3

    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert [row["selected_metric_value"] for row in payload["entries"]] == [1.4, None, None]
    assert payload["entries"][1]["promotion_status"] == "candidate"
    assert result.skipped_plots == {
        "strategy_metric_comparison": "Skipped strategy metric comparison because at least 2 numeric metric rows are required."
    }

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["review_metrics"] == {
        "alpha_entry_count": 0,
        "blocked_entry_count": 0,
        "eligible_entry_count": 2,
        "entry_count": 3,
        "portfolio_entry_count": 0,
        "promoted_entry_count": 0,
        "reviewed_entry_count": 3,
        "strategy_entry_count": 3,
    }


def test_compare_research_runs_is_deterministic_across_output_paths_with_review_decisions_and_gates(tmp_path: Path) -> None:
    strategy_root = tmp_path / "artifacts" / "strategies"
    _write_registry(
        strategy_root / "registry.jsonl",
        [
            _strategy_entry_with_review(
                run_id="strategy-promoted",
                strategy_name="momentum_v1",
                timestamp="2026-03-19T00:05:00Z",
                review_status="promoted",
                decision_reason="committee approved",
            ),
            _strategy_entry(
                run_id="strategy-candidate",
                strategy_name="carry_v1",
                timestamp="2026-03-19T00:04:00Z",
                sharpe_ratio=1.05,
                total_return=0.07,
            ),
        ],
    )

    first = compare_research_runs(
        run_types=["strategy"],
        strategy_artifacts_root=strategy_root,
        portfolio_artifacts_root=tmp_path / "artifacts" / "portfolios",
        alpha_artifacts_root=tmp_path / "artifacts" / "alpha",
        output_path=tmp_path / "run_one",
        emit_plots=False,
        promotion_gate_config={
            "status_on_pass": "review_ready",
            "status_on_fail": "needs_work",
            "gates": [
                {
                    "gate_id": "minimum_rows",
                    "source": "metrics",
                    "metric_path": "entry_count",
                    "comparator": "gte",
                    "threshold": 2,
                },
                {
                    "gate_id": "promoted_present",
                    "source": "metrics",
                    "metric_path": "promoted_entry_count",
                    "comparator": "gte",
                    "threshold": 1,
                },
            ],
        },
    )
    second = compare_research_runs(
        run_types=["strategy"],
        strategy_artifacts_root=strategy_root,
        portfolio_artifacts_root=tmp_path / "artifacts" / "portfolios",
        alpha_artifacts_root=tmp_path / "artifacts" / "alpha",
        output_path=tmp_path / "run_two",
        emit_plots=False,
        promotion_gate_config={
            "status_on_pass": "review_ready",
            "status_on_fail": "needs_work",
            "gates": [
                {
                    "gate_id": "minimum_rows",
                    "source": "metrics",
                    "metric_path": "entry_count",
                    "comparator": "gte",
                    "threshold": 2,
                },
                {
                    "gate_id": "promoted_present",
                    "source": "metrics",
                    "metric_path": "promoted_entry_count",
                    "comparator": "gte",
                    "threshold": 1,
                },
            ],
        },
    )

    assert first.review_id == second.review_id
    assert [entry.run_id for entry in first.entries] == [entry.run_id for entry in second.entries]
    assert [entry.promotion_status for entry in first.entries] == ["promoted", "eligible"]
    assert first.csv_path.read_bytes() == second.csv_path.read_bytes()
    assert first.promotion_gate_path is not None
    assert second.promotion_gate_path is not None
    assert first.promotion_gate_path.read_bytes() == second.promotion_gate_path.read_bytes()
    assert _without_output_path(_read_json(first.json_path)) == _without_output_path(_read_json(second.json_path))
    assert _without_output_path(_read_json(first.manifest_path)) == _without_output_path(_read_json(second.manifest_path))

    promotion_payload = json.loads(first.promotion_gate_path.read_text(encoding="utf-8"))
    assert promotion_payload["promotion_status"] == "review_ready"
    assert promotion_payload["passed_gate_count"] == 2
    assert promotion_payload["results"][1]["actual_value"] == pytest.approx(1.0)


def test_parse_args_supports_unified_review_inputs() -> None:
    args = parse_args(
        [
            "--from-registry",
            "--run-types",
            "alpha_evaluation,strategy",
            "portfolio",
            "--timeframe",
            "1D",
            "--dataset",
            "features_daily",
            "--alpha-name",
            "alpha_one",
            "--strategy-name",
            "momentum_v1",
            "--portfolio-name",
            "core_portfolio",
            "--top-k",
            "2",
            "--output-path",
            "artifacts/custom",
            "--review-config",
            "configs/review.yml",
            "--alpha-metric",
            "rank_ic_ir",
            "--strategy-secondary-metric",
            "cumulative_return",
            "--portfolio-metric",
            "total_return",
            "--promotion-gates",
            "configs/review_gates.yml",
            "--disable-plots",
        ]
    )

    assert args.from_registry is True
    assert args.run_types == ["alpha_evaluation,strategy", "portfolio"]
    assert args.timeframe == "1D"
    assert args.dataset == "features_daily"
    assert args.alpha_name == "alpha_one"
    assert args.strategy_name == "momentum_v1"
    assert args.portfolio_name == "core_portfolio"
    assert args.top_k == 2
    assert args.output_path == "artifacts/custom"
    assert args.review_config == "configs/review.yml"
    assert args.alpha_metric == "rank_ic_ir"
    assert args.strategy_secondary_metric == "cumulative_return"
    assert args.portfolio_metric == "total_return"
    assert args.promotion_gates == "configs/review_gates.yml"
    assert args.disable_plots is True


def test_parse_args_supports_unified_review_legacy_flag_aliases() -> None:
    args = parse_args(
        [
            "--from_registry",
            "--run_types",
            "strategy,portfolio",
            "--strategy_name",
            "momentum_v1",
            "--portfolio_name",
            "core_portfolio",
            "--top_k",
            "1",
            "--output_path",
            "artifacts/custom",
        ]
    )

    assert args.from_registry is True
    assert args.run_types == ["strategy,portfolio"]
    assert args.strategy_name == "momentum_v1"
    assert args.portfolio_name == "core_portfolio"
    assert args.top_k == 1
    assert args.output_path == "artifacts/custom"


def test_run_cli_prints_unified_review_summary(monkeypatch, capsys, tmp_path: Path) -> None:
    expected_result = ResearchReviewResult(
        review_id="registry_review_deadbeefcafe",
        filters={
            "run_types": ["strategy"],
            "timeframe": "1D",
            "dataset": None,
            "alpha_name": None,
            "strategy_name": None,
            "portfolio_name": None,
            "top_k_per_type": None,
        },
        entries=[],
        csv_path=tmp_path / "leaderboard.csv",
        json_path=tmp_path / "review_summary.json",
        manifest_path=tmp_path / "manifest.json",
        promotion_gate_path=None,
        review_config=load_review_config().to_dict(),
    )

    monkeypatch.setattr("src.cli.compare_research.compare_research_runs", lambda **kwargs: expected_result)

    result = run_cli(["--from-registry", "--run-types", "strategy"])

    assert result is expected_result
    stdout = capsys.readouterr().out
    assert "review_id: registry_review_deadbeefcafe" in stdout
    assert "plot_count: 0" in stdout
    assert "rows: 0" in stdout
    assert "leaderboard_csv:" in stdout


def test_run_cli_loads_review_config_and_applies_cli_overrides(monkeypatch, tmp_path: Path) -> None:
    review_config_path = tmp_path / "review.yml"
    review_config_path.write_text(
        """
review:
  filters:
    run_types: [strategy]
  ranking:
    strategy_primary_metric: total_return
""".strip(),
        encoding="utf-8",
    )
    gates_path = tmp_path / "review_gates.yml"
    gates_path.write_text(
        """
promotion_gates:
  gates:
    - gate_id: minimum_rows
      source: metrics
      metric_path: entry_count
      comparator: gte
      threshold: 1
""".strip(),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_compare_research_runs(**kwargs):
        captured.update(kwargs)
        return ResearchReviewResult(
            review_id="registry_review_deadbeefcafe",
            filters={
                "run_types": ["strategy"],
                "timeframe": None,
                "dataset": None,
                "alpha_name": None,
                "strategy_name": None,
                "portfolio_name": None,
                "top_k_per_type": None,
            },
            entries=[],
            csv_path=tmp_path / "leaderboard.csv",
            json_path=tmp_path / "review_summary.json",
            manifest_path=tmp_path / "manifest.json",
            promotion_gate_path=None,
            review_config={},
        )

    monkeypatch.setattr("src.cli.compare_research.compare_research_runs", fake_compare_research_runs)

    run_cli(
        [
            "--from-registry",
            "--review-config",
            str(review_config_path),
            "--strategy-metric",
            "sharpe_ratio",
            "--promotion-gates",
            str(gates_path),
            "--disable-plots",
        ]
    )

    assert isinstance(captured["review_config"], dict)
    assert captured["review_config"]["ranking"]["strategy_primary_metric"] == "total_return"
    assert captured["strategy_metric"] == "sharpe_ratio"
    assert captured["emit_plots"] is False
    assert captured["promotion_gate_config"] == {
        "gates": [
            {
                "comparator": "gte",
                "gate_id": "minimum_rows",
                "metric_path": "entry_count",
                "source": "metrics",
                "threshold": 1,
            }
        ]
    }


def test_run_cli_requires_registry_mode() -> None:
    with pytest.raises(ValueError, match="Pass --from-registry"):
        run_cli([])
