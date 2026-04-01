from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cli.compare_research import parse_args, run_cli
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
) -> dict[str, object]:
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
    assert result.json_path == tmp_path / "leaderboard.json"
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
        "artifact_path",
    ]
    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert payload["counts_by_run_type"] == {
        "alpha_evaluation": 2,
        "strategy": 2,
        "portfolio": 2,
    }
    assert frame["promotion_status"].tolist() == [
        "eligible",
        "eligible",
        "eligible",
        "eligible",
        "blocked",
        "blocked",
    ]


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
    assert [entry.run_type for entry in result.entries] == ["alpha_evaluation", "strategy"]
    assert [entry.entity_name for entry in result.entries] == ["alpha_intraday", "intraday_v1"]


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
    assert first.csv_path.read_bytes() == second.csv_path.read_bytes()
    assert first.json_path.read_bytes() == second.json_path.read_bytes()
    assert first.review_id == build_research_review_id(filters=first.filters, entries=first.entries)


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
        json_path=tmp_path / "leaderboard.json",
    )

    monkeypatch.setattr("src.cli.compare_research.compare_research_runs", lambda **kwargs: expected_result)

    result = run_cli(["--from-registry", "--run-types", "strategy"])

    assert result is expected_result
    stdout = capsys.readouterr().out
    assert "review_id: registry_review_deadbeefcafe" in stdout
    assert "leaderboard_csv:" in stdout


def test_run_cli_requires_registry_mode() -> None:
    with pytest.raises(ValueError, match="Pass --from-registry"):
        run_cli([])
