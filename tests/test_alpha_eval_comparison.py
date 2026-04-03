from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cli.compare_alpha import parse_args, run_cli
from src.research.alpha_eval import compare_alpha_evaluation_runs
from src.research.alpha_eval.compare import (
    AlphaEvaluationComparisonError,
    AlphaEvaluationComparisonResult,
    AlphaEvaluationLeaderboardEntry,
    build_alpha_comparison_id,
)


def _write_registry(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _entry(
    *,
    run_id: str,
    alpha_name: str,
    dataset: str = "features_daily",
    timeframe: str = "1D",
    evaluation_horizon: int = 1,
    mean_ic: object = 0.1,
    ic_ir: object = 0.5,
    mean_rank_ic: object = 0.2,
    rank_ic_ir: object = 0.4,
    n_periods: object = 10,
    mapping_name: str | None = None,
    mapping_policy: str | None = None,
    mapping_quantile: float | None = None,
    sleeve_metrics: dict[str, object] | None = None,
) -> dict[str, object]:
    signal_mapping: dict[str, object] | None = None
    if mapping_policy is not None or mapping_name is not None:
        signal_mapping = {
            "metadata": {} if mapping_name is None else {"name": mapping_name},
            "policy": "rank_long_short" if mapping_policy is None else mapping_policy,
        }
        if mapping_quantile is not None:
            signal_mapping["quantile"] = mapping_quantile

    manifest = {
        "artifact_paths": {},
        "metric_summary": {
            "ic_ir": ic_ir,
            "mean_ic": mean_ic,
            "mean_rank_ic": mean_rank_ic,
            "n_periods": n_periods,
            "rank_ic_ir": rank_ic_ir,
        },
        "run_id": run_id,
    }
    if sleeve_metrics is not None:
        manifest["sleeve"] = {"enabled": True, "metric_summary": dict(sleeve_metrics)}

    config = {"dataset": dataset}
    if signal_mapping is not None:
        config["signal_mapping"] = signal_mapping

    return {
        "run_id": run_id,
        "run_type": "alpha_evaluation",
        "timestamp": "2026-03-19T00:00:00Z",
        "alpha_name": alpha_name,
        "dataset": dataset,
        "timeframe": timeframe,
        "evaluation_horizon": evaluation_horizon,
        "artifact_path": f"artifacts/alpha/{run_id}",
        "config": config,
        "manifest": manifest,
        "metrics_summary": {
            "mean_ic": mean_ic,
            "ic_ir": ic_ir,
            "mean_rank_ic": mean_rank_ic,
            "rank_ic_ir": rank_ic_ir,
            "n_periods": n_periods,
        },
    }


def test_compare_alpha_evaluation_runs_filters_and_ranks_forecast_view_deterministically(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts" / "alpha"
    _write_registry(
        artifacts_root / "registry.jsonl",
        [
            _entry(run_id="run-b", alpha_name="alpha_b", ic_ir=0.7, mean_ic=0.03),
            _entry(run_id="run-a-2", alpha_name="alpha_a", ic_ir=0.8, mean_ic=0.02),
            _entry(run_id="run-a-1", alpha_name="alpha_a", ic_ir=0.8, mean_ic=0.02),
            _entry(run_id="run-min", alpha_name="alpha_min", dataset="features_minute", timeframe="1Min", ic_ir=1.1),
        ],
    )

    result = compare_alpha_evaluation_runs(
        artifacts_root=artifacts_root,
        dataset="features_daily",
    )

    assert [entry.run_id for entry in result.leaderboard] == ["run-a-1", "run-a-2", "run-b"]
    assert [entry.rank for entry in result.leaderboard] == [1, 2, 3]
    assert result.metric == "ic_ir"
    assert result.view == "forecast"
    assert result.forecast_metric == "ic_ir"
    assert result.sleeve_metric == "sharpe_ratio"
    assert result.filters == {
        "alpha_name": None,
        "dataset": "features_daily",
        "evaluation_horizon": None,
        "mapping_name": None,
        "timeframe": None,
    }


def test_compare_alpha_evaluation_runs_ranks_sleeve_view_and_filters_mapping_name(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts" / "alpha"
    _write_registry(
        artifacts_root / "registry.jsonl",
        [
            _entry(
                run_id="run-zscore",
                alpha_name="alpha_a",
                mapping_policy="zscore_continuous",
                sleeve_metrics={"sharpe_ratio": 1.1, "total_return": 0.12, "turnover": 0.20},
            ),
            _entry(
                run_id="run-quantile-low-turnover",
                alpha_name="alpha_b",
                mapping_policy="top_bottom_quantile",
                mapping_quantile=0.2,
                sleeve_metrics={"sharpe_ratio": 1.4, "total_return": 0.09, "turnover": 0.10},
            ),
            _entry(
                run_id="run-quantile-high-turnover",
                alpha_name="alpha_c",
                mapping_policy="top_bottom_quantile",
                mapping_quantile=0.2,
                sleeve_metrics={"sharpe_ratio": 1.4, "total_return": 0.08, "turnover": 0.25},
            ),
        ],
    )

    result = compare_alpha_evaluation_runs(
        artifacts_root=artifacts_root,
        view="sleeve",
        sleeve_metric="sharpe_ratio",
        mapping_name="top_bottom_quantile[q=0.2]",
    )

    assert [entry.run_id for entry in result.leaderboard] == [
        "run-quantile-low-turnover",
        "run-quantile-high-turnover",
    ]
    assert result.metric == "sharpe_ratio"
    assert result.filters["mapping_name"] == "top_bottom_quantile[q=0.2]"


def test_compare_alpha_evaluation_runs_ranks_combined_view_forecast_then_sleeve(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts" / "alpha"
    _write_registry(
        artifacts_root / "registry.jsonl",
        [
            _entry(
                run_id="run-best-combined",
                alpha_name="alpha_a",
                ic_ir=0.8,
                mean_ic=0.04,
                sleeve_metrics={"sharpe_ratio": 1.5, "total_return": 0.12, "turnover": 0.15},
            ),
            _entry(
                run_id="run-mid-combined",
                alpha_name="alpha_b",
                ic_ir=0.8,
                mean_ic=0.04,
                sleeve_metrics={"sharpe_ratio": 1.1, "total_return": 0.13, "turnover": 0.10},
            ),
            _entry(
                run_id="run-better-sleeve-worse-forecast",
                alpha_name="alpha_c",
                ic_ir=0.7,
                mean_ic=0.05,
                sleeve_metrics={"sharpe_ratio": 2.1, "total_return": 0.20, "turnover": 0.08},
            ),
        ],
    )

    result = compare_alpha_evaluation_runs(
        artifacts_root=artifacts_root,
        view="combined",
        metric="ic_ir",
        sleeve_metric="sharpe_ratio",
    )

    assert [entry.run_id for entry in result.leaderboard] == [
        "run-best-combined",
        "run-mid-combined",
        "run-better-sleeve-worse-forecast",
    ]
    assert result.metric == "ic_ir+sharpe_ratio"


def test_compare_alpha_evaluation_runs_places_missing_and_nan_metrics_last(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts" / "alpha"
    _write_registry(
        artifacts_root / "registry.jsonl",
        [
            _entry(run_id="run-good", alpha_name="alpha_good", ic_ir=0.8, mean_ic=0.05),
            _entry(run_id="run-none", alpha_name="alpha_none", ic_ir=None, mean_ic=0.9),
            _entry(run_id="run-nan", alpha_name="alpha_nan", ic_ir=float("nan"), mean_ic=0.7),
        ],
    )

    result = compare_alpha_evaluation_runs(artifacts_root=artifacts_root)

    assert [entry.run_id for entry in result.leaderboard] == ["run-good", "run-none", "run-nan"]
    assert result.leaderboard[1].ic_ir is None
    assert result.leaderboard[2].ic_ir is None


def test_compare_alpha_evaluation_runs_writes_stable_artifacts_and_comparison_id(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts" / "alpha"
    _write_registry(
        artifacts_root / "registry.jsonl",
        [
            _entry(
                run_id="run-2",
                alpha_name="alpha_two",
                ic_ir=0.7,
                mean_ic=0.03,
                mapping_name="named_mapping",
                sleeve_metrics={"sharpe_ratio": 1.0, "total_return": 0.05, "turnover": 0.20},
            ),
            _entry(
                run_id="run-1",
                alpha_name="alpha_one",
                ic_ir=0.9,
                mean_ic=0.05,
                mapping_name="named_mapping",
                sleeve_metrics={"sharpe_ratio": 1.2, "total_return": 0.07, "turnover": 0.10},
            ),
        ],
    )

    first = compare_alpha_evaluation_runs(
        artifacts_root=artifacts_root,
        view="combined",
        mapping_name="named_mapping",
    )
    second = compare_alpha_evaluation_runs(
        artifacts_root=artifacts_root,
        view="combined",
        mapping_name="named_mapping",
    )

    assert first.comparison_id == second.comparison_id
    assert first.csv_path == second.csv_path
    assert first.json_path == second.json_path
    assert first.csv_path.read_bytes() == second.csv_path.read_bytes()
    assert first.json_path.read_bytes() == second.json_path.read_bytes()

    frame = pd.read_csv(first.csv_path)
    assert list(frame.columns) == [
        "rank",
        "alpha_name",
        "run_id",
        "dataset",
        "timeframe",
        "evaluation_horizon",
        "mapping_name",
        "mean_ic",
        "ic_ir",
        "mean_rank_ic",
        "rank_ic_ir",
        "n_periods",
        "sharpe_ratio",
        "annualized_return",
        "total_return",
        "cumulative_return",
        "max_drawdown",
        "turnover",
        "average_turnover",
        "total_turnover",
        "win_rate",
        "hit_rate",
        "profit_factor",
        "trade_count",
        "exposure_pct",
        "total_transaction_cost",
        "total_slippage_cost",
        "total_execution_friction",
        "artifact_path",
    ]
    payload = json.loads(first.json_path.read_text(encoding="utf-8"))
    assert payload["comparison_id"] == first.comparison_id
    assert payload["run_ids"] == ["run-1", "run-2"]
    assert payload["view"] == "combined"
    assert payload["forecast_metric"] == "ic_ir"
    assert payload["sleeve_metric"] == "sharpe_ratio"
    assert first.comparison_id == build_alpha_comparison_id(
        metric="ic_ir",
        view="combined",
        sleeve_metric="sharpe_ratio",
        filters={
            "alpha_name": None,
            "dataset": None,
            "evaluation_horizon": None,
            "mapping_name": "named_mapping",
            "timeframe": None,
        },
        run_ids=["run-1", "run-2"],
    )


def test_compare_alpha_evaluation_runs_rejects_empty_registry_and_empty_match_set(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts" / "alpha"

    with pytest.raises(AlphaEvaluationComparisonError, match="registry is empty"):
        compare_alpha_evaluation_runs(artifacts_root=artifacts_root)

    _write_registry(artifacts_root / "registry.jsonl", [_entry(run_id="run-1", alpha_name="alpha_one")])

    with pytest.raises(AlphaEvaluationComparisonError, match="No alpha evaluation runs matched"):
        compare_alpha_evaluation_runs(artifacts_root=artifacts_root, dataset="missing_dataset")


def test_compare_alpha_evaluation_runs_rejects_malformed_registry_entries(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts" / "alpha"
    malformed = _entry(run_id="run-1", alpha_name="alpha_one")
    del malformed["metrics_summary"]
    _write_registry(artifacts_root / "registry.jsonl", [malformed])

    with pytest.raises(AlphaEvaluationComparisonError, match="missing metrics_summary"):
        compare_alpha_evaluation_runs(artifacts_root=artifacts_root)

    malformed_metric = _entry(run_id="run-2", alpha_name="alpha_two")
    malformed_metric["metrics_summary"] = {
        "mean_ic": 0.1,
        "ic_ir": "bad",
        "mean_rank_ic": 0.2,
        "rank_ic_ir": 0.3,
        "n_periods": 5,
    }
    _write_registry(artifacts_root / "registry.jsonl", [malformed_metric])

    with pytest.raises(AlphaEvaluationComparisonError, match="non-numeric ic_ir"):
        compare_alpha_evaluation_runs(artifacts_root=artifacts_root)


def test_compare_alpha_evaluation_runs_rejects_missing_summary_keys(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts" / "alpha"
    malformed = _entry(run_id="run-1", alpha_name="alpha_one")
    malformed["metrics_summary"] = {
        "mean_ic": 0.1,
        "ic_ir": 0.2,
        "mean_rank_ic": 0.3,
        "n_periods": 4,
    }
    _write_registry(artifacts_root / "registry.jsonl", [malformed])

    with pytest.raises(AlphaEvaluationComparisonError, match="missing metrics_summary fields: rank_ic_ir"):
        compare_alpha_evaluation_runs(artifacts_root=artifacts_root)


def test_parse_args_supports_alpha_comparison_inputs() -> None:
    args = parse_args(
        [
            "--from-registry",
            "--view",
            "combined",
            "--metric",
            "mean_rank_ic",
            "--sleeve-metric",
            "max_drawdown",
            "--alpha-name",
            "alpha_one",
            "--dataset",
            "features_daily",
            "--timeframe",
            "1D",
            "--mapping-name",
            "named_mapping",
            "--evaluation-horizon",
            "5",
            "--output-path",
            "artifacts/custom",
        ]
    )

    assert args.from_registry is True
    assert args.view == "combined"
    assert args.metric == "mean_rank_ic"
    assert args.sleeve_metric == "max_drawdown"
    assert args.alpha_name == "alpha_one"
    assert args.dataset == "features_daily"
    assert args.timeframe == "1D"
    assert args.mapping_name == "named_mapping"
    assert args.evaluation_horizon == 5
    assert args.output_path == "artifacts/custom"


def test_parse_args_supports_alpha_legacy_flag_aliases() -> None:
    args = parse_args(
        [
            "--from_registry",
            "--alpha_name",
            "alpha_one",
            "--mapping_name",
            "named_mapping",
            "--horizon",
            "3",
            "--output_path",
            "artifacts/custom",
        ]
    )

    assert args.from_registry is True
    assert args.alpha_name == "alpha_one"
    assert args.mapping_name == "named_mapping"
    assert args.evaluation_horizon == 3
    assert args.output_path == "artifacts/custom"


def test_run_cli_prints_alpha_leaderboard_summary(monkeypatch, capsys, tmp_path: Path) -> None:
    expected_result = AlphaEvaluationComparisonResult(
        comparison_id="registry_alpha_combined_deadbeefcafe",
        view="combined",
        metric="ic_ir+sharpe_ratio",
        forecast_metric="ic_ir",
        sleeve_metric="sharpe_ratio",
        filters={
            "alpha_name": None,
            "dataset": None,
            "evaluation_horizon": None,
            "mapping_name": "named_mapping",
            "timeframe": "1D",
        },
        leaderboard=[
            AlphaEvaluationLeaderboardEntry(
                rank=1,
                alpha_name="alpha_one",
                run_id="run-1",
                dataset="features_daily",
                timeframe="1D",
                evaluation_horizon=1,
                mapping_name="named_mapping",
                mean_ic=0.05,
                ic_ir=0.9,
                mean_rank_ic=0.04,
                rank_ic_ir=0.8,
                n_periods=12,
                sharpe_ratio=1.4,
                annualized_return=0.22,
                total_return=0.08,
                cumulative_return=0.08,
                max_drawdown=0.03,
                turnover=0.12,
                average_turnover=0.12,
                total_turnover=1.2,
                win_rate=0.55,
                hit_rate=0.57,
                profit_factor=1.3,
                trade_count=18.0,
                exposure_pct=100.0,
                total_transaction_cost=0.001,
                total_slippage_cost=0.0008,
                total_execution_friction=0.0018,
                artifact_path="artifacts/alpha/run-1",
            )
        ],
        csv_path=tmp_path / "leaderboard.csv",
        json_path=tmp_path / "leaderboard.json",
    )

    monkeypatch.setattr("src.cli.compare_alpha.compare_alpha_evaluation_runs", lambda **kwargs: expected_result)

    result = run_cli(["--from-registry", "--view", "combined", "--timeframe", "1D", "--mapping-name", "named_mapping"])

    assert result is expected_result
    stdout = capsys.readouterr().out
    assert "comparison_id: registry_alpha_combined_deadbeefcafe" in stdout
    assert "view: combined" in stdout
    assert "ranking: ic_ir+sharpe_ratio" in stdout
    assert "forecast_metric: ic_ir" in stdout
    assert "sleeve_metric: sharpe_ratio" in stdout
    assert "rows: 1" in stdout
    assert "named_mapping" in stdout
    assert "leaderboard_csv:" in stdout


def test_run_cli_requires_registry_mode() -> None:
    with pytest.raises(ValueError, match="Pass --from-registry"):
        run_cli([])
