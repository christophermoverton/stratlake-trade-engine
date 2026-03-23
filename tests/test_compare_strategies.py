from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.cli.compare_strategies import parse_args, parse_strategy_names, run_cli
from src.research import compare


def test_parse_args_supports_comparison_flags() -> None:
    args = parse_args(
        [
            "--strategies",
            "momentum_v1",
            "mean_reversion_v1",
            "--start",
            "2025-01-01",
            "--end",
            "2025-03-01",
            "--metric",
            "total_return",
            "--top_k",
            "2",
            "--from_registry",
            "--output_path",
            "artifacts/custom",
        ]
    )

    assert args.strategies == ["momentum_v1", "mean_reversion_v1"]
    assert args.start == "2025-01-01"
    assert args.end == "2025-03-01"
    assert args.metric == "total_return"
    assert args.top_k == 2
    assert args.from_registry is True
    assert args.output_path == "artifacts/custom"
    assert args.evaluation is None


def test_parse_strategy_names_supports_mixed_cli_formats() -> None:
    assert parse_strategy_names(
        ["momentum_v1,mean_reversion_v1", "buy_and_hold_v1", "momentum_v1"]
    ) == ["momentum_v1", "mean_reversion_v1", "buy_and_hold_v1", "momentum_v1"]


def test_compare_strategies_runs_fresh_execution_and_writes_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def fake_run_strategy_experiment(strategy_name: str, start=None, end=None):
        calls.append(strategy_name)
        assert start == "2025-01-01"
        assert end == "2025-03-01"
        return compare.StrategyRunResult(
            strategy_name=strategy_name,
            run_id=f"run-{strategy_name}",
            metrics={
                "total_return": 0.11 if strategy_name == "momentum_v1" else 0.05,
                "cumulative_return": 0.11 if strategy_name == "momentum_v1" else 0.05,
                "annualized_return": 0.18 if strategy_name == "momentum_v1" else 0.08,
                "sharpe_ratio": 1.4 if strategy_name == "momentum_v1" else 0.8,
                "max_drawdown": 0.07 if strategy_name == "momentum_v1" else 0.03,
                "win_rate": 0.62 if strategy_name == "momentum_v1" else 0.51,
            },
            experiment_dir=tmp_path / f"run-{strategy_name}",
            results_df=None,  # type: ignore[arg-type]
        )

    monkeypatch.setattr(compare, "run_strategy_experiment", fake_run_strategy_experiment)

    result = compare.compare_strategies(
        ["momentum_v1", "mean_reversion_v1"],
        start="2025-01-01",
        end="2025-03-01",
        output_path=tmp_path,
    )

    assert calls == ["momentum_v1", "mean_reversion_v1"]
    assert [entry.strategy_name for entry in result.leaderboard] == [
        "momentum_v1",
        "mean_reversion_v1",
    ]
    assert result.csv_path == tmp_path / "leaderboard.csv"
    assert result.json_path == tmp_path / "leaderboard.json"
    assert result.csv_path.exists()
    assert result.json_path.exists()

    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert payload["selection_mode"] == "fresh"
    assert payload["metric"] == "sharpe_ratio"
    assert payload["leaderboard"][0]["strategy_name"] == "momentum_v1"
    assert "run_id" not in payload["leaderboard"][0]
    assert payload["leaderboard"][0]["annualized_return"] == pytest.approx(0.18)
    assert payload["leaderboard"][0]["win_rate"] == pytest.approx(0.62)

    leaderboard_frame = pd.read_csv(result.csv_path)
    assert list(leaderboard_frame.columns) == [
        "rank",
        "strategy_name",
        "evaluation_mode",
        "selected_metric_name",
        "selected_metric_value",
        "cumulative_return",
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "volatility",
        "sharpe_ratio",
        "max_drawdown",
        "win_rate",
        "hit_rate",
        "profit_factor",
        "turnover",
        "exposure_pct",
    ]


def test_compare_strategies_runs_walk_forward_execution_when_evaluation_provided(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    strategy_configs = {
        "momentum_v1": {"dataset": "features_daily", "parameters": {"lookback_short": 5}},
        "mean_reversion_v1": {"dataset": "features_daily", "parameters": {"lookback": 10, "threshold": 1.0}},
    }
    calls: list[tuple[str, Path]] = []

    monkeypatch.setattr(compare, "get_strategy_config", lambda strategy_name: strategy_configs[strategy_name])
    monkeypatch.setattr(compare, "build_strategy", lambda strategy_name, config: type("S", (), {"name": strategy_name, "dataset": config["dataset"]})())

    def fake_run_walk_forward_experiment(strategy_name, strategy, evaluation_path, strategy_config):
        calls.append((strategy_name, evaluation_path))
        return compare.WalkForwardRunResult(
            strategy_name=strategy_name,
            run_id=f"wf-{strategy_name}",
            experiment_dir=tmp_path / f"wf-{strategy_name}",
            metrics={
                "total_return": 0.2 if strategy_name == "momentum_v1" else 0.15,
                "cumulative_return": 0.2 if strategy_name == "momentum_v1" else 0.15,
                "sharpe_ratio": 1.9 if strategy_name == "momentum_v1" else 1.1,
                "max_drawdown": 0.08,
            },
            aggregate_summary={"split_count": 2},
            splits=[],
        )

    monkeypatch.setattr(compare, "run_walk_forward_experiment", fake_run_walk_forward_experiment)

    result = compare.compare_strategies(
        ["momentum_v1", "mean_reversion_v1"],
        evaluation_path=Path("configs/evaluation.yml"),
        output_path=tmp_path / "wf.csv",
    )

    assert calls == [
        ("momentum_v1", Path("configs/evaluation.yml")),
        ("mean_reversion_v1", Path("configs/evaluation.yml")),
    ]
    assert result.evaluation_mode == "walk_forward"
    assert result.csv_path == tmp_path / "wf.csv"
    assert [entry.run_id for entry in result.leaderboard] == ["wf-momentum_v1", "wf-mean_reversion_v1"]


def test_compare_strategies_uses_latest_matching_registry_run_and_applies_metric_tie_breaks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = tmp_path / "artifacts" / "strategies"
    registry_path = artifact_root / "registry.jsonl"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "run_id": "run-old",
                        "timestamp": "2026-03-19T00:00:00Z",
                        "strategy_name": "momentum_v1",
                        "evaluation_mode": "single",
                        "metrics_summary": {"sharpe_ratio": 1.0, "total_return": 0.09},
                    }
                ),
                json.dumps(
                    {
                        "run_id": "run-new",
                        "timestamp": "2026-03-19T00:05:00Z",
                        "strategy_name": "momentum_v1",
                        "evaluation_mode": "single",
                        "metrics_summary": {"sharpe_ratio": 1.1, "total_return": 0.1},
                    }
                ),
                json.dumps(
                    {
                        "run_id": "run-alpha",
                        "timestamp": "2026-03-19T00:01:00Z",
                        "strategy_name": "alpha_v1",
                        "evaluation_mode": "single",
                        "metrics_summary": {"sharpe_ratio": 1.1, "total_return": 0.2},
                    }
                )
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(compare.experiment_tracker, "ARTIFACTS_ROOT", artifact_root)

    result = compare.compare_strategies(
        ["momentum_v1", "alpha_v1"],
        from_registry=True,
        output_path=tmp_path,
    )

    assert [entry.run_id for entry in result.leaderboard] == ["run-alpha", "run-new"]
    assert [entry.strategy_name for entry in result.leaderboard] == ["alpha_v1", "momentum_v1"]
    assert result.selection_mode == "registry"


def test_compare_strategies_handles_missing_metric_values_and_top_k(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run_strategy_experiment(strategy_name: str, start=None, end=None):
        sharpe_ratio = None if strategy_name == "buy_and_hold_v1" else 0.9
        return compare.StrategyRunResult(
            strategy_name=strategy_name,
            run_id=f"run-{strategy_name}",
            metrics={
                "total_return": 0.05,
                "cumulative_return": 0.05,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": 0.02,
            },
            experiment_dir=tmp_path / f"run-{strategy_name}",
            results_df=None,  # type: ignore[arg-type]
        )

    monkeypatch.setattr(compare, "run_strategy_experiment", fake_run_strategy_experiment)

    result = compare.compare_strategies(
        ["buy_and_hold_v1", "momentum_v1"],
        top_k=1,
        output_path=tmp_path / "custom.csv",
    )

    assert len(result.leaderboard) == 1
    assert result.leaderboard[0].strategy_name == "momentum_v1"


def test_compare_strategies_uses_deterministic_default_output_path_and_stable_file_contents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    run_counts: dict[str, int] = {}

    def fake_run_strategy_experiment(strategy_name: str, start=None, end=None):
        run_counts[strategy_name] = run_counts.get(strategy_name, 0) + 1
        return compare.StrategyRunResult(
            strategy_name=strategy_name,
            run_id=f"ephemeral-{run_counts[strategy_name]}-{strategy_name}",
            metrics={
                "total_return": 0.10 if strategy_name == "momentum_v1" else 0.06,
                "cumulative_return": 0.10 if strategy_name == "momentum_v1" else 0.06,
                "annualized_return": 0.12 if strategy_name == "momentum_v1" else 0.07,
                "sharpe_ratio": 1.5 if strategy_name == "momentum_v1" else 0.9,
                "max_drawdown": 0.04 if strategy_name == "momentum_v1" else 0.03,
                "win_rate": 0.58 if strategy_name == "momentum_v1" else 0.53,
            },
            experiment_dir=tmp_path / f"run-{strategy_name}",
            results_df=None,  # type: ignore[arg-type]
        )

    monkeypatch.setattr(compare, "run_strategy_experiment", fake_run_strategy_experiment)

    first = compare.compare_strategies(
        ["momentum_v1", "mean_reversion_v1", "buy_and_hold_v1"],
    )
    second = compare.compare_strategies(
        ["momentum_v1", "mean_reversion_v1", "buy_and_hold_v1"],
    )

    assert first.csv_path == second.csv_path
    assert first.csv_path == compare.DEFAULT_COMPARISONS_ROOT / (
        compare.build_comparison_id(
            strategies=["momentum_v1", "mean_reversion_v1", "buy_and_hold_v1"],
            metric="sharpe_ratio",
            evaluation_mode="single",
            selection_mode="fresh",
            evaluation_path=None,
            start=None,
            end=None,
            top_k=None,
        )
    ) / "leaderboard.csv"
    assert first.csv_path.read_text(encoding="utf-8") == second.csv_path.read_text(encoding="utf-8")
    assert first.json_path.read_text(encoding="utf-8") == second.json_path.read_text(encoding="utf-8")
    assert first.leaderboard[0].run_id != second.leaderboard[0].run_id


def test_compare_strategies_raises_for_missing_registry_strategy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = tmp_path / "artifacts" / "strategies"
    registry_path = artifact_root / "registry.jsonl"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(compare.experiment_tracker, "ARTIFACTS_ROOT", artifact_root)

    with pytest.raises(ValueError, match="No registry runs found for strategy 'missing_v1'"):
        compare.compare_strategies(["missing_v1"], from_registry=True, output_path=tmp_path)


def test_compare_strategies_rejects_date_filters_with_evaluation() -> None:
    with pytest.raises(ValueError, match="cannot be combined with --evaluation"):
        compare.compare_strategies(
            ["momentum_v1"],
            evaluation_path=Path("configs/evaluation.yml"),
            start="2025-01-01",
        )


def test_compare_strategies_rejects_date_filters_with_registry(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot be combined with --from_registry"):
        compare.compare_strategies(
            ["momentum_v1"],
            from_registry=True,
            start="2025-01-01",
            output_path=tmp_path,
        )


def test_run_cli_prints_leaderboard_summary(monkeypatch, capsys, tmp_path: Path) -> None:
    expected_result = compare.ComparisonResult(
        metric="sharpe_ratio",
        evaluation_mode="single",
        selection_mode="fresh",
        selection_rule="freshly executed run per strategy",
        leaderboard=[
            compare.LeaderboardEntry(
                rank=1,
                strategy_name="momentum_v1",
                run_id="run-momentum_v1",
                evaluation_mode="single",
                selected_metric_name="sharpe_ratio",
                selected_metric_value=1.23,
                cumulative_return=0.12,
                total_return=0.12,
                sharpe_ratio=1.23,
                max_drawdown=0.04,
                annualized_return=None,
                annualized_volatility=None,
                volatility=None,
                win_rate=None,
                hit_rate=None,
                profit_factor=None,
                turnover=None,
                exposure_pct=None,
            )
        ],
        csv_path=tmp_path / "leaderboard.csv",
        json_path=tmp_path / "leaderboard.json",
    )

    monkeypatch.setattr("src.cli.compare_strategies.compare_strategies", lambda *args, **kwargs: expected_result)

    result = run_cli(["--strategies", "momentum_v1"])

    assert result is expected_result
    stdout = capsys.readouterr().out
    assert "metric: sharpe_ratio" in stdout
    assert "strategy" in stdout
    assert "momentum_v1" in stdout
    assert "leaderboard_csv:" in stdout


def test_run_cli_accepts_space_separated_strategy_names(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    expected_result = compare.ComparisonResult(
        metric="sharpe_ratio",
        evaluation_mode="single",
        selection_mode="fresh",
        selection_rule="freshly executed run per strategy",
        leaderboard=[],
        csv_path=tmp_path / "leaderboard.csv",
        json_path=tmp_path / "leaderboard.json",
    )

    def fake_compare_strategies(strategies, **kwargs):
        captured["strategies"] = list(strategies)
        captured["kwargs"] = kwargs
        return expected_result

    monkeypatch.setattr("src.cli.compare_strategies.compare_strategies", fake_compare_strategies)

    result = run_cli(
        ["--strategies", "momentum_v1", "mean_reversion_v1", "buy_and_hold_v1"]
    )

    assert result is expected_result
    assert captured["strategies"] == ["momentum_v1", "mean_reversion_v1", "buy_and_hold_v1"]


def test_run_cli_passes_date_filters_to_compare_strategies(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    expected_result = compare.ComparisonResult(
        metric="sharpe_ratio",
        evaluation_mode="single",
        selection_mode="fresh",
        selection_rule="freshly executed run per strategy",
        leaderboard=[],
        csv_path=tmp_path / "leaderboard.csv",
        json_path=tmp_path / "leaderboard.json",
    )

    def fake_compare_strategies(strategies, **kwargs):
        captured["strategies"] = list(strategies)
        captured["kwargs"] = kwargs
        return expected_result

    monkeypatch.setattr("src.cli.compare_strategies.compare_strategies", fake_compare_strategies)

    result = run_cli(
        [
            "--strategies",
            "momentum_v1",
            "mean_reversion_v1",
            "--start",
            "2025-01-01",
            "--end",
            "2025-03-01",
        ]
    )

    assert result is expected_result
    assert captured["strategies"] == ["momentum_v1", "mean_reversion_v1"]
    assert captured["kwargs"]["start"] == "2025-01-01"
    assert captured["kwargs"]["end"] == "2025-03-01"


def test_compare_strategies_executes_against_curated_daily_features_layout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ts = pd.date_range("2025-01-01", periods=30, freq="D", tz="UTC")
    closes = pd.Series([100.0 + float(index) for index in range(len(ts))], dtype="float64")
    feature_df = pd.DataFrame(
        {
            "symbol": pd.Series(["AAPL"] * len(ts), dtype="string"),
            "ts_utc": ts,
            "timeframe": pd.Series(["1D"] * len(ts), dtype="string"),
            "date": pd.Series(ts.strftime("%Y-%m-%d"), dtype="string"),
            "close": closes,
            "feature_ret_1d": closes.div(closes.shift(1)).sub(1.0),
        }
    )
    dataset_path = tmp_path / "data" / "curated" / "features_daily" / "symbol=AAPL" / "year=2025"
    dataset_path.mkdir(parents=True, exist_ok=True)
    feature_df.to_parquet(dataset_path / "part-0.parquet", index=False)

    run_counter = {"count": 0}

    def fake_save_experiment(strategy_name, results_df, metrics, config):
        run_counter["count"] += 1
        return tmp_path / f"run-{run_counter['count']}-{strategy_name}"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.cli.run_strategy.save_experiment", fake_save_experiment)

    result = compare.compare_strategies(
        ["momentum_v1", "mean_reversion_v1", "buy_and_hold_v1"],
        output_path=tmp_path / "leaderboard.csv",
    )

    assert len(result.leaderboard) == 3
    assert result.csv_path == tmp_path / "leaderboard.csv"
    assert result.json_path == tmp_path / "leaderboard.json"
    assert result.csv_path.exists()
    assert result.json_path.exists()
    assert {entry.strategy_name for entry in result.leaderboard} == {
        "momentum_v1",
        "mean_reversion_v1",
        "buy_and_hold_v1",
    }


def test_compare_strategies_repeated_runs_keep_leaderboard_bytes_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_counts: dict[str, int] = {}

    def fake_run_strategy_experiment(strategy_name: str, start=None, end=None):
        run_counts[strategy_name] = run_counts.get(strategy_name, 0) + 1
        return compare.StrategyRunResult(
            strategy_name=strategy_name,
            run_id=f"ephemeral-{run_counts[strategy_name]}-{strategy_name}",
            metrics={
                "total_return": 0.11 if strategy_name == "momentum_v1" else 0.05,
                "cumulative_return": 0.11 if strategy_name == "momentum_v1" else 0.05,
                "annualized_return": 0.18 if strategy_name == "momentum_v1" else 0.08,
                "annualized_volatility": 0.21 if strategy_name == "momentum_v1" else 0.12,
                "volatility": 0.01 if strategy_name == "momentum_v1" else 0.02,
                "sharpe_ratio": 1.4 if strategy_name == "momentum_v1" else 0.8,
                "max_drawdown": 0.07 if strategy_name == "momentum_v1" else 0.03,
                "win_rate": 0.62 if strategy_name == "momentum_v1" else 0.51,
                "hit_rate": 0.6 if strategy_name == "momentum_v1" else 0.5,
                "profit_factor": 1.5 if strategy_name == "momentum_v1" else 1.1,
                "turnover": 0.2 if strategy_name == "momentum_v1" else 0.1,
                "exposure_pct": 80.0 if strategy_name == "momentum_v1" else 75.0,
            },
            experiment_dir=tmp_path / f"run-{strategy_name}",
            results_df=None,  # type: ignore[arg-type]
        )

    monkeypatch.setattr(compare, "run_strategy_experiment", fake_run_strategy_experiment)

    first = compare.compare_strategies(
        ["momentum_v1", "mean_reversion_v1", "buy_and_hold_v1"],
        output_path=tmp_path / "leaderboard.csv",
    )
    first_csv = first.csv_path.read_bytes()
    first_json = first.json_path.read_bytes()

    second = compare.compare_strategies(
        ["momentum_v1", "mean_reversion_v1", "buy_and_hold_v1"],
        output_path=tmp_path / "leaderboard.csv",
    )

    assert [entry.strategy_name for entry in first.leaderboard] == [
        entry.strategy_name for entry in second.leaderboard
    ]
    assert [entry.selected_metric_value for entry in first.leaderboard] == [
        entry.selected_metric_value for entry in second.leaderboard
    ]
    assert first_csv == second.csv_path.read_bytes()
    assert first_json == second.json_path.read_bytes()
