from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cli.run_strategy import (
    StrategyRunResult,
    WalkForwardRunResult,
    get_strategy_config,
    parse_args,
    run_cli,
    run_strategy_experiment,
)
from src.research.strategies import build_strategy


def _results_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal": [0, 1],
            "strategy_return": [0.0, 0.02],
            "equity_curve": [1.0, 1.02],
        }
    )


def _write_daily_features_dataset(root: Path, symbol: str = "AAPL", periods: int = 30) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=periods, freq="D", tz="UTC")
    closes = pd.Series([100.0 + float(index) for index in range(periods)], dtype="float64")
    feature_df = pd.DataFrame(
        {
            "symbol": pd.Series([symbol] * periods, dtype="string"),
            "ts_utc": ts,
            "timeframe": pd.Series(["1D"] * periods, dtype="string"),
            "date": pd.Series(ts.strftime("%Y-%m-%d"), dtype="string"),
            "close": closes,
            "feature_ret_1d": closes.div(closes.shift(1)).sub(1.0),
        }
    )
    dataset_path = root / "data" / "curated" / "features_daily" / f"symbol={symbol}" / "year=2025"
    dataset_path.mkdir(parents=True, exist_ok=True)
    feature_df.to_parquet(dataset_path / "part-0.parquet", index=False)
    return feature_df


def test_parse_args_accepts_strategy_and_date_filters() -> None:
    args = parse_args(["--strategy", "momentum_v1", "--start", "2025-01-01", "--end", "2025-02-01"])

    assert args.strategy == "momentum_v1"
    assert args.start == "2025-01-01"
    assert args.end == "2025-02-01"
    assert args.evaluation is None


def test_parse_args_accepts_evaluation_flag_without_path() -> None:
    args = parse_args(["--strategy", "momentum_v1", "--evaluation"])

    assert args.strategy == "momentum_v1"
    assert args.evaluation.endswith("configs\\evaluation.yml")


def test_get_strategy_config_raises_for_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="Unknown strategy 'missing_strategy'"):
        get_strategy_config("missing_strategy", {"momentum_v1": {"dataset": "features_daily"}})


def test_run_cli_invokes_research_pipeline_components(monkeypatch, capsys) -> None:
    feature_df = pd.DataFrame({"feature_ret_1d": [0.01, 0.02]})
    strategy_config = {
        "momentum_v1": {
            "dataset": "features_daily",
            "parameters": {"lookback_short": 5, "lookback_long": 20},
        }
    }
    calls: dict[str, object] = {}

    def fake_load_strategies_config(path=None):
        calls["config_path"] = path
        return strategy_config

    def fake_load_features(dataset, start=None, end=None):
        calls["load_features"] = {"dataset": dataset, "start": start, "end": end}
        return feature_df

    def fake_generate_signals(dataset, strategy):
        calls["generate_signals"] = {"dataset": dataset, "strategy_name": strategy.name}
        return dataset.assign(signal=[1, 0])

    def fake_run_backtest(signal_frame):
        calls["run_backtest"] = signal_frame.copy()
        return _results_frame()

    def fake_compute_metrics(results_df):
        calls["compute_metrics"] = results_df.copy()
        return {"cumulative_return": 0.02, "sharpe_ratio": 1.25}

    def fake_save_experiment(strategy_name, results_df, metrics, config):
        calls["save_experiment"] = {
            "strategy_name": strategy_name,
            "results_df": results_df.copy(),
            "metrics": metrics,
            "config": config,
        }
        return Path("artifacts/strategies/run-123")

    monkeypatch.setattr("src.cli.run_strategy.load_strategies_config", fake_load_strategies_config)
    monkeypatch.setattr("src.cli.run_strategy.load_features", fake_load_features)
    monkeypatch.setattr("src.cli.run_strategy.generate_signals", fake_generate_signals)
    monkeypatch.setattr("src.cli.run_strategy.run_backtest", fake_run_backtest)
    monkeypatch.setattr("src.cli.run_strategy.compute_metrics", fake_compute_metrics)
    monkeypatch.setattr("src.cli.run_strategy.save_experiment", fake_save_experiment)

    result = run_cli(["--strategy", "momentum_v1", "--start", "2025-01-01", "--end", "2025-02-01"])

    assert isinstance(result, StrategyRunResult)
    assert result.strategy_name == "momentum_v1"
    assert result.run_id == "run-123"
    assert calls["load_features"] == {
        "dataset": "features_daily",
        "start": "2025-01-01",
        "end": "2025-02-01",
    }
    assert calls["generate_signals"]["dataset"].equals(feature_df)
    assert calls["generate_signals"]["strategy_name"] == "momentum_v1"
    assert calls["run_backtest"]["signal"].tolist() == [1, 0]
    assert calls["save_experiment"]["strategy_name"] == "momentum_v1"
    assert calls["save_experiment"]["metrics"] == {
        "cumulative_return": 0.02,
        "sharpe_ratio": 1.25,
    }
    assert calls["save_experiment"]["config"] == {
        "strategy_name": "momentum_v1",
        "dataset": "features_daily",
        "parameters": {"lookback_short": 5, "lookback_long": 20},
        "start": "2025-01-01",
        "end": "2025-02-01",
    }

    stdout = capsys.readouterr().out
    assert "strategy: momentum_v1" in stdout
    assert "run_id: run-123" in stdout
    assert "cumulative_return: 0.020000" in stdout
    assert "sharpe_ratio: 1.250000" in stdout


def test_run_cli_invokes_walk_forward_mode(monkeypatch, capsys) -> None:
    strategy_config = {
        "momentum_v1": {
            "dataset": "features_daily",
            "parameters": {"lookback_short": 5, "lookback_long": 20},
        }
    }
    calls: dict[str, object] = {}
    walk_forward_result = WalkForwardRunResult(
        strategy_name="momentum_v1",
        run_id="wf-123",
        experiment_dir=Path("artifacts/strategies/wf-123"),
        metrics={
            "cumulative_return": 0.03,
            "sharpe_ratio": 1.5,
            "volatility": 0.1,
            "max_drawdown": 0.02,
            "win_rate": 0.5,
        },
        aggregate_summary={"split_count": 2, "cumulative_return": 0.03, "sharpe_ratio": 1.5},
        splits=[],
    )

    monkeypatch.setattr("src.cli.run_strategy.load_strategies_config", lambda path=None: strategy_config)

    def fake_run_walk_forward_experiment(strategy_name, strategy, evaluation_path, strategy_config):
        calls["walk_forward"] = {
            "strategy_name": strategy_name,
            "strategy_name_attr": strategy.name,
            "dataset": strategy.dataset,
            "evaluation_path": evaluation_path,
            "strategy_config": strategy_config,
        }
        return walk_forward_result

    monkeypatch.setattr("src.cli.run_strategy.run_walk_forward_experiment", fake_run_walk_forward_experiment)

    result = run_cli(["--strategy", "momentum_v1", "--evaluation", "configs/evaluation.yml"])

    assert result is walk_forward_result
    assert calls["walk_forward"]["strategy_name"] == "momentum_v1"
    assert calls["walk_forward"]["strategy_name_attr"] == "momentum_v1"
    assert calls["walk_forward"]["dataset"] == "features_daily"
    assert calls["walk_forward"]["evaluation_path"] == Path("configs/evaluation.yml")
    assert calls["walk_forward"]["strategy_config"] == strategy_config["momentum_v1"]

    stdout = capsys.readouterr().out
    assert "strategy: momentum_v1" in stdout
    assert "run_id: wf-123" in stdout
    assert "split_count: 2" in stdout


def test_run_cli_rejects_date_filters_in_evaluation_mode() -> None:
    with pytest.raises(ValueError, match="cannot be combined with --evaluation"):
        run_cli(["--strategy", "momentum_v1", "--evaluation", "--start", "2025-01-01"])


def test_run_strategy_experiment_loads_curated_daily_features_and_supports_mean_reversion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    feature_df = _write_daily_features_dataset(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.cli.run_strategy.save_experiment", lambda *args, **kwargs: tmp_path / "run-mean-reversion")

    result = run_strategy_experiment("mean_reversion_v1")

    assert isinstance(result, StrategyRunResult)
    assert result.strategy_name == "mean_reversion_v1"
    assert result.run_id == "run-mean-reversion"
    assert list(result.results_df.columns[-3:]) == ["signal", "strategy_return", "equity_curve"]
    assert result.results_df["close"].tolist() == pytest.approx(feature_df["close"].tolist())
    assert result.results_df["feature_ret_1d"].notna().sum() == len(feature_df) - 1


def test_build_strategy_supports_buy_and_hold_baseline() -> None:
    strategy = build_strategy("buy_and_hold_v1", {"dataset": "features_1m", "parameters": {}})

    assert strategy.name == "buy_and_hold_v1"
    assert strategy.dataset == "features_1m"


def test_run_strategy_experiment_is_reproducible_for_repeated_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_daily_features_dataset(tmp_path)
    monkeypatch.chdir(tmp_path)

    first = run_strategy_experiment("momentum_v1")
    first_metrics = dict(first.metrics)
    first_artifacts = {
        path.relative_to(first.experiment_dir).as_posix(): path.read_bytes()
        for path in sorted(first.experiment_dir.rglob("*"))
        if path.is_file()
    }

    second = run_strategy_experiment("momentum_v1")
    second_artifacts = {
        path.relative_to(second.experiment_dir).as_posix(): path.read_bytes()
        for path in sorted(second.experiment_dir.rglob("*"))
        if path.is_file()
    }

    assert first.run_id == second.run_id
    assert first_metrics == second.metrics
    assert first_artifacts == second_artifacts
