from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import pandas.testing as pdt
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cli.run_alpha import run_cli as run_alpha_cli
from src.cli.run_strategy import run_strategy_experiment
from src.cli.run_portfolio import run_cli as run_portfolio_cli
from src.research import compare, experiment_tracker
from src.research.alpha_eval import alpha_evaluation_registry_path
from src.research.registry import default_registry_path, load_registry
from src.research.strategy_base import BaseStrategy
from src.research.walk_forward import run_walk_forward_experiment


class SignStrategy(BaseStrategy):
    name = "sign_v1"
    dataset = "features_daily"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        return df["feature_ret_1d"].apply(
            lambda value: 1 if value > 0 else (-1 if value < 0 else 0)
        ).rename("signal")


def _write_daily_features_dataset(root: Path, symbol: str = "AAPL", periods: int = 80) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=periods, freq="D", tz="UTC")
    closes: list[float] = []
    close = 100.0
    for index in range(periods):
        close += 0.45 + ((index % 7) - 3) * 0.18
        closes.append(round(close, 6))

    close_series = pd.Series(closes, dtype="float64")
    feature_df = pd.DataFrame(
        {
            "symbol": pd.Series([symbol] * periods, dtype="string"),
            "ts_utc": ts,
            "timeframe": pd.Series(["1D"] * periods, dtype="string"),
            "date": pd.Series(ts.strftime("%Y-%m-%d"), dtype="string"),
            "close": close_series,
            "feature_ret_1d": close_series.div(close_series.shift(1)).sub(1.0),
        }
    )
    dataset_path = root / "data" / "curated" / "features_daily" / f"symbol={symbol}" / "year=2025"
    dataset_path.mkdir(parents=True, exist_ok=True)
    feature_df.to_parquet(dataset_path / "part-0.parquet", index=False)
    return feature_df


def _write_alpha_features_daily_dataset(root: Path) -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01", periods=8, freq="D", tz="UTC")
    rows: list[dict[str, object]] = []
    symbols = ["AAA", "BBB", "CCC"]
    for symbol_index, symbol in enumerate(symbols):
        base = float(symbol_index + 1)
        close = 100.0 + base * 10.0
        for ts_index, ts_utc in enumerate(timestamps):
            close += 1.2 + base * 0.5 + ts_index * 0.15
            rows.append(
                {
                    "symbol": symbol,
                    "ts_utc": ts_utc,
                    "timeframe": "1D",
                    "date": ts_utc.strftime("%Y-%m-%d"),
                    "feature_ret_1d": base * 0.03 + ts_index * 0.001,
                    "feature_ret_5d": base * 0.05 + ts_index * 0.002,
                    "feature_ret_20d": base * 0.07 + ts_index * 0.003,
                    "target_ret_1d": base * 0.01 + ts_index * 0.001,
                    "target_ret_5d": base * 0.02 + ts_index * 0.0015,
                    "close": close,
                }
            )

    frame = pd.DataFrame(rows).sort_values(["symbol", "ts_utc"], kind="stable").reset_index(drop=True)
    for column in ("symbol", "timeframe", "date"):
        frame[column] = frame[column].astype("string")
    for column in (
        "feature_ret_1d",
        "feature_ret_5d",
        "feature_ret_20d",
        "target_ret_1d",
        "target_ret_5d",
        "close",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("float64")

    for symbol in symbols:
        symbol_frame = frame.loc[frame["symbol"].eq(symbol)].copy(deep=True)
        dataset_path = root / "data" / "curated" / "features_daily" / f"symbol={symbol}" / "year=2025"
        dataset_path.mkdir(parents=True, exist_ok=True)
        symbol_frame.to_parquet(dataset_path / "part-0.parquet", index=False)
    return frame


def _write_evaluation_config(path: Path, payload: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump({"evaluation": payload}, sort_keys=False), encoding="utf-8")


def _feature_frame() -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", periods=7, freq="D")
    return pd.DataFrame(
        {
            "symbol": ["SPY"] * len(dates),
            "ts_utc": pd.to_datetime(dates, utc=True),
            "timeframe": ["1d"] * len(dates),
            "date": dates.strftime("%Y-%m-%d"),
            "feature_ret_1d": [0.01, -0.02, 0.03, -0.01, 0.02, 0.01, -0.03],
        }
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy().reset_index(drop=True)
    for column in frame.columns:
        if column == "ts_utc":
            timestamps = pd.to_datetime(frame[column], utc=True, errors="coerce")
            frame[column] = timestamps.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        elif column == "date":
            dates = pd.to_datetime(frame[column], utc=True, errors="coerce")
            frame[column] = dates.dt.strftime("%Y-%m-%d")
    return frame


def _assert_frame_values_stable(left: pd.DataFrame, right: pd.DataFrame) -> None:
    pdt.assert_frame_equal(
        _normalize_frame(left),
        _normalize_frame(right),
        check_dtype=False,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def _assert_structured_values_stable(left: Any, right: Any, *, path: str = "root") -> None:
    if isinstance(left, pd.DataFrame) and isinstance(right, pd.DataFrame):
        _assert_frame_values_stable(left, right)
        return

    if isinstance(left, dict) and isinstance(right, dict):
        assert set(left) == set(right), f"{path} keys differ: {set(left) ^ set(right)}"
        for key in sorted(left):
            _assert_structured_values_stable(left[key], right[key], path=f"{path}.{key}")
        return

    if isinstance(left, list) and isinstance(right, list):
        assert len(left) == len(right), f"{path} length differs"
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            _assert_structured_values_stable(left_item, right_item, path=f"{path}[{index}]")
        return

    if isinstance(left, bool) or isinstance(right, bool):
        assert left is right, f"{path} differs: {left!r} != {right!r}"
        return

    if isinstance(left, int | float) and isinstance(right, int | float):
        if isinstance(left, float) and math.isnan(left):
            assert isinstance(right, float) and math.isnan(right), f"{path} differs: {left!r} != {right!r}"
        else:
            assert left == pytest.approx(right, rel=1e-12, abs=1e-12), f"{path} differs"
        return

    assert left == right, f"{path} differs: {left!r} != {right!r}"


def _artifact_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _single_run_artifacts(run_dir: Path) -> dict[str, Any]:
    return {
        "metrics": _load_json(run_dir / "metrics.json"),
        "equity_curve": pd.read_csv(run_dir / "equity_curve.csv"),
        "signal_diagnostics": _load_json(run_dir / "signal_diagnostics.json"),
        "qa_summary": _load_json(run_dir / "qa_summary.json"),
        "signals": pd.read_parquet(run_dir / "signals.parquet"),
    }


def _write_portfolio_strategy_run(
    root: Path,
    *,
    run_id: str,
    strategy_name: str,
    rows: list[dict[str, object]],
) -> Path:
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(
        json.dumps({"strategy_name": strategy_name}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": run_id, "strategy_name": strategy_name}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    frame = pd.DataFrame(rows)
    frame["equity"] = 1.0
    frame["symbol"] = "SPY"
    frame["signal"] = 1.0
    frame["position"] = 1.0
    frame.to_csv(run_dir / "equity_curve.csv", index=False)
    return run_dir


@pytest.mark.parametrize("strategy_name", ["buy_and_hold_v1", "momentum_v1"])
def test_single_strategy_reruns_produce_stable_values_and_registry_entries(
    strategy_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_root = tmp_path / "artifacts" / "strategies"
    _write_daily_features_dataset(tmp_path)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(experiment_tracker, "ARTIFACTS_ROOT", artifact_root)

    first = run_strategy_experiment(strategy_name)
    second = run_strategy_experiment(strategy_name)

    assert first.run_id == second.run_id
    _assert_structured_values_stable(first.metrics, second.metrics, path="result.metrics")
    _assert_frame_values_stable(
        first.results_df.loc[:, ["signal", "strategy_return", "equity_curve"]],
        second.results_df.loc[:, ["signal", "strategy_return", "equity_curve"]],
    )
    _assert_structured_values_stable(
        first.signal_diagnostics,
        second.signal_diagnostics,
        path="result.signal_diagnostics",
    )
    _assert_structured_values_stable(first.qa_summary, second.qa_summary, path="result.qa_summary")

    first_artifacts = _single_run_artifacts(first.experiment_dir)
    second_artifacts = _single_run_artifacts(second.experiment_dir)
    _assert_structured_values_stable(first_artifacts, second_artifacts, path="artifacts")

    registry_entries = load_registry(default_registry_path(artifact_root))
    assert len(registry_entries) == 1
    assert registry_entries[0]["run_id"] == first.run_id
    assert registry_entries[0]["strategy_name"] == strategy_name
    assert registry_entries[0]["artifact_path"] == first.experiment_dir.as_posix()


def test_registry_latest_lookup_and_comparison_outputs_remain_stable_for_identical_reruns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_root = tmp_path / "artifacts" / "strategies"
    _write_daily_features_dataset(tmp_path)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(experiment_tracker, "ARTIFACTS_ROOT", artifact_root)

    first_run = run_strategy_experiment("momentum_v1")
    second_run = run_strategy_experiment("momentum_v1")
    buy_and_hold_run = run_strategy_experiment("buy_and_hold_v1")

    registry_entries = load_registry(default_registry_path(artifact_root))
    assert len(registry_entries) == 2
    assert {entry["run_id"] for entry in registry_entries} == {
        first_run.run_id,
        buy_and_hold_run.run_id,
    }

    first_comparison = compare.compare_strategies(
        ["momentum_v1", "buy_and_hold_v1"],
        from_registry=True,
        output_path=tmp_path / "leaderboard.csv",
    )
    second_comparison = compare.compare_strategies(
        ["momentum_v1", "buy_and_hold_v1"],
        from_registry=True,
        output_path=tmp_path / "leaderboard.csv",
    )

    assert [entry.strategy_name for entry in first_comparison.leaderboard] == [
        entry.strategy_name for entry in second_comparison.leaderboard
    ]
    assert [entry.run_id for entry in first_comparison.leaderboard] == [
        entry.run_id for entry in second_comparison.leaderboard
    ]
    assert {entry.run_id for entry in first_comparison.leaderboard} == {
        first_run.run_id,
        buy_and_hold_run.run_id,
    }

    _assert_frame_values_stable(
        pd.read_csv(first_comparison.csv_path),
        pd.read_csv(second_comparison.csv_path),
    )
    _assert_structured_values_stable(
        _load_json(first_comparison.json_path),
        _load_json(second_comparison.json_path),
        path="comparison.json",
    )


def test_fresh_comparison_leaderboard_is_stable_across_repeated_identical_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_root = tmp_path / "artifacts" / "strategies"
    _write_daily_features_dataset(tmp_path)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(experiment_tracker, "ARTIFACTS_ROOT", artifact_root)

    first = compare.compare_strategies(
        ["momentum_v1", "buy_and_hold_v1"],
        output_path=tmp_path / "fresh_leaderboard.csv",
    )
    second = compare.compare_strategies(
        ["momentum_v1", "buy_and_hold_v1"],
        output_path=tmp_path / "fresh_leaderboard.csv",
    )

    assert [entry.strategy_name for entry in first.leaderboard] == [
        entry.strategy_name for entry in second.leaderboard
    ]
    assert [entry.selected_metric_value for entry in first.leaderboard] == pytest.approx(
        [entry.selected_metric_value for entry in second.leaderboard]
    )
    assert [entry.run_id for entry in first.leaderboard] == [entry.run_id for entry in second.leaderboard]

    _assert_frame_values_stable(pd.read_csv(first.csv_path), pd.read_csv(second.csv_path))
    _assert_structured_values_stable(_load_json(first.json_path), _load_json(second.json_path), path="leaderboard")


def test_comparison_tie_breaking_is_deterministic_for_equal_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run_strategy_experiment(strategy_name: str, start=None, end=None):
        return compare.StrategyRunResult(
            strategy_name=strategy_name,
            run_id=f"run-{strategy_name}",
            metrics={
                "cumulative_return": 0.05,
                "total_return": 0.05,
                "sharpe_ratio": 1.0,
                "max_drawdown": 0.02,
            },
            experiment_dir=tmp_path / f"run-{strategy_name}",
            results_df=pd.DataFrame(),
        )

    monkeypatch.setattr(compare, "run_strategy_experiment", fake_run_strategy_experiment)

    result = compare.compare_strategies(
        ["momentum_v1", "buy_and_hold_v1", "alpha_v1"],
        output_path=tmp_path / "ties.csv",
    )

    assert [entry.strategy_name for entry in result.leaderboard] == [
        "alpha_v1",
        "buy_and_hold_v1",
        "momentum_v1",
    ]


def test_walk_forward_reruns_produce_stable_aggregate_and_split_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_root = tmp_path / "artifacts" / "strategies"
    config_path = tmp_path / "evaluation.yml"
    _write_evaluation_config(
        config_path,
        {
            "mode": "rolling",
            "timeframe": "1d",
            "start": "2022-01-01",
            "end": "2022-01-07",
            "train_window": "2D",
            "test_window": "1D",
            "step": "1D",
        },
    )

    monkeypatch.setattr(experiment_tracker, "ARTIFACTS_ROOT", artifact_root)
    monkeypatch.setattr("src.research.walk_forward.load_features", lambda dataset, start=None, end=None: _feature_frame())

    first = run_walk_forward_experiment(
        "sign_v1",
        SignStrategy(),
        evaluation_path=config_path,
        strategy_config={"dataset": "features_daily", "parameters": {"lookback": 2}},
    )
    second = run_walk_forward_experiment(
        "sign_v1",
        SignStrategy(),
        evaluation_path=config_path,
        strategy_config={"dataset": "features_daily", "parameters": {"lookback": 2}},
    )

    assert first.run_id == second.run_id
    _assert_structured_values_stable(first.metrics, second.metrics, path="walk_forward.metrics")
    _assert_structured_values_stable(
        first.aggregate_summary,
        second.aggregate_summary,
        path="walk_forward.aggregate_summary",
    )
    assert [split.split_id for split in first.splits] == [split.split_id for split in second.splits]
    assert [split.mode for split in first.splits] == [split.mode for split in second.splits]
    _assert_frame_values_stable(
        pd.read_csv(first.experiment_dir / "metrics_by_split.csv"),
        pd.read_csv(second.experiment_dir / "metrics_by_split.csv"),
    )
    _assert_frame_values_stable(
        pd.read_csv(first.experiment_dir / "equity_curve.csv"),
        pd.read_csv(second.experiment_dir / "equity_curve.csv"),
    )

    for first_split, second_split in zip(first.splits, second.splits, strict=True):
        _assert_structured_values_stable(first_split.metrics, second_split.metrics, path=f"split.{first_split.split_id}")
        _assert_frame_values_stable(first_split.results_df, second_split.results_df)

    registry_entries = load_registry(default_registry_path(artifact_root))
    assert len(registry_entries) == 1
    assert registry_entries[0]["run_id"] == first.run_id
    assert registry_entries[0]["split_count"] == len(first.splits)


def test_portfolio_reruns_produce_stable_outputs_artifacts_and_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy_root = tmp_path / "artifacts" / "strategies"
    portfolio_root = tmp_path / "artifacts" / "portfolios"
    _write_portfolio_strategy_run(
        strategy_root,
        run_id="run-alpha",
        strategy_name="alpha_v1",
        rows=[
            {"ts_utc": "2025-01-01T00:00:00Z", "strategy_return": 0.01},
            {"ts_utc": "2025-01-02T00:00:00Z", "strategy_return": 0.02},
        ],
    )
    _write_portfolio_strategy_run(
        strategy_root,
        run_id="run-beta",
        strategy_name="beta_v1",
        rows=[
            {"ts_utc": "2025-01-01T00:00:00Z", "strategy_return": 0.03},
            {"ts_utc": "2025-01-02T00:00:00Z", "strategy_return": -0.01},
        ],
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.cli.run_portfolio.experiment_tracker.ARTIFACTS_ROOT", strategy_root)
    monkeypatch.setattr("src.cli.run_portfolio.DEFAULT_PORTFOLIO_ARTIFACTS_ROOT", portfolio_root)

    first = run_portfolio_cli(
        [
            "--portfolio-name",
            "core_portfolio",
            "--run-ids",
            "run-alpha",
            "run-beta",
            "--timeframe",
            "1D",
        ]
    )
    second = run_portfolio_cli(
        [
            "--portfolio-name",
            "core_portfolio",
            "--run-ids",
            "run-alpha",
            "run-beta",
            "--timeframe",
            "1D",
        ]
    )

    assert first.run_id == second.run_id
    _assert_structured_values_stable(first.metrics, second.metrics, path="portfolio.metrics")
    _assert_frame_values_stable(first.portfolio_output, second.portfolio_output)
    assert _artifact_bytes(first.experiment_dir) == _artifact_bytes(second.experiment_dir)

    registry_entries = load_registry(default_registry_path(portfolio_root))
    portfolio_entries = [entry for entry in registry_entries if entry.get("run_id") == first.run_id]
    assert len(portfolio_entries) == 1
    _assert_structured_values_stable(portfolio_entries[0]["metrics"], first.metrics, path="portfolio.registry.metrics")


def test_builtin_alpha_full_reruns_produce_stable_artifacts_and_registry_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_alpha_features_daily_dataset(tmp_path)
    monkeypatch.chdir(tmp_path)

    argv = [
        "--alpha-name",
        "cs_linear_ret_1d",
        "--signal-policy",
        "top_bottom_quantile",
        "--signal-quantile",
        "0.34",
        "--start",
        "2025-01-01",
        "--end",
        "2025-01-09",
    ]
    first = run_alpha_cli(argv)
    second = run_alpha_cli(argv)

    assert first.run_id == second.run_id
    assert first.scaffold_path is not None
    assert second.scaffold_path is not None
    assert _artifact_bytes(first.artifact_dir) == _artifact_bytes(second.artifact_dir)

    _assert_frame_values_stable(
        pd.read_parquet(first.artifact_dir / "predictions.parquet"),
        pd.read_parquet(second.artifact_dir / "predictions.parquet"),
    )
    _assert_frame_values_stable(
        pd.read_parquet(first.artifact_dir / "signals.parquet"),
        pd.read_parquet(second.artifact_dir / "signals.parquet"),
    )
    _assert_frame_values_stable(
        pd.read_csv(first.artifact_dir / "sleeve_returns.csv"),
        pd.read_csv(second.artifact_dir / "sleeve_returns.csv"),
    )
    _assert_structured_values_stable(
        _load_json(first.scaffold_path),
        _load_json(second.scaffold_path),
        path="alpha.scaffold",
    )

    registry_entries = load_registry(alpha_evaluation_registry_path(tmp_path / "artifacts" / "alpha"))
    assert [entry["run_id"] for entry in registry_entries] == [first.run_id]
    assert registry_entries[0]["alpha_name"] == "cs_linear_ret_1d"
    assert registry_entries[0]["artifact_path"] == first.artifact_dir.as_posix()
