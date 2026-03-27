from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.research import experiment_tracker
from src.research.experiment_tracker import save_experiment
from src.research.registry import (
    RegistryError,
    append_registry_entry,
    default_registry_path,
    filter_by_allocator_name,
    filter_by_metric_threshold,
    filter_by_portfolio_name,
    filter_by_run_type,
    filter_by_strategy_name,
    generate_portfolio_run_id,
    load_registry,
    register_portfolio_run,
    serialize_canonical_json,
)
from src.research.strategy_base import BaseStrategy
from src.research.walk_forward import run_walk_forward_experiment


class SignStrategy(BaseStrategy):
    name = "sign_v1"
    dataset = "features_daily"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        return df["feature_ret_1d"].apply(
            lambda value: 1 if value > 0 else (-1 if value < 0 else 0)
        ).rename("signal")


def _single_run_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-03", "2025-01-06"],
            "timeframe": ["1D", "1D", "1D"],
            "feature_alpha": [0.15, -0.10, 0.25],
            "signal": [1, 0, -1],
            "strategy_return": [0.0, 0.02, -0.01],
            "equity_curve": [1.0, 1.02, 1.0098],
        }
    )


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


def _write_evaluation_config(path: Path, payload: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump({"evaluation": payload}, sort_keys=False), encoding="utf-8")


def _expected_metric_keys() -> set[str]:
    return {
        "cumulative_return",
        "total_return",
        "volatility",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "win_rate",
        "hit_rate",
        "profit_factor",
        "turnover",
        "total_turnover",
        "average_turnover",
        "trade_count",
        "rebalance_count",
        "percent_periods_traded",
        "average_trade_size",
        "total_transaction_cost",
        "total_slippage_cost",
        "total_execution_friction",
        "average_execution_friction_per_trade",
        "exposure_pct",
        "sanity_issue_count",
        "sanity_warning_count",
        "flagged_split_count",
    }


def test_save_experiment_writes_registry_entry_with_stable_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = tmp_path / "artifacts" / "strategies"
    registry_path = default_registry_path(artifact_root)
    results_df = _single_run_results()
    metrics = {
        "cumulative_return": 0.0098,
        "total_return": 0.0098,
        "volatility": 0.0153,
        "annualized_return": 1.2742,
        "annualized_volatility": 0.2429,
        "sharpe_ratio": 1.01,
        "max_drawdown": 0.01,
        "win_rate": 0.3333,
        "hit_rate": 0.5,
        "profit_factor": 2.0,
        "turnover": 0.6667,
        "total_turnover": 2.0,
        "average_turnover": 0.6667,
        "trade_count": 2.0,
        "rebalance_count": 2.0,
        "percent_periods_traded": 66.6667,
        "average_trade_size": 1.0,
        "total_transaction_cost": 0.0,
        "total_slippage_cost": 0.0,
        "total_execution_friction": 0.0,
        "average_execution_friction_per_trade": 0.0,
        "exposure_pct": 66.6667,
    }
    config = {
        "strategy_name": "mean_reversion",
        "dataset": "features_daily",
        "parameters": {"threshold": 0.75, "lookback": 20},
        "start": "2025-01-02",
        "end": "2025-01-07",
    }

    monkeypatch.setattr(experiment_tracker, "ARTIFACTS_ROOT", artifact_root)
    experiment_dir = save_experiment("mean_reversion", results_df, metrics, config)

    entries = load_registry(registry_path)

    assert len(entries) == 1
    entry = entries[0]
    assert entry["run_id"] == experiment_dir.name
    assert entry["timestamp"].endswith("Z")
    assert entry["strategy_name"] == "mean_reversion"
    assert entry["dataset"] == "features_daily"
    assert entry["strategy_params"] == {"lookback": 20, "threshold": 0.75}
    assert entry["evaluation_mode"] == "single"
    assert entry["evaluation_config"] is None
    assert entry["data_range"] == {"start": "2025-01-02", "end": "2025-01-06"}
    assert entry["timeframe"] == "1D"
    assert entry["metrics_summary"] == metrics
    assert entry["artifact_path"] == experiment_dir.as_posix()
    assert entry["split_count"] is None
    assert entry["evaluation_config_path"] is None


def test_save_experiment_appends_one_registry_entry_per_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = tmp_path / "artifacts" / "strategies"
    results_df = _single_run_results()
    metrics = {"cumulative_return": 0.0098, "total_return": 0.0098, "sharpe_ratio": 1.0}
    config = {"dataset": "features_daily", "parameters": {"lookback": 10}}

    monkeypatch.setattr(experiment_tracker, "ARTIFACTS_ROOT", artifact_root)

    first_dir = save_experiment("alpha_v1", results_df, metrics, config)
    second_dir = save_experiment("beta_v1", results_df, metrics, config)

    entries = load_registry(default_registry_path(artifact_root))

    assert [entry["run_id"] for entry in entries] == [first_dir.name, second_dir.name]
    assert [entry["strategy_name"] for entry in entries] == ["alpha_v1", "beta_v1"]


def test_append_registry_entry_rejects_duplicate_run_ids(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.jsonl"
    entry = {
        "run_id": "run-123",
        "timestamp": "2026-03-19T00:00:00Z",
        "strategy_name": "alpha_v1",
        "strategy_params": {},
        "evaluation_mode": "single",
        "evaluation_config": None,
        "data_range": {"start": "2025-01-01", "end": "2025-01-02"},
        "timeframe": "1D",
        "metrics_summary": {"cumulative_return": 0.1},
        "artifact_path": "artifacts/strategies/run-123",
        "split_count": None,
        "dataset": "features_daily",
        "evaluation_config_path": None,
    }

    append_registry_entry(registry_path, entry)

    with pytest.raises(RegistryError, match="already contains run_id 'run-123'"):
        append_registry_entry(registry_path, entry)


def test_serialize_canonical_json_is_deterministic() -> None:
    left = {"parameters": {"z": 1, "a": 2}, "evaluation": {"step": "1D", "mode": "rolling"}}
    right = {"evaluation": {"mode": "rolling", "step": "1D"}, "parameters": {"a": 2, "z": 1}}

    assert serialize_canonical_json(left) == serialize_canonical_json(right)
    assert serialize_canonical_json(left) == (
        '{"evaluation":{"mode":"rolling","step":"1D"},"parameters":{"a":2,"z":1}}'
    )


def test_walk_forward_run_writes_registry_entry_with_aggregate_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    monkeypatch.setattr(
        "src.research.walk_forward.load_features",
        lambda dataset, start=None, end=None: _feature_frame(),
    )

    result = run_walk_forward_experiment(
        "sign_v1",
        SignStrategy(),
        evaluation_path=config_path,
        strategy_config={"dataset": "features_daily", "parameters": {"lookback": 2}},
    )

    entries = load_registry(default_registry_path(artifact_root))

    assert len(entries) == 1
    entry = entries[0]
    assert entry["run_id"] == result.run_id
    assert entry["strategy_name"] == "sign_v1"
    assert entry["evaluation_mode"] == "walk_forward"
    assert entry["evaluation_config"] == {
        "end": "2022-01-07",
        "mode": "rolling",
        "start": "2022-01-01",
        "step": "1D",
        "test_window": "1D",
        "timeframe": "1d",
        "train_window": "2D",
    }
    assert entry["data_range"] == {"start": "2022-01-01", "end": "2022-01-07"}
    assert entry["timeframe"] == "1D"
    assert entry["split_count"] == 4
    assert entry["artifact_path"] == result.experiment_dir.as_posix()
    assert entry["evaluation_config_path"] == str(config_path)
    assert set(entry["metrics_summary"]) == _expected_metric_keys()
    assert "aggregation_method" not in entry["metrics_summary"]
    assert "split_count" not in entry["metrics_summary"]


def test_registry_load_and_filter_utilities_support_lightweight_queries(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.jsonl"
    rows = [
        {
            "run_id": "run-a",
            "timestamp": "2026-03-19T00:00:00Z",
            "strategy_name": "alpha_v1",
            "strategy_params": {"lookback": 5},
            "evaluation_mode": "single",
            "evaluation_config": None,
            "data_range": {"start": "2025-01-01", "end": "2025-01-02"},
            "timeframe": "1D",
            "metrics_summary": {"cumulative_return": 0.02, "sharpe_ratio": 0.9},
            "artifact_path": "artifacts/strategies/run-a",
            "split_count": None,
            "dataset": "features_daily",
            "evaluation_config_path": None,
        },
        {
            "run_id": "run-b",
            "timestamp": "2026-03-19T00:01:00Z",
            "strategy_name": "beta_v1",
            "strategy_params": {"lookback": 20},
            "evaluation_mode": "walk_forward",
            "evaluation_config": {"mode": "rolling"},
            "data_range": {"start": "2025-01-01", "end": "2025-01-31"},
            "timeframe": "1D",
            "metrics_summary": {"cumulative_return": 0.15, "sharpe_ratio": 1.3},
            "artifact_path": "artifacts/strategies/run-b",
            "split_count": 3,
            "dataset": "features_daily",
            "evaluation_config_path": "configs/evaluation.yml",
        },
    ]

    for row in rows:
        append_registry_entry(registry_path, row)

    entries = load_registry(registry_path)

    assert len(entries) == 2
    assert [entry["run_id"] for entry in filter_by_strategy_name(entries, "beta_v1")] == ["run-b"]
    assert [entry["run_id"] for entry in filter_by_metric_threshold(entries, "sharpe_ratio", min_value=1.0)] == [
        "run-b"
    ]
    assert [entry["run_id"] for entry in filter_by_metric_threshold(entries, "cumulative_return", max_value=0.05)] == [
        "run-a"
    ]


def test_registry_jsonl_lines_remain_valid_json_objects(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.jsonl"
    entry = {
        "run_id": "run-json",
        "timestamp": "2026-03-19T00:00:00Z",
        "strategy_name": "alpha_v1",
        "strategy_params": {"lookback": 5},
        "evaluation_mode": "single",
        "evaluation_config": None,
        "data_range": {"start": "2025-01-01", "end": "2025-01-02"},
        "timeframe": "1D",
        "metrics_summary": {"cumulative_return": 0.02},
        "artifact_path": "artifacts/strategies/run-json",
        "split_count": None,
        "dataset": "features_daily",
        "evaluation_config_path": None,
    }

    append_registry_entry(registry_path, entry)
    lines = registry_path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 1
    assert json.loads(lines[0])["run_id"] == "run-json"


def test_generate_portfolio_run_id_is_deterministic_for_equivalent_inputs() -> None:
    config_left = {
        "portfolio_name": "Core Portfolio",
        "allocator": "equal_weight",
        "settings": {"rebalance": "daily", "long_only": True},
    }
    config_right = {
        "allocator": "equal_weight",
        "settings": {"long_only": True, "rebalance": "daily"},
        "portfolio_name": "Core Portfolio",
    }

    left = generate_portfolio_run_id(
        portfolio_name="Core Portfolio",
        allocator_name="equal_weight",
        component_run_ids=["run-b", "run-a"],
        timeframe="1D",
        start_ts="2025-01-01",
        end_ts="2025-01-31",
        config=config_left,
        evaluation_config_path=Path("configs") / "portfolio.yml",
    )
    right = generate_portfolio_run_id(
        portfolio_name="Core Portfolio",
        allocator_name="equal_weight",
        component_run_ids=["run-a", "run-b"],
        timeframe="1D",
        start_ts="2025-01-01",
        end_ts="2025-01-31",
        config=config_right,
        evaluation_config_path="configs/portfolio.yml",
    )

    assert left == right
    assert left.startswith("core_portfolio_portfolio_")


def test_register_portfolio_run_appends_one_portfolio_entry_with_stable_schema(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.jsonl"
    config = {
        "portfolio_name": "Core Portfolio",
        "allocator": "equal_weight",
        "timeframe": "1D",
        "settings": {"rebalance": "daily"},
    }
    components = [
        {"strategy_name": "beta", "run_id": "run-b", "artifact_path": "artifacts/strategies/run-b"},
        {"strategy_name": "alpha", "run_id": "run-a", "artifact_path": "artifacts/strategies/run-a"},
    ]
    metrics = {"total_return": 0.12, "sharpe_ratio": 1.4}
    run_id = generate_portfolio_run_id(
        portfolio_name="Core Portfolio",
        allocator_name="equal_weight",
        component_run_ids=["run-b", "run-a"],
        timeframe="1D",
        start_ts="2025-01-01",
        end_ts="2025-01-31",
        config=config,
        evaluation_config_path="configs/portfolio.yml",
    )

    register_portfolio_run(
        registry_path=registry_path,
        run_id=run_id,
        config=config,
        components=components,
        metrics=metrics,
        artifact_path="artifacts/portfolios/core_portfolio",
        metadata={
            "portfolio_name": "Core Portfolio",
            "allocator_name": "equal_weight",
            "timeframe": "1D",
            "start_ts": "2025-01-01",
            "end_ts": "2025-01-31",
            "evaluation_config_path": "configs/portfolio.yml",
        },
    )

    entries = load_registry(registry_path)

    assert len(entries) == 1
    entry = entries[0]
    assert entry["run_id"] == run_id
    assert entry["run_type"] == "portfolio"
    assert entry["portfolio_name"] == "Core Portfolio"
    assert entry["allocator_name"] == "equal_weight"
    assert entry["component_run_ids"] == ["run-a", "run-b"]
    assert entry["component_strategy_names"] == ["alpha", "beta"]
    assert entry["timeframe"] == "1D"
    assert entry["start_ts"] == "2025-01-01"
    assert entry["end_ts"] == "2025-01-31"
    assert entry["artifact_path"] == "artifacts/portfolios/core_portfolio"
    assert entry["metrics"] == metrics
    assert entry["evaluation_config_path"] == "configs/portfolio.yml"
    assert entry["split_count"] is None


def test_register_portfolio_run_skips_duplicate_identical_run_ids(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.jsonl"
    payload = {
        "registry_path": registry_path,
        "run_id": "core_portfolio_portfolio_123abc456def",
        "config": {"portfolio_name": "Core Portfolio", "allocator": "equal_weight", "timeframe": "1D"},
        "components": [
            {"strategy_name": "alpha", "run_id": "run-a"},
            {"strategy_name": "beta", "run_id": "run-b"},
        ],
        "metrics": {"total_return": 0.12},
        "artifact_path": "artifacts/portfolios/core_portfolio",
        "metadata": {
            "portfolio_name": "Core Portfolio",
            "allocator_name": "equal_weight",
            "timeframe": "1D",
            "start_ts": "2025-01-01",
            "end_ts": "2025-01-31",
        },
    }

    register_portfolio_run(**payload)
    register_portfolio_run(**payload)

    entries = load_registry(registry_path)
    assert [entry["run_id"] for entry in entries] == ["core_portfolio_portfolio_123abc456def"]


def test_registry_filters_support_mixed_strategy_and_portfolio_entries(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.jsonl"
    append_registry_entry(
        registry_path,
        {
            "run_id": "strategy-run",
            "timestamp": "2026-03-19T00:00:00Z",
            "strategy_name": "alpha_v1",
            "strategy_params": {"lookback": 5},
            "evaluation_mode": "single",
            "evaluation_config": None,
            "data_range": {"start": "2025-01-01", "end": "2025-01-02"},
            "timeframe": "1D",
            "metrics_summary": {"cumulative_return": 0.02, "sharpe_ratio": 0.9},
            "artifact_path": "artifacts/strategies/strategy-run",
            "split_count": None,
            "dataset": "features_daily",
            "evaluation_config_path": None,
        },
    )
    register_portfolio_run(
        registry_path=registry_path,
        run_id="core_portfolio_portfolio_123abc456def",
        config={"portfolio_name": "Core Portfolio", "allocator": "equal_weight", "timeframe": "1D"},
        components=[
            {"strategy_name": "alpha_v1", "run_id": "strategy-run"},
            {"strategy_name": "beta_v1", "run_id": "strategy-run-2"},
        ],
        metrics={"total_return": 0.15, "sharpe_ratio": 1.3},
        artifact_path="artifacts/portfolios/core_portfolio",
        metadata={
            "portfolio_name": "Core Portfolio",
            "allocator_name": "equal_weight",
            "timeframe": "1D",
            "start_ts": "2025-01-01",
            "end_ts": "2025-01-31",
        },
    )

    entries = load_registry(registry_path)

    assert [entry["run_id"] for entry in filter_by_run_type(entries, "strategy")] == ["strategy-run"]
    assert [entry["run_id"] for entry in filter_by_run_type(entries, "portfolio")] == [
        "core_portfolio_portfolio_123abc456def"
    ]
    assert [entry["run_id"] for entry in filter_by_portfolio_name(entries, "Core Portfolio")] == [
        "core_portfolio_portfolio_123abc456def"
    ]
    assert [entry["run_id"] for entry in filter_by_allocator_name(entries, "equal_weight")] == [
        "core_portfolio_portfolio_123abc456def"
    ]
    assert [entry["run_id"] for entry in filter_by_strategy_name(entries, "alpha_v1")] == ["strategy-run"]
    assert [entry["run_id"] for entry in filter_by_metric_threshold(entries, "sharpe_ratio", min_value=1.0)] == [
        "core_portfolio_portfolio_123abc456def"
    ]


def test_load_registry_defaults_missing_run_type_for_legacy_strategy_entries(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.jsonl"
    registry_path.write_text(
        json.dumps(
            {
                "run_id": "legacy-run",
                "timestamp": "2026-03-19T00:00:00Z",
                "strategy_name": "alpha_v1",
                "strategy_params": {},
                "evaluation_mode": "single",
                "evaluation_config": None,
                "data_range": {"start": "2025-01-01", "end": "2025-01-02"},
                "timeframe": "1D",
                "metrics_summary": {"cumulative_return": 0.02},
                "artifact_path": "artifacts/strategies/legacy-run",
                "split_count": None,
                "dataset": "features_daily",
                "evaluation_config_path": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    entries = load_registry(registry_path)

    assert len(entries) == 1
    assert entries[0]["run_type"] == "strategy"


def test_register_portfolio_run_rejects_missing_component_run_ids(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="components must contain at least one portfolio component"):
        register_portfolio_run(
            registry_path=tmp_path / "registry.jsonl",
            run_id="core_portfolio_portfolio_123abc456def",
            config={"portfolio_name": "Core Portfolio", "allocator": "equal_weight", "timeframe": "1D"},
            components=[],
            metrics={"total_return": 0.12},
            artifact_path="artifacts/portfolios/core_portfolio",
            metadata={
                "portfolio_name": "Core Portfolio",
                "allocator_name": "equal_weight",
                "timeframe": "1D",
                "start_ts": "2025-01-01",
                "end_ts": "2025-01-31",
            },
        )
