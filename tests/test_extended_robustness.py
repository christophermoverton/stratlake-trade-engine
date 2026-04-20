from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.robustness import load_robustness_config, RobustnessConfig
from src.research.robustness import run_robustness_experiment


def _write_cross_section_dataset(root: Path, periods: int = 45) -> None:
    timestamps = pd.date_range("2025-01-01", periods=periods, freq="D", tz="UTC")
    symbols = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    for symbol_index, symbol in enumerate(symbols):
        closes = []
        level = 100.0 + float(symbol_index * 7)
        for index in range(periods):
            drift = (symbol_index - 2) * 0.6 + (0.4 if index % 5 else -0.3)
            level += drift
            closes.append(level)
        close_series = pd.Series(closes, dtype="float64")
        frame = pd.DataFrame(
            {
                "symbol": pd.Series([symbol] * periods, dtype="string"),
                "ts_utc": timestamps,
                "timeframe": pd.Series(["1D"] * periods, dtype="string"),
                "date": pd.Series(timestamps.strftime("%Y-%m-%d"), dtype="string"),
                "close": close_series,
                "feature_ret_1d": close_series.div(close_series.shift(1)).sub(1.0).fillna(0.0),
                "high": close_series + 1.0,
                "low": close_series - 1.0,
                "market_return": pd.Series([0.001] * periods, dtype="float64"),
            }
        )
        dataset_dir = root / "data" / "curated" / "features_daily" / f"symbol={symbol}" / "year=2025"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(dataset_dir / "part-0.parquet", index=False)


def test_load_robustness_config_parses_extended_sweep_mapping(tmp_path: Path) -> None:
    config_path = tmp_path / "extended_robustness.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "robustness": {
                    "sweep": {
                        "strategy": {"name": ["cross_section_momentum"]},
                        "signal": {"type": ["cross_section_rank", "binary_signal"], "params": {"quantile": [0.2]}},
                        "constructor": {"name": ["rank_dollar_neutral", "identity_weights"]},
                        "asymmetry": {"exclude_short": [False, True]},
                        "group_by": ["strategy.name", "signal.type"],
                    },
                    "ranking": {"primary_metric": "sharpe_ratio", "tie_breakers": ["total_return"]},
                    "statistical_controls": {
                        "primary_metric": "sharpe_ratio",
                        "validity_ranking_method": "adjusted_q_value",
                        "fdr_alpha": 0.1,
                        "min_splits_for_inference": 4,
                    },
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    config = load_robustness_config(config_path)

    assert config.is_extended_sweep is True
    assert config.research_sweep.strategy.names == ("cross_section_momentum",)
    assert config.research_sweep.signal.types == ("cross_section_rank", "binary_signal")
    assert config.research_sweep.constructor.names == ("rank_dollar_neutral", "identity_weights")
    assert config.research_sweep.asymmetry.options["exclude_short"] == (False, True)
    assert config.ranking.primary_metric == "sharpe_ratio"
    assert config.ranking.tie_breakers == ("total_return",)
    assert config.statistical_controls.primary_metric == "sharpe_ratio"
    assert config.statistical_controls.validity_ranking_method == "adjusted_q_value"
    assert config.statistical_controls.min_splits_for_inference == 4


def test_run_extended_robustness_experiment_rejects_invalid_combos_and_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_cross_section_dataset(tmp_path)
    monkeypatch.chdir(tmp_path)

    config = RobustnessConfig.from_mapping(
        {
            "sweep": {
                "strategy": {"name": ["cross_section_momentum"]},
                "signal": {"type": ["cross_section_rank", "binary_signal"], "params": {"quantile": [0.2]}},
                "constructor": {"name": ["rank_dollar_neutral", "identity_weights"]},
                "asymmetry": {"exclude_short": [False, True]},
            },
            "ranking": {"primary_metric": "sharpe_ratio", "tie_breakers": ["total_return"]},
            "thresholds": {"sharpe_ratio": {"min": -100.0}},
        }
    )

    first = run_robustness_experiment(None, robustness_config=config)
    second = run_robustness_experiment(None, robustness_config=config)

    assert first.run_id == second.run_id
    assert first.summary["config_count"] == 4
    assert first.summary["invalid_config_count"] == 4
    assert first.variant_metrics["variant_id"].tolist() == second.variant_metrics["variant_id"].tolist()
    assert (first.experiment_dir / "metrics_by_config.csv").exists()
    assert (first.experiment_dir / "ranked_configs.csv").exists()
    assert (first.experiment_dir / "runs").exists()
    manifest = json.loads((first.experiment_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["config_count"] == 4
    assert len(manifest["child_runs"]) == 4
    first_snapshot = {
        path.relative_to(first.experiment_dir).as_posix(): path.read_bytes()
        for path in sorted(first.experiment_dir.rglob("*"))
        if path.is_file()
    }
    second_snapshot = {
        path.relative_to(second.experiment_dir).as_posix(): path.read_bytes()
        for path in sorted(second.experiment_dir.rglob("*"))
        if path.is_file()
    }
    assert first_snapshot == second_snapshot


def test_run_extended_robustness_experiment_supports_ensemble_configs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_cross_section_dataset(tmp_path)
    monkeypatch.chdir(tmp_path)

    config = RobustnessConfig.from_mapping(
        {
            "sweep": {
                "strategy": {"name": ["cross_section_momentum", "mean_reversion"]},
                "signal": {"type": ["ternary_quantile"], "params": {"quantile": [0.2]}},
                "constructor": {"name": ["identity_weights"]},
                "ensemble": {
                    "definitions": [
                        {
                            "name": "momentum_plus_reversion",
                            "members": ["cross_section_momentum", "mean_reversion"],
                            "weighting_method": "equal_weight",
                        }
                    ]
                },
            },
            "ranking": {"primary_metric": "total_return"},
        }
    )

    result = run_robustness_experiment(None, robustness_config=config)

    assert "ensemble_name" in result.variant_metrics.columns
    assert "momentum_plus_reversion" in set(result.variant_metrics["ensemble_name"].dropna().astype("string"))
    ensemble_rows = result.variant_metrics.loc[result.variant_metrics["ensemble_name"].astype("string") == "momentum_plus_reversion"]
    assert len(ensemble_rows) == 1
    ensemble_run_dir = result.experiment_dir / "runs" / str(ensemble_rows.iloc[0]["variant_id"])
    assert (ensemble_run_dir / "manifest.json").exists()
    assert (ensemble_run_dir / "portfolio_returns.csv").exists()
