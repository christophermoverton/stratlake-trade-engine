from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.pipeline.builder import PipelineBuilder, PipelineBuilderError


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


def test_builder_selects_latest_registry_version_when_version_omitted(tmp_path: Path) -> None:
    strategy_registry = tmp_path / "strategies.jsonl"
    strategy_registry.write_text(
        "\n".join(
            [
                json.dumps({"strategy_id": "demo_strategy", "version": "1.0.0", "output_signal": {"signal_type": "cross_section_rank"}}),
                json.dumps({"strategy_id": "demo_strategy", "version": "2.0.0", "output_signal": {"signal_type": "cross_section_rank"}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    signal_registry = tmp_path / "signal_types.jsonl"
    signal_registry.write_text(
        json.dumps(
            {
                "signal_type_id": "cross_section_rank",
                "version": "1.0.0",
                "status": "active",
                "domain": "rank",
                "codomain": "[-1,1]",
                "description": "rank",
                "semantic_flags": {"directional": True, "ordinal": True, "executable": True},
                "validation_rules": {"required_columns": ["symbol", "ts_utc", "value"], "cross_sectional": True, "min_cross_section_size": 2, "min_value": -1.0, "max_value": 1.0},
                "transformation_policies": {},
                "compatible_position_constructors": ["rank_dollar_neutral"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    constructor_registry = tmp_path / "constructors.jsonl"
    constructor_registry.write_text(
        json.dumps(
            {
                "constructor_id": "rank_dollar_neutral",
                "version": "1.0.0",
                "inputs": ["cross_section_rank"],
                "parameters": {
                    "gross_long": {"type": "float"},
                    "gross_short": {"type": "float"},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    strategy_catalog = tmp_path / "strategies.yml"
    strategy_catalog.write_text(
        yaml.safe_dump(
            {
                "demo_strategy": {
                    "dataset": "features_daily",
                    "signal_type": "cross_section_rank",
                    "parameters": {},
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    built = (
        PipelineBuilder(
            "demo_pipeline",
            strategy_registry_path=strategy_registry,
            signal_registry_path=signal_registry,
            constructor_registry_path=constructor_registry,
            strategy_catalog_path=strategy_catalog,
        )
        .strategy("demo_strategy")
        .signal("cross_section_rank")
        .construct_positions("rank_dollar_neutral", params={"gross_long": 0.5, "gross_short": 0.5})
        .build()
    )

    strategy_payload = yaml.safe_load(built.support_files["demo_pipeline.strategy.builder.yml"])
    assert strategy_payload["strategy"]["version"] == "2.0.0"


def test_builder_rejects_incompatible_signal_constructor() -> None:
    with pytest.raises(PipelineBuilderError, match="not compatible"):
        (
            PipelineBuilder("invalid_pipeline")
            .strategy("cross_section_momentum")
            .signal("binary_signal", params={"quantile": 0.2})
            .construct_positions("rank_dollar_neutral", params={"gross_long": 0.5, "gross_short": 0.5})
            .build()
        )


def test_builder_emits_deterministic_yaml() -> None:
    first = (
        PipelineBuilder("stable_pipeline")
        .strategy("cross_section_momentum", params={"lookback_days": 1})
        .signal("cross_section_rank")
        .construct_positions("rank_dollar_neutral", params={"gross_long": 0.5, "gross_short": 0.5})
        .build()
    )
    second = (
        PipelineBuilder("stable_pipeline")
        .strategy("cross_section_momentum", params={"lookback_days": 1})
        .signal("cross_section_rank")
        .construct_positions("rank_dollar_neutral", params={"gross_long": 0.5, "gross_short": 0.5})
        .build()
    )

    assert first.pipeline_yaml == second.pipeline_yaml
    assert first.support_files == second.support_files


def test_builder_runs_single_pipeline_with_portfolio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_cross_section_dataset(tmp_path)
    monkeypatch.chdir(tmp_path)

    result = (
        PipelineBuilder("builder_single_pipeline")
        .strategy("cross_section_momentum", params={"lookback_days": 1})
        .signal("cross_section_rank")
        .construct_positions("rank_dollar_neutral", params={"gross_long": 0.5, "gross_short": 0.5})
        .portfolio("equal_weight", params={"timeframe": "1D"})
        .run()
    )

    pipeline_dir = tmp_path / "artifacts" / "pipelines" / result.pipeline_run_id
    manifest = json.loads((pipeline_dir / "manifest.json").read_text(encoding="utf-8"))
    assert result.status == "completed"
    assert result.execution_order == ("run_strategy", "run_portfolio")
    assert [step["step_id"] for step in manifest["steps"]] == ["run_strategy", "run_portfolio"]
    assert Path(manifest["steps"][0]["step_artifact_dir"]).exists()
    assert Path(manifest["steps"][1]["step_artifact_dir"]).exists()


def test_builder_runs_sweep_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_cross_section_dataset(tmp_path)
    monkeypatch.chdir(tmp_path)

    result = (
        PipelineBuilder("builder_sweep_pipeline")
        .strategy("cross_section_momentum", params={"lookback_days": 1})
        .signal("cross_section_rank")
        .construct_positions("rank_dollar_neutral", params={"gross_long": 0.5, "gross_short": 0.5})
        .asymmetry({"exclude_short": False})
        .sweep(
            {
                "signal": {"type": ["cross_section_rank", "binary_signal"], "params": {"quantile": [0.2]}},
                "constructor": {"name": ["rank_dollar_neutral", "identity_weights"]},
                "asymmetry": {"exclude_short": [False, True]},
                "ranking": {"primary_metric": "sharpe_ratio", "tie_breakers": ["total_return"]},
            }
        )
        .run()
    )

    pipeline_dir = tmp_path / "artifacts" / "pipelines" / result.pipeline_run_id
    manifest = json.loads((pipeline_dir / "manifest.json").read_text(encoding="utf-8"))
    step_artifact_dir = Path(manifest["steps"][0]["step_artifact_dir"])

    assert result.status == "completed"
    assert result.execution_order == ("research_sweep",)
    assert step_artifact_dir.exists()
    assert (step_artifact_dir / "metrics_by_config.csv").exists()
    assert (step_artifact_dir / "ranked_configs.csv").exists()
