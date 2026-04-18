from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.cli.run_pipeline import run_cli as run_pipeline_cli


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


def test_pipeline_runner_executes_extended_robustness_step(tmp_path: Path, monkeypatch) -> None:
    _write_cross_section_dataset(tmp_path)
    monkeypatch.chdir(tmp_path)

    robustness_path = tmp_path / "extended_robustness.yml"
    robustness_path.write_text(
        """
robustness:
  strategy_name: cross_section_momentum
  sweep:
    strategy:
      name: [cross_section_momentum]
    signal:
      type: [cross_section_rank, binary_signal]
      params:
        quantile: [0.2]
    constructor:
      name: [rank_dollar_neutral, identity_weights]
    asymmetry:
      exclude_short: [false, true]
  ranking:
    primary_metric: sharpe_ratio
    tie_breakers: [total_return]
""".strip(),
        encoding="utf-8",
    )
    pipeline_path = tmp_path / "extended_robustness_pipeline.yml"
    pipeline_path.write_text(
        f"""
id: extended_robustness_pipeline
steps:
  - id: research_sweep
    adapter: python_module
    module: src.cli.run_strategy
    argv:
      - --strategy
      - cross_section_momentum
      - --robustness
      - {json.dumps(str(robustness_path))}
""".strip(),
        encoding="utf-8",
    )

    result = run_pipeline_cli(["--config", str(pipeline_path)])
    pipeline_dir = tmp_path / "artifacts" / "pipelines" / result.pipeline_run_id
    manifest = json.loads((pipeline_dir / "manifest.json").read_text(encoding="utf-8"))

    assert result.status == "completed"
    assert result.execution_order == ("research_sweep",)
    step = manifest["steps"][0]
    step_artifact_dir = Path(step["step_artifact_dir"])
    assert step["step_id"] == "research_sweep"
    assert step_artifact_dir.exists()
    assert (step_artifact_dir / "metrics_by_config.csv").exists()
    assert (step_artifact_dir / "ranked_configs.csv").exists()
    assert (step_artifact_dir / "manifest.json").exists()
