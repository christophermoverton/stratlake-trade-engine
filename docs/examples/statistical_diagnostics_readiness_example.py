from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.research.metrics import (
    build_metrics_readiness_manifest,
    compute_performance_metrics,
    write_metrics_readiness_manifest,
)


OUTPUT_DIR = Path("docs/examples/output/statistical_diagnostics_readiness_example")
EXPECTED_METRIC_FIELDS = {
    "t_stat",
    "p_value",
    "hit_rate_p_value",
    "autocorr_lag1",
    "effective_n",
    "split_mean_diff",
    "split_mean_diff_p",
    "rolling_sharpe_mean",
    "rolling_sharpe_sd",
    "sharpe_stability_ratio",
}


def build_example_backtest_frame() -> pd.DataFrame:
    returns = [
        0.004,
        0.003,
        -0.002,
        0.005,
        0.001,
        -0.003,
        0.006,
        0.002,
        -0.001,
        0.004,
        0.003,
        -0.002,
        0.005,
        0.002,
        -0.004,
        0.006,
        0.001,
        -0.002,
        0.004,
        0.003,
        -0.001,
        0.005,
        0.002,
        -0.003,
        0.004,
        0.002,
        -0.002,
        0.005,
        0.001,
        -0.004,
        0.006,
        0.002,
        -0.001,
        0.004,
        0.003,
        -0.002,
    ]
    signals = [
        0.0,
        1.0,
        1.0,
        1.0,
        0.0,
        -1.0,
        -1.0,
        0.0,
        1.0,
        1.0,
        0.0,
        -1.0,
        -1.0,
        -1.0,
        0.0,
        1.0,
        1.0,
        0.0,
        -1.0,
        -1.0,
        0.0,
        1.0,
        1.0,
        0.0,
        -1.0,
        -1.0,
        0.0,
        1.0,
        1.0,
        0.0,
        -1.0,
        -1.0,
        0.0,
        1.0,
        1.0,
        0.0,
    ]
    equity = pd.Series(returns, dtype="float64").add(1.0).cumprod()
    return pd.DataFrame(
        {
            "ts_utc": pd.date_range("2025-01-01", periods=len(returns), freq="D", tz="UTC"),
            "timeframe": ["1D"] * len(returns),
            "signal": signals,
            "strategy_return": returns,
            "equity_curve": equity,
        }
    )


def main() -> int:
    backtest = build_example_backtest_frame()
    metrics = compute_performance_metrics(backtest)

    missing_fields = EXPECTED_METRIC_FIELDS.difference(metrics)
    if missing_fields:
        raise AssertionError(f"Missing expected metric fields: {sorted(missing_fields)}")

    readiness = build_metrics_readiness_manifest(metrics, run_id="docs_m30_diagnostics_example")
    if readiness["schema_version"] != 1:
        raise AssertionError("Unexpected readiness schema version.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUTPUT_DIR / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    readiness_path = write_metrics_readiness_manifest(
        OUTPUT_DIR,
        metrics,
        run_id="docs_m30_diagnostics_example",
    )

    print("M30 statistical diagnostics readiness example")
    print(f"metrics: {metrics_path.as_posix()}")
    print(f"readiness: {readiness_path.as_posix()}")
    print(f"readiness_status: {readiness['status']}")
    for field in sorted(EXPECTED_METRIC_FIELDS):
        print(f"{field}: {metrics[field]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
