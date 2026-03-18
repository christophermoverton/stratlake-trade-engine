from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import pandas as pd

ARTIFACTS_ROOT = Path("artifacts") / "strategies"
_BACKTEST_COLUMNS = ("strategy_return", "equity_curve")


def _sanitize_strategy_name(strategy_name: str) -> str:
    """Return a filesystem-friendly strategy name component for experiment paths."""

    cleaned = "".join(char if char.isalnum() else "_" for char in strategy_name.strip().lower())
    normalized = "_".join(part for part in cleaned.split("_") if part)
    return normalized or "strategy"


def _build_run_id(strategy_name: str) -> str:
    """Return a unique experiment run identifier using the current UTC timestamp."""

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}_{_sanitize_strategy_name(strategy_name)}"


def _signals_frame(results_df: pd.DataFrame) -> pd.DataFrame:
    """Return the signal-engine portion of an experiment output DataFrame."""

    signal_columns = [column for column in results_df.columns if column not in _BACKTEST_COLUMNS]
    return results_df.loc[:, signal_columns].copy()


def _equity_curve_frame(results_df: pd.DataFrame) -> pd.DataFrame:
    """Return the backtest artifact subset required for experiment inspection."""

    missing_columns = [column for column in _BACKTEST_COLUMNS if column not in results_df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Experiment results must include backtest columns: {missing}.")

    preferred_columns = [
        column for column in ("signal", *_BACKTEST_COLUMNS) if column in results_df.columns
    ]
    return results_df.loc[:, preferred_columns].copy()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Persist a dictionary to disk using stable JSON formatting."""

    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def save_experiment(
    strategy_name: str,
    results_df: pd.DataFrame,
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> Path:
    """
    Persist a strategy experiment's artifacts under the research artifact directory.

    Args:
        strategy_name: Strategy identifier associated with the experiment run.
        results_df: Backtest results containing signal data and equity curve outputs.
        metrics: Computed strategy performance metrics.
        config: Strategy configuration used for the experiment run.

    Returns:
        The created experiment directory path.

    Raises:
        ValueError: If ``results_df`` does not include ``strategy_return`` and
            ``equity_curve`` columns required for the equity curve artifact.
    """

    experiment_dir = ARTIFACTS_ROOT / _build_run_id(strategy_name)
    experiment_dir.mkdir(parents=True, exist_ok=False)

    _signals_frame(results_df).to_parquet(experiment_dir / "signals.parquet")
    _equity_curve_frame(results_df).to_parquet(experiment_dir / "equity_curve.parquet")
    _write_json(experiment_dir / "metrics.json", metrics)
    _write_json(experiment_dir / "config.json", config)

    return experiment_dir
