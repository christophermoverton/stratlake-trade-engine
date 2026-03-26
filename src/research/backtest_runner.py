from __future__ import annotations

import pandas as pd

from src.config.execution import ExecutionConfig, resolve_execution_config
from src.research.integrity import validate_research_integrity

RETURN_COLUMN_CANDIDATES: tuple[str, ...] = (
    "ret_1",
    "ret_1m",
    "ret_1d",
    "feature_ret_1m",
    "feature_ret_1d",
    "asset_return",
    "return",
    "returns",
)


def _resolve_return_column(df: pd.DataFrame) -> str:
    """Return the preferred asset return column name from a backtest input frame."""

    for column in RETURN_COLUMN_CANDIDATES:
        if column in df.columns:
            return column

    raise ValueError(
        "Run failed: missing returns. Backtest input must include an asset return column. "
        f"Expected one of: {', '.join(RETURN_COLUMN_CANDIDATES)}."
    )


def _validate_return_column(df: pd.DataFrame, column: str) -> None:
    usable_returns = pd.to_numeric(df[column], errors="coerce")
    if usable_returns.notna().any():
        return
    raise ValueError(
        f"Run failed: missing returns. Return column '{column}' is present but contains no usable values."
    )


def run_backtest(
    df: pd.DataFrame,
    execution_config: ExecutionConfig | None = None,
) -> pd.DataFrame:
    """
    Compute deterministic strategy returns and an equity curve from strategy signals.

    The strategy return uses the previous period's signal so the backtest does not
    look ahead at the same row's realized return.

    Args:
        df: Feature dataset containing a standardized ``signal`` column and a
            supported asset return column such as ``ret_1`` or ``feature_ret_1d``.

    Returns:
        A copy of ``df`` with added deterministic execution and equity columns.

    Raises:
        ValueError: If the required ``signal`` column or a supported asset return
            column is missing.
    """

    if "signal" not in df.columns:
        raise ValueError("Backtest input must include a 'signal' column.")

    return_column = _resolve_return_column(df)
    _validate_return_column(df, return_column)

    config = execution_config or resolve_execution_config()

    result = df.copy()
    result[return_column] = pd.to_numeric(result[return_column], errors="coerce").astype("float64")
    result["signal"] = pd.to_numeric(result["signal"], errors="coerce").fillna(0.0).astype("float64")
    executed_signal = result["signal"].shift(config.execution_delay).fillna(0.0).astype("float64")
    validate_research_integrity(
        result,
        result["signal"],
        positions=executed_signal,
        execution_delay=config.execution_delay,
    )

    delta_position = executed_signal.diff().fillna(executed_signal).astype("float64")
    gross_strategy_return = (executed_signal * result[return_column]).astype("float64")
    transaction_cost = _execution_cost(delta_position, config.transaction_cost_bps, enabled=config.enabled)
    slippage_cost = _execution_cost(delta_position, config.slippage_bps, enabled=config.enabled)

    result["executed_signal"] = executed_signal
    result["position"] = executed_signal
    result["delta_position"] = delta_position
    result["gross_strategy_return"] = gross_strategy_return
    result["transaction_cost"] = transaction_cost
    result["slippage_cost"] = slippage_cost
    result["net_strategy_return"] = (
        gross_strategy_return - transaction_cost - slippage_cost
    ).astype("float64")
    result["strategy_return"] = result["net_strategy_return"]
    result["equity_curve"] = (1.0 + result["strategy_return"]).cumprod()
    result.attrs["execution_config"] = config.to_dict()
    return result


def _execution_cost(delta_position: pd.Series, bps: float, *, enabled: bool) -> pd.Series:
    if not enabled or bps == 0.0:
        return pd.Series(0.0, index=delta_position.index, dtype="float64")
    return (delta_position.abs() * (float(bps) / 10_000.0)).astype("float64")
