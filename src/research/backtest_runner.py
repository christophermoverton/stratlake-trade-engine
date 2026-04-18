from __future__ import annotations

import pandas as pd

from src.config.execution import ExecutionConfig, resolve_execution_config
from src.research.consistency import validate_signals_to_backtest_consistency
from src.research.integrity import validate_research_integrity
from src.research.signal_semantics import (
    SignalSemanticsError,
    attach_signal_metadata,
    ensure_signal_type_compatible,
    executable_position_constructor_name,
    extract_signal_metadata,
    legacy_signal_type_from_values,
    validate_signal_frame,
)
from src.research.turnover import compute_position_change_frame, validate_position_change_frame

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


def _validate_backtest_signal_column(df: pd.DataFrame) -> pd.Series:
    normalized_signal = pd.to_numeric(df["signal"], errors="coerce")
    invalid_signal = normalized_signal.isna() | normalized_signal.isin([float("-inf"), float("inf")])
    if invalid_signal.any():
        bad_index = invalid_signal[invalid_signal].index[0]
        bad_value = df.loc[bad_index, "signal"]
        raise ValueError(
            "Backtest input contains invalid 'signal' values. Signals must be finite numeric exposures. "
            f"First invalid index: {bad_index!r}, value={bad_value!r}."
        )
    return normalized_signal.astype("float64")


def _validate_signal_semantics(df: pd.DataFrame) -> dict[str, object]:
    try:
        metadata = extract_signal_metadata(df)
        if metadata is None:
            inferred_type = legacy_signal_type_from_values(df["signal"])
            validated = validate_signal_frame(df, signal_type=inferred_type, value_column="signal")
            attach_signal_metadata(
                df,
                {
                    "signal_type": inferred_type,
                    "version": "1.0.0",
                    "value_column": "signal",
                    "parameters": {},
                    "source": {"layer": "legacy_backtest_input"},
                    "timestamp_normalization": "UTC",
                    "transformation_history": [],
                    "compatibility_mode": "legacy_inferred",
                },
            )
            return {
                "signal_type": inferred_type,
                "version": "1.0.0",
                "validated_row_count": int(len(validated)),
            }

        signal_type = str(metadata.get("signal_type", "")).strip()
        version = str(metadata.get("version", "")).strip()
        parameters = metadata.get("parameters", {})
        if not isinstance(parameters, dict):
            parameters = {}
        validate_signal_frame(
            df,
            signal_type=signal_type,
            value_column=str(metadata.get("value_column", "signal")),
            version=version,
            parameters=parameters,
        )
        ensure_signal_type_compatible(
            signal_type,
            position_constructor=executable_position_constructor_name(),
            version=version,
        )
        return {
            "signal_type": signal_type,
            "version": version,
            "validated_row_count": int(len(df)),
        }
    except (SignalSemanticsError, ValueError) as exc:
        message = str(exc)
        if "numeric column 'signal'" in message or "parse string" in message or "NaN values" in message or "finite numeric values" in message:
            raise ValueError(
                "Backtest input contains invalid 'signal' values. Signals must be finite numeric exposures."
            ) from exc
        if "sorted deterministically by ['symbol', 'ts_utc']" in message:
            raise ValueError("Backtest input must be sorted by (symbol, ts_utc).") from exc
        raise


def _compute_executed_signal(
    df: pd.DataFrame,
    signal: pd.Series,
    *,
    execution_delay: int,
) -> pd.Series:
    symbol_keys = df["symbol"].astype("string")
    return (
        signal.groupby(symbol_keys, sort=False, dropna=False)
        .shift(execution_delay)
        .fillna(0.0)
        .astype("float64")
    )


def run_backtest(
    df: pd.DataFrame,
    execution_config: ExecutionConfig | None = None,
) -> pd.DataFrame:
    """
    Compute deterministic strategy returns and an equity curve from strategy signals.

    The strategy return uses the previous period's signal so the backtest does not
    look ahead at the same row's realized return. Signals may be canonical discrete
    values in ``{-1, 0, 1}`` or any finite numeric exposure interpreted literally
    as lagged position size. This runner does not clip or normalize exposure values.

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
    signal_semantics = _validate_signal_semantics(df)

    return_column = _resolve_return_column(df)
    _validate_return_column(df, return_column)

    config = execution_config or resolve_execution_config()

    result = df.copy()
    result.attrs = dict(df.attrs)
    result[return_column] = pd.to_numeric(result[return_column], errors="coerce").fillna(0.0).astype("float64")
    result["signal"] = _validate_backtest_signal_column(result)
    executed_signal = _compute_executed_signal(
        result,
        result["signal"],
        execution_delay=config.execution_delay,
    )
    validate_research_integrity(
        result,
        result["signal"],
        positions=executed_signal,
        execution_delay=config.execution_delay,
        allow_continuous_signals=True,
    )

    position_change_frame = compute_position_change_frame(
        executed_signal,
        group_keys=result["symbol"],
    )
    validate_position_change_frame(position_change_frame)
    gross_strategy_return = (executed_signal * result[return_column]).astype("float64")
    transaction_cost = _execution_cost(position_change_frame["delta_position"], config.transaction_cost_bps, enabled=config.enabled)
    slippage_cost = _execution_cost(position_change_frame["delta_position"], config.slippage_bps, enabled=config.enabled)

    result["executed_signal"] = executed_signal
    result["position"] = position_change_frame["position"]
    result["delta_position"] = position_change_frame["delta_position"]
    result["abs_delta_position"] = position_change_frame["abs_delta_position"]
    result["turnover"] = position_change_frame["turnover"]
    result["trade_event"] = position_change_frame["trade_event"]
    result["gross_strategy_return"] = gross_strategy_return
    result["transaction_cost"] = transaction_cost
    result["slippage_cost"] = slippage_cost
    result["execution_friction"] = (transaction_cost + slippage_cost).astype("float64")
    result["net_strategy_return"] = (
        gross_strategy_return - result["execution_friction"]
    ).astype("float64")
    result["strategy_return"] = result["net_strategy_return"]
    result["equity_curve"] = (1.0 + result["strategy_return"]).cumprod()
    result.attrs["execution_config"] = config.to_dict()
    result.attrs["backtest_signal_semantics"] = signal_semantics
    validate_signals_to_backtest_consistency(df, result, return_column=return_column)
    return result


def _execution_cost(delta_position: pd.Series, bps: float, *, enabled: bool) -> pd.Series:
    if not enabled or bps == 0.0:
        return pd.Series(0.0, index=delta_position.index, dtype="float64")
    return (delta_position.abs() * (float(bps) / 10_000.0)).astype("float64")
