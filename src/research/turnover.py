from __future__ import annotations

import pandas as pd


def compute_position_change_frame(position: pd.Series) -> pd.DataFrame:
    """Return deterministic row-level change accounting for an executed position series."""

    executed_position = pd.to_numeric(position, errors="coerce").fillna(0.0).astype("float64")
    delta_position = executed_position.diff().fillna(executed_position).astype("float64")
    abs_delta_position = delta_position.abs().astype("float64")
    trade_event = abs_delta_position.gt(0.0)

    return pd.DataFrame(
        {
            "position": executed_position,
            "delta_position": delta_position,
            "abs_delta_position": abs_delta_position,
            "turnover": abs_delta_position.astype("float64"),
            "trade_event": trade_event,
        },
        index=executed_position.index,
    )


def compute_weight_change_frame(weights: pd.DataFrame) -> pd.DataFrame:
    """Return deterministic portfolio-level change accounting for a weight matrix."""

    normalized_weights = weights.astype("float64")
    weight_change = normalized_weights.diff().fillna(normalized_weights).astype("float64")
    abs_weight_change = weight_change.abs().astype("float64")
    portfolio_turnover = abs_weight_change.sum(axis=1).astype("float64")

    return pd.DataFrame(
        {
            "portfolio_weight_change": weight_change.sum(axis=1).astype("float64"),
            "portfolio_abs_weight_change": portfolio_turnover,
            "portfolio_turnover": portfolio_turnover,
            "portfolio_rebalance_event": portfolio_turnover.gt(0.0),
        },
        index=normalized_weights.index,
    )


def validate_position_change_frame(frame: pd.DataFrame) -> None:
    """Raise when deterministic position change accounting invariants are violated."""

    expected_abs_delta = pd.to_numeric(frame["delta_position"], errors="coerce").abs()
    actual_abs_delta = pd.to_numeric(frame["abs_delta_position"], errors="coerce")
    turnover = pd.to_numeric(frame["turnover"], errors="coerce")
    trade_event = frame["trade_event"].astype("bool")

    if not expected_abs_delta.equals(actual_abs_delta):
        raise ValueError("Position change accounting invariant failed: abs_delta_position must equal abs(delta_position).")
    if not actual_abs_delta.equals(turnover):
        raise ValueError("Position change accounting invariant failed: turnover must equal abs_delta_position.")
    if not turnover.ge(0.0).all():
        raise ValueError("Position change accounting invariant failed: turnover must be non-negative.")
    if not trade_event.equals(turnover.gt(0.0)):
        raise ValueError("Position change accounting invariant failed: trade_event must match non-zero turnover rows.")


def validate_weight_change_frame(frame: pd.DataFrame) -> None:
    """Raise when deterministic portfolio turnover invariants are violated."""

    turnover = pd.to_numeric(frame["portfolio_turnover"], errors="coerce")
    abs_change = pd.to_numeric(frame["portfolio_abs_weight_change"], errors="coerce")
    rebalance_event = frame["portfolio_rebalance_event"].astype("bool")

    if not turnover.equals(abs_change):
        raise ValueError("Portfolio turnover invariant failed: portfolio_turnover must equal portfolio_abs_weight_change.")
    if not turnover.ge(0.0).all():
        raise ValueError("Portfolio turnover invariant failed: portfolio_turnover must be non-negative.")
    if not rebalance_event.equals(turnover.gt(0.0)):
        raise ValueError("Portfolio turnover invariant failed: portfolio_rebalance_event must match non-zero turnover rows.")
