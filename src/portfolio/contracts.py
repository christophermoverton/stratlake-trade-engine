from __future__ import annotations

import json
from typing import Any

import pandas as pd

_STRATEGY_ID_CANDIDATES: tuple[str, ...] = ("strategy_name", "strategy_id", "strategy")
_PORTFOLIO_REQUIRED_COLUMNS: tuple[str, ...] = ("ts_utc", "portfolio_return")
_WEIGHT_SUM_TOLERANCE = 1e-8
_DEFAULT_INITIAL_CAPITAL = 1.0
_DEFAULT_ALIGNMENT_POLICY = "intersection"


class PortfolioContractError(ValueError):
    """Raised when portfolio-layer inputs violate deterministic contracts."""


def validate_strategy_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and normalize long-form strategy return input data.

    Expected columns:
      - ts_utc
      - strategy_return
      - one strategy identifier column from _STRATEGY_ID_CANDIDATES

    Returns a deterministic copy sorted by (ts_utc, strategy identifier).
    """

    if not isinstance(df, pd.DataFrame):
        raise PortfolioContractError("strategy returns input must be provided as a pandas DataFrame.")

    strategy_column = _resolve_strategy_identifier_column(df)
    required_columns = ("ts_utc", "strategy_return", strategy_column)
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        formatted = ", ".join(repr(column) for column in missing)
        raise PortfolioContractError(
            f"strategy returns input is missing required columns: {formatted}."
        )

    normalized = df.copy()
    normalized["ts_utc"] = _normalize_utc_timestamp_series(normalized["ts_utc"], column_name="ts_utc")
    normalized[strategy_column] = _normalize_non_null_string_series(
        normalized[strategy_column],
        column_name=strategy_column,
    )
    normalized["strategy_return"] = _normalize_float_series(
        normalized["strategy_return"],
        column_name="strategy_return",
    )

    duplicate_mask = normalized.duplicated(subset=["ts_utc", strategy_column], keep=False)
    if duplicate_mask.any():
        duplicate_row = normalized.loc[duplicate_mask, ["ts_utc", strategy_column]].iloc[0]
        raise PortfolioContractError(
            "strategy returns input contains duplicate (ts_utc, strategy) rows. "
            f"First duplicate key: ({duplicate_row['ts_utc']}, {duplicate_row[strategy_column]!r})."
        )

    normalized = normalized.sort_values(["ts_utc", strategy_column], kind="stable").reset_index(drop=True)
    normalized.attrs["portfolio_contract"] = {
        "contract": "strategy_returns",
        "strategy_identifier_column": strategy_column,
        "row_count": int(len(normalized)),
    }
    return normalized


def validate_aligned_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and normalize a wide strategy return matrix.

    Returns a deterministic copy with a UTC DatetimeIndex and lexicographically
    sorted strategy columns.
    """

    normalized = _validate_wide_numeric_matrix(
        df,
        matrix_name="aligned returns",
        require_row_sum_to_one=False,
        forbid_nans=False,
    )
    normalized.attrs["portfolio_contract"] = {
        "contract": "aligned_returns",
        "row_count": int(len(normalized)),
        "column_count": int(len(normalized.columns)),
    }
    return normalized


def validate_weights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and normalize a wide strategy weight matrix.

    Returns a deterministic copy with a UTC DatetimeIndex, sorted columns, no
    NaNs, and rows summing to 1.0 within tolerance.
    """

    normalized = _validate_wide_numeric_matrix(
        df,
        matrix_name="weights",
        require_row_sum_to_one=True,
        forbid_nans=True,
    )
    normalized.attrs["portfolio_contract"] = {
        "contract": "weights",
        "row_count": int(len(normalized)),
        "column_count": int(len(normalized.columns)),
        "row_sum_tolerance": _WEIGHT_SUM_TOLERANCE,
    }
    return normalized


def validate_portfolio_output(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and normalize a portfolio output frame.

    Required columns:
      - ts_utc
      - portfolio_return

    Optional traceability columns:
      - strategy_return__<strategy>
      - weight__<strategy>
      - portfolio_equity_curve

    Returns a deterministic copy sorted by ts_utc, with traceability columns
    sorted lexicographically by prefix and strategy identifier.
    """

    if not isinstance(df, pd.DataFrame):
        raise PortfolioContractError("portfolio output must be provided as a pandas DataFrame.")

    missing = [column for column in _PORTFOLIO_REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        formatted = ", ".join(repr(column) for column in missing)
        raise PortfolioContractError(f"portfolio output is missing required columns: {formatted}.")

    normalized = df.copy()
    normalized["ts_utc"] = _normalize_utc_timestamp_series(normalized["ts_utc"], column_name="ts_utc")
    normalized["portfolio_return"] = _normalize_float_series(
        normalized["portfolio_return"],
        column_name="portfolio_return",
    )

    if "portfolio_equity_curve" in normalized.columns:
        normalized["portfolio_equity_curve"] = _normalize_float_series(
            normalized["portfolio_equity_curve"],
            column_name="portfolio_equity_curve",
        )

    if normalized["ts_utc"].duplicated().any():
        duplicate_ts = normalized.loc[normalized["ts_utc"].duplicated(keep=False), "ts_utc"].iloc[0]
        raise PortfolioContractError(
            f"portfolio output contains duplicate timestamps. First duplicate ts_utc: {duplicate_ts}."
        )

    return_columns = [column for column in normalized.columns if column.startswith("strategy_return__")]
    weight_columns = [column for column in normalized.columns if column.startswith("weight__")]

    _validate_prefixed_numeric_columns(normalized, return_columns, prefix="strategy_return__")
    _validate_prefixed_numeric_columns(normalized, weight_columns, prefix="weight__")

    return_suffixes = {column.removeprefix("strategy_return__") for column in return_columns}
    weight_suffixes = {column.removeprefix("weight__") for column in weight_columns}
    if return_suffixes != weight_suffixes:
        missing_returns = sorted(weight_suffixes - return_suffixes)
        missing_weights = sorted(return_suffixes - weight_suffixes)
        details: list[str] = []
        if missing_returns:
            details.append(f"missing strategy_return__ columns for: {missing_returns}")
        if missing_weights:
            details.append(f"missing weight__ columns for: {missing_weights}")
        raise PortfolioContractError(
            "portfolio output traceability columns are inconsistent: " + "; ".join(details) + "."
        )

    ordered_columns = ["ts_utc"]
    ordered_columns.extend(sorted(return_columns))
    ordered_columns.extend(sorted(weight_columns))
    ordered_columns.append("portfolio_return")
    if "portfolio_equity_curve" in normalized.columns:
        ordered_columns.append("portfolio_equity_curve")
    extra_columns = [column for column in normalized.columns if column not in ordered_columns]
    ordered_columns.extend(sorted(extra_columns))

    normalized = normalized.sort_values("ts_utc", kind="stable").reset_index(drop=True)
    normalized = normalized.loc[:, ordered_columns]
    normalized.attrs["portfolio_contract"] = {
        "contract": "portfolio_output",
        "row_count": int(len(normalized)),
        "traceability_strategy_count": int(len(return_suffixes)),
    }
    return normalized


def validate_portfolio_config(config_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Validate and normalize a portfolio configuration mapping.

    Returns a JSON-serializable deterministic copy.
    """

    if not isinstance(config_dict, dict):
        raise PortfolioContractError("portfolio config must be provided as a dictionary.")

    required_fields = ("portfolio_name", "allocator", "components")
    missing = [field for field in required_fields if field not in config_dict]
    if missing:
        formatted = ", ".join(repr(field) for field in missing)
        raise PortfolioContractError(f"portfolio config is missing required fields: {formatted}.")

    portfolio_name = _normalize_config_string(
        config_dict.get("portfolio_name"),
        field_name="portfolio_name",
    )
    allocator = _normalize_config_string(
        config_dict.get("allocator"),
        field_name="allocator",
    )

    components = config_dict.get("components")
    if not isinstance(components, list) or not components:
        raise PortfolioContractError("portfolio config field 'components' must be a non-empty list.")

    normalized_components: list[dict[str, str]] = []
    seen_component_keys: set[tuple[str, str]] = set()
    for index, component in enumerate(components):
        if not isinstance(component, dict):
            raise PortfolioContractError(
                f"portfolio config component at index {index} must be a dictionary."
            )
        missing_component_fields = [
            field for field in ("strategy_name", "run_id") if field not in component
        ]
        if missing_component_fields:
            formatted = ", ".join(repr(field) for field in missing_component_fields)
            raise PortfolioContractError(
                f"portfolio config component at index {index} is missing required fields: {formatted}."
            )

        strategy_name = _normalize_config_string(
            component.get("strategy_name"),
            field_name=f"components[{index}].strategy_name",
        )
        run_id = _normalize_config_string(
            component.get("run_id"),
            field_name=f"components[{index}].run_id",
        )
        component_key = (strategy_name, run_id)
        if component_key in seen_component_keys:
            raise PortfolioContractError(
                "portfolio config components must be unique by (strategy_name, run_id). "
                f"Duplicate component: ({strategy_name!r}, {run_id!r})."
            )
        seen_component_keys.add(component_key)
        normalized_components.append({"strategy_name": strategy_name, "run_id": run_id})

    initial_capital_raw = config_dict.get("initial_capital", _DEFAULT_INITIAL_CAPITAL)
    try:
        initial_capital = float(initial_capital_raw)
    except (TypeError, ValueError) as exc:
        raise PortfolioContractError(
            "portfolio config field 'initial_capital' must be a finite float-compatible value."
        ) from exc
    if not pd.notna(initial_capital):
        raise PortfolioContractError(
            "portfolio config field 'initial_capital' must be a finite float-compatible value."
        )

    alignment_policy = _normalize_config_string(
        config_dict.get("alignment_policy", _DEFAULT_ALIGNMENT_POLICY),
        field_name="alignment_policy",
    )

    normalized_config = {
        "portfolio_name": portfolio_name,
        "allocator": allocator,
        "components": sorted(
            normalized_components,
            key=lambda component: (component["strategy_name"], component["run_id"]),
        ),
        "initial_capital": initial_capital,
        "alignment_policy": alignment_policy,
    }

    try:
        json.dumps(normalized_config, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise PortfolioContractError(
            "portfolio config must be JSON-serializable after normalization."
        ) from exc

    return normalized_config


def _validate_wide_numeric_matrix(
    df: pd.DataFrame,
    *,
    matrix_name: str,
    require_row_sum_to_one: bool,
    forbid_nans: bool,
) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise PortfolioContractError(f"{matrix_name} must be provided as a pandas DataFrame.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise PortfolioContractError(f"{matrix_name} must use a DatetimeIndex named 'ts_utc'.")
    if df.index.name != "ts_utc":
        raise PortfolioContractError(
            f"{matrix_name} index must be named 'ts_utc'; found {df.index.name!r}."
        )
    if df.columns.empty:
        raise PortfolioContractError(f"{matrix_name} must contain at least one strategy column.")
    if not df.columns.is_unique:
        duplicate_column = df.columns[df.columns.duplicated()].tolist()[0]
        raise PortfolioContractError(
            f"{matrix_name} contains duplicate strategy columns. First duplicate: {duplicate_column!r}."
        )

    normalized_index = _normalize_datetime_index(df.index, name="ts_utc", owner=matrix_name)
    normalized = df.copy()
    normalized.index = normalized_index

    normalized.columns = _normalize_column_index(normalized.columns, owner=matrix_name)
    for column in normalized.columns:
        normalized[column] = _normalize_float_series(normalized[column], column_name=str(column))

    if normalized.index.has_duplicates:
        duplicate_ts = normalized.index[normalized.index.duplicated()].tolist()[0]
        raise PortfolioContractError(
            f"{matrix_name} contains duplicate timestamps. First duplicate ts_utc: {duplicate_ts}."
        )

    if forbid_nans and normalized.isna().any().any():
        row_index, column_name = normalized.isna().stack()[lambda values: values].index[0]
        raise PortfolioContractError(
            f"{matrix_name} contains NaN values. First NaN at ts_utc={row_index}, column={column_name!r}."
        )

    sorted_columns = sorted(normalized.columns.tolist())
    normalized = normalized.loc[:, sorted_columns]
    normalized = normalized.sort_index(kind="stable")

    if require_row_sum_to_one and not normalized.empty:
        row_sums = normalized.sum(axis=1)
        invalid_mask = (row_sums - 1.0).abs() > _WEIGHT_SUM_TOLERANCE
        if invalid_mask.any():
            bad_ts = row_sums.index[invalid_mask][0]
            bad_sum = row_sums.loc[bad_ts]
            raise PortfolioContractError(
                "weights rows must sum to 1.0 within tolerance "
                f"{_WEIGHT_SUM_TOLERANCE}. First failing ts_utc={bad_ts}, row_sum={bad_sum}."
            )

    return normalized


def _resolve_strategy_identifier_column(df: pd.DataFrame) -> str:
    candidates = [column for column in _STRATEGY_ID_CANDIDATES if column in df.columns]
    if not candidates:
        raise PortfolioContractError(
            "strategy returns input must include one strategy identifier column from "
            f"{list(_STRATEGY_ID_CANDIDATES)!r}."
        )
    if len(candidates) > 1:
        raise PortfolioContractError(
            "strategy returns input must include exactly one strategy identifier column. "
            f"Found multiple candidates: {candidates}."
        )
    return candidates[0]


def _normalize_utc_timestamp_series(series: pd.Series, *, column_name: str) -> pd.Series:
    if series.isna().any():
        raise PortfolioContractError(f"column {column_name!r} contains null timestamps.")

    raw_dtype = series.dtype
    if isinstance(raw_dtype, pd.DatetimeTZDtype):
        if str(raw_dtype.tz) != "UTC":
            raise PortfolioContractError(
                f"column {column_name!r} must already be timezone-aware UTC; found timezone {raw_dtype.tz!r}."
            )
        parsed = pd.to_datetime(series, errors="coerce")
    elif pd.api.types.is_datetime64_dtype(raw_dtype):
        raise PortfolioContractError(
            f"column {column_name!r} must be timezone-aware UTC, but received tz-naive datetimes."
        )
    else:
        parsed = pd.to_datetime(series, errors="coerce", utc=False)
        if parsed.isna().any():
            raise PortfolioContractError(
                f"column {column_name!r} contains unparsable datetime values."
            )
        if not isinstance(parsed.dtype, pd.DatetimeTZDtype):
            raise PortfolioContractError(
                f"column {column_name!r} must be timezone-aware UTC, but received tz-naive values."
            )
        if str(parsed.dtype.tz) != "UTC":
            raise PortfolioContractError(
                f"column {column_name!r} must already be timezone-aware UTC; found timezone {parsed.dtype.tz!r}."
            )
    if parsed.isna().any():
        raise PortfolioContractError(f"column {column_name!r} contains unparsable datetime values.")
    return pd.Series(parsed, index=series.index, name=series.name)


def _normalize_datetime_index(index: pd.Index, *, name: str, owner: str) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        raise PortfolioContractError(f"{owner} must use a DatetimeIndex named {name!r}.")
    if index.tz is None:
        raise PortfolioContractError(f"{owner} index {name!r} must be timezone-aware UTC.")
    if str(index.tz) != "UTC":
        raise PortfolioContractError(f"{owner} index {name!r} must use timezone UTC; found {index.tz!r}.")
    if index.hasnans:
        raise PortfolioContractError(f"{owner} index {name!r} contains null timestamps.")
    return pd.DatetimeIndex(index, name=name)


def _normalize_non_null_string_series(series: pd.Series, *, column_name: str) -> pd.Series:
    normalized = series.astype("string")
    if normalized.isna().any():
        raise PortfolioContractError(f"column {column_name!r} contains null values.")
    normalized = normalized.str.strip()
    if (normalized == "").any():
        raise PortfolioContractError(f"column {column_name!r} contains blank string values.")
    return normalized


def _normalize_float_series(series: pd.Series, *, column_name: str) -> pd.Series:
    try:
        normalized = pd.to_numeric(series, errors="raise").astype("float64")
    except (TypeError, ValueError) as exc:
        raise PortfolioContractError(
            f"column {column_name!r} must contain float-compatible numeric values."
        ) from exc
    return normalized


def _normalize_column_index(columns: pd.Index, *, owner: str) -> pd.Index:
    normalized = pd.Index([str(column).strip() for column in columns], name=columns.name)
    if normalized.hasnans:
        raise PortfolioContractError(f"{owner} contains null strategy column identifiers.")
    if any(column == "" for column in normalized):
        raise PortfolioContractError(f"{owner} contains blank strategy column identifiers.")
    return normalized


def _validate_prefixed_numeric_columns(df: pd.DataFrame, columns: list[str], *, prefix: str) -> None:
    seen_suffixes: set[str] = set()
    for column in columns:
        suffix = column.removeprefix(prefix)
        if not suffix:
            raise PortfolioContractError(
                f"portfolio output column {column!r} must include a strategy identifier suffix."
            )
        if suffix in seen_suffixes:
            raise PortfolioContractError(
                f"portfolio output contains duplicate {prefix!r} strategy identifiers for {suffix!r}."
            )
        seen_suffixes.add(suffix)
        df[column] = _normalize_float_series(df[column], column_name=column)


def _normalize_config_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise PortfolioContractError(f"portfolio config field {field_name!r} must be a non-empty string.")
    normalized = value.strip()
    if not normalized:
        raise PortfolioContractError(f"portfolio config field {field_name!r} must be a non-empty string.")
    return normalized
