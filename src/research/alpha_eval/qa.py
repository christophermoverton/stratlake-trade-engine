from __future__ import annotations

from typing import Any

import pandas as pd

from src.research.alpha.signals import AlphaSignalMappingResult
from src.research.alpha_eval.evaluator import AlphaEvaluationResult

_FAIL_NULL_RATE = 0.10
_WARN_NULL_RATE = 0.02
_FAIL_MIN_VALID_TIMESTAMPS = 2
_WARN_MIN_VALID_TIMESTAMPS = 10
_WARN_VALID_TIMESTAMP_RATE = 0.70
_FAIL_VALID_TIMESTAMP_RATE = 0.40
_WARN_MEAN_TURNOVER = 0.75
_FAIL_MEAN_TURNOVER = 1.25
_WARN_MAX_SINGLE_NAME_SHARE = 0.50
_FAIL_MAX_SINGLE_NAME_SHARE = 0.75
_WARN_MAX_ABS_NET_EXPOSURE_SHARE = 0.35
_FAIL_MAX_ABS_NET_EXPOSURE_SHARE = 0.60


def generate_alpha_qa_summary(
    aligned_frame: pd.DataFrame,
    result: AlphaEvaluationResult,
    *,
    alpha_name: str | None,
    run_id: str | None,
    signal_mapping_result: AlphaSignalMappingResult | None = None,
) -> dict[str, Any]:
    """Build a deterministic QA summary for one alpha evaluation run."""

    normalized = aligned_frame.copy(deep=True)
    normalized.attrs = {}
    timestamps = _timestamp_series(normalized)
    prediction_series = _numeric_series(normalized, result.prediction_column)
    forward_return_series = _numeric_series(normalized, result.forward_return_column)
    valid_mask = prediction_series.notna() & forward_return_series.notna()

    timestamp_coverage = _timestamp_coverage_payload(
        timestamps=timestamps,
        valid_mask=valid_mask,
        result=result,
    )
    post_warmup_nulls = _post_warmup_null_payload(
        frame=normalized,
        timestamps=timestamps,
        prediction_column=result.prediction_column,
        forward_return_column=result.forward_return_column,
    )
    signal_payload = _signal_payload(signal_mapping_result)
    checks = _checks_payload(
        timestamp_coverage=timestamp_coverage,
        post_warmup_nulls=post_warmup_nulls,
        signal_payload=signal_payload,
    )
    flags = {key: value["status"] != "pass" for key, value in checks.items()}
    overall_status = _overall_status(checks)

    return {
        "run_id": run_id,
        "alpha_name": alpha_name,
        "timeframe": _coerce_optional_string(result.metadata.get("timeframe")),
        "row_count": int(result.row_count),
        "timestamp_count": int(result.timestamp_count),
        "symbol_count": int(result.symbol_count),
        "date_range": [
            _format_timestamp(result.metadata.get("ts_utc_start")),
            _format_timestamp(result.metadata.get("ts_utc_end")),
        ],
        "forecast": {
            "ic_ir": _coerce_number(result.summary.get("ic_ir")),
            "mean_ic": _coerce_number(result.summary.get("mean_ic")),
            "rank_ic_ir": _coerce_number(result.summary.get("rank_ic_ir")),
            "mean_rank_ic": _coerce_number(result.summary.get("mean_rank_ic")),
            "ic_positive_rate": _coerce_number(result.summary.get("ic_positive_rate")),
            "valid_timestamps": int(result.summary.get("n_periods", 0)),
        },
        "cross_section": timestamp_coverage,
        "nulls": post_warmup_nulls,
        "signals": signal_payload,
        "checks": checks,
        "flags": flags,
        "overall_status": overall_status,
    }


def _timestamp_coverage_payload(
    *,
    timestamps: pd.Series,
    valid_mask: pd.Series,
    result: AlphaEvaluationResult,
) -> dict[str, Any]:
    if timestamps.empty:
        return {
            "configured_min_cross_section_size": int(result.min_cross_section_size),
            "total_timestamps": 0,
            "valid_timestamps": int(result.summary.get("n_periods", 0)),
            "valid_timestamp_rate": None,
            "min_valid_cross_section_size": None,
            "mean_valid_cross_section_size": None,
            "max_valid_cross_section_size": None,
        }

    valid_frame = pd.DataFrame({"ts_utc": timestamps, "valid": valid_mask.astype(bool)})
    valid_sizes = (
        valid_frame.loc[valid_frame["valid"]]
        .groupby("ts_utc", sort=False)
        .size()
        .astype("int64")
    )
    total_timestamps = int(timestamps.nunique())
    valid_timestamps = int(valid_sizes.shape[0])
    return {
        "configured_min_cross_section_size": int(result.min_cross_section_size),
        "total_timestamps": total_timestamps,
        "valid_timestamps": valid_timestamps,
        "valid_timestamp_rate": (
            None if total_timestamps == 0 else float(valid_timestamps / total_timestamps)
        ),
        "min_valid_cross_section_size": None if valid_sizes.empty else int(valid_sizes.min()),
        "mean_valid_cross_section_size": None if valid_sizes.empty else float(valid_sizes.mean()),
        "max_valid_cross_section_size": None if valid_sizes.empty else int(valid_sizes.max()),
    }


def _post_warmup_null_payload(
    *,
    frame: pd.DataFrame,
    timestamps: pd.Series,
    prediction_column: str,
    forward_return_column: str,
) -> dict[str, Any]:
    if frame.empty or "symbol" not in frame.columns:
        return {
            "post_warmup_row_count": 0,
            "prediction_null_count": 0,
            "prediction_null_rate": None,
            "forward_return_null_count": 0,
            "forward_return_null_rate": None,
        }

    working = frame.copy(deep=True)
    working["_ts_utc"] = timestamps
    working["_prediction"] = _numeric_series(frame, prediction_column)
    working["_forward_return"] = _numeric_series(frame, forward_return_column)
    working["_row_order"] = working.groupby("symbol", sort=False).cumcount()
    post_warmup = working.loc[working["_row_order"] > 0]
    row_count = int(len(post_warmup))
    prediction_null_count = int(post_warmup["_prediction"].isna().sum())
    forward_return_null_count = int(post_warmup["_forward_return"].isna().sum())
    return {
        "post_warmup_row_count": row_count,
        "prediction_null_count": prediction_null_count,
        "prediction_null_rate": _safe_rate(prediction_null_count, row_count),
        "forward_return_null_count": forward_return_null_count,
        "forward_return_null_rate": _safe_rate(forward_return_null_count, row_count),
    }


def _signal_payload(signal_mapping_result: AlphaSignalMappingResult | None) -> dict[str, Any]:
    if signal_mapping_result is None:
        return {
            "enabled": False,
            "policy": None,
            "row_count": 0,
            "timestamp_count": 0,
            "symbol_count": 0,
            "mean_turnover": None,
            "max_turnover": None,
            "mean_gross_exposure": None,
            "max_gross_exposure": None,
            "mean_net_exposure": None,
            "max_abs_net_exposure": None,
            "max_single_name_abs_share": None,
            "mean_active_name_count": None,
            "min_active_name_count": None,
            "max_abs_net_exposure_share": None,
        }

    signals = signal_mapping_result.signals.copy(deep=True)
    signals.attrs = {}
    signals["ts_utc"] = pd.to_datetime(signals["ts_utc"], utc=True, errors="coerce")
    signals["signal"] = pd.to_numeric(signals["signal"], errors="coerce")
    grouped = signals.groupby("ts_utc", sort=False)

    active_name_count = grouped["signal"].apply(lambda values: int(values.fillna(0.0).ne(0.0).sum()))
    gross_exposure = grouped["signal"].apply(lambda values: float(values.abs().sum()))
    net_exposure = grouped["signal"].apply(lambda values: float(values.sum()))
    max_single_name_abs_share = grouped["signal"].apply(_max_single_name_abs_share)
    abs_net_share = []
    ordered = grouped["signal"].sum()
    gross_by_ts = gross_exposure.reindex(ordered.index)
    for ts_utc, net_value in ordered.items():
        gross_value = float(gross_by_ts.loc[ts_utc])
        abs_net_share.append(0.0 if gross_value <= 0.0 else abs(float(net_value)) / gross_value)
    abs_net_share_series = pd.Series(abs_net_share, index=ordered.index, dtype="float64")

    turnover = _signal_turnover_series(signals)
    return {
        "enabled": True,
        "policy": signal_mapping_result.config.policy,
        "row_count": int(signal_mapping_result.row_count),
        "timestamp_count": int(signal_mapping_result.timestamp_count),
        "symbol_count": int(signal_mapping_result.symbol_count),
        "mean_turnover": _coerce_number(turnover.mean() if not turnover.empty else None),
        "max_turnover": _coerce_number(turnover.max() if not turnover.empty else None),
        "mean_gross_exposure": _coerce_number(gross_exposure.mean() if not gross_exposure.empty else None),
        "max_gross_exposure": _coerce_number(gross_exposure.max() if not gross_exposure.empty else None),
        "mean_net_exposure": _coerce_number(net_exposure.mean() if not net_exposure.empty else None),
        "max_abs_net_exposure": _coerce_number(net_exposure.abs().max() if not net_exposure.empty else None),
        "max_single_name_abs_share": _coerce_number(
            max_single_name_abs_share.max() if not max_single_name_abs_share.empty else None
        ),
        "mean_active_name_count": _coerce_number(
            active_name_count.mean() if not active_name_count.empty else None
        ),
        "min_active_name_count": _coerce_number(
            active_name_count.min() if not active_name_count.empty else None
        ),
        "max_abs_net_exposure_share": _coerce_number(
            abs_net_share_series.max() if not abs_net_share_series.empty else None
        ),
    }


def _checks_payload(
    *,
    timestamp_coverage: dict[str, Any],
    post_warmup_nulls: dict[str, Any],
    signal_payload: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    valid_timestamps = int(timestamp_coverage.get("valid_timestamps") or 0)
    valid_timestamp_rate = _coerce_number(timestamp_coverage.get("valid_timestamp_rate"))
    max_null_rate = max(
        value
        for value in (
            _coerce_number(post_warmup_nulls.get("prediction_null_rate")),
            _coerce_number(post_warmup_nulls.get("forward_return_null_rate")),
            0.0,
        )
        if value is not None
    )
    mean_turnover = _coerce_number(signal_payload.get("mean_turnover"))
    max_single_name_abs_share = _coerce_number(signal_payload.get("max_single_name_abs_share"))
    max_abs_net_exposure_share = _coerce_number(signal_payload.get("max_abs_net_exposure_share"))

    checks = {
        "minimum_valid_timestamps": _bounded_check(
            actual_value=float(valid_timestamps),
            warn_threshold=float(_WARN_MIN_VALID_TIMESTAMPS),
            fail_threshold=float(_FAIL_MIN_VALID_TIMESTAMPS),
            direction="gte",
            description="Minimum usable evaluation timestamps after alignment and null filtering.",
        ),
        "valid_timestamp_rate": _bounded_check(
            actual_value=valid_timestamp_rate,
            warn_threshold=_WARN_VALID_TIMESTAMP_RATE,
            fail_threshold=_FAIL_VALID_TIMESTAMP_RATE,
            direction="gte",
            description="Share of timestamps that retained enough valid names for IC evaluation.",
        ),
        "post_warmup_null_limit": _bounded_check(
            actual_value=max_null_rate,
            warn_threshold=_WARN_NULL_RATE,
            fail_threshold=_FAIL_NULL_RATE,
            direction="lte",
            description="Maximum post-warmup null rate across prediction and forward-return fields.",
        ),
    }

    if signal_payload["enabled"]:
        checks["sleeve_turnover"] = _bounded_check(
            actual_value=mean_turnover,
            warn_threshold=_WARN_MEAN_TURNOVER,
            fail_threshold=_FAIL_MEAN_TURNOVER,
            direction="lte",
            description="Average timestamp-to-timestamp sleeve turnover implied by mapped signals.",
        )
        checks["concentration"] = _bounded_check(
            actual_value=max_single_name_abs_share,
            warn_threshold=_WARN_MAX_SINGLE_NAME_SHARE,
            fail_threshold=_FAIL_MAX_SINGLE_NAME_SHARE,
            direction="lte",
            description="Largest single-name share of gross signal exposure at any timestamp.",
        )
        checks["net_exposure_sanity"] = _bounded_check(
            actual_value=max_abs_net_exposure_share,
            warn_threshold=_WARN_MAX_ABS_NET_EXPOSURE_SHARE,
            fail_threshold=_FAIL_MAX_ABS_NET_EXPOSURE_SHARE,
            direction="lte",
            description="Largest absolute net exposure share implied by mapped signals.",
        )
    else:
        checks["signal_mapping_present"] = {
            "actual_value": 0.0,
            "warn_threshold": 1.0,
            "fail_threshold": None,
            "direction": "gte",
            "status": "warn",
            "description": "Signal mapping is recommended when you want tradability QA checks.",
        }
    return checks


def _bounded_check(
    *,
    actual_value: float | None,
    warn_threshold: float | None,
    fail_threshold: float | None,
    direction: str,
    description: str,
) -> dict[str, Any]:
    if actual_value is None:
        status = "warn"
    elif fail_threshold is not None and not _passes_threshold(actual_value, fail_threshold, direction=direction):
        status = "fail"
    elif warn_threshold is not None and not _passes_threshold(actual_value, warn_threshold, direction=direction):
        status = "warn"
    else:
        status = "pass"
    return {
        "actual_value": actual_value,
        "warn_threshold": warn_threshold,
        "fail_threshold": fail_threshold,
        "direction": direction,
        "status": status,
        "description": description,
    }


def _passes_threshold(actual_value: float, threshold: float, *, direction: str) -> bool:
    if direction == "gte":
        return actual_value >= threshold
    if direction == "lte":
        return actual_value <= threshold
    raise ValueError(f"Unsupported threshold direction {direction!r}.")


def _overall_status(checks: dict[str, dict[str, Any]]) -> str:
    statuses = [check["status"] for check in checks.values()]
    if any(status == "fail" for status in statuses):
        return "fail"
    if any(status == "warn" for status in statuses):
        return "warn"
    return "pass"


def _signal_turnover_series(signals: pd.DataFrame) -> pd.Series:
    if signals.empty:
        return pd.Series(dtype="float64")
    ordered = signals.loc[:, ["symbol", "ts_utc", "signal"]].copy(deep=True)
    ordered["signal"] = pd.to_numeric(ordered["signal"], errors="coerce").fillna(0.0)
    ordered = ordered.sort_values(["symbol", "ts_utc"], kind="stable").reset_index(drop=True)
    ordered["abs_delta"] = (
        ordered.groupby("symbol", sort=False)["signal"].diff().abs().fillna(0.0)
    )
    turnover = (
        ordered.groupby("ts_utc", sort=False)["abs_delta"].mean().astype("float64")
    )
    return turnover


def _max_single_name_abs_share(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0).abs()
    gross = float(numeric.sum())
    if gross <= 0.0:
        return 0.0
    return float(numeric.max() / gross)


def _timestamp_series(frame: pd.DataFrame) -> pd.Series:
    if "ts_utc" not in frame.columns:
        return pd.Series(dtype="datetime64[ns, UTC]")
    return pd.to_datetime(frame["ts_utc"], utc=True, errors="coerce")


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").astype("float64")


def _safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def _coerce_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _format_timestamp(value: Any) -> str | None:
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat().replace("+00:00", "Z")


__all__ = ["generate_alpha_qa_summary"]
