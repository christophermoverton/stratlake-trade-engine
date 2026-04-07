from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Literal

import pandas as pd

SUPPORTED_ALPHA_SPLIT_MODES = frozenset({"fixed", "rolling"})
SUPPORTED_WINDOW_UNITS = {"D", "W", "M", "Y", "H", "MIN", "T"}

AlphaSplitMode = Literal["fixed", "rolling"]


class AlphaTimeSplitError(ValueError):
    """Raised when alpha split boundaries are invalid or ambiguous."""


@dataclass(frozen=True)
class AlphaTimeSplit:
    """Deterministic alpha-model time split using half-open UTC intervals."""

    split_id: str
    train_start: pd.Timestamp | None
    train_end: pd.Timestamp
    predict_start: pd.Timestamp
    predict_end: pd.Timestamp
    mode: AlphaSplitMode
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_alpha_time_split(self)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serialization-friendly representation of the split."""

        return {
            "split_id": self.split_id,
            "mode": self.mode,
            "train_start": _format_timestamp(self.train_start),
            "train_end": _format_timestamp(self.train_end),
            "predict_start": _format_timestamp(self.predict_start),
            "predict_end": _format_timestamp(self.predict_end),
            "metadata": dict(self.metadata),
        }


def validate_alpha_time_split(split: AlphaTimeSplit) -> AlphaTimeSplit:
    """Validate one alpha time split and return it unchanged when valid."""

    if not isinstance(split, AlphaTimeSplit):
        raise TypeError("split must be an AlphaTimeSplit instance.")

    _validate_mode(split.mode)
    _require_utc_timestamp(split.train_start, field_name="train_start", allow_none=True)
    _require_utc_timestamp(split.train_end, field_name="train_end")
    _require_utc_timestamp(split.predict_start, field_name="predict_start")
    _require_utc_timestamp(split.predict_end, field_name="predict_end")
    _validate_boundaries(
        train_start=split.train_start,
        train_end=split.train_end,
        predict_start=split.predict_start,
        predict_end=split.predict_end,
    )
    return split


def make_alpha_fixed_split(
    *,
    train_start: str | pd.Timestamp | None,
    train_end: str | pd.Timestamp,
    predict_start: str | pd.Timestamp,
    predict_end: str | pd.Timestamp,
    split_id: str | None = None,
    metadata: dict[str, object] | None = None,
) -> AlphaTimeSplit:
    """Build one validated fixed alpha split with half-open UTC semantics."""

    normalized_train_start = _coerce_timestamp(train_start, field_name="train_start", allow_none=True)
    normalized_train_end = _coerce_timestamp(train_end, field_name="train_end")
    normalized_predict_start = _coerce_timestamp(predict_start, field_name="predict_start")
    normalized_predict_end = _coerce_timestamp(predict_end, field_name="predict_end")
    return _build_alpha_split(
        mode="fixed",
        train_start=normalized_train_start,
        train_end=normalized_train_end,
        predict_start=normalized_predict_start,
        predict_end=normalized_predict_end,
        split_id=split_id,
        metadata=metadata,
    )


def generate_alpha_rolling_splits(
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    train_window: str | pd.Timedelta,
    predict_window: str | pd.Timedelta,
    step: str | pd.Timedelta | None = None,
) -> list[AlphaTimeSplit]:
    """Generate deterministic rolling alpha splits using UTC half-open windows."""

    overall_start = _coerce_timestamp(start, field_name="start")
    overall_end = _coerce_timestamp(end, field_name="end")
    if overall_start >= overall_end:
        raise AlphaTimeSplitError("start must be earlier than end.")

    normalized_train_window = _coerce_window(train_window, field_name="train_window")
    normalized_predict_window = _coerce_window(predict_window, field_name="predict_window")
    normalized_step = _coerce_window(
        normalized_predict_window if step is None else step,
        field_name="step",
    )

    initial_train_end = overall_start + normalized_train_window
    initial_predict_start = initial_train_end
    initial_predict_end = initial_predict_start + normalized_predict_window
    if initial_predict_end > overall_end:
        raise AlphaTimeSplitError(
            "The requested range does not contain enough data to form at least one rolling alpha split."
        )

    splits: list[AlphaTimeSplit] = []
    index = 0
    train_start = overall_start
    train_end = initial_train_end
    predict_start = initial_predict_start
    predict_end = initial_predict_end

    while predict_end <= overall_end:
        splits.append(
            _build_alpha_split(
                mode="rolling",
                train_start=train_start,
                train_end=train_end,
                predict_start=predict_start,
                predict_end=predict_end,
                split_id=f"rolling_{index:04d}",
                metadata={"sequence_index": index},
            )
        )
        index += 1
        train_start = train_start + normalized_step
        train_end = train_end + normalized_step
        predict_start = predict_start + normalized_step
        predict_end = predict_end + normalized_step

    if not splits:
        raise AlphaTimeSplitError(
            "The requested range does not contain enough data to form at least one rolling alpha split."
        )

    return splits


def _build_alpha_split(
    *,
    mode: AlphaSplitMode,
    train_start: pd.Timestamp | None,
    train_end: pd.Timestamp,
    predict_start: pd.Timestamp,
    predict_end: pd.Timestamp,
    split_id: str | None,
    metadata: dict[str, object] | None,
) -> AlphaTimeSplit:
    _validate_mode(mode)
    _validate_boundaries(
        train_start=train_start,
        train_end=train_end,
        predict_start=predict_start,
        predict_end=predict_end,
    )

    normalized_metadata = _build_metadata(
        mode=mode,
        train_start=train_start,
        train_end=train_end,
        predict_start=predict_start,
        predict_end=predict_end,
        metadata=metadata,
    )
    resolved_split_id = _resolve_split_id(
        mode=mode,
        train_start=train_start,
        train_end=train_end,
        predict_start=predict_start,
        predict_end=predict_end,
        split_id=split_id,
    )
    return AlphaTimeSplit(
        split_id=resolved_split_id,
        train_start=train_start,
        train_end=train_end,
        predict_start=predict_start,
        predict_end=predict_end,
        mode=mode,
        metadata=normalized_metadata,
    )


def _validate_mode(mode: str) -> None:
    if mode not in SUPPORTED_ALPHA_SPLIT_MODES:
        supported = ", ".join(sorted(SUPPORTED_ALPHA_SPLIT_MODES))
        raise AlphaTimeSplitError(f"mode must be one of the supported alpha split modes: {supported}.")


def _validate_boundaries(
    *,
    train_start: pd.Timestamp | None,
    train_end: pd.Timestamp,
    predict_start: pd.Timestamp,
    predict_end: pd.Timestamp,
) -> None:
    if train_start is not None and train_start >= train_end:
        raise AlphaTimeSplitError("train_start must be earlier than train_end.")
    if predict_start >= predict_end:
        raise AlphaTimeSplitError("predict_start must be earlier than predict_end.")
    if train_end > predict_start:
        raise AlphaTimeSplitError(
            "Alpha split windows must not overlap: train_end must be less than or equal to predict_start."
        )


def _build_metadata(
    *,
    mode: AlphaSplitMode,
    train_start: pd.Timestamp | None,
    train_end: pd.Timestamp,
    predict_start: pd.Timestamp,
    predict_end: pd.Timestamp,
    metadata: dict[str, object] | None,
) -> dict[str, object]:
    normalized = dict(metadata or {})
    normalized.setdefault("mode", mode)
    normalized.setdefault("window_semantics", {"train": "[train_start, train_end)", "predict": "[predict_start, predict_end)"})
    normalized.setdefault("train_start", _format_timestamp(train_start))
    normalized.setdefault("train_end", _format_timestamp(train_end))
    normalized.setdefault("predict_start", _format_timestamp(predict_start))
    normalized.setdefault("predict_end", _format_timestamp(predict_end))
    normalized.setdefault(
        "train_duration_seconds",
        None if train_start is None else int((train_end - train_start).total_seconds()),
    )
    normalized.setdefault("predict_duration_seconds", int((predict_end - predict_start).total_seconds()))
    return normalized


def _resolve_split_id(
    *,
    mode: AlphaSplitMode,
    train_start: pd.Timestamp | None,
    train_end: pd.Timestamp,
    predict_start: pd.Timestamp,
    predict_end: pd.Timestamp,
    split_id: str | None,
) -> str:
    if split_id is not None:
        if not isinstance(split_id, str) or not split_id.strip():
            raise AlphaTimeSplitError("split_id must be a non-empty string when provided.")
        return split_id.strip()

    payload = json.dumps(
        {
            "mode": mode,
            "train_start": _format_timestamp(train_start),
            "train_end": _format_timestamp(train_end),
            "predict_start": _format_timestamp(predict_start),
            "predict_end": _format_timestamp(predict_end),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"{mode}_{digest}"


def _coerce_timestamp(
    value: str | pd.Timestamp | None,
    *,
    field_name: str,
    allow_none: bool = False,
) -> pd.Timestamp | None:
    if value is None:
        if allow_none:
            return None
        raise AlphaTimeSplitError(f"{field_name} is required.")

    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise AlphaTimeSplitError(f"{field_name} must be a valid timestamp or timestamp-like string.") from exc

    if pd.isna(timestamp):
        raise AlphaTimeSplitError(f"{field_name} must be a valid timestamp or timestamp-like string.")

    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _coerce_window(value: str | pd.Timedelta | pd.DateOffset, *, field_name: str) -> pd.DateOffset:
    if isinstance(value, pd.DateOffset):
        return value

    if isinstance(value, pd.Timedelta):
        if value <= pd.Timedelta(0):
            raise AlphaTimeSplitError(f"{field_name} must be greater than zero.")
        return pd.DateOffset(seconds=int(value.total_seconds()))

    if not isinstance(value, str) or not value.strip():
        raise AlphaTimeSplitError(
            f"{field_name} must be a non-empty duration string such as '30D', '4W', '3M', '1Y', or '12H'."
        )

    compact = value.replace(" ", "")
    match = re.fullmatch(r"(?P<count>\d+)(?P<unit>MIN|[DWMYHT])", compact, re.IGNORECASE)
    if match is None:
        raise AlphaTimeSplitError(
            f"{field_name} must use a positive duration like '30D', '4W', '3M', '1Y', or '12H'."
        )

    count = int(match.group("count"))
    unit = match.group("unit").upper()
    if count <= 0:
        raise AlphaTimeSplitError(f"{field_name} must be greater than zero.")
    if unit not in SUPPORTED_WINDOW_UNITS:
        supported = ", ".join(sorted(SUPPORTED_WINDOW_UNITS))
        raise AlphaTimeSplitError(f"{field_name} must use one of the supported units: {supported}.")

    if unit == "D":
        return pd.DateOffset(days=count)
    if unit == "W":
        return pd.DateOffset(weeks=count)
    if unit == "M":
        return pd.DateOffset(months=count)
    if unit == "Y":
        return pd.DateOffset(years=count)
    if unit == "H":
        return pd.DateOffset(hours=count)
    return pd.DateOffset(minutes=count)


def _require_utc_timestamp(
    value: pd.Timestamp | None,
    *,
    field_name: str,
    allow_none: bool = False,
) -> None:
    if value is None:
        if allow_none:
            return
        raise AlphaTimeSplitError(f"{field_name} is required.")
    if not isinstance(value, pd.Timestamp):
        raise TypeError(f"{field_name} must be a pandas Timestamp.")
    if value.tzinfo is None:
        raise AlphaTimeSplitError(f"{field_name} must be timezone-aware and normalized to UTC.")
    if value.tzname() != "UTC":
        raise AlphaTimeSplitError(f"{field_name} must be timezone-aware and normalized to UTC.")


def _format_timestamp(value: pd.Timestamp | None) -> str | None:
    if value is None:
        return None
    return value.tz_convert("UTC").isoformat().replace("+00:00", "Z")
