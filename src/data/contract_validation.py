from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd


@dataclass(frozen=True)
class BarsContract:
    """
    Consumer-side contract validation for curated OHLCV bars.

    Fail-fast checks:
      - required columns present
      - ts_utc timezone-aware and normalized to UTC (or optionally assume naive is UTC)
      - no nulls in primary key columns
    """
    required_columns: Tuple[str, ...] = (
        "symbol",
        "ts_utc",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "source",
        "timeframe",
    )
    primary_key: Tuple[str, ...] = ("symbol", "ts_utc", "timeframe")

    def validate(
        self,
        df: pd.DataFrame,
        *,
        strict: bool = True,
        normalize_ts_utc: bool = True,
    ) -> List[str]:
        """
        Returns a list of warnings (possibly empty).
        Raises ValueError on violations when strict=True.
        """
        warnings: List[str] = []

        # required columns present
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            msg = f"Missing required columns: {missing}"
            if strict:
                raise ValueError(msg)
            warnings.append(msg)
            
        if df.empty:
            return warnings

        # bail early if ts_utc missing
        if "ts_utc" not in df.columns:
            return warnings

        # ts_utc tz-aware + UTC normalized (optionally)
        if normalize_ts_utc:
            df["ts_utc"] = _normalize_ts_utc_to_utc(df["ts_utc"], strict=strict)

        ts = df["ts_utc"]
        if not _is_tz_aware_series(ts):
            msg = "ts_utc must be timezone-aware (UTC)."
            if strict:
                raise ValueError(msg)
            warnings.append(msg)
        else:
            try:
                _ = ts.dt.tz_convert("UTC")
            except Exception as e:
                msg = f"ts_utc could not be tz-converted to UTC: {e}"
                if strict:
                    raise ValueError(msg)
                warnings.append(msg)

        # no nulls in PK
        for col in self.primary_key:
            if col not in df.columns:
                msg = f"Primary key column missing: {col}"
                if strict:
                    raise ValueError(msg)
                warnings.append(msg)
                continue

            nulls = int(df[col].isna().sum())
            if nulls > 0:
                msg = f"Nulls in primary key column '{col}': {nulls}"
                if strict:
                    raise ValueError(msg)
                warnings.append(msg)

        return warnings


def _is_tz_aware_series(s: pd.Series) -> bool:
    return hasattr(s.dtype, "tz") and s.dtype.tz is not None


def _normalize_ts_utc_to_utc(ts: pd.Series, *, strict: bool) -> pd.Series:
    # parse strings/objects
    try:
        parsed = pd.to_datetime(ts, errors="raise")
    except Exception as e:
        raise ValueError(f"ts_utc could not be parsed to datetime: {e}")

    # tz-naive policy
    if not _is_tz_aware_series(parsed):
        if strict:
            raise ValueError(
                "ts_utc is tz-naive. Provide tz-aware timestamps or run strict=False to assume UTC."
            )
        parsed = parsed.dt.tz_localize("UTC")

    # normalize to UTC
    try:
        parsed = parsed.dt.tz_convert("UTC")
    except Exception as e:
        raise ValueError(f"ts_utc could not be tz-converted to UTC: {e}")

    return parsed