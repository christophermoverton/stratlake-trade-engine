from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from src.artifacts.safety import portable_path
from src.corporate_actions.dividend_contract import (
    DIVIDEND_PRIMARY_KEY_FIELDS,
    DIVIDEND_REQUIRED_FIELDS,
    DIVIDEND_REQUIRED_NULLABLE_FIELDS,
    DIVIDEND_SCHEMA_VERSION,
    DIVIDEND_UPSTREAM_FIELD_MAPPING,
    DividendContractError,
    validate_dividend_event_schema,
)


DIVIDEND_SORT_COLUMNS: tuple[str, ...] = (
    "symbol",
    "ex_date",
    "event_type",
    "process_date",
    "source_event_id",
    "source",
)


@dataclass(frozen=True)
class DividendImportResult:
    source_data_path: str
    source_metadata_path: str
    output_root: str
    start: str
    end: str
    input_row_count: int
    normalized_row_count: int
    written_row_count: int
    filtered_row_count: int
    duplicate_event_count: int
    invalid_event_count: int
    symbols: tuple[str, ...]
    partitions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_data_path": self.source_data_path,
            "source_metadata_path": self.source_metadata_path,
            "output_root": self.output_root,
            "start": self.start,
            "end": self.end,
            "input_row_count": self.input_row_count,
            "normalized_row_count": self.normalized_row_count,
            "written_row_count": self.written_row_count,
            "filtered_row_count": self.filtered_row_count,
            "duplicate_event_count": self.duplicate_event_count,
            "invalid_event_count": self.invalid_event_count,
            "symbols": list(self.symbols),
            "partitions": list(self.partitions),
        }


class DividendImportError(ValueError):
    """Raised when local dividend event import cannot complete deterministically."""


def read_upstream_dividend_artifacts(
    source_data_path: str | Path,
    source_metadata_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read local upstream dividend Parquet and metadata JSON artifacts."""

    data_path = _resolve_local_file(source_data_path, label="source dividend data")
    metadata_path = _resolve_local_file(source_metadata_path, label="source dividend metadata")

    data = pd.read_parquet(data_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise DividendImportError("source dividend metadata JSON must contain an object.")
    return data, metadata


def normalize_upstream_dividend_events(upstream: pd.DataFrame) -> pd.DataFrame:
    """Map upstream dividend artifacts into the StratLake dividend event contract."""

    if not isinstance(upstream, pd.DataFrame):
        raise DividendImportError("upstream dividend events must be provided as a pandas DataFrame.")

    normalized = pd.DataFrame(index=upstream.index)
    for stratlake_field in DIVIDEND_REQUIRED_FIELDS + DIVIDEND_REQUIRED_NULLABLE_FIELDS:
        upstream_field = DIVIDEND_UPSTREAM_FIELD_MAPPING[stratlake_field]
        if stratlake_field == "schema_version":
            normalized[stratlake_field] = DIVIDEND_SCHEMA_VERSION
        elif stratlake_field == "as_of_date" and stratlake_field in upstream.columns:
            normalized[stratlake_field] = upstream[stratlake_field]
        elif upstream_field is not None and upstream_field in upstream.columns:
            normalized[stratlake_field] = upstream[upstream_field]
        else:
            normalized[stratlake_field] = pd.NA

    normalized["symbol"] = _normalize_string_series(normalized["symbol"], uppercase=True)
    for column in ("event_type", "source", "source_event_id", "source_payload_fingerprint", "currency"):
        normalized[column] = _normalize_string_series(normalized[column])

    for column in ("ex_date", "process_date", "as_of_date"):
        normalized[column] = _normalize_date_series(normalized[column], column_name=column, nullable=False)
    for column in ("declaration_date", "record_date", "payable_date"):
        normalized[column] = _normalize_date_series(normalized[column], column_name=column, nullable=True)

    normalized["raw_payload"] = normalized["raw_payload"].map(_normalize_raw_payload)
    normalized["year"] = pd.to_datetime(normalized["ex_date"], errors="raise").dt.strftime("%Y")
    return normalized


def filter_dividend_events_by_ex_date(
    events: pd.DataFrame,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    """Apply half-open StratLake import windows: start <= ex_date < end."""

    start_date = _normalize_boundary_date(start, label="start")
    end_date = _normalize_boundary_date(end, label="end")
    if start_date >= end_date:
        raise DividendImportError("dividend import start must be earlier than end.")

    ex_dates = pd.to_datetime(events["ex_date"], errors="raise")
    mask = (ex_dates >= pd.Timestamp(start_date)) & (ex_dates < pd.Timestamp(end_date))
    return events.loc[mask].copy()


def write_dividend_event_dataset(events: pd.DataFrame, output_root: str | Path) -> tuple[str, ...]:
    """Write deterministic symbol/year partitioned dividend event Parquet output."""

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    if events.empty:
        return ()

    required_for_write = set(DIVIDEND_REQUIRED_FIELDS + DIVIDEND_REQUIRED_NULLABLE_FIELDS + ("year",))
    missing = sorted(required_for_write - set(events.columns))
    if missing:
        formatted = ", ".join(repr(column) for column in missing)
        raise DividendImportError(f"dividend events are missing required writer columns: {formatted}.")

    ordered = _sort_dividend_events(events)
    partitions: list[str] = []
    for (symbol, year), partition in ordered.groupby(["symbol", "year"], sort=True):
        partition_path = root / f"symbol={symbol}" / f"year={year}"
        partition_path.mkdir(parents=True, exist_ok=True)
        for existing in sorted(partition_path.glob("part-*.parquet")):
            existing.unlink()

        output_columns = [
            column
            for column in DIVIDEND_REQUIRED_FIELDS + DIVIDEND_REQUIRED_NULLABLE_FIELDS
            if column != "symbol"
        ]
        partition.loc[:, output_columns].reset_index(drop=True).to_parquet(
            partition_path / "part-0.parquet",
            index=False,
        )
        partitions.append(portable_path(partition_path, roots=(Path.cwd(), root)))

    return tuple(sorted(partitions))


def import_dividend_events(
    *,
    source_data_path: str | Path,
    source_metadata_path: str | Path,
    output_root: str | Path,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    strict: bool = True,
) -> DividendImportResult:
    """Import local upstream dividend artifacts into StratLake-curated event evidence."""

    source_data = _resolve_local_file(source_data_path, label="source dividend data")
    source_metadata = _resolve_local_file(source_metadata_path, label="source dividend metadata")
    output = Path(output_root)
    start_date = _normalize_boundary_date(start, label="start")
    end_date = _normalize_boundary_date(end, label="end")
    if start_date >= end_date:
        raise DividendImportError("dividend import start must be earlier than end.")

    upstream, _metadata = read_upstream_dividend_artifacts(source_data, source_metadata)
    normalized = normalize_upstream_dividend_events(upstream)
    filtered = filter_dividend_events_by_ex_date(normalized, start=start_date, end=end_date)

    preliminary_duplicate_event_count = _duplicate_event_count(filtered)
    if preliminary_duplicate_event_count and strict:
        duplicate_key = _first_duplicate_key(filtered)
        raise DividendImportError(
            "dividend import contains duplicate primary-key rows. "
            f"First duplicate key: {duplicate_key}."
        )

    valid_events, invalid_event_count = _validate_or_filter_invalid_events(filtered, strict=strict)
    duplicate_event_count = _duplicate_event_count(valid_events)
    if duplicate_event_count:
        valid_events = valid_events.drop_duplicates(subset=list(DIVIDEND_PRIMARY_KEY_FIELDS), keep="first")

    valid_events = _sort_dividend_events(valid_events).reset_index(drop=True)
    validate_dividend_event_schema(valid_events.drop(columns=["year"], errors="ignore"))
    partitions = write_dividend_event_dataset(valid_events, output)

    symbols = tuple(sorted(valid_events["symbol"].dropna().astype("string").unique().tolist()))
    return DividendImportResult(
        source_data_path=portable_path(source_data, roots=(Path.cwd(),)),
        source_metadata_path=portable_path(source_metadata, roots=(Path.cwd(),)),
        output_root=portable_path(output, roots=(Path.cwd(),)),
        start=start_date,
        end=end_date,
        input_row_count=int(len(upstream)),
        normalized_row_count=int(len(normalized)),
        written_row_count=int(len(valid_events)),
        filtered_row_count=int(len(normalized) - len(filtered)),
        duplicate_event_count=duplicate_event_count,
        invalid_event_count=invalid_event_count,
        symbols=symbols,
        partitions=partitions,
    )


def _resolve_local_file(path: str | Path, *, label: str) -> Path:
    raw = str(path)
    if "://" in raw:
        parsed = urlparse(raw)
        if parsed.scheme != "file":
            raise DividendImportError(f"{label} path must be a local file path, not a URI: {raw!r}.")
        resolved = Path(parsed.path)
    else:
        resolved = Path(path)
    if not resolved.exists():
        raise DividendImportError(f"{label} path does not exist: {raw!r}.")
    if not resolved.is_file():
        raise DividendImportError(f"{label} path must be a file: {raw!r}.")
    return resolved


def _normalize_string_series(series: pd.Series, *, uppercase: bool = False) -> pd.Series:
    normalized = series.astype("string").str.strip()
    if uppercase:
        normalized = normalized.str.upper()
    return normalized.mask(normalized == "", pd.NA)


def _normalize_date_series(series: pd.Series, *, column_name: str, nullable: bool) -> pd.Series:
    values = series.copy()
    if nullable:
        non_null = values.dropna()
        if non_null.empty:
            return values.where(values.isna(), pd.NA)
        parsed = pd.to_datetime(non_null, errors="raise", format="mixed", utc=True)
        normalized = pd.Series(pd.NA, index=values.index, dtype="string")
        normalized.loc[non_null.index] = parsed.dt.strftime("%Y-%m-%d")
        return normalized

    parsed = pd.to_datetime(values, errors="raise", format="mixed", utc=True)
    if parsed.isna().any():
        null_count = int(parsed.isna().sum())
        raise DividendImportError(f"date column {column_name!r} contains {null_count} null value(s).")
    return parsed.dt.strftime("%Y-%m-%d")


def _normalize_boundary_date(value: str | pd.Timestamp, *, label: str) -> str:
    try:
        parsed = pd.to_datetime(value, errors="raise", format="mixed", utc=True)
    except Exception as exc:
        raise DividendImportError(f"dividend import {label} date is not parseable: {value!r}.") from exc
    if pd.isna(parsed):
        raise DividendImportError(f"dividend import {label} date must not be null.")
    return pd.Timestamp(parsed).strftime("%Y-%m-%d")


def _normalize_raw_payload(value: Any) -> Any:
    if value is None:
        return pd.NA
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return pd.NA
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return stripped
        return json.dumps(parsed, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if isinstance(value, Mapping) or isinstance(value, list | tuple):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if pd.isna(value):
        return pd.NA
    return str(value)


def _validate_or_filter_invalid_events(events: pd.DataFrame, *, strict: bool) -> tuple[pd.DataFrame, int]:
    try:
        validate_dividend_event_schema(events.drop(columns=["year"], errors="ignore"))
    except DividendContractError:
        if strict:
            raise
    else:
        return events.copy(), 0

    valid_indices: list[Any] = []
    invalid_count = 0
    for index, row in events.iterrows():
        row_frame = pd.DataFrame([row.drop(labels=["year"], errors="ignore")])
        try:
            validate_dividend_event_schema(row_frame)
        except DividendContractError:
            invalid_count += 1
        else:
            valid_indices.append(index)

    return events.loc[valid_indices].copy(), invalid_count


def _duplicate_event_count(events: pd.DataFrame) -> int:
    if events.empty:
        return 0
    duplicate_mask = events.duplicated(subset=list(DIVIDEND_PRIMARY_KEY_FIELDS), keep=False)
    return int(duplicate_mask.sum())


def _first_duplicate_key(events: pd.DataFrame) -> dict[str, Any]:
    duplicate_mask = events.duplicated(subset=list(DIVIDEND_PRIMARY_KEY_FIELDS), keep=False)
    if not duplicate_mask.any():
        return {}
    return events.loc[duplicate_mask, list(DIVIDEND_PRIMARY_KEY_FIELDS)].iloc[0].to_dict()


def _sort_dividend_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    return events.sort_values(list(DIVIDEND_SORT_COLUMNS), kind="mergesort").reset_index(drop=True)
