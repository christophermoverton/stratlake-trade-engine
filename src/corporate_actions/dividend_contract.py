from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pandas as pd


DIVIDEND_SCHEMA_NAME = "corporate_actions.dividends.v1"
DIVIDEND_SCHEMA_VERSION = "1.0.0"

DIVIDEND_REQUIRED_FIELDS: tuple[str, ...] = (
    "symbol",
    "event_type",
    "ex_date",
    "process_date",
    "source",
    "source_event_id",
    "source_payload_fingerprint",
    "as_of_date",
    "schema_version",
)

DIVIDEND_REQUIRED_NULLABLE_FIELDS: tuple[str, ...] = (
    "declaration_date",
    "record_date",
    "payable_date",
    "cash_amount",
    "stock_amount",
    "currency",
    "raw_payload",
)

DIVIDEND_SUPPORTED_EVENT_TYPES: tuple[str, ...] = ("cash_dividend", "stock_dividend")

DIVIDEND_PRIMARY_KEY_FIELDS: tuple[str, ...] = (
    "symbol",
    "event_type",
    "source_event_id",
    "ex_date",
    "process_date",
    "source",
)

DIVIDEND_FALLBACK_KEY_FIELDS: tuple[str, ...] = (
    "symbol",
    "event_type",
    "ex_date",
    "process_date",
    "cash_amount",
    "stock_amount",
    "currency",
    "source_payload_fingerprint",
)

DIVIDEND_UPSTREAM_FIELD_MAPPING: dict[str, str | None] = {
    "symbol": "symbol",
    "event_type": "corporate_action_type",
    "ex_date": "ex_date",
    "process_date": "process_date",
    "source": "source",
    "source_event_id": "corporate_action_id",
    "source_payload_fingerprint": "source_payload_hash",
    "as_of_date": "process_date",
    "schema_version": None,
    "declaration_date": "declaration_date",
    "record_date": "record_date",
    "payable_date": "payable_date",
    "cash_amount": "cash_amount",
    "stock_amount": "stock_amount",
    "currency": "currency",
    "raw_payload": "raw",
}

_DATE_FIELDS: tuple[str, ...] = ("ex_date", "process_date", "as_of_date")
_NULLABLE_DATE_FIELDS: tuple[str, ...] = ("declaration_date", "record_date", "payable_date")
_ALL_REQUIRED_COLUMNS: tuple[str, ...] = DIVIDEND_REQUIRED_FIELDS + DIVIDEND_REQUIRED_NULLABLE_FIELDS


class DividendContractError(ValueError):
    """Raised when dividend event evidence violates the StratLake contract."""


def build_dividend_schema_contract() -> dict[str, Any]:
    """Return the deterministic machine-readable dividend event contract."""

    return {
        "schema_name": DIVIDEND_SCHEMA_NAME,
        "schema_version": DIVIDEND_SCHEMA_VERSION,
        "required_fields": list(DIVIDEND_REQUIRED_FIELDS),
        "required_nullable_fields": list(DIVIDEND_REQUIRED_NULLABLE_FIELDS),
        "supported_event_types": list(DIVIDEND_SUPPORTED_EVENT_TYPES),
        "primary_key_fields": list(DIVIDEND_PRIMARY_KEY_FIELDS),
        "fallback_key_fields": list(DIVIDEND_FALLBACK_KEY_FIELDS),
        "upstream_field_mapping": dict(sorted(DIVIDEND_UPSTREAM_FIELD_MAPPING.items())),
        "fields": [
            {
                "name": field,
                "required": True,
                "nullable": field in DIVIDEND_REQUIRED_NULLABLE_FIELDS,
            }
            for field in _ALL_REQUIRED_COLUMNS
        ],
        "non_goals": [
            "OHLCV price adjustment",
            "adjusted price dataset creation",
            "total-return reconstruction",
            "dividend reinvestment modeling",
            "strategy, alpha, portfolio, or backtest mutation",
            "live ingestion or external-service access",
        ],
    }


def serialize_dividend_schema_contract() -> str:
    """Serialize the schema contract with stable JSON formatting."""

    return json.dumps(build_dividend_schema_contract(), indent=2, sort_keys=True) + "\n"


def validate_dividend_event_schema(data: pd.DataFrame | Mapping[str, Any]) -> pd.DataFrame:
    """
    Validate dividend event evidence and return a deterministic copy.

    This is a contract-only validator. It does not import upstream data, register
    catalog evidence, adjust prices, or mutate research artifacts.
    """

    normalized = _coerce_to_dataframe(data)

    missing_required = [column for column in DIVIDEND_REQUIRED_FIELDS if column not in normalized.columns]
    if missing_required:
        _raise_missing("required dividend event columns", missing_required)

    missing_nullable = [
        column for column in DIVIDEND_REQUIRED_NULLABLE_FIELDS if column not in normalized.columns
    ]
    if missing_nullable:
        _raise_missing("required nullable dividend event columns", missing_nullable)

    missing_primary_key = [column for column in DIVIDEND_PRIMARY_KEY_FIELDS if column not in normalized.columns]
    if missing_primary_key:
        _raise_missing("primary-key dividend event columns", missing_primary_key)

    if normalized.empty:
        return normalized.loc[:, list(_ALL_REQUIRED_COLUMNS)].copy()

    _validate_non_null_columns(normalized)
    _validate_event_types(normalized)
    _validate_schema_version(normalized)
    _validate_date_columns(normalized)
    _validate_duplicate_primary_keys(normalized)

    return normalized.loc[:, list(_ALL_REQUIRED_COLUMNS)].copy()


def _coerce_to_dataframe(data: pd.DataFrame | Mapping[str, Any]) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, Mapping):
        return pd.DataFrame([dict(data)])
    raise DividendContractError("dividend event evidence must be a pandas DataFrame or mapping.")


def _raise_missing(label: str, columns: list[str]) -> None:
    formatted = ", ".join(repr(column) for column in columns)
    raise DividendContractError(f"missing {label}: {formatted}.")


def _validate_non_null_columns(df: pd.DataFrame) -> None:
    for column in DIVIDEND_REQUIRED_FIELDS:
        null_count = int(df[column].isna().sum())
        if null_count:
            raise DividendContractError(
                f"non-nullable dividend event column {column!r} contains {null_count} null value(s)."
            )


def _validate_event_types(df: pd.DataFrame) -> None:
    observed = set(df["event_type"].astype("string").dropna().tolist())
    invalid = sorted(observed - set(DIVIDEND_SUPPORTED_EVENT_TYPES))
    if invalid:
        supported = ", ".join(repr(event_type) for event_type in DIVIDEND_SUPPORTED_EVENT_TYPES)
        raise DividendContractError(
            f"unsupported dividend event_type value(s): {invalid}. Supported values: {supported}."
        )


def _validate_schema_version(df: pd.DataFrame) -> None:
    observed = sorted(set(df["schema_version"].astype("string").dropna().tolist()))
    invalid = [version for version in observed if version != DIVIDEND_SCHEMA_VERSION]
    if invalid:
        raise DividendContractError(
            f"unsupported dividend schema_version value(s): {invalid}. "
            f"Expected {DIVIDEND_SCHEMA_VERSION!r}."
        )


def _validate_date_columns(df: pd.DataFrame) -> None:
    for column in _DATE_FIELDS:
        _parse_date_series(df[column], column_name=column, nullable=False)
    for column in _NULLABLE_DATE_FIELDS:
        _parse_date_series(df[column], column_name=column, nullable=True)


def _parse_date_series(series: pd.Series, *, column_name: str, nullable: bool) -> None:
    values = series.dropna() if nullable else series
    if values.empty:
        return
    try:
        pd.to_datetime(values, errors="raise")
    except Exception as exc:
        raise DividendContractError(f"dividend date column {column_name!r} is not parseable: {exc}") from exc


def _validate_duplicate_primary_keys(df: pd.DataFrame) -> None:
    duplicate_mask = df.duplicated(subset=list(DIVIDEND_PRIMARY_KEY_FIELDS), keep=False)
    if not duplicate_mask.any():
        return

    duplicate_key = df.loc[duplicate_mask, list(DIVIDEND_PRIMARY_KEY_FIELDS)].iloc[0].to_dict()
    raise DividendContractError(
        "dividend event evidence contains duplicate primary-key rows. "
        f"First duplicate key: {duplicate_key}."
    )
