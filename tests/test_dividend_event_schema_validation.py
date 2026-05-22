from __future__ import annotations

import pandas as pd
import pytest

from src.corporate_actions.dividend_contract import (
    DIVIDEND_SCHEMA_VERSION,
    DividendContractError,
    validate_dividend_event_schema,
)


def _dividend_event(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "symbol": "AAPL",
        "event_type": "cash_dividend",
        "ex_date": "2026-02-10",
        "process_date": "2026-02-01",
        "source": "fintech-market-ingestion",
        "source_event_id": "ca_001",
        "source_payload_fingerprint": "sha256:abc123",
        "as_of_date": "2026-02-01",
        "schema_version": DIVIDEND_SCHEMA_VERSION,
        "declaration_date": "2026-01-20",
        "record_date": "2026-02-11",
        "payable_date": "2026-02-20",
        "cash_amount": 0.24,
        "stock_amount": None,
        "currency": "USD",
        "raw_payload": {"corporate_action_id": "ca_001"},
    }
    record.update(overrides)
    return record


def test_validate_dividend_event_schema_accepts_cash_dividend() -> None:
    normalized = validate_dividend_event_schema(pd.DataFrame([_dividend_event()]))

    assert normalized.loc[0, "event_type"] == "cash_dividend"
    assert normalized.loc[0, "cash_amount"] == pytest.approx(0.24)


def test_validate_dividend_event_schema_accepts_stock_dividend() -> None:
    normalized = validate_dividend_event_schema(
        _dividend_event(
            event_type="stock_dividend",
            source_event_id="ca_stock_001",
            cash_amount=None,
            stock_amount=0.05,
            currency=None,
        )
    )

    assert normalized.loc[0, "event_type"] == "stock_dividend"
    assert normalized.loc[0, "stock_amount"] == pytest.approx(0.05)


def test_validate_dividend_event_schema_rejects_missing_required_field() -> None:
    record = _dividend_event()
    record.pop("source_payload_fingerprint")

    with pytest.raises(DividendContractError, match="source_payload_fingerprint"):
        validate_dividend_event_schema(record)


def test_validate_dividend_event_schema_rejects_invalid_event_type() -> None:
    with pytest.raises(DividendContractError, match="unsupported dividend event_type"):
        validate_dividend_event_schema(_dividend_event(event_type="split"))


def test_validate_dividend_event_schema_allows_nullable_date_fields() -> None:
    normalized = validate_dividend_event_schema(
        _dividend_event(declaration_date=None, record_date=None, payable_date=None)
    )

    assert pd.isna(normalized.loc[0, "declaration_date"])
    assert pd.isna(normalized.loc[0, "record_date"])
    assert pd.isna(normalized.loc[0, "payable_date"])


def test_validate_dividend_event_schema_rejects_unparseable_nullable_date_when_present() -> None:
    with pytest.raises(DividendContractError, match="payable_date"):
        validate_dividend_event_schema(_dividend_event(payable_date="not-a-date"))


def test_validate_dividend_event_schema_requires_amount_columns_even_when_null() -> None:
    normalized = validate_dividend_event_schema(
        _dividend_event(cash_amount=None, stock_amount=None, currency=None)
    )

    assert "cash_amount" in normalized.columns
    assert "stock_amount" in normalized.columns
    assert pd.isna(normalized.loc[0, "cash_amount"])
    assert pd.isna(normalized.loc[0, "stock_amount"])


def test_validate_dividend_event_schema_rejects_missing_amount_column() -> None:
    record = _dividend_event()
    record.pop("stock_amount")

    with pytest.raises(DividendContractError, match="stock_amount"):
        validate_dividend_event_schema(record)


def test_validate_dividend_event_schema_rejects_null_non_nullable_value() -> None:
    with pytest.raises(DividendContractError, match="source_event_id"):
        validate_dividend_event_schema(_dividend_event(source_event_id=None))


def test_validate_dividend_event_schema_rejects_duplicate_primary_keys() -> None:
    duplicate = _dividend_event(raw_payload={"row": 2})
    frame = pd.DataFrame([_dividend_event(), duplicate])

    with pytest.raises(DividendContractError, match="duplicate primary-key rows"):
        validate_dividend_event_schema(frame)


def test_validate_dividend_event_schema_rejects_unparseable_required_date() -> None:
    with pytest.raises(DividendContractError, match="ex_date"):
        validate_dividend_event_schema(_dividend_event(ex_date="2026-99-99"))
