from __future__ import annotations

import json

from src.corporate_actions.dividend_contract import (
    DIVIDEND_FALLBACK_KEY_FIELDS,
    DIVIDEND_PRIMARY_KEY_FIELDS,
    DIVIDEND_REQUIRED_FIELDS,
    DIVIDEND_REQUIRED_NULLABLE_FIELDS,
    DIVIDEND_SCHEMA_NAME,
    DIVIDEND_SCHEMA_VERSION,
    DIVIDEND_SUPPORTED_EVENT_TYPES,
    DIVIDEND_UPSTREAM_FIELD_MAPPING,
    build_dividend_schema_contract,
    serialize_dividend_schema_contract,
)


def test_dividend_contract_defines_required_and_nullable_fields() -> None:
    assert DIVIDEND_SCHEMA_NAME == "corporate_actions.dividends.v1"
    assert DIVIDEND_SCHEMA_VERSION == "1.0.0"
    assert DIVIDEND_REQUIRED_FIELDS == (
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
    assert "cash_amount" in DIVIDEND_REQUIRED_NULLABLE_FIELDS
    assert "stock_amount" in DIVIDEND_REQUIRED_NULLABLE_FIELDS
    assert "raw_payload" in DIVIDEND_REQUIRED_NULLABLE_FIELDS


def test_dividend_contract_defines_supported_event_types() -> None:
    assert DIVIDEND_SUPPORTED_EVENT_TYPES == ("cash_dividend", "stock_dividend")


def test_dividend_contract_defines_primary_key_fields() -> None:
    assert DIVIDEND_PRIMARY_KEY_FIELDS == (
        "symbol",
        "event_type",
        "source_event_id",
        "ex_date",
        "process_date",
        "source",
    )


def test_dividend_contract_defines_fallback_key_fields() -> None:
    assert DIVIDEND_FALLBACK_KEY_FIELDS == (
        "symbol",
        "event_type",
        "ex_date",
        "process_date",
        "cash_amount",
        "stock_amount",
        "currency",
        "source_payload_fingerprint",
    )


def test_dividend_contract_documents_upstream_field_mapping() -> None:
    assert DIVIDEND_UPSTREAM_FIELD_MAPPING["source_event_id"] == "corporate_action_id"
    assert DIVIDEND_UPSTREAM_FIELD_MAPPING["source_payload_fingerprint"] == "source_payload_hash"
    assert DIVIDEND_UPSTREAM_FIELD_MAPPING["raw_payload"] == "raw"
    assert DIVIDEND_UPSTREAM_FIELD_MAPPING["as_of_date"] == "process_date"
    assert DIVIDEND_UPSTREAM_FIELD_MAPPING["schema_version"] is None


def test_build_dividend_schema_contract_is_machine_readable() -> None:
    contract = build_dividend_schema_contract()

    assert contract["schema_name"] == DIVIDEND_SCHEMA_NAME
    assert contract["schema_version"] == DIVIDEND_SCHEMA_VERSION
    assert contract["primary_key_fields"] == list(DIVIDEND_PRIMARY_KEY_FIELDS)
    assert contract["fallback_key_fields"] == list(DIVIDEND_FALLBACK_KEY_FIELDS)
    assert contract["upstream_field_mapping"]["event_type"] == "corporate_action_type"


def test_serialize_dividend_schema_contract_is_deterministic() -> None:
    first = serialize_dividend_schema_contract()
    second = serialize_dividend_schema_contract()

    assert first == second
    assert first.endswith("\n")
    assert json.loads(first) == build_dividend_schema_contract()
    assert first == json.dumps(build_dividend_schema_contract(), indent=2, sort_keys=True) + "\n"
