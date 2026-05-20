"""Corporate-action event contracts for downstream research evidence."""

from src.corporate_actions.dividend_contract import (
    DIVIDEND_FALLBACK_KEY_FIELDS,
    DIVIDEND_PRIMARY_KEY_FIELDS,
    DIVIDEND_REQUIRED_FIELDS,
    DIVIDEND_REQUIRED_NULLABLE_FIELDS,
    DIVIDEND_SCHEMA_NAME,
    DIVIDEND_SCHEMA_VERSION,
    DIVIDEND_SUPPORTED_EVENT_TYPES,
    DIVIDEND_UPSTREAM_FIELD_MAPPING,
    DividendContractError,
    build_dividend_schema_contract,
    serialize_dividend_schema_contract,
    validate_dividend_event_schema,
)

__all__ = [
    "DIVIDEND_FALLBACK_KEY_FIELDS",
    "DIVIDEND_PRIMARY_KEY_FIELDS",
    "DIVIDEND_REQUIRED_FIELDS",
    "DIVIDEND_REQUIRED_NULLABLE_FIELDS",
    "DIVIDEND_SCHEMA_NAME",
    "DIVIDEND_SCHEMA_VERSION",
    "DIVIDEND_SUPPORTED_EVENT_TYPES",
    "DIVIDEND_UPSTREAM_FIELD_MAPPING",
    "DividendContractError",
    "build_dividend_schema_contract",
    "serialize_dividend_schema_contract",
    "validate_dividend_event_schema",
]
