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
from src.corporate_actions.dividend_importer import (
    DIVIDEND_SORT_COLUMNS,
    DividendImportError,
    DividendImportResult,
    filter_dividend_events_by_ex_date,
    import_dividend_events,
    normalize_upstream_dividend_events,
    read_upstream_dividend_artifacts,
    write_dividend_event_dataset,
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
    "DIVIDEND_SORT_COLUMNS",
    "DividendImportError",
    "DividendImportResult",
    "build_dividend_schema_contract",
    "filter_dividend_events_by_ex_date",
    "import_dividend_events",
    "normalize_upstream_dividend_events",
    "read_upstream_dividend_artifacts",
    "serialize_dividend_schema_contract",
    "validate_dividend_event_schema",
    "write_dividend_event_dataset",
]
