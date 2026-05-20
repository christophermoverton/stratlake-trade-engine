from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.corporate_actions.dividend_contract import DIVIDEND_SCHEMA_VERSION
from src.corporate_actions.dividend_importer import (
    DividendImportError,
    filter_dividend_events_by_ex_date,
    import_dividend_events,
    normalize_upstream_dividend_events,
    read_upstream_dividend_artifacts,
)


def _upstream_dividend_rows() -> list[dict[str, object]]:
    return [
        {
            "corporate_action_id": " start-boundary ",
            "symbol": " aapl ",
            "corporate_action_type": " cash_dividend ",
            "source": " alpaca ",
            "process_date": "2024-01-02T15:30:00Z",
            "declaration_date": "2023-12-15",
            "ex_date": "2024-01-01T00:00:00Z",
            "record_date": "2024-01-03",
            "payable_date": "2024-01-15",
            "cash_amount": 0.24,
            "stock_amount": None,
            "currency": " usd ",
            "source_payload_hash": " hash-start ",
            "raw": json.dumps({"b": 2, "a": 1}),
        },
        {
            "corporate_action_id": "middle-stock",
            "symbol": " msft ",
            "corporate_action_type": "stock_dividend",
            "source": "alpaca",
            "process_date": "2024-06-01",
            "declaration_date": None,
            "ex_date": "2024-06-15",
            "record_date": None,
            "payable_date": None,
            "cash_amount": None,
            "stock_amount": 0.05,
            "currency": None,
            "source_payload_hash": "hash-stock",
            "raw": {"event": "stock"},
        },
        {
            "corporate_action_id": "end-boundary",
            "symbol": "AAPL",
            "corporate_action_type": "cash_dividend",
            "source": "alpaca",
            "process_date": "2025-01-01",
            "declaration_date": None,
            "ex_date": "2025-01-01",
            "record_date": None,
            "payable_date": None,
            "cash_amount": 0.25,
            "stock_amount": None,
            "currency": "USD",
            "source_payload_hash": "hash-end",
            "raw": None,
        },
    ]


def _write_upstream_artifacts(tmp_path: Path, rows: list[dict[str, object]] | None = None) -> tuple[Path, Path]:
    source_root = tmp_path / "upstream" / "data" / "curated" / "corporate_actions" / "dividends"
    source_root.mkdir(parents=True)
    data_path = source_root / "dividends.parquet"
    metadata_path = source_root / "metadata.json"
    fixture_rows = [_parquet_safe_row(row) for row in rows or _upstream_dividend_rows()]
    pd.DataFrame(fixture_rows).to_parquet(data_path, index=False)
    metadata_path.write_text(
        json.dumps({"source": "fixture", "row_count": len(rows or _upstream_dividend_rows())}, sort_keys=True),
        encoding="utf-8",
    )
    return data_path, metadata_path


def _parquet_safe_row(row: dict[str, object]) -> dict[str, object]:
    safe = dict(row)
    if isinstance(safe.get("raw"), dict | list):
        safe["raw"] = json.dumps(safe["raw"], sort_keys=True)
    return safe


def test_read_upstream_dividend_artifacts_reads_local_parquet_and_metadata(tmp_path: Path) -> None:
    data_path, metadata_path = _write_upstream_artifacts(tmp_path)

    frame, metadata = read_upstream_dividend_artifacts(data_path, metadata_path)

    assert len(frame) == 3
    assert metadata == {"row_count": 3, "source": "fixture"}


def test_read_upstream_dividend_artifacts_rejects_non_local_uri(tmp_path: Path) -> None:
    _data_path, metadata_path = _write_upstream_artifacts(tmp_path)

    with pytest.raises(DividendImportError, match="local file path"):
        read_upstream_dividend_artifacts("https://example.test/dividends.parquet", metadata_path)


def test_normalize_upstream_dividend_events_maps_contract_fields() -> None:
    normalized = normalize_upstream_dividend_events(pd.DataFrame(_upstream_dividend_rows()))

    assert normalized.loc[0, "symbol"] == "AAPL"
    assert normalized.loc[0, "event_type"] == "cash_dividend"
    assert normalized.loc[0, "source_event_id"] == "start-boundary"
    assert normalized.loc[0, "source_payload_fingerprint"] == "hash-start"
    assert normalized.loc[0, "as_of_date"] == "2024-01-02"
    assert normalized.loc[0, "currency"] == "usd"
    assert normalized.loc[0, "raw_payload"] == '{"a":1,"b":2}'
    assert normalized.loc[0, "schema_version"] == DIVIDEND_SCHEMA_VERSION
    assert normalized.loc[1, "raw_payload"] == '{"event":"stock"}'
    assert normalized.loc[1, "year"] == "2024"


def test_filter_dividend_events_by_ex_date_uses_half_open_window() -> None:
    normalized = normalize_upstream_dividend_events(pd.DataFrame(_upstream_dividend_rows()))

    filtered = filter_dividend_events_by_ex_date(normalized, start="2024-01-01", end="2025-01-01")

    assert filtered["source_event_id"].tolist() == ["start-boundary", "middle-stock"]
    assert "end-boundary" not in filtered["source_event_id"].tolist()


def test_import_dividend_events_writes_contract_valid_partitioned_output(tmp_path: Path) -> None:
    data_path, metadata_path = _write_upstream_artifacts(tmp_path)
    output_root = tmp_path / "data" / "curated" / "events" / "dividends"

    result = import_dividend_events(
        source_data_path=data_path,
        source_metadata_path=metadata_path,
        output_root=output_root,
        start="2024-01-01",
        end="2025-01-01",
    )

    assert result.input_row_count == 3
    assert result.normalized_row_count == 3
    assert result.filtered_row_count == 1
    assert result.written_row_count == 2
    assert result.duplicate_event_count == 0
    assert result.invalid_event_count == 0
    assert result.symbols == ("AAPL", "MSFT")
    assert result.partitions == (
        "symbol=AAPL/year=2024",
        "symbol=MSFT/year=2024",
    )

    output = pd.read_parquet(output_root)
    assert output.sort_values(["symbol", "source_event_id"])["source_event_id"].tolist() == [
        "start-boundary",
        "middle-stock",
    ]
    assert "year" in output.columns


def test_import_dividend_events_rejects_invalid_contract_rows_in_strict_mode(tmp_path: Path) -> None:
    rows = _upstream_dividend_rows()
    rows[0]["corporate_action_type"] = "split"
    data_path, metadata_path = _write_upstream_artifacts(tmp_path, rows)

    with pytest.raises(Exception, match="unsupported dividend event_type"):
        import_dividend_events(
            source_data_path=data_path,
            source_metadata_path=metadata_path,
            output_root=tmp_path / "output",
            start="2024-01-01",
            end="2025-01-01",
            strict=True,
        )


def test_import_dividend_events_reports_invalid_rows_in_advisory_mode(tmp_path: Path) -> None:
    rows = _upstream_dividend_rows()
    rows[0]["corporate_action_type"] = "split"
    data_path, metadata_path = _write_upstream_artifacts(tmp_path, rows)

    result = import_dividend_events(
        source_data_path=data_path,
        source_metadata_path=metadata_path,
        output_root=tmp_path / "output",
        start="2024-01-01",
        end="2025-01-01",
        strict=False,
    )

    assert result.invalid_event_count == 1
    assert result.written_row_count == 1


def test_importer_does_not_require_upstream_package_import() -> None:
    assert "fintech_market_ingestion" not in sys.modules
