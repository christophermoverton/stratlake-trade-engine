from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.corporate_actions.dividend_importer import DividendImportError, import_dividend_events


def _duplicate_rows() -> list[dict[str, object]]:
    base = {
        "corporate_action_id": "dup-1",
        "symbol": "AAPL",
        "corporate_action_type": "cash_dividend",
        "source": "alpaca",
        "process_date": "2024-02-01",
        "declaration_date": None,
        "ex_date": "2024-02-15",
        "record_date": None,
        "payable_date": None,
        "cash_amount": 0.24,
        "stock_amount": None,
        "currency": "USD",
        "source_payload_hash": "hash-1",
        "raw": {"row": 1},
    }
    duplicate = dict(base)
    duplicate["raw"] = {"row": 2}
    return [base, duplicate]


def _write_duplicate_source(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    source.mkdir()
    data_path = source / "dividends.parquet"
    metadata_path = source / "metadata.json"
    safe_rows = []
    for row in _duplicate_rows():
        safe = dict(row)
        safe["raw"] = json.dumps(safe["raw"], sort_keys=True)
        safe_rows.append(safe)
    pd.DataFrame(safe_rows).to_parquet(data_path, index=False)
    metadata_path.write_text(json.dumps({"fixture": "duplicates"}, sort_keys=True), encoding="utf-8")
    return data_path, metadata_path


def test_import_dividend_events_detects_duplicate_primary_keys_in_strict_mode(tmp_path: Path) -> None:
    data_path, metadata_path = _write_duplicate_source(tmp_path)

    with pytest.raises(DividendImportError, match="duplicate primary-key rows"):
        import_dividend_events(
            source_data_path=data_path,
            source_metadata_path=metadata_path,
            output_root=tmp_path / "output",
            start="2024-01-01",
            end="2025-01-01",
            strict=True,
        )


def test_import_dividend_events_reports_and_deduplicates_in_advisory_mode(tmp_path: Path) -> None:
    data_path, metadata_path = _write_duplicate_source(tmp_path)
    output_root = tmp_path / "output"

    result = import_dividend_events(
        source_data_path=data_path,
        source_metadata_path=metadata_path,
        output_root=output_root,
        start="2024-01-01",
        end="2025-01-01",
        strict=False,
    )

    assert result.duplicate_event_count == 2
    assert result.written_row_count == 1
    output = pd.read_parquet(output_root)
    assert len(output) == 1
    assert output.loc[0, "source_event_id"] == "dup-1"
