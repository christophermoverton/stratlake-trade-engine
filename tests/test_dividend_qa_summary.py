from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.corporate_actions.dividend_importer import import_dividend_events


def test_dividend_qa_summary_reports_required_counts_and_advisory_policy(tmp_path: Path) -> None:
    rows = [
        _row("dup-1", cash_amount=0.24, currency="USD"),
        _row("dup-1", cash_amount=0.24, currency="USD"),
        _row("bad-type", corporate_action_type="split", cash_amount=0.10, currency="USD"),
        _row("negative-cash", cash_amount=-0.01, currency="USD"),
        _row("missing-currency", cash_amount=0.15, currency=None),
        _row("outside", ex_date="2025-01-01", cash_amount=0.20, currency="USD"),
    ]
    data_path, metadata_path = _write_source(tmp_path, rows)

    result = import_dividend_events(
        source_data_path=data_path,
        source_metadata_path=metadata_path,
        output_root=tmp_path / "output",
        artifact_root=tmp_path / "artifacts" / "corporate_actions",
        start="2024-01-01",
        end="2025-01-01",
        strict=False,
    )
    qa_summary = _read_json(tmp_path / "artifacts" / "corporate_actions" / result.run_id / "qa_summary.json")

    assert qa_summary["input_row_count"] == 6
    assert qa_summary["normalized_row_count"] == 6
    assert qa_summary["filtered_row_count"] == 1
    assert qa_summary["written_row_count"] == 3
    assert qa_summary["duplicate_event_count"] == 2
    assert qa_summary["invalid_event_count"] == 1
    assert qa_summary["invalid_event_type_count"] == 1
    assert qa_summary["negative_cash_amount_count"] == 1
    assert qa_summary["negative_stock_amount_count"] == 0
    assert qa_summary["currency_missing_for_cash_dividend_count"] == 1
    assert qa_summary["rows_outside_import_window_count"] == 1
    assert qa_summary["strict_mode"] is False
    assert qa_summary["qa_status"] == "warn"
    assert "keeps the first deterministic occurrence" in qa_summary["advisory_duplicate_policy"]
    assert qa_summary["event_evidence_policy"].endswith("not adjusted price data")


def test_duplicate_and_invalid_event_csvs_are_deterministically_ordered(tmp_path: Path) -> None:
    rows = [
        _row("z-dup", symbol="MSFT"),
        _row("z-dup", symbol="MSFT"),
        _row("a-invalid", corporate_action_type="split"),
    ]
    data_path, metadata_path = _write_source(tmp_path, rows)

    result = import_dividend_events(
        source_data_path=data_path,
        source_metadata_path=metadata_path,
        output_root=tmp_path / "output",
        artifact_root=tmp_path / "artifacts" / "corporate_actions",
        start="2024-01-01",
        end="2025-01-01",
        strict=False,
    )
    run_dir = tmp_path / "artifacts" / "corporate_actions" / result.run_id

    duplicate_events = pd.read_csv(run_dir / "duplicate_events.csv")
    invalid_events = pd.read_csv(run_dir / "invalid_events.csv")

    assert duplicate_events["source_event_id"].tolist() == ["z-dup", "z-dup"]
    assert invalid_events["source_event_id"].tolist() == ["a-invalid"]
    assert "unsupported dividend event_type" in invalid_events.loc[0, "issue"]


def _row(
    corporate_action_id: str,
    *,
    symbol: str = "AAPL",
    corporate_action_type: str = "cash_dividend",
    ex_date: str = "2024-02-15",
    cash_amount: float | None = 0.24,
    stock_amount: float | None = None,
    currency: str | None = "USD",
) -> dict[str, object]:
    return {
        "corporate_action_id": corporate_action_id,
        "symbol": symbol,
        "corporate_action_type": corporate_action_type,
        "source": "alpaca",
        "process_date": "2024-02-01",
        "declaration_date": None,
        "ex_date": ex_date,
        "record_date": None,
        "payable_date": None,
        "cash_amount": cash_amount,
        "stock_amount": stock_amount,
        "currency": currency,
        "source_payload_hash": f"hash-{corporate_action_id}",
        "raw": json.dumps({"id": corporate_action_id}, sort_keys=True),
    }


def _write_source(tmp_path: Path, rows: list[dict[str, object]]) -> tuple[Path, Path]:
    source = tmp_path / "source"
    source.mkdir()
    data_path = source / "dividends.parquet"
    metadata_path = source / "metadata.json"
    pd.DataFrame(rows).to_parquet(data_path, index=False)
    metadata_path.write_text(json.dumps({"source_vendor": "alpaca"}, sort_keys=True), encoding="utf-8")
    return data_path, metadata_path


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))
