from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.cli.import_corporate_actions_dividends import main, run_cli


def test_dividend_cli_help_documents_local_file_boundaries(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    output = capsys.readouterr().out
    assert exc_info.value.code == 0
    assert "local upstream" in output
    assert "no Alpaca calls" in output
    assert "adjusted price data" in output


def test_dividend_cli_happy_path_prints_deterministic_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data_path, metadata_path = _write_source(tmp_path)
    output_root = tmp_path / "repo" / "data" / "curated" / "events" / "dividends"
    artifact_root = tmp_path / "repo" / "artifacts" / "corporate_actions"

    payload = run_cli(
        [
            "--source-data",
            str(data_path),
            "--source-metadata",
            str(metadata_path),
            "--output-root",
            str(output_root),
            "--artifact-root",
            str(artifact_root),
            "--start",
            "2024-01-01",
            "--end",
            "2025-01-01",
            "--strict",
        ]
    )
    printed = json.loads(capsys.readouterr().out)

    assert printed == payload
    assert payload["qa_status"] == "pass"
    assert payload["written_row_count"] == 2
    assert payload["output_root"] == "data/curated/events/dividends"
    assert payload["artifact_path"].startswith("artifacts/corporate_actions/dividend_import_")


def test_dividend_cli_invalid_source_path_exits_nonzero(tmp_path: Path) -> None:
    _data_path, metadata_path = _write_source(tmp_path)

    with pytest.raises(Exception, match="does not exist"):
        run_cli(
            [
                "--source-data",
                str(tmp_path / "missing.parquet"),
                "--source-metadata",
                str(metadata_path),
                "--output-root",
                str(tmp_path / "output"),
                "--artifact-root",
                str(tmp_path / "artifacts" / "corporate_actions"),
                "--start",
                "2024-01-01",
                "--end",
                "2025-01-01",
                "--strict",
            ]
        )


def _write_source(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    source.mkdir()
    data_path = source / "dividends.parquet"
    metadata_path = source / "metadata.json"
    pd.DataFrame(
        [
            _row("cash-1", "AAPL", "cash_dividend", "2024-02-15", 0.24, None, "USD"),
            _row("stock-1", "MSFT", "stock_dividend", "2024-06-15", None, 0.05, None),
        ]
    ).to_parquet(data_path, index=False)
    metadata_path.write_text(json.dumps({"source_vendor": "synthetic_fixture"}, sort_keys=True), encoding="utf-8")
    return data_path, metadata_path


def _row(
    event_id: str,
    symbol: str,
    event_type: str,
    ex_date: str,
    cash_amount: float | None,
    stock_amount: float | None,
    currency: str | None,
) -> dict[str, object]:
    return {
        "corporate_action_id": event_id,
        "symbol": symbol,
        "corporate_action_type": event_type,
        "source": "synthetic_fixture",
        "process_date": "2024-01-01",
        "declaration_date": None,
        "ex_date": ex_date,
        "record_date": None,
        "payable_date": None,
        "cash_amount": cash_amount,
        "stock_amount": stock_amount,
        "currency": currency,
        "source_payload_hash": f"hash-{event_id}",
        "raw": json.dumps({"id": event_id}, sort_keys=True),
    }
