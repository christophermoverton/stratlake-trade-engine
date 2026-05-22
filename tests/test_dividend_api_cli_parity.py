from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pandas.testing as pdt

from src.cli.import_corporate_actions_dividends import run_cli
from src.corporate_actions import import_dividend_events, load_dividend_events


def test_dividend_api_cli_parity_for_same_local_inputs(tmp_path: Path, capsys) -> None:
    data_path, metadata_path = _write_source(tmp_path)
    repo_root = tmp_path / "repo"
    output_root = repo_root / "data" / "curated" / "events" / "dividends"
    artifact_root = repo_root / "artifacts" / "corporate_actions"

    api_result = import_dividend_events(
        source_data_path=data_path,
        source_metadata_path=metadata_path,
        output_root=output_root,
        artifact_root=artifact_root,
        start="2024-01-01",
        end="2025-01-01",
        strict=True,
    )
    api_rows = load_dividend_events(output_root)
    api_artifacts = _artifact_texts(artifact_root / api_result.run_id)

    cli_payload = run_cli(
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
    capsys.readouterr()
    cli_rows = load_dividend_events(output_root)
    cli_artifacts = _artifact_texts(artifact_root / str(cli_payload["run_id"]))
    qa_summary = json.loads((artifact_root / api_result.run_id / "qa_summary.json").read_text(encoding="utf-8"))

    assert cli_payload == {
        "artifact_path": api_result.artifact_path,
        "output_root": api_result.output_root,
        "qa_status": qa_summary["qa_status"],
        "run_id": api_result.run_id,
        "written_row_count": api_result.written_row_count,
    }
    pdt.assert_frame_equal(api_rows, cli_rows)
    assert api_artifacts == cli_artifacts


def test_load_dividend_events_reads_dataset_root_and_reconstructs_partitions(tmp_path: Path) -> None:
    data_path, metadata_path = _write_source(tmp_path)
    output_root = tmp_path / "repo" / "data" / "curated" / "events" / "dividends"

    import_dividend_events(
        source_data_path=data_path,
        source_metadata_path=metadata_path,
        output_root=output_root,
        artifact_root=tmp_path / "repo" / "artifacts" / "corporate_actions",
        start="2024-01-01",
        end="2025-01-01",
        strict=True,
    )

    loaded = load_dividend_events(output_root)
    first_part = next(output_root.rglob("part-0.parquet"))
    raw_part = pd.read_parquet(first_part)

    assert loaded["symbol"].tolist() == ["AAPL", "MSFT"]
    assert "year" in loaded.columns
    assert "symbol" not in raw_part.columns


def _write_source(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    source.mkdir()
    data_path = source / "dividends.parquet"
    metadata_path = source / "metadata.json"
    pd.DataFrame(
        [
            _row("stock-1", "MSFT", "stock_dividend", "2024-06-15", None, 0.05, None),
            _row("cash-1", "AAPL", "cash_dividend", "2024-02-15", 0.24, None, "USD"),
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


def _artifact_texts(root: Path) -> dict[str, str]:
    return {path.name: path.read_text(encoding="utf-8") for path in sorted(root.iterdir())}
