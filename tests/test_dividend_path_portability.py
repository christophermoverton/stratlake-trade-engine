from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.corporate_actions.dividend_importer import import_dividend_events


def test_dividend_import_artifacts_do_not_leak_absolute_paths_or_secrets(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    data_path = source / "dividends.parquet"
    metadata_path = source / "metadata.json"
    pd.DataFrame(
        [
            {
                "corporate_action_id": "cash-1",
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
                "source_payload_hash": "hash-cash",
                "raw": json.dumps({"row": 1}, sort_keys=True),
            }
        ]
    ).to_parquet(data_path, index=False)
    metadata_path.write_text(
        json.dumps({"access_token": "super-secret-token", "source_vendor": "alpaca"}, sort_keys=True),
        encoding="utf-8",
    )

    result = import_dividend_events(
        source_data_path=data_path,
        source_metadata_path=metadata_path,
        output_root=tmp_path / "output",
        artifact_root=tmp_path / "artifacts" / "corporate_actions",
        start="2024-01-01",
        end="2025-01-01",
    )
    run_dir = tmp_path / "artifacts" / "corporate_actions" / result.run_id

    artifact_text = "\n".join(path.read_text(encoding="utf-8") for path in sorted(run_dir.iterdir()))

    assert str(tmp_path) not in artifact_text
    assert "\\" not in artifact_text
    assert "super-secret-token" not in artifact_text
    assert "C:" not in artifact_text
    assert "file://" not in artifact_text
