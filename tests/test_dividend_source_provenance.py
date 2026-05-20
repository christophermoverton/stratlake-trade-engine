from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.corporate_actions.dividend_contract import DIVIDEND_SCHEMA_NAME, DIVIDEND_SCHEMA_VERSION
from src.corporate_actions.dividend_importer import import_dividend_events


def test_source_provenance_records_local_source_metadata_without_live_access(tmp_path: Path) -> None:
    data_path, metadata_path = _write_source(tmp_path)

    result = import_dividend_events(
        source_data_path=data_path,
        source_metadata_path=metadata_path,
        output_root=tmp_path / "output",
        artifact_root=tmp_path / "artifacts" / "corporate_actions",
        start="2024-01-01",
        end="2025-01-01",
    )

    provenance = _read_json(
        tmp_path / "artifacts" / "corporate_actions" / result.run_id / "source_provenance.json"
    )

    assert provenance["upstream_project"] == "fintech-market-ingestion"
    assert provenance["upstream_package_name"] == "fintech-market-ingestion"
    assert provenance["upstream_package_version"] == "1.2.3"
    assert provenance["upstream_source_repository"] == "christophermoverton/fintech-market-ingestion"
    assert provenance["source_vendor"] == "alpaca"
    assert provenance["schema_name"] == DIVIDEND_SCHEMA_NAME
    assert provenance["schema_version"] == DIVIDEND_SCHEMA_VERSION
    assert provenance["live_network_used"] is False
    assert provenance["credentials_used"] is False
    assert len(provenance["source_dataset_fingerprint"]) == 64
    assert len(provenance["import_config_fingerprint"]) == 64
    assert provenance["upstream_metadata"]["api_token"] == "[redacted]"
    assert provenance["fallback_key_behavior"] == "deferred_inactive"


def test_import_config_fingerprint_is_stable_for_same_config(tmp_path: Path) -> None:
    data_path, metadata_path = _write_source(tmp_path)
    kwargs = {
        "source_data_path": data_path,
        "source_metadata_path": metadata_path,
        "output_root": tmp_path / "output",
        "artifact_root": tmp_path / "artifacts" / "corporate_actions",
        "start": "2024-01-01",
        "end": "2025-01-01",
    }

    first = import_dividend_events(**kwargs)
    first_provenance = _read_json(
        tmp_path / "artifacts" / "corporate_actions" / first.run_id / "source_provenance.json"
    )
    second = import_dividend_events(**kwargs)
    second_provenance = _read_json(
        tmp_path / "artifacts" / "corporate_actions" / second.run_id / "source_provenance.json"
    )

    assert first.run_id == second.run_id
    assert first_provenance["import_config_fingerprint"] == second_provenance["import_config_fingerprint"]
    assert first_provenance["source_dataset_fingerprint"] == second_provenance["source_dataset_fingerprint"]


def _write_source(tmp_path: Path) -> tuple[Path, Path]:
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
        json.dumps(
            {
                "api_token": "secret-value",
                "source_vendor": "alpaca",
                "upstream_package_name": "fintech-market-ingestion",
                "upstream_package_version": "1.2.3",
                "upstream_project": "fintech-market-ingestion",
                "upstream_source_repository": "christophermoverton/fintech-market-ingestion",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return data_path, metadata_path


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))
