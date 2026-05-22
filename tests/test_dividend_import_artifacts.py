from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.corporate_actions.dividend_contract import build_dividend_schema_contract
from src.corporate_actions.dividend_importer import import_dividend_events


def _rows() -> list[dict[str, object]]:
    base = {
        "corporate_action_id": "cash-1",
        "symbol": "aapl",
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
    duplicate = dict(base)
    duplicate["raw"] = json.dumps({"row": 2}, sort_keys=True)
    invalid = dict(base)
    invalid.update(
        {
            "corporate_action_id": "invalid-1",
            "corporate_action_type": "split",
            "source_payload_hash": "hash-invalid",
        }
    )
    outside = dict(base)
    outside.update(
        {
            "corporate_action_id": "outside-1",
            "ex_date": "2025-01-01",
            "source_payload_hash": "hash-outside",
        }
    )
    return [base, duplicate, invalid, outside]


def _write_source(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    data_path = source_dir / "dividends.parquet"
    metadata_path = source_dir / "metadata.json"
    pd.DataFrame(_rows()).to_parquet(data_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "source_vendor": "alpaca",
                "upstream_package_version": "0.0.test",
                "upstream_project": "fintech-market-ingestion",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "data" / "curated" / "events" / "dividends"
    artifact_root = tmp_path / "artifacts" / "corporate_actions"
    return data_path, metadata_path, output_root, artifact_root


def test_import_dividend_events_writes_deterministic_artifact_bundle(tmp_path: Path) -> None:
    data_path, metadata_path, output_root, artifact_root = _write_source(tmp_path)

    result = import_dividend_events(
        source_data_path=data_path,
        source_metadata_path=metadata_path,
        output_root=output_root,
        artifact_root=artifact_root,
        start="2024-01-01",
        end="2025-01-01",
        strict=False,
    )

    run_dir = artifact_root / result.run_id
    assert sorted(path.name for path in run_dir.iterdir()) == [
        "duplicate_events.csv",
        "import_config.json",
        "invalid_events.csv",
        "manifest.json",
        "qa_summary.json",
        "schema_contract.json",
        "source_provenance.json",
        "summary.json",
    ]

    manifest = _read_json(run_dir / "manifest.json")
    assert manifest["artifact_type"] == "corporate_action_event_import"
    assert manifest["event_evidence_policy"] == (
        "dividend events are explicit event evidence, not adjusted price data"
    )
    assert manifest["fallback_key_behavior"] == "deferred_inactive"
    assert set(manifest["artifact_files"]) == {path.name for path in run_dir.iterdir()}

    summary = _read_json(run_dir / "summary.json")
    assert summary["import_result"]["run_id"] == result.run_id
    assert summary["qa_status"] == "warn"


def test_schema_contract_artifact_matches_m40_contract(tmp_path: Path) -> None:
    data_path, metadata_path, output_root, artifact_root = _write_source(tmp_path)

    result = import_dividend_events(
        source_data_path=data_path,
        source_metadata_path=metadata_path,
        output_root=output_root,
        artifact_root=artifact_root,
        start="2024-01-01",
        end="2025-01-01",
        strict=False,
    )

    assert _read_json(artifact_root / result.run_id / "schema_contract.json") == build_dividend_schema_contract()


def test_import_artifacts_are_deterministic_across_reruns(tmp_path: Path) -> None:
    data_path, metadata_path, output_root, artifact_root = _write_source(tmp_path)
    kwargs = {
        "source_data_path": data_path,
        "source_metadata_path": metadata_path,
        "output_root": output_root,
        "artifact_root": artifact_root,
        "start": "2024-01-01",
        "end": "2025-01-01",
        "strict": False,
    }

    first = import_dividend_events(**kwargs)
    first_files = _read_artifact_texts(artifact_root / first.run_id)
    second = import_dividend_events(**kwargs)
    second_files = _read_artifact_texts(artifact_root / second.run_id)

    assert first.run_id == second.run_id
    assert first_files == second_files


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_artifact_texts(root: Path) -> dict[str, str]:
    return {path.name: path.read_text(encoding="utf-8") for path in sorted(root.iterdir())}
