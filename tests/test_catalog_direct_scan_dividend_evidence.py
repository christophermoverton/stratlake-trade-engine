from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.catalog.indexer import build_catalog
from src.corporate_actions.dividend_importer import import_dividend_events


def test_direct_catalog_scan_discovers_dividend_evidence_without_registry(tmp_path: Path) -> None:
    result = _make_dividend_import(tmp_path)
    before = _snapshot(tmp_path)

    first = build_catalog(tmp_path / "artifacts", repo_root=tmp_path)
    second = build_catalog(tmp_path / "artifacts", repo_root=tmp_path)
    after = _snapshot(tmp_path)

    dividend_records = [record for record in first if record.run_id == result.run_id]
    assert len(dividend_records) == 1
    assert before == after
    assert [record.to_dict() for record in first] == [record.to_dict() for record in second]
    assert dividend_records[0].source_registry_path is None
    assert "artifact_root_no_registry_entry" in dividend_records[0].validation.validation_warnings


def test_dividend_catalog_record_preserves_canonical_dataset_and_artifact_links(tmp_path: Path) -> None:
    result = _make_dividend_import(tmp_path)
    record = next(record for record in build_catalog(tmp_path / "artifacts", repo_root=tmp_path) if record.run_id == result.run_id)
    evidence = record.metadata["evidence"]

    expected_artifact_root = f"artifacts/corporate_actions/{result.run_id}"
    assert record.artifact_root == expected_artifact_root
    assert record.source_manifest_path == f"{expected_artifact_root}/manifest.json"
    assert evidence["canonical_dataset_root"] == "data/curated/events/dividends"
    assert evidence["qa_summary_path"] == f"{expected_artifact_root}/qa_summary.json"
    assert evidence["schema_contract_path"] == f"{expected_artifact_root}/schema_contract.json"
    assert evidence["source_provenance_path"] == f"{expected_artifact_root}/source_provenance.json"
    assert evidence["summary_path"] == f"{expected_artifact_root}/summary.json"
    assert all(not Path(path).is_absolute() and "\\" not in path for path in record.source_files)


def _make_dividend_import(tmp_path: Path):
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
    metadata_path.write_text(json.dumps({"source_vendor": "alpaca"}, sort_keys=True), encoding="utf-8")
    return import_dividend_events(
        source_data_path=data_path,
        source_metadata_path=metadata_path,
        output_root=tmp_path / "data" / "curated" / "events" / "dividends",
        artifact_root=tmp_path / "artifacts" / "corporate_actions",
        start="2024-01-01",
        end="2025-01-01",
    )


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
