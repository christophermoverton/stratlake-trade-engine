from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.catalog.indexer import build_artifact_records, build_catalog
from src.catalog.query import CatalogQuery, query_catalog
from src.catalog.validation import validate_record
from src.corporate_actions.dividend_importer import import_dividend_events


def test_dividend_import_artifacts_are_registered_as_event_evidence(tmp_path: Path) -> None:
    result = _make_dividend_import(tmp_path)

    records = build_catalog(tmp_path / "artifacts", repo_root=tmp_path)
    record = next(item for item in records if item.run_id == result.run_id)
    evidence = record.metadata["evidence"]

    assert record.run_type == "corporate_action_event_dataset"
    assert record.record_family == "corporate_action_event_dataset"
    assert record.qa_status == "pass"
    assert evidence["artifact_type"] == "corporate_action_event_dataset"
    assert evidence["evidence_type"] == "dividend_events"
    assert evidence["source_domain"] == "corporate_actions"
    assert evidence["event_domain"] == "dividends"
    assert evidence["schema_version"] == "corporate_actions.dividends.v1"
    assert evidence["contract_schema_version"] == "1.0.0"
    assert evidence["canonicality"] == "canonical_import_artifact"
    assert evidence["canonical_dataset_root"] == "data/curated/events/dividends"
    assert evidence["qa_summary_path"].endswith("/qa_summary.json")
    assert evidence["source_provenance_path"].endswith("/source_provenance.json")
    assert evidence["summary_path"].endswith("/summary.json")
    assert evidence["source_dataset_fingerprint"]
    assert evidence["import_config_fingerprint"]
    assert evidence["live_network_used"] is False
    assert evidence["credentials_used"] is False

    artifact_records = build_artifact_records(record, repo_root=tmp_path)
    assert {artifact.relative_path for artifact in artifact_records} >= {
        "manifest.json",
        "qa_summary.json",
        "source_provenance.json",
        "schema_contract.json",
        "summary.json",
    }


def test_dividend_catalog_query_filters_match_evidence_facets(tmp_path: Path) -> None:
    result = _make_dividend_import(tmp_path)
    records = build_catalog(tmp_path / "artifacts", repo_root=tmp_path)

    checks = [
        CatalogQuery(artifact_type="corporate_action_event_dataset"),
        CatalogQuery(evidence_type="dividend_events"),
        CatalogQuery(source_domain="corporate_actions"),
        CatalogQuery(event_domain="dividends"),
        CatalogQuery(schema_version="corporate_actions.dividends.v1"),
    ]

    for query in checks:
        assert [record.run_id for record in query_catalog(records, query)] == [result.run_id]


def test_dividend_catalog_validation_accepts_required_import_artifacts(tmp_path: Path) -> None:
    result = _make_dividend_import(tmp_path)
    record = next(item for item in build_catalog(tmp_path / "artifacts", repo_root=tmp_path) if item.run_id == result.run_id)

    issues = validate_record(record, repo_root=tmp_path, include_info=True)
    codes = [issue.code for issue in issues]

    assert "dividend_evidence_required_artifact_missing" not in codes
    assert "dividend_evidence_non_portable_path" not in codes


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
    metadata_path.write_text(
        json.dumps(
            {
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
    return import_dividend_events(
        source_data_path=data_path,
        source_metadata_path=metadata_path,
        output_root=tmp_path / "data" / "curated" / "events" / "dividends",
        artifact_root=tmp_path / "artifacts" / "corporate_actions",
        start="2024-01-01",
        end="2025-01-01",
    )
