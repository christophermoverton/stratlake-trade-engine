from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.catalog.indexer import build_catalog
from src.catalog.query import records_to_rows
from src.corporate_actions.dividend_importer import import_dividend_events


def test_dividend_evidence_is_not_classified_as_research_or_price_artifacts(tmp_path: Path) -> None:
    result = _make_dividend_import(tmp_path)
    record = next(record for record in build_catalog(tmp_path / "artifacts", repo_root=tmp_path) if record.run_id == result.run_id)
    evidence = record.metadata["evidence"]

    forbidden_run_types = {
        "strategy",
        "alpha_evaluation",
        "portfolio",
        "governance_bundle",
        "promotion_governance",
        "ohlcv",
        "adjusted_price",
    }
    assert record.run_type not in forbidden_run_types
    assert record.strategy_name is None
    assert record.alpha_model_name is None
    assert record.portfolio_name is None
    assert record.promotion_status is None
    assert record.review_status is None
    assert evidence["artifact_type"] == "corporate_action_event_dataset"
    assert evidence["event_evidence_policy"] == (
        "dividend events are explicit event evidence, not adjusted price data"
    )
    assert "adjusted_price" not in json.dumps(record.to_dict(), sort_keys=True)


def test_dividend_evidence_rows_include_queryable_classification_facets(tmp_path: Path) -> None:
    result = _make_dividend_import(tmp_path)
    records = build_catalog(tmp_path / "artifacts", repo_root=tmp_path)

    row = next(row for row in records_to_rows(records) if row["run_id"] == result.run_id)

    assert row["artifact_type"] == "corporate_action_event_dataset"
    assert row["evidence_type"] == "dividend_events"
    assert row["source_domain"] == "corporate_actions"
    assert row["event_domain"] == "dividends"
    assert row["schema_version"] == "corporate_actions.dividends.v1"


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
