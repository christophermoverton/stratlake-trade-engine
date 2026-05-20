from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from docs.examples.m40_cross_repo_q1_dividend_smoke_workflow import (
    main,
    run_cross_repo_q1_dividend_smoke_workflow,
)
from src.catalog import CatalogQuery, build_catalog, query_catalog
from src.corporate_actions import load_dividend_events


def test_cross_repo_smoke_helper_imports_explicit_local_artifacts_and_catalogs_evidence(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    source_data, source_metadata = _write_synthetic_upstream_artifacts(repo_root)
    output_root = repo_root / "data" / "curated" / "events" / "dividends"
    artifact_root = repo_root / "artifacts" / "corporate_actions"

    summary = run_cross_repo_q1_dividend_smoke_workflow(
        source_data=source_data,
        source_metadata=source_metadata,
        output_root=output_root,
        artifact_root=artifact_root,
        start="2024-01-01",
        end="2024-04-01",
        symbol="aapl",
        strict=True,
        repo_root=repo_root,
    )

    assert summary["symbol"] == "AAPL"
    assert summary["written_row_count"] == 1
    assert summary["loaded_row_count"] == 1
    assert summary["symbol_row_count"] == 1
    assert summary["qa_status"] == "pass"
    assert summary["catalog_match_count"] == 1
    assert summary["run_id"] in summary["catalog_run_ids"]

    loaded = load_dividend_events(output_root)
    assert loaded["symbol"].tolist() == ["AAPL"]
    assert loaded["ex_date"].tolist() == ["2024-02-15"]

    records = build_catalog(repo_root / "artifacts", repo_root=repo_root)
    matches = query_catalog(records, CatalogQuery(evidence_type="dividend_events"))
    assert [record.run_id for record in matches] == [summary["run_id"]]
    assert matches[0].record_family == "corporate_action_event_dataset"


def test_cross_repo_smoke_helper_cli_prints_deterministic_summary(tmp_path: Path, capsys) -> None:
    repo_root = tmp_path / "repo"
    source_data, source_metadata = _write_synthetic_upstream_artifacts(repo_root)
    output_root = repo_root / "data" / "curated" / "events" / "dividends"
    artifact_root = repo_root / "artifacts" / "corporate_actions"

    code = main(
        [
            "--source-data",
            str(source_data),
            "--source-metadata",
            str(source_metadata),
            "--output-root",
            str(output_root),
            "--artifact-root",
            str(artifact_root),
            "--start",
            "2024-01-01",
            "--end",
            "2024-04-01",
            "--symbol",
            "AAPL",
            "--repo-root",
            str(repo_root),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["qa_status"] == "pass"
    assert payload["written_row_count"] == 1
    assert payload["catalog_match_count"] == 1
    assert "credential" not in json.dumps(payload, sort_keys=True).lower()


def test_cross_repo_smoke_helper_keeps_generated_outputs_under_explicit_roots(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    source_data, source_metadata = _write_synthetic_upstream_artifacts(repo_root)
    output_root = repo_root / "generated" / "data"
    artifact_root = repo_root / "generated" / "artifacts" / "corporate_actions"

    run_cross_repo_q1_dividend_smoke_workflow(
        source_data=source_data,
        source_metadata=source_metadata,
        output_root=output_root,
        artifact_root=artifact_root,
        start="2024-01-01",
        end="2024-04-01",
        repo_root=repo_root,
    )

    generated_files = [path for path in (repo_root / "generated").rglob("*") if path.is_file()]
    assert generated_files
    for path in generated_files:
        assert path.is_relative_to(repo_root / "generated")


def _write_synthetic_upstream_artifacts(repo_root: Path) -> tuple[Path, Path]:
    source_root = repo_root / "upstream" / "data" / "curated" / "corporate_actions" / "dividends"
    source_root.mkdir(parents=True, exist_ok=True)
    data_path = source_root / "dividends.parquet"
    metadata_path = source_root / "metadata.json"

    rows = [
        {
            "corporate_action_id": "aapl-q1-cash",
            "symbol": " aapl ",
            "corporate_action_type": " cash_dividend ",
            "source": "synthetic_fixture",
            "process_date": "2024-02-01",
            "declaration_date": "2024-01-15",
            "ex_date": "2024-02-15",
            "record_date": "2024-02-16",
            "payable_date": "2024-03-01",
            "cash_amount": 0.24,
            "stock_amount": None,
            "currency": "USD",
            "source_payload_hash": "fixture-aapl-q1-cash",
            "raw": json.dumps({"symbol": "AAPL", "window": "2024Q1"}, sort_keys=True),
        },
        {
            "corporate_action_id": "aapl-q2-cash",
            "symbol": "AAPL",
            "corporate_action_type": "cash_dividend",
            "source": "synthetic_fixture",
            "process_date": "2024-04-01",
            "declaration_date": None,
            "ex_date": "2024-04-01",
            "record_date": None,
            "payable_date": None,
            "cash_amount": 0.25,
            "stock_amount": None,
            "currency": "USD",
            "source_payload_hash": "fixture-aapl-q2-cash",
            "raw": json.dumps({"symbol": "AAPL", "window": "outside"}, sort_keys=True),
        },
    ]
    pd.DataFrame(rows).to_parquet(data_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "source_vendor": "synthetic_fixture",
                "upstream_package_name": "fintech-market-ingestion",
                "upstream_project": "fintech-market-ingestion",
                "upstream_source_repository": "christophermoverton/fintech-market-ingestion",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return data_path, metadata_path
