"""CI-safe M40 dividend evidence import example.

This example creates tiny local upstream artifacts, imports them through the
public Python API, and prints a deterministic summary. It does not call Alpaca,
read credentials, use network access, or create adjusted price data.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.corporate_actions import import_dividend_events, load_dividend_events


EXAMPLE_ROOT = Path("docs/examples/output/m40_dividend_events")


def run_m40_dividend_evidence_import_example() -> dict[str, Any]:
    fixture_root = EXAMPLE_ROOT / "fixtures" / "corporate_actions"
    output_root = EXAMPLE_ROOT / "data"
    artifact_root = EXAMPLE_ROOT / "artifacts"
    source_data, source_metadata = write_synthetic_dividend_fixture(fixture_root)

    result = import_dividend_events(
        source_data_path=source_data,
        source_metadata_path=source_metadata,
        output_root=output_root,
        artifact_root=artifact_root,
        start="2024-01-01",
        end="2025-01-01",
        strict=True,
    )
    loaded = load_dividend_events(output_root)
    qa_summary = json.loads((artifact_root / result.run_id / "qa_summary.json").read_text(encoding="utf-8"))

    return {
        "artifact_path": result.artifact_path,
        "dataset_root": result.output_root,
        "loaded_row_count": int(len(loaded)),
        "qa_status": qa_summary["qa_status"],
        "run_id": result.run_id,
        "written_row_count": result.written_row_count,
    }


def write_synthetic_dividend_fixture(fixture_root: Path) -> tuple[Path, Path]:
    fixture_root.mkdir(parents=True, exist_ok=True)
    data_path = fixture_root / "dividends.parquet"
    metadata_path = fixture_root / "metadata.json"
    rows = [
        {
            "corporate_action_id": "cash-2024-aapl",
            "symbol": "AAPL",
            "corporate_action_type": "cash_dividend",
            "source": "synthetic_fixture",
            "process_date": "2024-02-01",
            "declaration_date": "2024-01-15",
            "ex_date": "2024-02-15",
            "record_date": "2024-02-16",
            "payable_date": "2024-03-01",
            "cash_amount": 0.24,
            "stock_amount": None,
            "currency": "USD",
            "source_payload_hash": "fixture-hash-cash-aapl",
            "raw": json.dumps({"fixture": "cash"}, sort_keys=True),
        },
        {
            "corporate_action_id": "stock-2024-msft",
            "symbol": "MSFT",
            "corporate_action_type": "stock_dividend",
            "source": "synthetic_fixture",
            "process_date": "2024-06-01",
            "declaration_date": None,
            "ex_date": "2024-06-15",
            "record_date": None,
            "payable_date": None,
            "cash_amount": None,
            "stock_amount": 0.05,
            "currency": None,
            "source_payload_hash": "fixture-hash-stock-msft",
            "raw": json.dumps({"fixture": "stock"}, sort_keys=True),
        },
    ]
    pd.DataFrame(rows).to_parquet(data_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "source_vendor": "synthetic_fixture",
                "upstream_package_name": "fintech-market-ingestion",
                "upstream_project": "fintech-market-ingestion",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return data_path, metadata_path


if __name__ == "__main__":
    print(json.dumps(run_m40_dividend_evidence_import_example(), indent=2, sort_keys=True))
