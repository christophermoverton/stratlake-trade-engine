from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pandas.testing as pdt

from src.corporate_actions.dividend_importer import (
    load_dividend_events,
    import_dividend_events,
    normalize_upstream_dividend_events,
    write_dividend_event_dataset,
)


def _rows() -> list[dict[str, object]]:
    return [
        {
            "corporate_action_id": "b",
            "symbol": "msft",
            "corporate_action_type": "stock_dividend",
            "source": "alpaca",
            "process_date": "2024-03-02",
            "declaration_date": None,
            "ex_date": "2024-03-05",
            "record_date": None,
            "payable_date": None,
            "cash_amount": None,
            "stock_amount": 0.1,
            "currency": None,
            "source_payload_hash": "hash-b",
            "raw": {"id": "b"},
        },
        {
            "corporate_action_id": "a",
            "symbol": "aapl",
            "corporate_action_type": "cash_dividend",
            "source": "alpaca",
            "process_date": "2024-02-02",
            "declaration_date": None,
            "ex_date": "2024-02-05",
            "record_date": None,
            "payable_date": None,
            "cash_amount": 0.24,
            "stock_amount": None,
            "currency": "USD",
            "source_payload_hash": "hash-a",
            "raw": {"id": "a"},
        },
    ]


def _write_source(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    source.mkdir()
    data_path = source / "dividends.parquet"
    metadata_path = source / "metadata.json"
    safe_rows = []
    for row in reversed(_rows()):
        safe = dict(row)
        safe["raw"] = json.dumps(safe["raw"], sort_keys=True)
        safe_rows.append(safe)
    pd.DataFrame(safe_rows).to_parquet(data_path, index=False)
    metadata_path.write_text(json.dumps({"fixture": True}, sort_keys=True), encoding="utf-8")
    return data_path, metadata_path


def test_write_dividend_event_dataset_sorts_rows_and_uses_stable_part_names(tmp_path: Path) -> None:
    events = normalize_upstream_dividend_events(pd.DataFrame(_rows()))
    output_root = tmp_path / "events" / "dividends"

    partitions = write_dividend_event_dataset(events, output_root)

    assert partitions == (
        "symbol=AAPL/year=2024",
        "symbol=MSFT/year=2024",
    )
    assert sorted(path.name for path in output_root.rglob("*.parquet")) == [
        "part-0.parquet",
        "part-0.parquet",
    ]
    output = pd.read_parquet(output_root).sort_values(["symbol", "ex_date"]).reset_index(drop=True)
    assert output["source_event_id"].tolist() == ["a", "b"]


def test_import_dividend_events_rerun_produces_equivalent_output_and_metadata(tmp_path: Path) -> None:
    data_path, metadata_path = _write_source(tmp_path)
    output_root = tmp_path / "events" / "dividends"

    first = import_dividend_events(
        source_data_path=data_path,
        source_metadata_path=metadata_path,
        output_root=output_root,
        start="2024-01-01",
        end="2025-01-01",
    )
    first_frame = pd.read_parquet(output_root).sort_values(["symbol", "source_event_id"]).reset_index(drop=True)
    first_digest = _logical_dataset_digest(output_root)

    second = import_dividend_events(
        source_data_path=data_path,
        source_metadata_path=metadata_path,
        output_root=output_root,
        start="2024-01-01",
        end="2025-01-01",
    )
    second_frame = pd.read_parquet(output_root).sort_values(["symbol", "source_event_id"]).reset_index(drop=True)
    second_digest = _logical_dataset_digest(output_root)

    assert first.to_dict() == second.to_dict()
    assert first_digest == second_digest
    pdt.assert_frame_equal(first_frame, second_frame)


def test_write_dividend_event_dataset_replaces_previous_snapshot(tmp_path: Path) -> None:
    output_root = tmp_path / "events" / "dividends"

    wider = normalize_upstream_dividend_events(pd.DataFrame(_rows()))
    write_dividend_event_dataset(wider, output_root)

    narrower = wider.loc[
        (wider["symbol"] == "AAPL") & (wider["year"] == "2024"),
        :,
    ].reset_index(drop=True)
    write_dividend_event_dataset(narrower, output_root)

    loaded = load_dividend_events(output_root)
    loaded_normalized = loaded.assign(
        symbol=loaded["symbol"].astype(str),
        year=loaded["year"].astype(str),
    ).reset_index(drop=True)
    expected = narrower.loc[:, loaded_normalized.columns].assign(
        symbol=narrower["symbol"].astype(str),
        year=narrower["year"].astype(str),
    ).reset_index(drop=True)
    pdt.assert_frame_equal(loaded_normalized, expected, check_dtype=False)
    assert not (output_root / "symbol=MSFT").exists()

    first_digest = _logical_dataset_digest(output_root)
    write_dividend_event_dataset(narrower, output_root)
    second_digest = _logical_dataset_digest(output_root)
    assert first_digest == second_digest


def _logical_dataset_digest(root: Path) -> str:
    frame = pd.read_parquet(root).sort_values(["symbol", "source_event_id"]).reset_index(drop=True)
    payload = frame.to_json(orient="records", date_format="iso")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
