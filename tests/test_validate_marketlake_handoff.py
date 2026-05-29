from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.cli.validate_marketlake_handoff import main as validate_marketlake_handoff_main
from src.cli.validate_marketlake_handoff import run_cli as run_validate_marketlake_handoff_cli
from src.validation.marketlake_handoff import validate_marketlake_handoff


def test_validate_marketlake_handoff_passes_on_valid_bundle(tmp_path: Path) -> None:
    project_root = tmp_path / "stratlake"
    marketlake_root = project_root / "data" / "curated"
    universe_path = project_root / "configs" / "universe.yml"
    paths_path = project_root / "configs" / "paths.yml"
    tickers_path = project_root / "configs" / "tickers_sample.txt"

    _write_valid_bundle(
        project_root=project_root,
        marketlake_root=marketlake_root,
        universe_path=universe_path,
        paths_path=paths_path,
        tickers_path=tickers_path,
    )

    before_files = _snapshot_files(project_root)
    result = validate_marketlake_handoff(
        root=project_root,
        marketlake_root=marketlake_root,
        universe=universe_path,
        start="2025-01-02",
        end="2025-01-04",
        timeframe="1D",
    )
    after_files = _snapshot_files(project_root)

    assert result.status == "pass"
    assert result.errors == ()
    assert result.warnings == ()
    assert result.coverage["coverage_pct"] == 1.0
    assert result.symbols["missing"] == []
    assert before_files == after_files
    assert result.to_dict()["checks"]


def test_validate_marketlake_handoff_warns_without_paths_config(tmp_path: Path) -> None:
    project_root = tmp_path / "stratlake"
    marketlake_root = project_root / "data" / "curated"
    universe_path = project_root / "configs" / "universe.yml"
    tickers_path = project_root / "configs" / "tickers_sample.txt"

    _write_dataset(
        marketlake_root=marketlake_root,
        rows=_rows_for_symbols(("AAPL", "MSFT"), ("2025-01-02", "2025-01-03")),
    )
    project_root.mkdir(parents=True, exist_ok=True)
    (project_root / "configs").mkdir(parents=True, exist_ok=True)
    tickers_path.write_text("AAPL\nMSFT\n", encoding="utf-8")
    universe_path.write_text(
        yaml.safe_dump(
            {"name": "demo", "tickers_file": "configs/tickers_sample.txt"}, sort_keys=True
        ),
        encoding="utf-8",
    )

    result = validate_marketlake_handoff(
        root=project_root,
        marketlake_root=marketlake_root,
        universe=universe_path,
        start="2025-01-02",
        end="2025-01-04",
        timeframe="1D",
    )

    assert result.status == "warn"
    assert any(
        check.name == "paths_config_alignment" and check.status == "warn" for check in result.checks
    )
    assert result.errors == ()


def test_validate_marketlake_handoff_fails_when_marketlake_root_looks_like_archive_pack(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "stratlake"
    marketlake_root = project_root / "data" / "curated"
    universe_path = project_root / "configs" / "universe.yml"
    paths_path = project_root / "configs" / "paths.yml"
    tickers_path = project_root / "configs" / "tickers_sample.txt"

    _write_valid_bundle(
        project_root=project_root,
        marketlake_root=marketlake_root,
        universe_path=universe_path,
        paths_path=paths_path,
        tickers_path=tickers_path,
    )
    (marketlake_root / "manifest.json").write_text("{}\n", encoding="utf-8")

    result = validate_marketlake_handoff(
        root=project_root,
        marketlake_root=marketlake_root,
        universe=universe_path,
        start="2025-01-02",
        end="2025-01-04",
        timeframe="1D",
    )

    assert result.status == "fail"
    assert any(
        check.name == "marketlake_root_not_archive_pack" and check.status == "fail"
        for check in result.checks
    )


def test_validate_marketlake_handoff_fails_on_missing_symbol(tmp_path: Path) -> None:
    project_root = tmp_path / "stratlake"
    marketlake_root = project_root / "data" / "curated"
    universe_path = project_root / "configs" / "universe.yml"
    paths_path = project_root / "configs" / "paths.yml"
    tickers_path = project_root / "configs" / "tickers_sample.txt"

    _write_dataset(
        marketlake_root=marketlake_root,
        rows=_rows_for_symbols(("AAPL",), ("2025-01-02", "2025-01-03")),
    )
    _write_bundle_configs(
        project_root=project_root,
        marketlake_root=marketlake_root,
        universe_path=universe_path,
        paths_path=paths_path,
        tickers_path=tickers_path,
        symbols=("AAPL", "MSFT"),
    )

    result = validate_marketlake_handoff(
        root=project_root,
        marketlake_root=marketlake_root,
        universe=universe_path,
        start="2025-01-02",
        end="2025-01-04",
        timeframe="1D",
    )

    assert result.status == "fail"
    assert result.symbols["missing"] == ["MSFT"]
    assert any(
        check.name == "symbol_coverage" and check.status == "fail" for check in result.checks
    )


def test_validate_marketlake_handoff_fails_on_missing_window_rows(tmp_path: Path) -> None:
    project_root = tmp_path / "stratlake"
    marketlake_root = project_root / "data" / "curated"
    universe_path = project_root / "configs" / "universe.yml"
    paths_path = project_root / "configs" / "paths.yml"
    tickers_path = project_root / "configs" / "tickers_sample.txt"

    _write_dataset(
        marketlake_root=marketlake_root,
        rows=_rows_for_symbols(("AAPL", "MSFT"), ("2025-01-01",)),
    )
    _write_bundle_configs(
        project_root=project_root,
        marketlake_root=marketlake_root,
        universe_path=universe_path,
        paths_path=paths_path,
        tickers_path=tickers_path,
        symbols=("AAPL", "MSFT"),
    )

    result = validate_marketlake_handoff(
        root=project_root,
        marketlake_root=marketlake_root,
        universe=universe_path,
        start="2025-01-02",
        end="2025-01-04",
        timeframe="1D",
    )

    assert result.status == "fail"
    assert any(
        check.name == "window_coverage" and check.status == "fail" for check in result.checks
    )
    assert result.coverage["window_row_count"] == 0


def test_validate_marketlake_handoff_fails_on_invalid_schema(tmp_path: Path) -> None:
    project_root = tmp_path / "stratlake"
    marketlake_root = project_root / "data" / "curated"
    universe_path = project_root / "configs" / "universe.yml"
    paths_path = project_root / "configs" / "paths.yml"
    tickers_path = project_root / "configs" / "tickers_sample.txt"

    project_root.mkdir(parents=True, exist_ok=True)
    (marketlake_root / "bars_daily").mkdir(parents=True, exist_ok=True)
    _write_bundle_configs(
        project_root=project_root,
        marketlake_root=marketlake_root,
        universe_path=universe_path,
        paths_path=paths_path,
        tickers_path=tickers_path,
        symbols=("AAPL",),
    )
    invalid_rows = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "ts_utc": pd.Timestamp("2025-01-02T16:00:00Z"),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "timeframe": "1D",
                "date": "2025-01-02",
            }
        ]
    )
    invalid_rows.to_parquet(marketlake_root / "bars_daily" / "part-0.parquet", index=False)

    result = validate_marketlake_handoff(
        root=project_root,
        marketlake_root=marketlake_root,
        universe=universe_path,
        start="2025-01-02",
        end="2025-01-04",
        timeframe="1D",
    )

    assert result.status == "fail"
    assert any(check.name == "dataset_schema" and check.status == "fail" for check in result.checks)


def test_validate_marketlake_handoff_cli_json_is_deterministic(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project_root = tmp_path / "stratlake"
    marketlake_root = project_root / "data" / "curated"
    universe_path = project_root / "configs" / "universe.yml"
    paths_path = project_root / "configs" / "paths.yml"
    tickers_path = project_root / "configs" / "tickers_sample.txt"

    _write_valid_bundle(
        project_root=project_root,
        marketlake_root=marketlake_root,
        universe_path=universe_path,
        paths_path=paths_path,
        tickers_path=tickers_path,
    )

    args = [
        "--root",
        str(project_root),
        "--marketlake-root",
        str(marketlake_root),
        "--universe",
        str(universe_path),
        "--start",
        "2025-01-02",
        "--end",
        "2025-01-04",
        "--timeframe",
        "1D",
        "--json",
    ]

    first = run_validate_marketlake_handoff_cli(args)
    first_output = capsys.readouterr().out
    second = run_validate_marketlake_handoff_cli(args)
    second_output = capsys.readouterr().out

    assert first == second
    assert first_output == second_output
    assert json.loads(first_output)["status"] == "pass"


def test_validate_marketlake_handoff_main_succeeds_for_valid_bundle(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project_root = tmp_path / "stratlake"
    marketlake_root = project_root / "data" / "curated"
    universe_path = project_root / "configs" / "universe.yml"
    paths_path = project_root / "configs" / "paths.yml"
    tickers_path = project_root / "configs" / "tickers_sample.txt"

    _write_valid_bundle(
        project_root=project_root,
        marketlake_root=marketlake_root,
        universe_path=universe_path,
        paths_path=paths_path,
        tickers_path=tickers_path,
    )

    exit_code = validate_marketlake_handoff_main(
        [
            "--root",
            str(project_root),
            "--marketlake-root",
            str(marketlake_root),
            "--universe",
            str(universe_path),
            "--start",
            "2025-01-02",
            "--end",
            "2025-01-04",
            "--timeframe",
            "1D",
            "--json",
        ]
    )
    capsys.readouterr()

    assert exit_code == 0


def test_validate_marketlake_handoff_output_outside_marketlake_root_succeeds(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "stratlake"
    marketlake_root = project_root / "data" / "curated"
    universe_path = project_root / "configs" / "universe.yml"
    paths_path = project_root / "configs" / "paths.yml"
    tickers_path = project_root / "configs" / "tickers_sample.txt"
    output_path = project_root / "artifacts" / "_derived" / "handoff_validation" / "report.json"

    _write_valid_bundle(
        project_root=project_root,
        marketlake_root=marketlake_root,
        universe_path=universe_path,
        paths_path=paths_path,
        tickers_path=tickers_path,
    )

    result = validate_marketlake_handoff(
        root=project_root,
        marketlake_root=marketlake_root,
        universe=universe_path,
        start="2025-01-02",
        end="2025-01-04",
        timeframe="1D",
        output=output_path,
    )

    assert result.status == "pass"
    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["authoritative"] is False
    assert payload["status"] == "pass"
    assert output_path.resolve() != marketlake_root.resolve()
    assert marketlake_root.resolve() not in output_path.resolve().parents


def test_validate_marketlake_handoff_output_inside_marketlake_root_is_rejected(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "stratlake"
    marketlake_root = project_root / "data" / "curated"
    universe_path = project_root / "configs" / "universe.yml"
    paths_path = project_root / "configs" / "paths.yml"
    tickers_path = project_root / "configs" / "tickers_sample.txt"
    output_path = marketlake_root / "handoff_report.json"

    _write_valid_bundle(
        project_root=project_root,
        marketlake_root=marketlake_root,
        universe_path=universe_path,
        paths_path=paths_path,
        tickers_path=tickers_path,
    )
    before_files = _snapshot_files(marketlake_root)

    with pytest.raises(
        ValueError, match="Refusing to write handoff validation report inside MarketLake root."
    ):
        validate_marketlake_handoff(
            root=project_root,
            marketlake_root=marketlake_root,
            universe=universe_path,
            start="2025-01-02",
            end="2025-01-04",
            timeframe="1D",
            output=output_path,
        )

    after_files = _snapshot_files(marketlake_root)
    assert not output_path.exists()
    assert before_files == after_files


def test_validate_marketlake_handoff_cli_output_inside_marketlake_root_surfaces_error(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "stratlake"
    marketlake_root = project_root / "data" / "curated"
    universe_path = project_root / "configs" / "universe.yml"
    paths_path = project_root / "configs" / "paths.yml"
    tickers_path = project_root / "configs" / "tickers_sample.txt"
    output_path = marketlake_root / "handoff_report.json"

    _write_valid_bundle(
        project_root=project_root,
        marketlake_root=marketlake_root,
        universe_path=universe_path,
        paths_path=paths_path,
        tickers_path=tickers_path,
    )

    with pytest.raises(
        ValueError, match="Refusing to write handoff validation report inside MarketLake root."
    ):
        validate_marketlake_handoff_main(
            [
                "--root",
                str(project_root),
                "--marketlake-root",
                str(marketlake_root),
                "--universe",
                str(universe_path),
                "--start",
                "2025-01-02",
                "--end",
                "2025-01-04",
                "--timeframe",
                "1D",
                "--output",
                str(output_path),
            ]
        )

    assert not output_path.exists()


def _write_valid_bundle(
    *,
    project_root: Path,
    marketlake_root: Path,
    universe_path: Path,
    paths_path: Path,
    tickers_path: Path,
) -> None:
    _write_dataset(
        marketlake_root=marketlake_root,
        rows=_rows_for_symbols(("AAPL", "MSFT"), ("2025-01-02", "2025-01-03")),
    )
    _write_bundle_configs(
        project_root=project_root,
        marketlake_root=marketlake_root,
        universe_path=universe_path,
        paths_path=paths_path,
        tickers_path=tickers_path,
        symbols=("AAPL", "MSFT"),
    )


def _write_bundle_configs(
    *,
    project_root: Path,
    marketlake_root: Path,
    universe_path: Path,
    paths_path: Path,
    tickers_path: Path,
    symbols: tuple[str, ...],
) -> None:
    project_root.mkdir(parents=True, exist_ok=True)
    (project_root / "configs").mkdir(parents=True, exist_ok=True)
    tickers_path.write_text("\n".join(symbols) + "\n", encoding="utf-8")
    universe_path.write_text(
        yaml.safe_dump(
            {
                "name": "demo",
                "tickers_file": "configs/tickers_sample.txt",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    paths_path.write_text(
        yaml.safe_dump(
            {
                "project_root": ".",
                "configs_root": "configs",
                "marketlake_root": marketlake_root.resolve().as_posix(),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _write_dataset(*, marketlake_root: Path, rows: list[dict[str, object]]) -> None:
    dataset_root = marketlake_root / "bars_daily"
    dataset_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(dataset_root / "part-0.parquet", index=False)


def _rows_for_symbols(symbols: tuple[str, ...], dates: tuple[str, ...]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for symbol in symbols:
        for date in dates:
            rows.append(
                {
                    "symbol": symbol,
                    "ts_utc": pd.Timestamp(f"{date}T16:00:00Z"),
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.5,
                    "volume": 1000,
                    "source": "synthetic",
                    "timeframe": "1D",
                    "date": date,
                }
            )
    return rows


def _snapshot_files(root: Path) -> set[str]:
    return {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
