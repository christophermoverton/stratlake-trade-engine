from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.config.settings import Settings
from src.pipeline.feature_pipeline import (
    run_daily_feature_pipeline,
    run_minute_feature_pipeline,
)

LOGGER = logging.getLogger(__name__)
SUPPORTED_TIMEFRAMES = ("1Min", "1D")
NON_FEATURE_COLUMNS = {"symbol", "ts_utc", "timeframe", "date"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build feature datasets from curated marketlake bars.")
    parser.add_argument("--timeframe", choices=SUPPORTED_TIMEFRAMES, required=True)
    parser.add_argument("--start", required=True, dest="start")
    parser.add_argument("--end", required=True, dest="end")
    parser.add_argument("--tickers", required=True, dest="tickers")
    return parser.parse_args(argv)


def load_tickers(tickers_file: str | Path) -> list[str]:
    path = Path(tickers_file)
    tickers = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [ticker for ticker in tickers if ticker]


def generate_run_id(now: datetime | None = None) -> str:
    current = now or datetime.now(timezone.utc)
    return current.strftime("%Y%m%dT%H%M%SZ")


def compute_missingness(df: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    feature_columns = [column for column in df.columns if column not in NON_FEATURE_COLUMNS]
    if df.empty:
        return {
            column: {
                "missing_count": 0,
                "missing_fraction": 0.0,
            }
            for column in feature_columns
        }

    row_count = len(df.index)
    summary: dict[str, dict[str, float | int]] = {}
    for column in feature_columns:
        missing_count = int(df[column].isna().sum())
        summary[column] = {
            "missing_count": missing_count,
            "missing_fraction": missing_count / row_count,
        }
    return summary


def resolve_input_partitions(
    timeframe: str,
    symbols: Sequence[str],
    start_date: str,
    end_date: str,
    marketlake_root: Path,
) -> list[str]:
    dataset = "bars_1m" if timeframe == "1Min" else "bars_daily"
    selected_symbols = {symbol.strip().upper() for symbol in symbols if symbol and symbol.strip()}
    partitions: set[str] = set()

    dataset_root = Path(marketlake_root) / dataset
    if not dataset_root.exists():
        return []

    for parquet_file in dataset_root.glob("symbol=*/date=*/*.parquet"):
        symbol_value = parquet_file.parent.parent.name.removeprefix("symbol=")
        date_value = parquet_file.parent.name.removeprefix("date=")
        if selected_symbols and symbol_value.upper() not in selected_symbols:
            continue
        if date_value < start_date or date_value >= end_date:
            continue
        partitions.add(str(parquet_file.parent.resolve()))

    return sorted(partitions)


def write_summary(summary: dict[str, Any], artifact_root: Path) -> Path:
    artifact_root.mkdir(parents=True, exist_ok=True)
    summary_path = artifact_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def build_summary(
    *,
    run_id: str,
    timeframe: str,
    start: str,
    end: str,
    tickers_file: str | Path,
    requested_symbols: Sequence[str],
    features: pd.DataFrame,
    marketlake_root: Path,
    input_partitions_used: Sequence[str],
) -> dict[str, Any]:
    processed_symbols = []
    if "symbol" in features.columns and not features.empty:
        processed_symbols = sorted(features["symbol"].dropna().astype(str).unique().tolist())

    return {
        "run_id": run_id,
        "timeframe": timeframe,
        "start": start,
        "end": end,
        "tickers_file": str(Path(tickers_file)),
        "symbols_requested": list(requested_symbols),
        "symbols_processed": processed_symbols,
        "feature_row_count": int(len(features.index)),
        "marketlake_root": str(marketlake_root),
        "input_partitions_used": list(input_partitions_used),
        "missingness_by_feature_column": compute_missingness(features),
    }


def configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def run_cli(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    settings = Settings.load()
    configure_logging(settings.log_level)

    tickers = load_tickers(args.tickers)
    run_id = generate_run_id()
    input_partitions = resolve_input_partitions(
        timeframe=args.timeframe,
        symbols=tickers,
        start_date=args.start,
        end_date=args.end,
        marketlake_root=settings.marketlake_root,
    )

    LOGGER.info("Resolved MARKETLAKE_ROOT=%s", settings.marketlake_root)
    LOGGER.info("Resolved timeframe=%s", args.timeframe)
    LOGGER.info("Input partitions used=%s", input_partitions)
    LOGGER.info("Run ID=%s", run_id)

    pipeline = run_minute_feature_pipeline if args.timeframe == "1Min" else run_daily_feature_pipeline
    features = pipeline(
        tickers,
        start_date=args.start,
        end_date=args.end,
    )

    summary = build_summary(
        run_id=run_id,
        timeframe=args.timeframe,
        start=args.start,
        end=args.end,
        tickers_file=args.tickers,
        requested_symbols=tickers,
        features=features,
        marketlake_root=settings.marketlake_root,
        input_partitions_used=input_partitions,
    )
    artifact_root = settings.artifacts_root / "feature_runs" / run_id
    summary_path = write_summary(summary, artifact_root)
    LOGGER.info("Summary artifact written=%s", summary_path)
    return summary_path


def main() -> None:
    run_cli()


if __name__ == "__main__":
    main()
