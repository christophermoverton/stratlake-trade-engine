from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import yaml

from cli.build_features import (
    build_summary,
    compute_missingness,
    load_tickers,
    parse_args,
    resolve_input_partitions,
    run_cli,
    write_summary,
)
from src.config.settings import Settings


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        marketlake_root=tmp_path / "marketlake",
        features_root=tmp_path / "features",
        artifacts_root=tmp_path / "artifacts",
        duckdb_path=":memory:",
        log_level="INFO",
        default_timezone="UTC",
        paths_config={},
        universe_config={},
        features_config={},
    )


def _features_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": pd.Series(["AAPL", "MSFT"], dtype="string"),
            "ts_utc": pd.to_datetime(["2025-11-01T14:30:00Z", "2025-11-01T14:31:00Z"], utc=True),
            "timeframe": pd.Series(["1Min", "1Min"], dtype="string"),
            "feature_alpha": [1.0, None],
            "feature_beta": [None, None],
        }
    )


def _write_yaml(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle)


def test_parse_args_and_load_tickers(tmp_path: Path) -> None:
    tickers_file = tmp_path / "tickers.txt"
    tickers_file.write_text(" AAPL \n\nMSFT\n", encoding="utf-8")

    args = parse_args(
        ["--timeframe", "1Min", "--start", "2025-11-01", "--end", "2025-12-01", "--tickers", str(tickers_file)]
    )

    assert args.timeframe == "1Min"
    assert args.start == "2025-11-01"
    assert args.end == "2025-12-01"
    assert args.tickers == str(tickers_file)
    assert load_tickers(tickers_file) == ["AAPL", "MSFT"]


def test_compute_missingness_summary() -> None:
    summary = compute_missingness(_features_df())

    assert summary["feature_alpha"]["missing_count"] == 1
    assert summary["feature_alpha"]["missing_fraction"] == 0.5
    assert summary["feature_beta"]["missing_count"] == 2
    assert summary["feature_beta"]["missing_fraction"] == 1.0


def test_write_summary_creates_json(tmp_path: Path) -> None:
    summary = build_summary(
        run_id="run-123",
        timeframe="1D",
        start="2025-11-01",
        end="2025-11-03",
        tickers_file="configs/tickers.txt",
        requested_symbols=["AAPL"],
        features=_features_df().iloc[0:1].assign(timeframe="1D"),
        marketlake_root=tmp_path / "marketlake",
        input_partitions_used=["partition-a"],
    )

    summary_path = write_summary(summary, tmp_path / "artifacts" / "feature_runs" / "run-123")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary_path == tmp_path / "artifacts" / "feature_runs" / "run-123" / "summary.json"
    assert payload["run_id"] == "run-123"
    assert payload["symbols_processed"] == ["AAPL"]
    assert payload["input_partitions_used"] == ["partition-a"]
    assert "missingness_by_feature_column" in payload


def test_resolve_input_partitions_filters_by_symbol_and_date(tmp_path: Path) -> None:
    marketlake_root = tmp_path / "marketlake"
    partition_a = marketlake_root / "bars_1m" / "symbol=AAPL" / "date=2025-11-01"
    partition_b = marketlake_root / "bars_1m" / "symbol=AAPL" / "date=2025-12-01"
    partition_c = marketlake_root / "bars_1m" / "symbol=MSFT" / "date=2025-11-01"
    partition_a.mkdir(parents=True)
    partition_b.mkdir(parents=True)
    partition_c.mkdir(parents=True)
    (partition_a / "part-0.parquet").write_text("stub", encoding="utf-8")
    (partition_b / "part-0.parquet").write_text("stub", encoding="utf-8")
    (partition_c / "part-0.parquet").write_text("stub", encoding="utf-8")

    partitions = resolve_input_partitions("1Min", ["AAPL"], "2025-11-01", "2025-12-01", marketlake_root)

    assert partitions == [str(partition_a.resolve())]


def test_run_cli_dispatches_minute_pipeline_and_logs_metadata(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    tickers_file = tmp_path / "tickers.txt"
    tickers_file.write_text("AAPL\nMSFT\n", encoding="utf-8")

    settings = _settings(tmp_path)
    used_partition = settings.marketlake_root / "bars_1m" / "symbol=AAPL" / "date=2025-11-01"
    used_partition.mkdir(parents=True, exist_ok=True)
    (used_partition / "part-0.parquet").write_text("stub", encoding="utf-8")

    calls: dict[str, object] = {}

    def fake_minute_pipeline(symbols, *, start_date=None, end_date=None, **kwargs):
        calls["minute"] = {
            "symbols": symbols,
            "start_date": start_date,
            "end_date": end_date,
        }
        return _features_df()

    monkeypatch.setattr("cli.build_features.Settings.load", lambda: settings)
    monkeypatch.setattr("cli.build_features.run_minute_feature_pipeline", fake_minute_pipeline)
    monkeypatch.setattr(
        "cli.build_features.run_daily_feature_pipeline",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("daily pipeline should not run")),
    )
    monkeypatch.setattr("cli.build_features.generate_run_id", lambda now=None: "run-minute")

    caplog.set_level(logging.INFO)
    summary_path = run_cli(
        ["--timeframe", "1Min", "--start", "2025-11-01", "--end", "2025-12-01", "--tickers", str(tickers_file)]
    )

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert calls["minute"] == {
        "symbols": ["AAPL", "MSFT"],
        "start_date": "2025-11-01",
        "end_date": "2025-12-01",
    }
    assert payload["run_id"] == "run-minute"
    assert payload["timeframe"] == "1Min"
    assert payload["symbols_requested"] == ["AAPL", "MSFT"]
    assert payload["symbols_processed"] == ["AAPL", "MSFT"]
    assert payload["feature_row_count"] == 2
    assert payload["input_partitions_used"] == [str(used_partition.resolve())]
    assert "Resolved MARKETLAKE_ROOT=" in caplog.text
    assert "Resolved timeframe=1Min" in caplog.text
    assert "Input partitions used=" in caplog.text
    assert "Run ID=run-minute" in caplog.text


def test_run_cli_dispatches_daily_pipeline(tmp_path: Path, monkeypatch) -> None:
    tickers_file = tmp_path / "tickers.txt"
    tickers_file.write_text("AAPL\n", encoding="utf-8")

    settings = _settings(tmp_path)
    calls: dict[str, object] = {}

    def fake_daily_pipeline(symbols, *, start_date=None, end_date=None, **kwargs):
        calls["daily"] = {
            "symbols": symbols,
            "start_date": start_date,
            "end_date": end_date,
        }
        return _features_df().iloc[0:1].assign(timeframe="1D")

    monkeypatch.setattr("cli.build_features.Settings.load", lambda: settings)
    monkeypatch.setattr("cli.build_features.run_daily_feature_pipeline", fake_daily_pipeline)
    monkeypatch.setattr(
        "cli.build_features.run_minute_feature_pipeline",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("minute pipeline should not run")),
    )
    monkeypatch.setattr("cli.build_features.generate_run_id", lambda now=None: "run-daily")

    summary_path = run_cli(
        ["--timeframe", "1D", "--start", "2025-11-01", "--end", "2025-12-01", "--tickers", str(tickers_file)]
    )

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert calls["daily"] == {
        "symbols": ["AAPL"],
        "start_date": "2025-11-01",
        "end_date": "2025-12-01",
    }
    assert summary_path == settings.artifacts_root / "feature_runs" / "run-daily" / "summary.json"
    assert payload["timeframe"] == "1D"


def test_run_cli_uses_env_when_paths_yaml_is_missing(tmp_path: Path, monkeypatch) -> None:
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    _write_yaml(configs_dir / "universe.yml", {})
    _write_yaml(configs_dir / "features.yml", {})

    tickers_file = tmp_path / "tickers.txt"
    tickers_file.write_text("AAPL\n", encoding="utf-8")

    marketlake_root = tmp_path / "env-marketlake"
    artifacts_root = tmp_path / "env-artifacts"
    marketlake_root.mkdir()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MARKETLAKE_ROOT", str(marketlake_root))
    monkeypatch.setenv("ARTIFACTS_ROOT", str(artifacts_root))
    monkeypatch.setenv("FEATURES_ROOT", str(tmp_path / "env-features"))
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    monkeypatch.setenv("DEFAULT_TIMEZONE", "UTC")

    monkeypatch.setattr(
        "cli.build_features.run_minute_feature_pipeline",
        lambda *args, **kwargs: _features_df().iloc[0:1],
    )
    monkeypatch.setattr("cli.build_features.generate_run_id", lambda now=None: "run-env-fallback")

    summary_path = run_cli(
        ["--timeframe", "1Min", "--start", "2025-11-01", "--end", "2025-12-01", "--tickers", str(tickers_file)]
    )

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_path == artifacts_root / "feature_runs" / "run-env-fallback" / "summary.json"
    assert payload["marketlake_root"] == str(marketlake_root)
