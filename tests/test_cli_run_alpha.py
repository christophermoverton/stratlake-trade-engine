from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.cli.run_alpha import AlphaRunResult, FULL_RUN_SCAFFOLD_FILE, parse_args, run_cli


def _write_builtin_alpha_dataset(root: Path) -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01", periods=6, freq="D", tz="UTC")
    rows: list[dict[str, object]] = []
    symbols = ["AAA", "BBB", "CCC"]
    for symbol_index, symbol in enumerate(symbols):
        base = float(symbol_index + 1)
        for ts_index, ts_utc in enumerate(timestamps):
            rows.append(
                {
                    "symbol": symbol,
                    "ts_utc": ts_utc,
                    "timeframe": "1D",
                    "date": ts_utc.strftime("%Y-%m-%d"),
                    "feature_ret_1d": base * 1.5 + ts_index * 0.2,
                    "feature_ret_5d": base * 1.0 + ts_index * 0.15,
                    "feature_ret_20d": base * 0.5 + ts_index * 0.1,
                    "target_ret_1d": base * 0.01 + ts_index * 0.001,
                    "target_ret_5d": base * 0.02 + ts_index * 0.002,
                    "close": 100.0 + base * 10.0 + ts_index * (base + 1.0),
                }
            )

    frame = pd.DataFrame(rows).sort_values(["symbol", "ts_utc"], kind="stable").reset_index(drop=True)
    for column in ("symbol", "timeframe", "date"):
        frame[column] = frame[column].astype("string")
    numeric_columns = [
        "feature_ret_1d",
        "feature_ret_5d",
        "feature_ret_20d",
        "target_ret_1d",
        "target_ret_5d",
        "close",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("float64")

    for symbol in symbols:
        symbol_frame = frame.loc[frame["symbol"].eq(symbol)].copy(deep=True)
        dataset_dir = root / "data" / "curated" / "features_daily" / f"symbol={symbol}" / "year=2025"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        symbol_frame.to_parquet(dataset_dir / "part-0.parquet", index=False)
    return frame


def test_parse_args_requires_builtin_alpha_name() -> None:
    with pytest.raises(SystemExit):
        parse_args([])


def test_run_cli_supports_evaluation_only_mode_for_builtin_alpha(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_builtin_alpha_dataset(tmp_path)
    monkeypatch.chdir(tmp_path)

    result = run_cli(
        [
            "--alpha-name",
            "cs_linear_ret_1d",
            "--mode",
            "evaluate",
            "--start",
            "2025-01-01",
            "--end",
            "2025-01-07",
        ]
    )

    assert isinstance(result, AlphaRunResult)
    assert result.mode == "evaluate"
    assert result.alpha_name == "cs_linear_ret_1d"
    assert result.scaffold_path is None
    assert result.artifact_dir.exists()
    assert (result.artifact_dir / "alpha_metrics.json").exists()
    assert not (result.artifact_dir / FULL_RUN_SCAFFOLD_FILE).exists()

    stdout = capsys.readouterr().out
    assert "alpha_name: cs_linear_ret_1d" in stdout
    assert "mode: evaluate" in stdout
    assert f"run_id: {result.run_id}" in stdout
    assert f"artifact_dir: {result.artifact_dir.as_posix()}" in stdout


def test_run_cli_full_mode_writes_deterministic_scaffold_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_builtin_alpha_dataset(tmp_path)
    monkeypatch.chdir(tmp_path)

    first = run_cli(
        [
            "--alpha-name",
            "cs_linear_ret_1d",
            "--start",
            "2025-01-01",
            "--end",
            "2025-01-07",
        ]
    )
    second = run_cli(
        [
            "--alpha-name",
            "cs_linear_ret_1d",
            "--start",
            "2025-01-01",
            "--end",
            "2025-01-07",
        ]
    )

    assert first.mode == "full"
    assert first.run_id == second.run_id
    assert first.scaffold_path is not None
    assert second.scaffold_path is not None
    assert first.scaffold_path.read_bytes() == second.scaffold_path.read_bytes()
    assert (first.artifact_dir / "signals.parquet").exists()
    assert (first.artifact_dir / "sleeve_returns.csv").exists()
    assert (first.artifact_dir / "sleeve_equity_curve.csv").exists()
    assert (first.artifact_dir / "sleeve_metrics.json").exists()
    assert (first.artifact_dir / "signals.parquet").read_bytes() == (second.artifact_dir / "signals.parquet").read_bytes()
    assert (first.artifact_dir / "sleeve_returns.csv").read_bytes() == (second.artifact_dir / "sleeve_returns.csv").read_bytes()
    assert (first.artifact_dir / "sleeve_equity_curve.csv").read_bytes() == (second.artifact_dir / "sleeve_equity_curve.csv").read_bytes()
    assert (first.artifact_dir / "sleeve_metrics.json").read_bytes() == (second.artifact_dir / "sleeve_metrics.json").read_bytes()

    payload = json.loads(first.scaffold_path.read_text(encoding="utf-8"))
    assert payload["alpha_name"] == "cs_linear_ret_1d"
    assert payload["run_id"] == first.run_id
    assert payload["status"] == "completed"
    assert payload["next_stage"] is None
    assert payload["signal_mapping"]["config"]["policy"] == "rank_long_short"
    assert payload["sleeve"]["sleeve_returns_path"] == "sleeve_returns.csv"
    assert payload["sleeve"]["sleeve_equity_curve_path"] == "sleeve_equity_curve.csv"
    assert payload["sleeve"]["sleeve_metrics_path"] == "sleeve_metrics.json"
    assert payload["resolved_config"]["dataset"] == "features_daily"


def test_run_cli_rejects_unknown_builtin_alpha(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_builtin_alpha_dataset(tmp_path)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="Unknown alpha 'does_not_exist'"):
        run_cli(["--alpha-name", "does_not_exist"])


def test_run_cli_rejects_non_daily_dataset_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_builtin_alpha_dataset(tmp_path)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="require dataset 'features_daily'"):
        run_cli(
            [
                "--alpha-name",
                "cs_linear_ret_1d",
                "--dataset",
                "features_1m",
            ]
        )
