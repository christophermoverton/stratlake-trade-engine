from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pandas.testing as pdt
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.research.alpha.registry as alpha_registry
from src.cli.run_alpha_evaluation import (
    AlphaEvaluationRunResult,
    load_alpha_evaluation_config,
    parse_args,
    run_cli,
)
from src.research.alpha.base import BaseAlphaModel
from src.research.alpha_eval import alpha_evaluation_registry_path
from src.research.registry import load_registry

TARGET_COLUMN = "target_ret_1d"
MODEL_NAME = "cli_alpha_eval_weighted_model"


class CliAlphaEvalWeightedModel(BaseAlphaModel):
    name = MODEL_NAME

    def __init__(self) -> None:
        self.feature_columns: list[str] = []
        self.feature_means: dict[str, float] = {}

    def _fit(self, df: pd.DataFrame) -> None:
        self.feature_columns = [
            column
            for column in df.columns
            if column.startswith("feature_") and column not in {"symbol", "ts_utc", TARGET_COLUMN}
        ]
        self.feature_means = {
            column: float(pd.to_numeric(df[column], errors="coerce").mean())
            for column in self.feature_columns
        }

    def _predict(self, df: pd.DataFrame) -> pd.Series:
        score = pd.Series(0.0, index=df.index, dtype="float64")
        for idx, column in enumerate(self.feature_columns, start=1):
            centered = pd.to_numeric(df[column], errors="coerce").astype("float64") - self.feature_means[column]
            score = score + centered * float(idx)
        return score.rename("prediction")


class ConstantCliAlphaEvalModel(BaseAlphaModel):
    name = "constant_cli_alpha_eval_model"

    def _fit(self, df: pd.DataFrame) -> None:
        return None

    def _predict(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=df.index, dtype="float64", name="prediction")


@pytest.fixture(autouse=True)
def reset_alpha_registry() -> None:
    original_registry = dict(alpha_registry._ALPHA_MODEL_REGISTRY)
    alpha_registry._ALPHA_MODEL_REGISTRY.clear()
    alpha_registry._ALPHA_MODEL_REGISTRY.update(original_registry)
    yield
    alpha_registry._ALPHA_MODEL_REGISTRY.clear()
    alpha_registry._ALPHA_MODEL_REGISTRY.update(original_registry)


def _write_alpha_eval_dataset(root: Path) -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01", periods=5, freq="D", tz="UTC")
    rows: list[dict[str, object]] = []
    symbols = ["AAA", "BBB", "CCC"]
    for symbol_index, symbol in enumerate(symbols):
        for ts_index, ts_utc in enumerate(timestamps):
            feature_alpha = float((symbol_index + 1) * 10 + ts_index)
            feature_beta = float((symbol_index + 1) * 5 - ts_index)
            close = float((symbol_index + 2) * 100 + (ts_index * (symbol_index + 2)) ** 2 + ts_index)
            rows.append(
                {
                    "symbol": symbol,
                    "ts_utc": ts_utc,
                    "timeframe": "1D",
                    "date": ts_utc.strftime("%Y-%m-%d"),
                    "feature_alpha": feature_alpha,
                    "feature_beta": feature_beta,
                    TARGET_COLUMN: float((symbol_index + 1) * 0.01 + ts_index * 0.001),
                    "close": close,
                }
            )

    frame = pd.DataFrame(rows).sort_values(["symbol", "ts_utc"], kind="stable").reset_index(drop=True)
    for column in ("symbol", "timeframe", "date"):
        frame[column] = frame[column].astype("string")
    for column in ("feature_alpha", "feature_beta", TARGET_COLUMN, "close"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("float64")

    for symbol in symbols:
        symbol_frame = frame.loc[frame["symbol"].eq(symbol)].copy(deep=True)
        dataset_dir = root / "data" / "curated" / "features_daily" / f"symbol={symbol}" / "year=2025"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        symbol_frame.to_parquet(dataset_dir / "part-0.parquet", index=False)
    return frame


def test_parse_args_accepts_alpha_eval_inputs() -> None:
    args = parse_args(
        [
            "--alpha-model",
            MODEL_NAME,
            "--dataset",
            "features_daily",
            "--target-column",
            TARGET_COLUMN,
            "--price-column",
            "close",
            "--alpha-horizon",
            "2",
            "--tickers",
            "configs/tickers_50.txt",
        ]
    )

    assert args.alpha_model == MODEL_NAME
    assert args.dataset == "features_daily"
    assert args.target_column == TARGET_COLUMN
    assert args.price_column == "close"
    assert args.alpha_horizon == 2
    assert args.tickers == "configs/tickers_50.txt"


def test_load_alpha_evaluation_config_supports_nested_section(tmp_path: Path) -> None:
    config_path = tmp_path / "alpha_eval.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "alpha_evaluation": {
                    "alpha_model": MODEL_NAME,
                    "dataset": "features_daily",
                    "target_column": TARGET_COLUMN,
                    "price_column": "close",
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    payload = load_alpha_evaluation_config(config_path)

    assert payload == {
        "alpha_model": MODEL_NAME,
        "dataset": "features_daily",
        "target_column": TARGET_COLUMN,
        "price_column": "close",
    }


def test_run_cli_executes_full_alpha_evaluation_pipeline_and_persists_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_alpha_eval_dataset(tmp_path)
    monkeypatch.chdir(tmp_path)

    result = run_cli(
        [
            "--alpha-model",
            MODEL_NAME,
            "--model-class",
            f"{__name__}:CliAlphaEvalWeightedModel",
            "--dataset",
            "features_daily",
            "--target-column",
            TARGET_COLUMN,
            "--feature-columns",
            "feature_alpha",
            "feature_beta",
            "--price-column",
            "close",
            "--start",
            "2025-01-01",
            "--end",
            "2025-01-06",
        ]
    )

    assert isinstance(result, AlphaEvaluationRunResult)
    assert result.alpha_name == MODEL_NAME
    assert result.evaluation_result.summary["n_periods"] == 4
    assert result.artifact_dir.exists()
    assert (result.artifact_dir / "alpha_metrics.json").exists()
    assert (result.artifact_dir / "ic_timeseries.csv").exists()
    assert (result.artifact_dir / "manifest.json").exists()

    metrics_payload = json.loads((result.artifact_dir / "alpha_metrics.json").read_text(encoding="utf-8"))
    assert metrics_payload["metadata"]["alpha_name"] == MODEL_NAME
    assert metrics_payload["metadata"]["run_id"] == result.run_id
    assert metrics_payload["mean_ic"] == pytest.approx(result.evaluation_result.summary["mean_ic"])
    assert metrics_payload["ic_ir"] == pytest.approx(result.evaluation_result.summary["ic_ir"])
    assert metrics_payload["n_periods"] == result.evaluation_result.summary["n_periods"]

    registry_entries = load_registry(alpha_evaluation_registry_path(tmp_path / "artifacts" / "alpha"))
    assert len(registry_entries) == 1
    assert registry_entries[0]["run_id"] == result.run_id
    assert registry_entries[0]["alpha_name"] == MODEL_NAME
    assert registry_entries[0]["artifact_path"] == result.artifact_dir.as_posix()
    assert registry_entries[0]["metrics_path"] == (result.artifact_dir / "alpha_metrics.json").as_posix()
    assert registry_entries[0]["ic_timeseries_path"] == (result.artifact_dir / "ic_timeseries.csv").as_posix()

    stdout = capsys.readouterr().out
    assert f"alpha_model: {MODEL_NAME}" in stdout
    assert f"run_id: {result.run_id}" in stdout
    assert f"mean_ic: {result.evaluation_result.summary['mean_ic']:.6f}" in stdout
    assert f"ic_ir: {result.evaluation_result.summary['ic_ir']:.6f}" in stdout
    assert "n_periods: 4" in stdout


def test_run_cli_supports_config_and_ticker_file_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_alpha_eval_dataset(tmp_path)
    tickers_path = tmp_path / "tickers.txt"
    tickers_path.write_text("AAA\nCCC\n", encoding="utf-8")
    config_path = tmp_path / "alpha_eval.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "alpha_evaluation": {
                    "alpha_model": MODEL_NAME,
                    "dataset": "features_daily",
                    "target_column": TARGET_COLUMN,
                    "feature_columns": ["feature_alpha", "feature_beta"],
                    "price_column": "close",
                    "alpha_horizon": 1,
                    "tickers": ["AAA", "BBB", "CCC"],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    result = run_cli(
        [
            "--config",
            str(config_path),
            "--model-class",
            f"{__name__}:CliAlphaEvalWeightedModel",
            "--tickers",
            str(tickers_path),
        ]
    )

    assert sorted(result.loaded_frame["symbol"].astype(str).unique().tolist()) == ["AAA", "CCC"]
    assert result.evaluation_result.symbol_count == 2


def test_run_cli_is_deterministic_for_repeated_identical_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_alpha_eval_dataset(tmp_path)
    monkeypatch.chdir(tmp_path)

    argv = [
        "--alpha-model",
        MODEL_NAME,
        "--model-class",
        f"{__name__}:CliAlphaEvalWeightedModel",
        "--dataset",
        "features_daily",
        "--target-column",
        TARGET_COLUMN,
        "--feature-columns",
        "feature_alpha",
        "feature_beta",
        "--price-column",
        "close",
        "--start",
        "2025-01-01",
        "--end",
        "2025-01-06",
    ]
    first = run_cli(argv)
    first_snapshot = {
        path.relative_to(first.artifact_dir).as_posix(): path.read_bytes()
        for path in sorted(first.artifact_dir.rglob("*"))
        if path.is_file()
    }

    second = run_cli(argv)
    second_snapshot = {
        path.relative_to(second.artifact_dir).as_posix(): path.read_bytes()
        for path in sorted(second.artifact_dir.rglob("*"))
        if path.is_file()
    }
    registry_entries = load_registry(alpha_evaluation_registry_path(tmp_path / "artifacts" / "alpha"))

    assert first.run_id == second.run_id
    assert first_snapshot == second_snapshot
    assert [entry["run_id"] for entry in registry_entries] == [first.run_id]
    pdt.assert_frame_equal(first.prediction_frame, second.prediction_frame, check_dtype=True, check_exact=True)
    pdt.assert_frame_equal(first.aligned_frame, second.aligned_frame, check_dtype=True, check_exact=True)
    pdt.assert_frame_equal(
        first.evaluation_result.ic_timeseries,
        second.evaluation_result.ic_timeseries,
        check_dtype=True,
        check_exact=True,
    )
    assert first.evaluation_result.summary == second.evaluation_result.summary


def test_run_cli_surfaces_validation_failures_for_constant_predictions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_alpha_eval_dataset(tmp_path)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(Exception, match="cross-sectional variation in the prediction column"):
        run_cli(
            [
                "--alpha-model",
                ConstantCliAlphaEvalModel.name,
                "--model-class",
                f"{__name__}:ConstantCliAlphaEvalModel",
                "--dataset",
                "features_daily",
                "--target-column",
                TARGET_COLUMN,
                "--price-column",
                "close",
                "--start",
                "2025-01-01",
                "--end",
                "2025-01-06",
            ]
        )

    assert not alpha_evaluation_registry_path(tmp_path / "artifacts" / "alpha").exists()
