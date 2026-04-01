from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Iterator

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cli.run_alpha_evaluation import run_cli
from src.research.alpha.base import BaseAlphaModel
from src.research.alpha_eval import compare_alpha_evaluation_runs


OUTPUT_ROOT = Path(__file__).resolve().parent / "output" / "alpha_evaluation_end_to_end"
WORKSPACE_ROOT = OUTPUT_ROOT / "workspace"
ARTIFACTS_ROOT = OUTPUT_ROOT / "artifacts" / "alpha"
COMPARISON_OUTPUT = OUTPUT_ROOT / "comparisons"
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"
DATASET_NAME = "features_daily"
TARGET_COLUMN = "target_ret_1d"
FEATURE_COLUMNS = ["feature_alpha", "feature_beta"]
TIMEFRAME = "1D"


class WeightedAlphaModel(BaseAlphaModel):
    name = "docs_example_weighted_alpha_model"

    def __init__(self) -> None:
        self.feature_columns: list[str] = []
        self.feature_means: dict[str, float] = {}

    def _fit(self, df: pd.DataFrame) -> None:
        self.feature_columns = list(FEATURE_COLUMNS)
        self.feature_means = {
            column: float(pd.to_numeric(df[column], errors="coerce").mean())
            for column in self.feature_columns
        }

    def _predict(self, df: pd.DataFrame) -> pd.Series:
        score = pd.Series(0.0, index=df.index, dtype="float64")
        for index, column in enumerate(self.feature_columns, start=1):
            centered = pd.to_numeric(df[column], errors="coerce").astype("float64") - self.feature_means[column]
            score = score + centered * float(index)
        return score.rename("prediction")


class InvertedWeightedAlphaModel(WeightedAlphaModel):
    name = "docs_example_inverted_alpha_model"

    def _predict(self, df: pd.DataFrame) -> pd.Series:
        return super()._predict(df).mul(-1.0)


@dataclass(frozen=True)
class ExampleRunSnapshot:
    alpha_name: str
    run_id: str
    artifact_dir: str
    mean_ic: float
    ic_ir: float | None
    mean_rank_ic: float
    rank_ic_ir: float | None
    n_periods: int


def build_demo_dataset(root: Path) -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01", periods=6, freq="D", tz="UTC")
    rows: list[dict[str, object]] = []
    symbols = ["AAA", "BBB", "CCC"]
    for symbol_index, symbol in enumerate(symbols):
        base_level = symbol_index + 1
        for ts_index, ts_utc in enumerate(timestamps):
            feature_alpha = float(base_level * 10 + ts_index)
            feature_beta = float(base_level * 5 - ts_index)
            close = float((base_level + 1) * 100 + ts_index * (base_level + 2))
            rows.append(
                {
                    "symbol": symbol,
                    "ts_utc": ts_utc,
                    "timeframe": TIMEFRAME,
                    "date": ts_utc.strftime("%Y-%m-%d"),
                    "feature_alpha": feature_alpha,
                    "feature_beta": feature_beta,
                    TARGET_COLUMN: float(base_level * 0.01 + ts_index * 0.001),
                    "close": close,
                }
            )

    frame = pd.DataFrame(rows).sort_values(["symbol", "ts_utc"], kind="stable").reset_index(drop=True)
    for column in ("symbol", "timeframe", "date"):
        frame[column] = frame[column].astype("string")
    for column in ("feature_alpha", "feature_beta", TARGET_COLUMN, "close"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("float64")

    dataset_root = root / "data" / "curated" / DATASET_NAME
    for symbol in sorted(frame["symbol"].astype(str).unique().tolist()):
        symbol_frame = frame.loc[frame["symbol"].eq(symbol)].copy(deep=True)
        output_dir = dataset_root / f"symbol={symbol}" / "year=2025"
        output_dir.mkdir(parents=True, exist_ok=True)
        symbol_frame.to_parquet(output_dir / "part-0.parquet", index=False)
    return frame


@contextmanager
def example_environment() -> Iterator[None]:
    previous_features_root = os.environ.get("FEATURES_ROOT")
    os.environ["FEATURES_ROOT"] = str(WORKSPACE_ROOT / "data")
    try:
        yield
    finally:
        if previous_features_root is None:
            os.environ.pop("FEATURES_ROOT", None)
        else:
            os.environ["FEATURES_ROOT"] = previous_features_root


def run_alpha_evaluation(model_class_name: str, alpha_name: str) -> Any:
    return run_cli(
        [
            "--alpha-model",
            alpha_name,
            "--model-class",
            f"{Path(__file__).resolve()}:{model_class_name}",
            "--dataset",
            DATASET_NAME,
            "--target-column",
            TARGET_COLUMN,
            "--feature-columns",
            *FEATURE_COLUMNS,
            "--price-column",
            "close",
            "--start",
            "2025-01-01",
            "--end",
            "2025-01-07",
            "--artifacts-root",
            str(ARTIFACTS_ROOT),
        ]
    )


def snapshot_run(result: Any) -> ExampleRunSnapshot:
    summary = result.evaluation_result.summary
    return ExampleRunSnapshot(
        alpha_name=result.alpha_name,
        run_id=result.run_id,
        artifact_dir=result.artifact_dir.as_posix(),
        mean_ic=float(summary["mean_ic"]),
        ic_ir=None if summary["ic_ir"] is None else float(summary["ic_ir"]),
        mean_rank_ic=float(summary["mean_rank_ic"]),
        rank_ic_ir=None if summary["rank_ic_ir"] is None else float(summary["rank_ic_ir"]),
        n_periods=int(summary["n_periods"]),
    )


def _normalize_json_value(value: Any) -> Any:
    if value is None or isinstance(value, bool | str | int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {key: _normalize_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_json_value(item) for item in value]
    return value


def write_summary(
    *,
    dataset: pd.DataFrame,
    weighted_result: Any,
    inverted_result: Any,
    comparison_result: Any,
) -> dict[str, Any]:
    summary = {
        "dataset": {
            "row_count": int(len(dataset)),
            "symbols": sorted(dataset["symbol"].astype(str).unique().tolist()),
            "timeframe": TIMEFRAME,
        },
        "workflow": [
            "predict",
            "align",
            "validate",
            "evaluate",
            "aggregate",
            "persist",
            "register",
            "compare",
        ],
        "runs": [
            asdict(snapshot_run(weighted_result)),
            asdict(snapshot_run(inverted_result)),
        ],
        "alignment_preview": weighted_result.aligned_frame.loc[
            :,
            ["symbol", "ts_utc", "prediction_score", "close", "forward_return"],
        ]
        .head(6)
        .assign(ts_utc=lambda frame: frame["ts_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
        .to_dict(orient="records"),
        "ic_timeseries_preview": weighted_result.evaluation_result.ic_timeseries.head(5)
        .assign(ts_utc=lambda frame: frame["ts_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
        .to_dict(orient="records"),
        "comparison": {
            "comparison_id": comparison_result.comparison_id,
            "metric": comparison_result.metric,
            "leaderboard_csv": comparison_result.csv_path.as_posix(),
            "leaderboard_json": comparison_result.json_path.as_posix(),
            "leaderboard": [
                {
                    "rank": entry.rank,
                    "alpha_name": entry.alpha_name,
                    "run_id": entry.run_id,
                    "mean_ic": entry.mean_ic,
                    "ic_ir": entry.ic_ir,
                    "mean_rank_ic": entry.mean_rank_ic,
                    "rank_ic_ir": entry.rank_ic_ir,
                }
                for entry in comparison_result.leaderboard
            ],
        },
    }
    normalized = _normalize_json_value(summary)
    SUMMARY_PATH.write_text(json.dumps(normalized, indent=2, sort_keys=True), encoding="utf-8")
    return normalized


def print_summary(summary: dict[str, Any]) -> None:
    print("Alpha Evaluation End-to-End Example")
    print(f"Dataset rows: {summary['dataset']['row_count']}")
    print(f"Symbols: {', '.join(summary['dataset']['symbols'])}")
    print()
    print("Weighted run:")
    print(pd.DataFrame([summary["runs"][0]]).to_string(index=False))
    print()
    print("Inverted run:")
    print(pd.DataFrame([summary["runs"][1]]).to_string(index=False))
    print()
    print("Alignment preview:")
    print(pd.DataFrame(summary["alignment_preview"]).to_string(index=False))
    print()
    print("IC preview:")
    print(pd.DataFrame(summary["ic_timeseries_preview"]).to_string(index=False))
    print()
    print("Leaderboard:")
    print(pd.DataFrame(summary["comparison"]["leaderboard"]).to_string(index=False))
    print()
    print(f"Summary: {SUMMARY_PATH.as_posix()}")
    print(f"Artifacts root: {ARTIFACTS_ROOT.as_posix()}")
    print(f"Leaderboard CSV: {summary['comparison']['leaderboard_csv']}")


def run_example() -> dict[str, Any]:
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    COMPARISON_OUTPUT.mkdir(parents=True, exist_ok=True)
    dataset = build_demo_dataset(WORKSPACE_ROOT)

    with example_environment():
        weighted_result = run_alpha_evaluation("WeightedAlphaModel", WeightedAlphaModel.name)
        inverted_result = run_alpha_evaluation("InvertedWeightedAlphaModel", InvertedWeightedAlphaModel.name)
        comparison_result = compare_alpha_evaluation_runs(
            metric="ic_ir",
            artifacts_root=ARTIFACTS_ROOT,
            output_path=COMPARISON_OUTPUT,
        )

    summary = write_summary(
        dataset=dataset,
        weighted_result=weighted_result,
        inverted_result=inverted_result,
        comparison_result=comparison_result,
    )
    print_summary(summary)
    return summary


if __name__ == "__main__":
    run_example()
