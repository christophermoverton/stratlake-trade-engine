from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Iterator

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cli.run_alpha import run_cli as run_alpha_cli
from src.cli.run_portfolio import run_cli as run_portfolio_cli
from src.research.review import compare_research_runs


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "output" / "real_alpha_workflow"
ALPHA_NAME = "rank_composite_momentum"
ALPHA_CATALOG_PATH = REPO_ROOT / "configs" / "alphas.yml"
PORTFOLIO_NAME = "rank_composite_momentum_sleeve_portfolio"
DATASET_NAME = "features_daily"
SUMMARY_FILENAME = "summary.json"


@dataclass(frozen=True)
class ExampleArtifacts:
    summary: dict[str, Any]
    dataset: pd.DataFrame
    signals: pd.DataFrame
    sleeve_returns: pd.DataFrame
    portfolio_returns: pd.DataFrame | None
    review_leaderboard: pd.DataFrame


def build_demo_dataset(workspace_root: Path) -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01", periods=30, freq="D", tz="UTC")
    symbols = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    symbol_frames: list[pd.DataFrame] = []

    for symbol_index, symbol in enumerate(symbols):
        price = 95.0 + symbol_index * 7.5
        rows: list[dict[str, object]] = []
        for ts_index, ts_utc in enumerate(timestamps):
            momentum_level = 1.25 - symbol_index * 0.35 + ts_index * 0.09
            short_term = momentum_level * 0.8 + ((ts_index + symbol_index) % 4 - 1.5) * 0.05
            long_term = momentum_level * 1.1 + (2 - symbol_index) * 0.03
            distance_to_sma = momentum_level * 0.55 + ((ts_index % 5) - 2.0) * 0.015
            daily_return = max(
                -0.02,
                min(
                    0.03,
                    0.0015
                    + momentum_level * 0.0022
                    + distance_to_sma * 0.0015
                    + ((ts_index % 3) - 1.0) * 0.0004,
                ),
            )
            price = price * (1.0 + daily_return)
            rows.append(
                {
                    "symbol": symbol,
                    "ts_utc": ts_utc,
                    "timeframe": "1D",
                    "date": ts_utc.strftime("%Y-%m-%d"),
                    "feature_ret_1d": short_term * 0.45,
                    "feature_ret_5d": short_term,
                    "feature_ret_20d": long_term,
                    "feature_close_to_sma20": distance_to_sma,
                    "feature_vol_20d": 0.15 + symbol_index * 0.02 + abs((ts_index % 6) - 2.5) * 0.01,
                    "close": price,
                }
            )

        symbol_frame = pd.DataFrame(rows)
        close = pd.to_numeric(symbol_frame["close"], errors="raise").astype("float64")
        symbol_frame.loc[:2, ["feature_close_to_sma20", "feature_vol_20d"]] = pd.NA
        symbol_frame["target_ret_1d"] = close.shift(-1).div(close).sub(1.0)
        symbol_frame["target_ret_5d"] = close.shift(-5).div(close).sub(1.0)
        symbol_frames.append(symbol_frame)

    frame = pd.concat(symbol_frames, ignore_index=True).sort_values(["symbol", "ts_utc"], kind="stable").reset_index(drop=True)
    for column in ("symbol", "timeframe", "date"):
        frame[column] = frame[column].astype("string")
    numeric_columns = [
        "feature_ret_1d",
        "feature_ret_5d",
        "feature_ret_20d",
        "feature_close_to_sma20",
        "feature_vol_20d",
        "close",
        "target_ret_1d",
        "target_ret_5d",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("float64")

    dataset_root = workspace_root / "data" / "curated" / DATASET_NAME
    for symbol in symbols:
        symbol_frame = frame.loc[frame["symbol"].eq(symbol)].copy(deep=True)
        output_dir = dataset_root / f"symbol={symbol}" / "year=2025"
        output_dir.mkdir(parents=True, exist_ok=True)
        symbol_frame.to_parquet(output_dir / "part-0.parquet", index=False)
    return frame


@contextmanager
def example_environment(workspace_root: Path) -> Iterator[None]:
    previous_features_root = os.environ.get("FEATURES_ROOT")
    previous_cwd = Path.cwd()
    os.environ["FEATURES_ROOT"] = str(workspace_root / "data")
    os.chdir(workspace_root)
    try:
        yield
    finally:
        os.chdir(previous_cwd)
        if previous_features_root is None:
            os.environ.pop("FEATURES_ROOT", None)
        else:
            os.environ["FEATURES_ROOT"] = previous_features_root


def write_portfolio_config(
    workspace_root: Path,
    *,
    alpha_run_id: str,
    alpha_name: str,
) -> Path:
    config_path = workspace_root / "portfolio_from_alpha_sleeve.yml"
    payload = {
        "portfolio_name": PORTFOLIO_NAME,
        "allocator": "equal_weight",
        "components": [
            {
                "strategy_name": f"{alpha_name}_sleeve",
                "run_id": alpha_run_id,
                "artifact_type": "alpha_sleeve",
            }
        ],
    }
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def _relative_to_output(path: Path, output_root: Path) -> str:
    try:
        return path.relative_to(output_root).as_posix()
    except ValueError:
        return path.as_posix()


def _normalize_json_value(value: Any) -> Any:
    if value is None or isinstance(value, bool | str | int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, pd.Timestamp):
        timestamp = value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")
        return timestamp.isoformat().replace("+00:00", "Z")
    if isinstance(value, dict):
        return {key: _normalize_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_json_value(item) for item in value]
    if pd.isna(value):
        return None
    return value


def _json_preview(frame: pd.DataFrame, *, limit: int, timestamp_columns: tuple[str, ...] = ("ts_utc",)) -> list[dict[str, Any]]:
    preview = frame.head(limit).copy(deep=True)
    for column in timestamp_columns:
        if column in preview.columns:
            preview[column] = pd.to_datetime(preview[column], utc=True, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return _normalize_json_value(preview.to_dict(orient="records"))


def write_summary(
    output_root: Path,
    *,
    dataset: pd.DataFrame,
    alpha_result: Any,
    alpha_artifact_dir: Path,
    signals: pd.DataFrame,
    sleeve_returns: pd.DataFrame,
    portfolio_result: Any | None,
    portfolio_artifact_dir: Path | None,
    portfolio_returns: pd.DataFrame | None,
    review_result: Any,
    review_leaderboard: pd.DataFrame,
) -> dict[str, Any]:
    alpha_summary = alpha_result.evaluation.evaluation_result.summary
    signal_mapping_result = alpha_result.evaluation.signal_mapping_result
    alpha_row = review_leaderboard.loc[review_leaderboard["run_type"].eq("alpha_evaluation")].iloc[0].to_dict()
    portfolio_rows = review_leaderboard.loc[review_leaderboard["run_type"].eq("portfolio")]
    portfolio_row = None if portfolio_rows.empty else portfolio_rows.iloc[0].to_dict()

    summary = {
        "workflow": [
            "build_features_daily_fixture",
            "resolve_builtin_alpha_from_catalog",
            "train_and_predict",
            "evaluate_alpha",
            "map_predictions_to_signals",
            "generate_sleeve_artifacts",
            "optionally_construct_portfolio",
            "write_review_outputs",
        ],
        "dataset": {
            "name": DATASET_NAME,
            "row_count": int(len(dataset)),
            "symbol_count": int(dataset["symbol"].astype("string").nunique()),
            "symbols": sorted(dataset["symbol"].astype(str).unique().tolist()),
            "date_range": {
                "start": str(dataset["date"].min()),
                "end": str(dataset["date"].max()),
            },
        },
        "alpha": {
            "alpha_name": alpha_result.alpha_name,
            "catalog_path": _relative_to_output(ALPHA_CATALOG_PATH, output_root),
            "run_id": alpha_result.run_id,
            "mode": alpha_result.mode,
            "artifact_dir": _relative_to_output(alpha_artifact_dir, output_root),
            "resolved_config": {
                "dataset": alpha_result.resolved_config["dataset"],
                "target_column": alpha_result.resolved_config["target_column"],
                "feature_columns": list(alpha_result.resolved_config["feature_columns"]),
                "train_start": alpha_result.resolved_config.get("train_start"),
                "train_end": alpha_result.resolved_config.get("train_end"),
                "predict_start": alpha_result.resolved_config.get("predict_start"),
                "predict_end": alpha_result.resolved_config.get("predict_end"),
                "alpha_horizon": int(alpha_result.resolved_config["alpha_horizon"]),
            },
            "evaluation": {
                "mean_ic": float(alpha_summary["mean_ic"]),
                "ic_ir": float(alpha_summary["ic_ir"]),
                "mean_rank_ic": float(alpha_summary["mean_rank_ic"]),
                "rank_ic_ir": float(alpha_summary["rank_ic_ir"]),
                "n_periods": int(alpha_summary["n_periods"]),
            },
            "signal_mapping": {
                "policy": signal_mapping_result.config.policy if signal_mapping_result is not None else None,
                "quantile": signal_mapping_result.config.quantile if signal_mapping_result is not None else None,
                "row_count": None if signal_mapping_result is None else int(signal_mapping_result.row_count),
                "signal_counts": {
                    str(key): int(value)
                    for key, value in signals["signal"].value_counts(dropna=False).sort_index().items()
                },
            },
            "signals_preview": _json_preview(
                signals.loc[:, [column for column in ("symbol", "ts_utc", "prediction_score", "signal") if column in signals.columns]],
                limit=8,
            ),
            "sleeve": {
                "metrics": json.loads((alpha_artifact_dir / "sleeve_metrics.json").read_text(encoding="utf-8")),
                "returns_preview": _json_preview(sleeve_returns, limit=6),
            },
        },
        "portfolio": {
            "included": portfolio_result is not None,
            "portfolio_name": None if portfolio_result is None else portfolio_result.portfolio_name,
            "run_id": None if portfolio_result is None else portfolio_result.run_id,
            "artifact_dir": None if portfolio_artifact_dir is None else _relative_to_output(portfolio_artifact_dir, output_root),
            "metrics": None
            if portfolio_result is None
            else {
                "total_return": portfolio_result.metrics.get("total_return"),
                "sharpe_ratio": portfolio_result.metrics.get("sharpe_ratio"),
                "max_drawdown": portfolio_result.metrics.get("max_drawdown"),
                "realized_volatility": portfolio_result.metrics.get("realized_volatility"),
            },
            "returns_preview": None if portfolio_returns is None else _json_preview(portfolio_returns, limit=6),
        },
        "review": {
            "review_id": review_result.review_id,
            "selected_run_ids": [entry.run_id for entry in review_result.entries],
            "artifact_dir": _relative_to_output(review_result.csv_path.parent, output_root),
            "artifact_files": json.loads(review_result.manifest_path.read_text(encoding="utf-8"))["artifact_files"],
            "alpha_entry": {
                "selected_metric_name": alpha_row["selected_metric_name"],
                "selected_metric_value": alpha_row["selected_metric_value"],
                "mapping_name": alpha_row["mapping_name"],
                "sleeve_metric_name": alpha_row["sleeve_metric_name"],
                "sleeve_metric_value": alpha_row["sleeve_metric_value"],
                "linked_portfolio_count": alpha_row["linked_portfolio_count"],
                "linked_portfolio_names": alpha_row["linked_portfolio_names"],
            },
            "portfolio_entry": None
            if portfolio_row is None
            else {
                "selected_metric_name": portfolio_row["selected_metric_name"],
                "selected_metric_value": portfolio_row["selected_metric_value"],
                "entity_name": portfolio_row["entity_name"],
            },
        },
    }
    normalized = _normalize_json_value(summary)
    (output_root / SUMMARY_FILENAME).write_text(json.dumps(normalized, indent=2, sort_keys=True), encoding="utf-8")
    return normalized


def print_summary(summary: dict[str, Any], output_root: Path) -> None:
    print("Real Alpha Workflow Example")
    print(f"Alpha: {summary['alpha']['alpha_name']}")
    print(f"Alpha run id: {summary['alpha']['run_id']}")
    print(
        "Alpha evaluation: "
        f"mean_ic={summary['alpha']['evaluation']['mean_ic']:.6f}, "
        f"ic_ir={summary['alpha']['evaluation']['ic_ir']:.6f}, "
        f"mean_rank_ic={summary['alpha']['evaluation']['mean_rank_ic']:.6f}"
    )
    print(
        "Signal mapping: "
        f"{summary['alpha']['signal_mapping']['policy']} "
        f"(quantile={summary['alpha']['signal_mapping']['quantile']})"
    )
    if summary["portfolio"]["included"]:
        print(f"Portfolio run id: {summary['portfolio']['run_id']}")
        print(
            "Portfolio metrics: "
            f"total_return={summary['portfolio']['metrics']['total_return']:.6f}, "
            f"sharpe_ratio={summary['portfolio']['metrics']['sharpe_ratio']:.6f}"
        )
    print(f"Review id: {summary['review']['review_id']}")
    print(f"Review selected runs: {', '.join(summary['review']['selected_run_ids'])}")
    print(f"Output directory: {output_root.as_posix()}")


def run_example(*, output_root: Path | None = None, include_portfolio: bool = True, verbose: bool = True) -> ExampleArtifacts:
    resolved_output_root = DEFAULT_OUTPUT_ROOT if output_root is None else Path(output_root)
    if resolved_output_root.exists():
        shutil.rmtree(resolved_output_root)
    resolved_output_root.mkdir(parents=True, exist_ok=True)

    workspace_root = resolved_output_root / "workspace"
    dataset = build_demo_dataset(workspace_root)

    portfolio_result = None
    portfolio_returns: pd.DataFrame | None = None
    with example_environment(workspace_root):
        alpha_result = run_alpha_cli(
            [
                "--alpha-name",
                ALPHA_NAME,
                "--config",
                str(ALPHA_CATALOG_PATH),
                "--start",
                "2025-01-01",
                "--end",
                "2025-01-30",
                "--train-start",
                "2025-01-04",
                "--train-end",
                "2025-01-18",
                "--predict-start",
                "2025-01-18",
                "--predict-end",
                "2025-01-25",
                "--artifacts-root",
                "artifacts/alpha",
                "--signal-policy",
                "top_bottom_quantile",
                "--signal-quantile",
                "0.34",
            ]
        )
        if include_portfolio:
            portfolio_config_path = write_portfolio_config(
                workspace_root,
                alpha_run_id=alpha_result.run_id,
                alpha_name=alpha_result.alpha_name,
            )
            portfolio_result = run_portfolio_cli(
                [
                    "--portfolio-config",
                    str(portfolio_config_path),
                    "--timeframe",
                    "1D",
                    "--output-dir",
                    "artifacts/portfolios",
                ]
            )
            portfolio_returns = pd.read_csv(portfolio_result.experiment_dir / "portfolio_returns.csv")

    resolved_alpha_artifact_dir = Path(alpha_result.artifact_dir)
    if not resolved_alpha_artifact_dir.is_absolute():
        resolved_alpha_artifact_dir = workspace_root / resolved_alpha_artifact_dir
    resolved_portfolio_artifact_dir = None
    if portfolio_result is not None:
        resolved_portfolio_artifact_dir = Path(portfolio_result.experiment_dir)
        if not resolved_portfolio_artifact_dir.is_absolute():
            resolved_portfolio_artifact_dir = workspace_root / resolved_portfolio_artifact_dir

    signals = pd.read_parquet(resolved_alpha_artifact_dir / "signals.parquet")
    sleeve_returns = pd.read_csv(resolved_alpha_artifact_dir / "sleeve_returns.csv")
    review_result = compare_research_runs(
        run_types=["alpha_evaluation", "portfolio"] if include_portfolio else ["alpha_evaluation"],
        alpha_name=ALPHA_NAME,
        top_k_per_type=1,
        alpha_artifacts_root=workspace_root / "artifacts" / "alpha",
        portfolio_artifacts_root=workspace_root / "artifacts" / "portfolios",
        output_path=resolved_output_root / "review",
        review_config={"output": {"emit_plots": False}},
    )
    review_leaderboard = pd.read_csv(review_result.csv_path)
    summary = write_summary(
        resolved_output_root,
        dataset=dataset,
        alpha_result=alpha_result,
        alpha_artifact_dir=resolved_alpha_artifact_dir,
        signals=signals,
        sleeve_returns=sleeve_returns,
        portfolio_result=portfolio_result,
        portfolio_artifact_dir=resolved_portfolio_artifact_dir,
        portfolio_returns=portfolio_returns,
        review_result=review_result,
        review_leaderboard=review_leaderboard,
    )

    if verbose:
        print_summary(summary, resolved_output_root)

    return ExampleArtifacts(
        summary=summary,
        dataset=dataset,
        signals=signals,
        sleeve_returns=sleeve_returns,
        portfolio_returns=portfolio_returns,
        review_leaderboard=review_leaderboard,
    )


def main() -> None:
    run_example()


if __name__ == "__main__":
    main()
