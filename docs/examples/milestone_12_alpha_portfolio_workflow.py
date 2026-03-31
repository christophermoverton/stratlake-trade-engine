from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.execution import ExecutionConfig
from src.portfolio import EqualWeightAllocator, compute_portfolio_metrics, construct_portfolio, write_portfolio_artifacts
from src.research.alpha import (
    BaseAlphaModel,
    generate_alpha_rolling_splits,
    get_cross_section,
    list_cross_section_timestamps,
    make_alpha_fixed_split,
    predict_alpha_model,
    register_alpha_model,
    train_alpha_model,
)
import src.research.alpha.registry as alpha_registry
from src.research.backtest_runner import run_backtest


DATASET_PATH = Path(__file__).resolve().parent / "data" / "milestone_12_alpha_portfolio_dataset.csv"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "output" / "milestone_12_alpha_portfolio_workflow"
FEATURE_COLUMNS = ["feature_alpha", "feature_beta"]
TARGET_COLUMN = "target_ret_1d"
EXAMPLE_MODEL_NAME = "docs_example_centered_linear_alpha_model"
RISK_CONFIG = {
    "volatility_window": 2,
    "target_volatility": 0.10,
    "allow_scale_up": True,
    "max_volatility_scale": 100.0,
}


class ExampleCenteredLinearAlphaModel(BaseAlphaModel):
    name = EXAMPLE_MODEL_NAME

    def __init__(self) -> None:
        self.feature_columns: list[str] = []
        self.feature_means: dict[str, float] = {}

    def _fit(self, df: pd.DataFrame) -> None:
        self.feature_columns = [
            column
            for column in df.columns
            if column not in {"symbol", "ts_utc", TARGET_COLUMN}
        ]
        self.feature_means = {
            column: float(pd.to_numeric(df[column], errors="coerce").mean())
            for column in self.feature_columns
        }

    def _predict(self, df: pd.DataFrame) -> pd.Series:
        if not self.feature_columns:
            raise RuntimeError("Model must be fit before predict.")

        score = pd.Series(0.0, index=df.index, dtype="float64")
        for index, column in enumerate(self.feature_columns, start=1):
            centered = pd.to_numeric(df[column], errors="coerce").astype("float64") - self.feature_means[column]
            score = score + centered * (6.0 / float(index))
        return score.rename("prediction")


@dataclass(frozen=True)
class ExampleArtifacts:
    summary: dict[str, Any]
    prediction_frame: pd.DataFrame
    cross_section: pd.DataFrame
    single_symbol_backtest: pd.DataFrame
    returns_wide: pd.DataFrame
    baseline_portfolio: pd.DataFrame
    targeted_portfolio: pd.DataFrame


def load_example_dataset(dataset_path: Path = DATASET_PATH) -> pd.DataFrame:
    frame = pd.read_csv(dataset_path)
    frame["ts_utc"] = pd.to_datetime(frame["ts_utc"], utc=True, errors="raise")
    frame["symbol"] = frame["symbol"].astype("string")
    frame["timeframe"] = frame["timeframe"].astype("string")
    frame["date"] = frame["date"].astype("string")
    for column in [*FEATURE_COLUMNS, "feature_ret_1d", TARGET_COLUMN]:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("float64")
    return frame.sort_values(["symbol", "ts_utc"], kind="stable").reset_index(drop=True)


def build_fixed_split(frame: pd.DataFrame) -> Any:
    timestamps = frame["ts_utc"].drop_duplicates().tolist()
    return make_alpha_fixed_split(
        train_start=timestamps[0],
        train_end=timestamps[4],
        predict_start=timestamps[4],
        predict_end=timestamps[-1] + pd.Timedelta(days=1),
        split_id="fixed_demo_split",
        metadata={"example": "milestone_12"},
    )


def build_rolling_splits(frame: pd.DataFrame) -> list[Any]:
    timestamps = frame["ts_utc"].drop_duplicates().tolist()
    return generate_alpha_rolling_splits(
        start=timestamps[0],
        end=timestamps[-1] + pd.Timedelta(days=1),
        train_window="4D",
        predict_window="2D",
        step="2D",
    )


def register_example_model() -> None:
    if EXAMPLE_MODEL_NAME in alpha_registry._ALPHA_MODEL_REGISTRY:
        return
    register_alpha_model(EXAMPLE_MODEL_NAME, ExampleCenteredLinearAlphaModel)


def train_and_predict(frame: pd.DataFrame, split: Any) -> pd.DataFrame:
    register_example_model()
    trained = train_alpha_model(
        frame,
        model_name=EXAMPLE_MODEL_NAME,
        target_column=TARGET_COLUMN,
        feature_columns=FEATURE_COLUMNS,
        train_start=split.train_start,
        train_end=split.train_end,
    )
    prediction_result = predict_alpha_model(
        trained,
        frame,
        predict_start=split.predict_start,
        predict_end=split.predict_end,
    )
    predictions = prediction_result.predictions.copy(deep=True)
    predictions["signal"] = np.sign(predictions["prediction_score"]).astype("float64")
    predictions["continuous_signal"] = predictions["prediction_score"].astype("float64")
    return predictions


def build_cross_section(predictions: pd.DataFrame) -> tuple[list[pd.Timestamp], pd.DataFrame]:
    timestamps = list_cross_section_timestamps(predictions)
    cross_section = get_cross_section(
        predictions,
        timestamps[0],
        columns=["prediction_score", "signal", "continuous_signal"],
    )
    return timestamps, cross_section


def run_single_symbol_backtest(frame: pd.DataFrame, predictions: pd.DataFrame, symbol: str) -> pd.DataFrame:
    symbol_frame = frame.loc[frame["symbol"].eq(symbol)].copy(deep=True)
    symbol_predictions = predictions.loc[predictions["symbol"].eq(symbol), ["ts_utc", "continuous_signal"]].copy(deep=True)
    backtest_input = symbol_frame.merge(symbol_predictions, on="ts_utc", how="inner", sort=True)
    backtest_input["signal"] = backtest_input.pop("continuous_signal").astype("float64")
    return run_backtest(
        backtest_input,
        ExecutionConfig(enabled=False, execution_delay=1, transaction_cost_bps=0.0, slippage_bps=0.0),
    )


def build_strategy_return_matrix(frame: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    sleeves: dict[str, pd.Series] = {}
    for symbol in predictions["symbol"].drop_duplicates().tolist():
        sleeve_backtest = run_single_symbol_backtest(frame, predictions, str(symbol))
        sleeves[str(symbol)] = sleeve_backtest.set_index("ts_utc")["strategy_return"].astype("float64")

    returns_wide = pd.DataFrame(sleeves).sort_index(kind="stable")
    returns_wide.index = pd.DatetimeIndex(returns_wide.index, tz="UTC", name="ts_utc")
    returns_wide = returns_wide.loc[:, sorted(returns_wide.columns)]
    return returns_wide.astype("float64")


def build_components(returns_wide: pd.DataFrame) -> list[dict[str, object]]:
    return [
        {"strategy_name": str(column), "run_id": f"example_{str(column).lower()}"}
        for column in returns_wide.columns
    ]


def portfolio_config_dict(
    *,
    portfolio_name: str,
    execution_config: ExecutionConfig,
    volatility_targeting: dict[str, object],
) -> dict[str, object]:
    return {
        "portfolio_name": portfolio_name,
        "allocator": "equal_weight",
        "timeframe": "1D",
        "initial_capital": 1.0,
        "execution": execution_config.to_dict(),
        "risk": dict(RISK_CONFIG),
        "volatility_targeting": volatility_targeting,
        "validation": {"target_weight_sum": 1.0},
    }


def summarize_weights(portfolio_output: pd.DataFrame) -> dict[str, float]:
    latest = portfolio_output.filter(regex=r"^weight__").iloc[-1]
    return {
        column.removeprefix("weight__"): round(float(value), 8)
        for column, value in latest.items()
    }


def _rounded(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 10)
    if isinstance(value, pd.Timestamp):
        return value.isoformat().replace("+00:00", "Z")
    if isinstance(value, dict):
        return {key: _rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rounded(item) for item in value]
    return value


def write_summary(
    output_root: Path,
    *,
    dataset_row_count: int,
    dataset_symbols: list[str],
    predictions: pd.DataFrame,
    cross_section: pd.DataFrame,
    single_symbol_backtest: pd.DataFrame,
    returns_wide: pd.DataFrame,
    baseline_portfolio: pd.DataFrame,
    targeted_portfolio: pd.DataFrame,
    fixed_split: Any,
    rolling_splits: list[Any],
) -> dict[str, Any]:
    baseline_targeting = baseline_portfolio.attrs["portfolio_volatility_targeting"]
    targeted_targeting = targeted_portfolio.attrs["portfolio_volatility_targeting"]
    baseline_metrics = compute_portfolio_metrics(baseline_portfolio, "1D", risk_config=RISK_CONFIG)
    targeted_metrics = compute_portfolio_metrics(targeted_portfolio, "1D", risk_config=RISK_CONFIG)

    summary = {
        "dataset": {
            "row_count": int(dataset_row_count),
            "symbols": sorted(dataset_symbols),
            "prediction_row_count": int(len(predictions)),
        },
        "splits": {
            "fixed": fixed_split.to_dict(),
            "rolling": [split.to_dict() for split in rolling_splits],
        },
        "predictions": {
            "sample": predictions.loc[:, ["symbol", "ts_utc", "prediction_score", "signal", "continuous_signal"]]
            .head(6)
            .assign(ts_utc=lambda df: df["ts_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
            .to_dict(orient="records"),
        },
        "cross_section": {
            "timestamp_count": len(list_cross_section_timestamps(predictions)),
            "sample": cross_section.assign(ts_utc=lambda df: df["ts_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")).to_dict(orient="records"),
        },
        "backtest": {
            "symbol": str(single_symbol_backtest["symbol"].iloc[0]),
            "row_count": int(len(single_symbol_backtest)),
            "strategy_return_sum": round(float(single_symbol_backtest["strategy_return"].sum()), 10),
            "final_equity_curve": round(float(single_symbol_backtest["equity_curve"].iloc[-1]), 10),
            "sample": single_symbol_backtest.loc[
                :,
                ["symbol", "ts_utc", "signal", "executed_signal", "feature_ret_1d", "strategy_return", "equity_curve"],
            ]
            .assign(ts_utc=lambda df: df["ts_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
            .to_dict(orient="records"),
        },
        "portfolio": {
            "return_matrix_rows": int(len(returns_wide)),
            "return_matrix_columns": returns_wide.columns.tolist(),
            "baseline": {
                "weights": summarize_weights(baseline_portfolio),
                "metrics": {
                    "total_return": round(float(baseline_metrics["total_return"]), 10),
                    "annualized_volatility": round(float(baseline_metrics["annualized_volatility"]), 10),
                },
            },
            "targeted": {
                "weights": summarize_weights(targeted_portfolio),
                "metrics": {
                    "total_return": round(float(targeted_metrics["total_return"]), 10),
                    "annualized_volatility": round(float(targeted_metrics["annualized_volatility"]), 10),
                },
                "targeting": {
                    "target_volatility": targeted_targeting["target_volatility"],
                    "estimated_pre_target_volatility": targeted_targeting["estimated_pre_target_volatility"],
                    "estimated_post_target_volatility": targeted_targeting["estimated_post_target_volatility"],
                    "scaling_factor": targeted_targeting["volatility_scaling_factor"],
                },
            },
            "baseline_targeting_enabled": bool(baseline_targeting["enabled"]),
        },
    }

    summary = _rounded(summary)
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    predictions.to_csv(output_root / "predictions.csv", index=False, lineterminator="\n")
    cross_section.to_csv(output_root / "cross_section.csv", index=False, lineterminator="\n")
    single_symbol_backtest.to_csv(output_root / "single_symbol_backtest.csv", index=False, lineterminator="\n")
    returns_wide.reset_index().to_csv(output_root / "portfolio_returns_matrix.csv", index=False, lineterminator="\n")
    return summary


def print_summary(summary: dict[str, Any], output_root: Path) -> None:
    print("Milestone 12 Alpha & Portfolio Workflow Example")
    print(f"Dataset rows: {summary['dataset']['row_count']}")
    print(f"Prediction rows: {summary['dataset']['prediction_row_count']}")
    print(f"Symbols: {', '.join(summary['dataset']['symbols'])}")
    print(f"Fixed split: {summary['splits']['fixed']['split_id']}")
    print(f"Rolling splits: {len(summary['splits']['rolling'])}")
    print()
    print("Prediction sample:")
    print(pd.DataFrame(summary["predictions"]["sample"]).to_string(index=False))
    print()
    print("Cross-section sample:")
    print(pd.DataFrame(summary["cross_section"]["sample"]).to_string(index=False))
    print()
    print("Single-symbol backtest sample:")
    print(pd.DataFrame(summary["backtest"]["sample"]).to_string(index=False))
    print()
    print("Portfolio baseline weights:")
    print(pd.Series(summary["portfolio"]["baseline"]["weights"]).to_string())
    print()
    print("Portfolio targeted weights:")
    print(pd.Series(summary["portfolio"]["targeted"]["weights"]).to_string())
    print()
    print(
        "Volatility targeting scale: "
        f"{summary['portfolio']['targeted']['targeting']['scaling_factor']}"
    )
    print(f"Output directory: {output_root.as_posix()}")


def run_example(
    *,
    output_root: Path | None = None,
    verbose: bool = True,
) -> ExampleArtifacts:
    frame = load_example_dataset()
    resolved_output_root = DEFAULT_OUTPUT_ROOT if output_root is None else Path(output_root)
    resolved_output_root.mkdir(parents=True, exist_ok=True)

    fixed_split = build_fixed_split(frame)
    rolling_splits = build_rolling_splits(frame)
    predictions = train_and_predict(frame, fixed_split)
    timestamps, cross_section = build_cross_section(predictions)
    del timestamps

    single_symbol_backtest = run_single_symbol_backtest(frame, predictions, symbol="NVDA")
    returns_wide = build_strategy_return_matrix(frame, predictions)

    execution_config = ExecutionConfig(
        enabled=True,
        execution_delay=1,
        transaction_cost_bps=2.0,
        slippage_bps=1.0,
    )
    baseline_portfolio = construct_portfolio(
        returns_wide,
        allocator=EqualWeightAllocator(),
        initial_capital=1.0,
        execution_config=execution_config,
        volatility_targeting_config={"enabled": False, "target_volatility": 0.10, "lookback_periods": 2},
        periods_per_year=252,
    )
    targeted_portfolio = construct_portfolio(
        returns_wide,
        allocator=EqualWeightAllocator(),
        initial_capital=1.0,
        execution_config=execution_config,
        volatility_targeting_config={"enabled": True, "target_volatility": 0.10, "lookback_periods": 2},
        periods_per_year=252,
    )

    components = build_components(returns_wide)
    baseline_metrics = compute_portfolio_metrics(baseline_portfolio, "1D", risk_config=RISK_CONFIG)
    targeted_metrics = compute_portfolio_metrics(targeted_portfolio, "1D", risk_config=RISK_CONFIG)
    write_portfolio_artifacts(
        resolved_output_root / "artifacts" / "baseline",
        baseline_portfolio,
        baseline_metrics,
        portfolio_config_dict(
            portfolio_name="milestone_12_example_baseline",
            execution_config=execution_config,
            volatility_targeting={"enabled": False, "target_volatility": 0.10, "lookback_periods": 2},
        ),
        components,
    )
    write_portfolio_artifacts(
        resolved_output_root / "artifacts" / "targeted",
        targeted_portfolio,
        targeted_metrics,
        portfolio_config_dict(
            portfolio_name="milestone_12_example_targeted",
            execution_config=execution_config,
            volatility_targeting={"enabled": True, "target_volatility": 0.10, "lookback_periods": 2},
        ),
        components,
    )

    summary = write_summary(
        resolved_output_root,
        dataset_row_count=len(frame),
        dataset_symbols=sorted(frame["symbol"].astype(str).unique().tolist()),
        predictions=predictions,
        cross_section=cross_section,
        single_symbol_backtest=single_symbol_backtest,
        returns_wide=returns_wide,
        baseline_portfolio=baseline_portfolio,
        targeted_portfolio=targeted_portfolio,
        fixed_split=fixed_split,
        rolling_splits=rolling_splits,
    )
    if verbose:
        print_summary(summary, resolved_output_root)

    return ExampleArtifacts(
        summary=summary,
        prediction_frame=predictions,
        cross_section=cross_section,
        single_symbol_backtest=single_symbol_backtest,
        returns_wide=returns_wide,
        baseline_portfolio=baseline_portfolio,
        targeted_portfolio=targeted_portfolio,
    )


def main() -> None:
    run_example()


if __name__ == "__main__":
    main()
