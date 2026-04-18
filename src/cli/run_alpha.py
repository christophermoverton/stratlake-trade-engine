from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Sequence

from src.cli.run_alpha_evaluation import (
    AlphaEvaluationRunResult,
    _format_run_failure,
    _inherit_compatible_position_constructor,
    DEFAULT_ARTIFACTS_ROOT,
    load_ticker_file,
    run_resolved_config,
)
from src.data.load_features import SUPPORTED_FEATURE_DATASETS
from src.research.alpha import AlphaPredictionError, AlphaTrainingError
from src.research.alpha.catalog import ALPHAS_CONFIG, resolve_alpha_config
from src.research.alpha_eval import (
    AlphaEvaluationError,
    ForwardReturnAlignmentError,
    alpha_evaluation_registry_path,
    generate_alpha_sleeve,
    register_alpha_evaluation_run,
    write_alpha_sleeve_artifacts,
)
from src.research.position_constructors import normalize_position_constructor_config
from src.research.promotion import load_promotion_gate_config
from src.research.registry import RegistryError
from src.pipeline.cli_adapter import build_pipeline_cli_result

SUPPORTED_RUN_MODES = ("evaluate", "full")
FULL_RUN_SCAFFOLD_FILE = "alpha_run_scaffold.json"
DEFAULT_FULL_RUN_SIGNAL_MAPPING = {"policy": "rank_long_short"}


@dataclass(frozen=True)
class AlphaRunResult:
    """Structured result returned from one built-in alpha CLI run."""

    alpha_name: str
    mode: str
    run_id: str
    artifact_dir: Path
    evaluation: AlphaEvaluationRunResult
    scaffold_path: Path | None
    resolved_config: dict[str, Any]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the first-class alpha runner."""

    parser = argparse.ArgumentParser(
        description="Run one named built-in alpha config on features_daily without dynamic model imports."
    )
    parser.add_argument("--alpha-name", required=True, help="Built-in alpha name defined in configs/alphas.yml.")
    parser.add_argument(
        "--config",
        default=str(ALPHAS_CONFIG),
        help="Optional alpha catalog YAML path. Defaults to configs/alphas.yml.",
    )
    parser.add_argument(
        "--mode",
        choices=SUPPORTED_RUN_MODES,
        default="full",
        help="Run mode. 'evaluate' persists only evaluation artifacts. 'full' also writes a deterministic scaffold for later sleeve generation.",
    )
    parser.add_argument(
        "--evaluation-only",
        action="store_true",
        help="Shortcut for --mode evaluate.",
    )
    parser.add_argument("--dataset", help="Optional dataset override. Built-in alpha runs currently require features_daily.")
    parser.add_argument("--target-column", help="Optional target column override.")
    parser.add_argument(
        "--feature-columns",
        nargs="+",
        help="Optional explicit feature column override.",
    )
    parser.add_argument("--price-column", help="Price column used to derive forward returns.")
    parser.add_argument(
        "--realized-return-column",
        help="Realized return column used to compound forward returns.",
    )
    parser.add_argument("--alpha-horizon", type=int, help="Forward-return horizon in bars.")
    parser.add_argument("--start", help="Inclusive dataset load start date or timestamp.")
    parser.add_argument("--end", help="Exclusive dataset load end date or timestamp.")
    parser.add_argument("--train-start", help="Inclusive training start timestamp.")
    parser.add_argument("--train-end", help="Exclusive training end timestamp.")
    parser.add_argument("--predict-start", help="Inclusive prediction start timestamp.")
    parser.add_argument("--predict-end", help="Exclusive prediction end timestamp.")
    parser.add_argument(
        "--tickers",
        help="Optional path to a newline-delimited ticker file used to filter the loaded feature universe.",
    )
    parser.add_argument(
        "--artifacts-root",
        help="Optional artifact root directory. Defaults to ARTIFACTS_ROOT/alpha or artifacts/alpha.",
    )
    parser.add_argument(
        "--min-cross-section-size",
        type=int,
        help="Minimum number of non-null rows required per evaluation timestamp.",
    )
    parser.add_argument(
        "--promotion-gates",
        help="Optional YAML/JSON promotion gate config override.",
    )
    parser.add_argument(
        "--signal-policy",
        choices=(
            "rank_long_short",
            "zscore_continuous",
            "top_bottom_quantile",
            "long_only_top_quantile",
        ),
        help="Optional explicit post-prediction signal mapping policy.",
    )
    parser.add_argument(
        "--signal-quantile",
        type=float,
        help="Optional quantile used by top/bottom or long-only signal mapping policies.",
    )
    return parser.parse_args(argv)


def resolve_cli_config(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve a built-in alpha run config from the alpha catalog plus CLI overrides."""

    config_path = Path(args.config)
    mode = "evaluate" if args.evaluation_only else str(args.mode)
    resolved = resolve_alpha_config({}, alpha_name=args.alpha_name, path=config_path)
    cli_overrides = {
        "alpha_name": args.alpha_name,
        "alpha_catalog_path": str(config_path),
        "dataset": args.dataset,
        "target_column": args.target_column,
        "feature_columns": args.feature_columns,
        "price_column": args.price_column,
        "realized_return_column": args.realized_return_column,
        "alpha_horizon": args.alpha_horizon,
        "start": args.start,
        "end": args.end,
        "train_start": args.train_start,
        "train_end": args.train_end,
        "predict_start": args.predict_start,
        "predict_end": args.predict_end,
        "tickers": None if args.tickers is None else load_ticker_file(args.tickers),
        "artifacts_root": args.artifacts_root,
        "min_cross_section_size": args.min_cross_section_size,
        "promotion_gates": None if args.promotion_gates is None else load_promotion_gate_config(args.promotion_gates),
        "run_mode": mode,
    }
    signal_mapping = dict(resolved.get("signal_mapping", {})) if isinstance(resolved.get("signal_mapping"), dict) else None
    if args.signal_policy is not None or args.signal_quantile is not None:
        signal_mapping = {} if signal_mapping is None else dict(signal_mapping)
        if args.signal_policy is not None:
            signal_mapping["policy"] = args.signal_policy
        if args.signal_quantile is not None:
            signal_mapping["quantile"] = args.signal_quantile
    for key, value in cli_overrides.items():
        if value is not None:
            resolved[key] = value
    if mode == "full" and signal_mapping is None:
        signal_mapping = dict(DEFAULT_FULL_RUN_SIGNAL_MAPPING)
    position_constructor = normalize_position_constructor_config(resolved.get("position_constructor"))
    if position_constructor is not None:
        resolved["position_constructor"] = position_constructor
    if signal_mapping is not None:
        signal_mapping = _inherit_compatible_position_constructor(
            signal_mapping,
            position_constructor=position_constructor,
        )
        resolved["signal_mapping"] = signal_mapping

    dataset = str(resolved.get("dataset", "")).strip()
    if dataset not in SUPPORTED_FEATURE_DATASETS:
        expected = ", ".join(sorted(SUPPORTED_FEATURE_DATASETS))
        raise ValueError(f"Unknown feature dataset: {dataset!r}. Expected one of: {expected}.")
    if dataset != "features_daily":
        raise ValueError("Built-in alpha runs currently require dataset 'features_daily'.")

    if bool(resolved.get("price_column")) == bool(resolved.get("realized_return_column")):
        raise ValueError("Specify exactly one of price_column or realized_return_column.")

    feature_columns = resolved.get("feature_columns")
    if feature_columns is not None:
        if not isinstance(feature_columns, list):
            raise ValueError("feature_columns must be provided as a list when specified.")
        resolved["feature_columns"] = [str(column) for column in feature_columns]

    tickers = resolved.get("tickers")
    if tickers is not None:
        if not isinstance(tickers, list):
            raise ValueError("tickers must resolve to a list of ticker strings.")
        resolved["tickers"] = [str(symbol) for symbol in tickers]

    if "alpha_horizon" in resolved:
        resolved["alpha_horizon"] = int(resolved["alpha_horizon"])
    if "min_cross_section_size" in resolved:
        resolved["min_cross_section_size"] = int(resolved["min_cross_section_size"])
    resolved["artifacts_root"] = str(Path(resolved.get("artifacts_root") or DEFAULT_ARTIFACTS_ROOT))

    return resolved


def run_cli(
    argv: Sequence[str] | None = None,
    *,
    state: dict[str, Any] | None = None,
    pipeline_context: dict[str, Any] | None = None,
) -> AlphaRunResult | dict[str, Any]:
    """Execute the first-class built-in alpha runner."""

    args = parse_args(argv)
    resolved_config = resolve_cli_config(args)
    evaluation_result = run_resolved_config(resolved_config)

    scaffold_path: Path | None = None
    if resolved_config["run_mode"] == "full":
        if evaluation_result.signal_mapping_result is None:
            raise ValueError("Full alpha runs require signal mapping to generate a sleeve.")
        sleeve_result = generate_alpha_sleeve(
            signals=evaluation_result.signal_mapping_result.signals,
            dataset=evaluation_result.loaded_frame,
            price_column=_optional_string(resolved_config.get("price_column")),
            realized_return_column=_optional_string(resolved_config.get("realized_return_column")),
            position_constructor=resolved_config.get("position_constructor"),
            alpha_name=str(resolved_config["alpha_name"]),
            run_id=evaluation_result.run_id,
        )
        updated_manifest = write_alpha_sleeve_artifacts(
            evaluation_result.artifact_dir,
            sleeve_result,
            update_manifest=True,
        )
        register_alpha_evaluation_run(
            run_id=evaluation_result.run_id,
            alpha_name=str(resolved_config["alpha_model"]),
            effective_config=evaluation_result.resolved_config,
            evaluation_result=evaluation_result.evaluation_result,
            artifact_dir=evaluation_result.artifact_dir,
            manifest=updated_manifest or evaluation_result.manifest,
            registry_path=alpha_evaluation_registry_path(Path(str(resolved_config["artifacts_root"]))),
        )
        scaffold_path = write_full_run_scaffold(
            artifact_dir=evaluation_result.artifact_dir,
            evaluation=evaluation_result,
            resolved_config=resolved_config,
        )

    result = AlphaRunResult(
        alpha_name=str(resolved_config["alpha_name"]),
        mode=str(resolved_config["run_mode"]),
        run_id=evaluation_result.run_id,
        artifact_dir=evaluation_result.artifact_dir,
        evaluation=evaluation_result,
        scaffold_path=scaffold_path,
        resolved_config=dict(evaluation_result.resolved_config),
    )
    print_summary(result)
    if pipeline_context is not None:
        manifest_path = result.artifact_dir / "manifest.json"
        output_paths = {
            "scaffold_path": result.scaffold_path,
            "alpha_metrics_json": result.artifact_dir / "alpha_metrics.json",
            "predictions_parquet": result.artifact_dir / "predictions.parquet",
            "signals_parquet": result.artifact_dir / "signals.parquet",
            "sleeve_metrics_json": result.artifact_dir / "sleeve_metrics.json",
        }
        return build_pipeline_cli_result(
            identifier=result.run_id,
            name=result.alpha_name,
            artifact_dir=result.artifact_dir,
            manifest_path=(manifest_path if manifest_path.exists() else None),
            output_paths=output_paths,
            metrics={
                "mean_ic": float(result.evaluation.evaluation_result.summary["mean_ic"]),
                "ic_ir": float(result.evaluation.evaluation_result.summary["ic_ir"]),
                "n_periods": int(result.evaluation.evaluation_result.summary["n_periods"]),
            },
            extra={
                "alpha_name": result.alpha_name,
                "mode": result.mode,
            },
        )
    return result


def write_full_run_scaffold(
    *,
    artifact_dir: Path,
    evaluation: AlphaEvaluationRunResult,
    resolved_config: dict[str, Any],
) -> Path:
    """Write one deterministic scaffold artifact for the later full alpha run stages."""

    scaffold_path = artifact_dir / FULL_RUN_SCAFFOLD_FILE
    payload = {
        "alpha_name": str(evaluation.alpha_name),
        "run_id": str(evaluation.run_id),
        "mode": "full",
        "status": "completed",
        "next_stage": None,
        "artifact_dir": artifact_dir.as_posix(),
        "evaluation_metrics": {
            "mean_ic": float(evaluation.evaluation_result.summary["mean_ic"]),
            "ic_ir": float(evaluation.evaluation_result.summary["ic_ir"]),
            "n_periods": int(evaluation.evaluation_result.summary["n_periods"]),
        },
        "signal_mapping": (
            None
            if evaluation.signal_mapping_result is None
            else {
                "config": dict(evaluation.signal_mapping_result.config.__dict__),
                "row_count": int(evaluation.signal_mapping_result.row_count),
                "signals_path": "signals.parquet",
            }
        ),
        "sleeve": {
            "sleeve_returns_path": "sleeve_returns.csv",
            "sleeve_equity_curve_path": "sleeve_equity_curve.csv",
            "sleeve_metrics_path": "sleeve_metrics.json",
        },
        "resolved_config": {
            key: value
            for key, value in resolved_config.items()
            if key != "model_class"
        },
    }
    scaffold_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return scaffold_path


def print_summary(result: AlphaRunResult) -> None:
    """Print a concise alpha run summary for CLI users."""

    summary = result.evaluation.evaluation_result.summary
    print(f"alpha_name: {result.alpha_name}")
    print(f"mode: {result.mode}")
    print(f"run_id: {result.run_id}")
    print(f"mean_ic: {float(summary['mean_ic']):.6f}")
    print(f"ic_ir: {float(summary['ic_ir']):.6f}")
    print(f"n_periods: {int(summary['n_periods'])}")
    print(f"artifact_dir: {result.artifact_dir.as_posix()}")
    if result.scaffold_path is not None:
        print("status: completed")
        print(f"scaffold_path: {result.scaffold_path.as_posix()}")


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def main() -> None:
    """CLI entrypoint used by direct module execution."""

    try:
        run_cli()
    except (
        AlphaEvaluationError,
        AlphaPredictionError,
        AlphaTrainingError,
        FileNotFoundError,
        ForwardReturnAlignmentError,
        RegistryError,
        TypeError,
        ValueError,
    ) as exc:
        print(_format_run_failure(exc), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
