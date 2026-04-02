from __future__ import annotations

import argparse
import hashlib
import importlib
from importlib.util import module_from_spec, spec_from_file_location
import os
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Sequence

import pandas as pd
import yaml

import src.research.alpha.registry as alpha_registry
from src.data.load_features import load_features
from src.research.alpha import (
    AlphaPredictionError,
    AlphaTrainingError,
    AlphaPredictionResult,
    BaseAlphaModel,
    TrainedAlphaModel,
    predict_alpha_model,
    register_alpha_model,
    train_alpha_model,
)
from src.research.alpha.catalog import register_builtin_alpha_catalog, resolve_alpha_config
from src.research.alpha_eval import (
    AlphaEvaluationError,
    AlphaEvaluationResult,
    ForwardReturnAlignmentError,
    align_forward_returns,
    alpha_evaluation_registry_path,
    evaluate_alpha_predictions,
    register_alpha_evaluation_run,
    resolve_alpha_evaluation_artifact_dir,
    write_alpha_evaluation_artifacts,
)
from src.research.promotion import load_promotion_gate_config
from src.research.registry import RegistryError, canonicalize_value, serialize_canonical_json

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS_ROOT = Path(os.getenv("ARTIFACTS_ROOT", "artifacts")) / "alpha"


@dataclass(frozen=True)
class AlphaEvaluationRunResult:
    """Structured result returned from one deterministic alpha-evaluation CLI run."""

    alpha_name: str
    run_id: str
    artifact_dir: Path
    loaded_frame: pd.DataFrame
    trained_model: TrainedAlphaModel
    prediction_result: AlphaPredictionResult
    prediction_frame: pd.DataFrame
    aligned_frame: pd.DataFrame
    evaluation_result: AlphaEvaluationResult
    manifest: dict[str, Any]
    resolved_config: dict[str, Any]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the alpha evaluation workflow."""

    parser = argparse.ArgumentParser(
        description="Run a deterministic alpha evaluation workflow through the StratLake research pipeline."
    )
    parser.add_argument("--config", help="Optional YAML config path for alpha evaluation inputs.")
    parser.add_argument("--alpha-name", help="Built-in alpha name defined in configs/alphas.yml.")
    parser.add_argument("--alpha-model", help="Registered alpha model name.")
    parser.add_argument(
        "--model-class",
        help="Optional import target in the form module:Class or path.py:Class to register before running.",
    )
    parser.add_argument("--dataset", help="Feature dataset name to load, such as features_daily.")
    parser.add_argument("--target-column", help="Training target column.")
    parser.add_argument(
        "--feature-columns",
        nargs="+",
        help="Optional explicit feature columns. Defaults to deterministic feature_* derivation.",
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
    return parser.parse_args(argv)


def load_alpha_evaluation_config(path: str | Path) -> dict[str, Any]:
    """Load one alpha evaluation YAML config file."""

    resolved_path = Path(path)
    with resolved_path.open("r", encoding="utf-8") as file_obj:
        payload = yaml.safe_load(file_obj) or {}

    if not isinstance(payload, dict):
        raise ValueError("Alpha evaluation config must deserialize to a mapping.")

    nested = payload.get("alpha_evaluation")
    if nested is None:
        return payload
    if not isinstance(nested, dict):
        raise ValueError("alpha_evaluation config section must be a mapping.")
    return nested


def run_cli(argv: Sequence[str] | None = None) -> AlphaEvaluationRunResult:
    """Execute the alpha evaluation CLI flow from parsed command-line arguments."""

    args = parse_args(argv)
    resolved_config = resolve_cli_config(args)
    result = run_resolved_config(resolved_config)
    print_summary(result)
    return result


def run_resolved_config(resolved_config: dict[str, Any]) -> AlphaEvaluationRunResult:
    """Execute one alpha evaluation run from a fully resolved config mapping."""

    ensure_model_registered(
        resolved_config.get("alpha_model"),
        model_class_import=resolved_config.get("model_class"),
    )
    loaded_frame = load_features(
        str(resolved_config["dataset"]),
        start=_optional_string(resolved_config.get("start")),
        end=_optional_string(resolved_config.get("end")),
        symbols=_optional_symbol_list(resolved_config.get("tickers")),
    )
    trained_model = train_alpha_model(
        loaded_frame,
        model_name=str(resolved_config["alpha_model"]),
        target_column=str(resolved_config["target_column"]),
        feature_columns=_optional_feature_columns(resolved_config.get("feature_columns")),
        train_start=_optional_string(resolved_config.get("train_start")),
        train_end=_optional_string(resolved_config.get("train_end")),
    )
    prediction_result = predict_alpha_model(
        trained_model,
        loaded_frame,
        predict_start=_optional_string(resolved_config.get("predict_start")),
        predict_end=_optional_string(resolved_config.get("predict_end")),
    )

    alignment_input = build_alignment_input(
        prediction_result.predictions,
        loaded_frame,
        source_columns=_alignment_source_columns(resolved_config),
    )
    aligned_frame = align_forward_returns(
        alignment_input,
        prediction_column="prediction_score",
        price_column=_optional_string(resolved_config.get("price_column")),
        realized_return_column=_optional_string(resolved_config.get("realized_return_column")),
        horizon=int(resolved_config["alpha_horizon"]),
    )
    evaluation_result = evaluate_alpha_predictions(
        aligned_frame,
        prediction_column="prediction_score",
        forward_return_column="forward_return",
        min_cross_section_size=int(resolved_config["min_cross_section_size"]),
    )

    effective_config = {
        **resolved_config,
        "feature_columns": list(trained_model.feature_columns),
        "predict_columns": list(prediction_result.predictions.columns),
    }
    run_id = build_alpha_evaluation_run_id(
        alpha_name=str(resolved_config["alpha_model"]),
        effective_config=effective_config,
        evaluation_result=evaluation_result,
    )
    artifact_dir = resolve_alpha_evaluation_artifact_dir(
        run_id,
        artifacts_root=Path(str(resolved_config["artifacts_root"])),
    )
    manifest = write_alpha_evaluation_artifacts(
        artifact_dir,
        evaluation_result,
        trained_model=trained_model,
        prediction_result=prediction_result,
        aligned_frame=aligned_frame,
        run_id=run_id,
        alpha_name=str(resolved_config["alpha_model"]),
        promotion_gate_config=resolved_config.get("promotion_gates"),
    )
    register_alpha_evaluation_run(
        run_id=run_id,
        alpha_name=str(resolved_config["alpha_model"]),
        effective_config=effective_config,
        evaluation_result=evaluation_result,
        artifact_dir=artifact_dir,
        manifest=manifest,
        registry_path=alpha_evaluation_registry_path(Path(str(resolved_config["artifacts_root"]))),
    )

    result = AlphaEvaluationRunResult(
        alpha_name=str(resolved_config["alpha_model"]),
        run_id=run_id,
        artifact_dir=artifact_dir,
        loaded_frame=loaded_frame,
        trained_model=trained_model,
        prediction_result=prediction_result,
        prediction_frame=prediction_result.predictions,
        aligned_frame=aligned_frame,
        evaluation_result=evaluation_result,
        manifest=manifest,
        resolved_config=effective_config,
    )
    return result


def resolve_cli_config(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve one deterministic config mapping from YAML and CLI overrides."""

    config_payload = {} if args.config is None else load_alpha_evaluation_config(args.config)
    if not isinstance(config_payload, dict):
        raise ValueError("Resolved alpha evaluation config must be a mapping.")
    config_payload = resolve_alpha_config(config_payload, alpha_name=args.alpha_name)

    cli_payload = {
        "alpha_name": args.alpha_name,
        "alpha_model": args.alpha_model,
        "model_class": args.model_class,
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
    }

    resolved = dict(config_payload)
    for key, value in cli_payload.items():
        if value is not None:
            resolved[key] = value

    resolved = resolve_alpha_config(resolved)
    if resolved.get("alpha_name") is not None and resolved.get("alpha_model") is None:
        resolved["alpha_model"] = resolved["alpha_name"]

    required_keys = ("alpha_model", "dataset", "target_column")
    missing = [key for key in required_keys if not _has_non_empty_value(resolved.get(key))]
    if missing:
        formatted = ", ".join(missing)
        raise ValueError(f"Alpha evaluation config is missing required fields: {formatted}.")

    if bool(resolved.get("price_column")) == bool(resolved.get("realized_return_column")):
        raise ValueError("Specify exactly one of price_column or realized_return_column.")

    resolved["alpha_horizon"] = int(resolved.get("alpha_horizon", 1))
    resolved["min_cross_section_size"] = int(resolved.get("min_cross_section_size", 2))
    resolved["artifacts_root"] = str(Path(resolved.get("artifacts_root") or DEFAULT_ARTIFACTS_ROOT))

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

    return resolved


def load_ticker_file(path: str | Path) -> list[str]:
    """Load one deterministic ticker file."""

    symbols: list[str] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        normalized = raw_line.strip()
        if not normalized or normalized.startswith("#"):
            continue
        symbols.append(normalized)
    return symbols


def ensure_model_registered(
    model_name: str | None,
    *,
    model_class_import: str | None = None,
) -> None:
    """Register one model class dynamically when requested by the CLI."""

    register_builtin_alpha_catalog()
    if model_class_import is None:
        return

    model_cls = import_model_class(model_class_import)
    if not issubclass(model_cls, BaseAlphaModel):
        raise TypeError("Imported model class must inherit from BaseAlphaModel.")

    resolved_name = getattr(model_cls, "name", None)
    if not isinstance(resolved_name, str) or not resolved_name.strip():
        raise ValueError("Imported alpha model class must define a non-empty 'name' attribute.")
    if model_name is not None and model_name.strip() != resolved_name.strip():
        raise ValueError(
            f"--alpha-model ({model_name!r}) must match imported model class name ({resolved_name!r})."
        )

    existing = alpha_registry._ALPHA_MODEL_REGISTRY.get(resolved_name)
    if existing is model_cls:
        return
    if existing is not None and existing is not model_cls:
        raise ValueError(f"Alpha model '{resolved_name}' is already registered with a different class.")
    register_alpha_model(resolved_name, model_cls)


def import_model_class(import_target: str) -> type[BaseAlphaModel]:
    """Import one alpha model class from a module path or Python file path."""

    module_target, _, class_name = import_target.rpartition(":")
    if not module_target or not class_name:
        raise ValueError("model_class must use the format module:Class or path.py:Class.")

    module = _import_module_target(module_target)
    model_cls = getattr(module, class_name, None)
    if not isinstance(model_cls, type):
        raise ValueError(f"Could not resolve class '{class_name}' from '{module_target}'.")
    return model_cls


def _import_module_target(module_target: str) -> Any:
    module_path = Path(module_target)
    if module_target.endswith(".py") or module_path.exists():
        resolved_path = module_path if module_path.is_absolute() else (Path.cwd() / module_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Model class module path not found: {resolved_path}")
        module_name = f"alpha_eval_cli_{hashlib.sha256(str(resolved_path).encode('utf-8')).hexdigest()[:12]}"
        if module_name in sys.modules:
            return sys.modules[module_name]
        spec = spec_from_file_location(module_name, resolved_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Could not import module from path: {resolved_path}")
        module = module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(module_target)


def build_alignment_input(
    prediction_frame: pd.DataFrame,
    dataset: pd.DataFrame,
    *,
    source_columns: list[str],
) -> pd.DataFrame:
    """Join predictions with their forward-return source columns using the canonical alpha keys."""

    merge_keys = [column for column in ("symbol", "ts_utc", "timeframe") if column in prediction_frame.columns and column in dataset.columns]
    selected_source_columns = [column for column in source_columns if column not in merge_keys]
    merged = prediction_frame.merge(
        dataset.loc[:, [*merge_keys, *selected_source_columns]],
        on=merge_keys,
        how="inner",
        sort=False,
    )
    sort_columns = [column for column in ("symbol", "ts_utc", "timeframe") if column in merged.columns]
    if sort_columns:
        merged = merged.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    merged.attrs = {}
    return merged


def build_alpha_evaluation_run_id(
    *,
    alpha_name: str,
    effective_config: dict[str, Any],
    evaluation_result: AlphaEvaluationResult,
) -> str:
    """Build a deterministic run id from effective configuration and evaluation outputs."""

    payload = {
        "alpha_name": alpha_name,
        "config": canonicalize_value(effective_config),
        "summary": canonicalize_value(dict(evaluation_result.summary)),
        "metadata": canonicalize_value(dict(evaluation_result.metadata)),
        "ic_timeseries": dataframe_snapshot(evaluation_result.ic_timeseries),
    }
    digest = hashlib.sha256(serialize_canonical_json(payload).encode("utf-8")).hexdigest()[:12]
    return f"{_sanitize_name_component(alpha_name)}_alpha_eval_{digest}"


def dataframe_snapshot(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Return a deterministic JSON-serializable snapshot of one DataFrame."""

    snapshot = df.copy(deep=True)
    snapshot.attrs = {}
    for column in snapshot.columns:
        if pd.api.types.is_datetime64_any_dtype(snapshot[column]):
            timestamps = pd.to_datetime(snapshot[column], utc=True, errors="coerce")
            snapshot[column] = timestamps.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return canonicalize_value(snapshot.to_dict(orient="records"))


def print_summary(result: AlphaEvaluationRunResult) -> None:
    """Print a concise alpha evaluation summary for CLI users."""

    summary = result.evaluation_result.summary
    print(f"alpha_model: {result.alpha_name}")
    print(f"run_id: {result.run_id}")
    print(f"mean_ic: {float(summary['mean_ic']):.6f}")
    print(f"ic_ir: {float(summary['ic_ir']):.6f}")
    print(f"n_periods: {int(summary['n_periods'])}")
    print(f"artifact_dir: {result.artifact_dir.as_posix()}")


def _alignment_source_columns(config: dict[str, Any]) -> list[str]:
    columns = ["symbol", "ts_utc"]
    if _has_non_empty_value(config.get("price_column")):
        columns.append(str(config["price_column"]))
    if _has_non_empty_value(config.get("realized_return_column")):
        columns.append(str(config["realized_return_column"]))
    if "timeframe" not in columns:
        columns.append("timeframe")
    return list(dict.fromkeys(columns))


def _optional_feature_columns(value: Any) -> list[str] | None:
    if value is None:
        return None
    return [str(column) for column in value]


def _optional_symbol_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    return [str(symbol) for symbol in value]


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _has_non_empty_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _sanitize_name_component(name: str) -> str:
    cleaned = "".join(char if char.isalnum() else "_" for char in name.strip().lower())
    normalized = "_".join(part for part in cleaned.split("_") if part)
    return normalized or "alpha"


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


def _format_run_failure(exc: Exception) -> str:
    message = str(exc).strip()
    if message.startswith("Run failed:"):
        return message
    return f"Run failed: {message}"


if __name__ == "__main__":
    main()
