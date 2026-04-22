from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from src.execution.result import ExecutionResult, summarize_execution_result


def run_alpha_evaluation(
    *,
    config: dict[str, Any] | None = None,
    config_path: str | Path | None = None,
    **overrides: Any,
) -> ExecutionResult:
    """Run an alpha evaluation from config data without invoking a subprocess."""

    from src.cli import run_alpha_evaluation as cli

    if config is not None and config_path is not None:
        raise ValueError("Specify either config or config_path, not both.")
    resolved = dict(config or {})
    if config_path is not None:
        resolved = cli.load_alpha_evaluation_config(config_path)
    for key, value in overrides.items():
        if value is not None:
            resolved[key] = value
    if "alpha_name" in resolved:
        resolved = cli.resolve_alpha_config(resolved, alpha_name=resolved.get("alpha_name"))
    raw_result = cli.run_resolved_config(cli.resolve_cli_config(_namespace_from_config(resolved)))
    return _summarize_alpha_result(raw_result)


def run_alpha_evaluation_from_cli_args(args: Any) -> ExecutionResult:
    from src.cli import run_alpha_evaluation as cli

    raw_result = cli.run_resolved_config(cli.resolve_cli_config(args))
    return _summarize_alpha_result(raw_result)


def run_alpha_evaluation_from_argv(argv: Sequence[str] | None = None) -> ExecutionResult:
    from src.cli.run_alpha_evaluation import parse_args

    return run_alpha_evaluation_from_cli_args(parse_args(argv))


def run_alpha(
    alpha_name: str,
    *,
    config_path: str | Path | None = None,
    mode: str = "full",
    **overrides: Any,
) -> ExecutionResult:
    """Run one built-in alpha workflow through the shared execution surface."""

    from src.cli import run_alpha as cli

    argv = ["--alpha-name", alpha_name, "--mode", mode]
    if config_path is not None:
        argv.extend(["--config", str(config_path)])
    for key, value in overrides.items():
        if value is None:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        elif isinstance(value, (list, tuple)):
            argv.append(flag)
            argv.extend(str(item) for item in value)
        else:
            argv.extend([flag, str(value)])
    return run_alpha_from_cli_args(cli.parse_args(argv))


def run_alpha_from_cli_args(args: Any) -> ExecutionResult:
    from src.cli import run_alpha as cli

    resolved_config = cli.resolve_cli_config(args)
    evaluation_result = cli.run_resolved_config(resolved_config)
    scaffold_path: Path | None = None
    if resolved_config["run_mode"] == "full":
        if evaluation_result.signal_mapping_result is None:
            raise ValueError("Full alpha runs require signal mapping to generate a sleeve.")
        sleeve_result = cli.generate_alpha_sleeve(
            signals=evaluation_result.signal_mapping_result.signals,
            dataset=evaluation_result.loaded_frame,
            price_column=cli._optional_string(resolved_config.get("price_column")),
            realized_return_column=cli._optional_string(resolved_config.get("realized_return_column")),
            position_constructor=cli._resolve_sleeve_position_constructor(resolved_config),
            alpha_name=str(resolved_config["alpha_name"]),
            run_id=evaluation_result.run_id,
        )
        updated_manifest = cli.write_alpha_sleeve_artifacts(
            evaluation_result.artifact_dir,
            sleeve_result,
            update_manifest=True,
        )
        cli.register_alpha_evaluation_run(
            run_id=evaluation_result.run_id,
            alpha_name=str(resolved_config["alpha_model"]),
            effective_config=evaluation_result.resolved_config,
            evaluation_result=evaluation_result.evaluation_result,
            artifact_dir=evaluation_result.artifact_dir,
            manifest=updated_manifest or evaluation_result.manifest,
            registry_path=cli.alpha_evaluation_registry_path(Path(str(resolved_config["artifacts_root"]))),
        )
        scaffold_path = cli.write_full_run_scaffold(
            artifact_dir=evaluation_result.artifact_dir,
            evaluation=evaluation_result,
            resolved_config=resolved_config,
        )

    raw_result = cli.AlphaRunResult(
        alpha_name=str(resolved_config["alpha_name"]),
        mode=str(resolved_config["run_mode"]),
        run_id=evaluation_result.run_id,
        artifact_dir=evaluation_result.artifact_dir,
        evaluation=evaluation_result,
        scaffold_path=scaffold_path,
        resolved_config=dict(evaluation_result.resolved_config),
    )
    return summarize_execution_result(
        workflow="alpha",
        raw_result=raw_result,
        metrics={
            "mean_ic": float(raw_result.evaluation.evaluation_result.summary["mean_ic"]),
            "ic_ir": float(raw_result.evaluation.evaluation_result.summary["ic_ir"]),
            "n_periods": int(raw_result.evaluation.evaluation_result.summary["n_periods"]),
        },
        output_paths=_alpha_output_paths(raw_result.artifact_dir, scaffold_path=scaffold_path),
        extra={"mode": raw_result.mode},
    )


def run_alpha_from_argv(argv: Sequence[str] | None = None) -> ExecutionResult:
    from src.cli.run_alpha import parse_args

    return run_alpha_from_cli_args(parse_args(argv))


def _summarize_alpha_result(raw_result: Any) -> ExecutionResult:
    return summarize_execution_result(
        workflow="alpha_evaluation",
        raw_result=raw_result,
        metrics=dict(raw_result.evaluation_result.summary),
        output_paths=_alpha_output_paths(raw_result.artifact_dir),
    )


def _alpha_output_paths(artifact_dir: Path, *, scaffold_path: Path | None = None) -> dict[str, Path | None]:
    return {
        "manifest_json": artifact_dir / "manifest.json",
        "alpha_metrics_json": artifact_dir / "alpha_metrics.json",
        "predictions_parquet": artifact_dir / "predictions.parquet",
        "signals_parquet": artifact_dir / "signals.parquet",
        "sleeve_metrics_json": artifact_dir / "sleeve_metrics.json",
        "scaffold_path": scaffold_path,
    }


def _namespace_from_config(config: dict[str, Any]) -> Any:
    import argparse

    defaults = {
        "config": None,
        "alpha_name": None,
        "alpha_model": None,
        "model_class": None,
        "dataset": None,
        "target_column": None,
        "feature_columns": None,
        "price_column": None,
        "realized_return_column": None,
        "alpha_horizon": None,
        "start": None,
        "end": None,
        "train_start": None,
        "train_end": None,
        "predict_start": None,
        "predict_end": None,
        "tickers": None,
        "artifacts_root": None,
        "min_cross_section_size": None,
        "promotion_gates": None,
        "signal_policy": None,
        "signal_quantile": None,
    }
    defaults.update(config)
    return argparse.Namespace(**defaults)
