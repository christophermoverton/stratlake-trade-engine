from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

PIPELINES_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINES_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINES_ROOT))

from _common import (
    ensure_output_root,
    example_environment,
    read_json,
    relative_to_output,
    resolve_output_path,
    sanitize_json_outputs,
    write_cross_section_dataset,
    write_json,
)
from src.cli.build_pipeline import run_cli as run_build_pipeline_cli
from src.pipeline.builder import PipelineBuilder, load_builder_config


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "output" / "pipelines" / "declarative_builder"


@dataclass(frozen=True)
class ExampleArtifacts:
    summary: dict[str, Any]


def run_example(*, output_root: Path | None = None, verbose: bool = True, reset_output: bool = True) -> ExampleArtifacts:
    resolved_output_root = ensure_output_root(
        DEFAULT_OUTPUT_ROOT if output_root is None else Path(output_root),
        reset_output=reset_output,
    )
    workspace_root = resolved_output_root / "workspace"
    write_cross_section_dataset(workspace_root)

    config_path = Path(__file__).resolve().parent / "config.yml"
    output_pipeline_path = workspace_root / "configs" / "m21_declarative_builder.pipeline.yml"

    with example_environment(workspace_root):
        config_payload = load_builder_config(config_path)
        declarative_builder = PipelineBuilder.from_mapping(config_payload)
        imperative_builder = (
            PipelineBuilder("m21_declarative_builder", start="2025-01-10", end="2025-02-25", strict=True)
            .strategy("cross_section_momentum", params={"lookback_days": 5})
            .signal("cross_section_rank")
            .construct_positions("rank_dollar_neutral", params={"gross_long": 0.5, "gross_short": 0.5})
            .portfolio(
                "equal_weight",
                params={"timeframe": "1D", "alignment_policy": "intersection", "initial_capital": 1.0},
            )
        )

        declarative_built = declarative_builder.build()
        imperative_built = imperative_builder.build()
        cli_result = run_build_pipeline_cli(
            ["--config", str(config_path), "--output", str(output_pipeline_path), "--run"]
        )

    manifest_path = workspace_root / "artifacts" / "pipelines" / cli_result.pipeline_run_id / "manifest.json"
    manifest = read_json(manifest_path)
    strategy_artifact_dir = resolve_output_path(manifest["steps"][0]["step_artifact_dir"], base_dir=workspace_root)
    portfolio_artifact_dir = resolve_output_path(manifest["steps"][1]["step_artifact_dir"], base_dir=workspace_root)
    summary = {
        "pipeline": "declarative_builder",
        "purpose": "Demonstrate config-defined execution via Pipeline Builder and equivalence with imperative composition.",
        "components": {
            "builder_config_path": "config.yml",
            "declarative_pipeline_cli": "python -m src.cli.build_pipeline --config ... --run",
            "includes_portfolio_stage": True,
        },
        "equivalence": {
            "yaml_equal": declarative_built.pipeline_yaml == imperative_built.pipeline_yaml,
            "support_files_equal": declarative_built.support_files == imperative_built.support_files,
        },
        "run": {
            "pipeline_run_id": cli_result.pipeline_run_id,
            "status": cli_result.status,
            "execution_order": list(cli_result.execution_order),
        },
        "artifacts": {
            "generated_pipeline": relative_to_output(output_pipeline_path, resolved_output_root),
            "pipeline_manifest": relative_to_output(manifest_path, resolved_output_root),
            "strategy_artifact_dir": relative_to_output(strategy_artifact_dir, resolved_output_root),
            "portfolio_artifact_dir": relative_to_output(portfolio_artifact_dir, resolved_output_root),
            "summary": "summary.json",
        },
    }

    write_json(resolved_output_root / "summary.json", summary)
    sanitize_json_outputs(resolved_output_root)
    sanitized = read_json(resolved_output_root / "summary.json")

    if verbose:
        print("Declarative builder pipeline")
        print(f"pipeline_run_id: {sanitized['run']['pipeline_run_id']}")
        print(f"declarative-imperative yaml equal: {sanitized['equivalence']['yaml_equal']}")
        print(f"output_root: {resolved_output_root.as_posix()}")

    return ExampleArtifacts(summary=sanitized)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the canonical declarative Pipeline Builder example.")
    parser.add_argument("--output-root", help="Optional output root override.")
    parser.add_argument("--no-reset", action="store_true", help="Do not reset output directory before running.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_root = None if args.output_root is None else Path(args.output_root)
    run_example(output_root=output_root, reset_output=not args.no_reset)


if __name__ == "__main__":
    main()
