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
from src.pipeline.builder import PipelineBuilder, load_builder_config


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "output" / "pipelines" / "long_short_risk_controls"


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
    builder = PipelineBuilder.from_mapping(load_builder_config(config_path))
    generated_pipeline = workspace_root / "configs" / "m21_long_short_risk_controls.pipeline.yml"

    with example_environment(workspace_root):
        result = builder.run(generated_pipeline)

    manifest = read_json(workspace_root / "artifacts" / "pipelines" / result.pipeline_run_id / "manifest.json")
    strategy_artifact_dir = resolve_output_path(
        manifest["steps"][0]["step_artifact_dir"],
        base_dir=workspace_root,
    )
    strategy_manifest = read_json(strategy_artifact_dir / "manifest.json")
    summary = {
        "pipeline": "long_short_risk_controls",
        "purpose": "Demonstrate asymmetric long/short controls and constructor constraints in canonical M21 execution.",
        "components": {
            "signal_semantics": "cross_section_rank",
            "position_constructor": {
                "name": "rank_dollar_neutral",
                "params": {
                    "gross_long": 0.6,
                    "gross_short": 0.4,
                },
            },
            "asymmetry": {
                "exclude_short": False,
                "max_short_positions": 2,
                "short_position_scale": 0.8,
            },
        },
        "run": {
            "pipeline_run_id": result.pipeline_run_id,
            "status": result.status,
            "execution_order": list(result.execution_order),
        },
        "artifacts": {
            "pipeline_manifest": relative_to_output(
                workspace_root / "artifacts" / "pipelines" / result.pipeline_run_id / "manifest.json",
                resolved_output_root,
            ),
            "strategy_artifact_dir": relative_to_output(strategy_artifact_dir, resolved_output_root),
            "strategy_signal_semantics": strategy_manifest.get("signal_semantics"),
            "summary": "summary.json",
        },
    }
    write_json(resolved_output_root / "summary.json", summary)
    sanitize_json_outputs(resolved_output_root)
    sanitized = read_json(resolved_output_root / "summary.json")

    if verbose:
        print("Long/short asymmetry and risk-controls pipeline")
        print(f"pipeline_run_id: {sanitized['run']['pipeline_run_id']}")
        print(f"status: {sanitized['run']['status']}")
        print(f"output_root: {resolved_output_root.as_posix()}")

    return ExampleArtifacts(summary=sanitized)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the canonical long/short risk-controls M21 pipeline.")
    parser.add_argument("--output-root", help="Optional output root override.")
    parser.add_argument("--no-reset", action="store_true", help="Do not reset output directory before running.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_root = None if args.output_root is None else Path(args.output_root)
    run_example(output_root=output_root, reset_output=not args.no_reset)


if __name__ == "__main__":
    main()
