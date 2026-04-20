from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import yaml

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
from src.pipeline.builder import PipelineBuilder


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "output" / "pipelines" / "regime_ensemble_showcase"


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
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    profiles = payload.get("profiles", [])

    runs: list[dict[str, Any]] = []
    with example_environment(workspace_root):
        for entry in profiles:
            strategy_name = str(entry["name"])
            strategy_params = dict(entry.get("strategy_params") or {})
            signal_config = dict(entry["signal"])
            constructor_config = dict(entry["constructor"])
            pipeline_id = f"m22_regime_{strategy_name}"
            builder = (
                PipelineBuilder(pipeline_id, start="2025-01-10", end="2025-02-25", strict=False)
                .strategy(strategy_name, params=strategy_params)
                .signal(str(signal_config["type"]), params=dict(signal_config.get("params") or {}))
                .construct_positions(
                    str(constructor_config["name"]),
                    params=dict(constructor_config.get("params") or {}),
                )
            )
            generated_pipeline = workspace_root / "configs" / f"{pipeline_id}.pipeline.yml"
            result = builder.run(generated_pipeline)
            manifest = read_json(workspace_root / "artifacts" / "pipelines" / result.pipeline_run_id / "manifest.json")
            strategy_artifact_dir = resolve_output_path(
                manifest["steps"][0]["step_artifact_dir"],
                base_dir=workspace_root,
            )
            runs.append(
                {
                    "strategy": strategy_name,
                    "pipeline_run_id": result.pipeline_run_id,
                    "status": result.status,
                    "signal_type": signal_config["type"],
                    "constructor": constructor_config["name"],
                    "strategy_artifact_dir": relative_to_output(strategy_artifact_dir, resolved_output_root),
                    "pipeline_manifest": relative_to_output(
                        workspace_root / "artifacts" / "pipelines" / result.pipeline_run_id / "manifest.json",
                        resolved_output_root,
                    ),
                }
            )

    summary = {
        "pipeline": "regime_ensemble_showcase",
        "purpose": "Show regime-aware and ensemble strategies under the canonical builder-driven execution path.",
        "components": {
            "strategy_archetypes": [run["strategy"] for run in runs],
            "signal_semantics": [run["signal_type"] for run in runs],
            "position_constructors": [run["constructor"] for run in runs],
        },
        "runs": runs,
        "artifacts": {
            "summary": "summary.json",
        },
    }
    write_json(resolved_output_root / "summary.json", summary)
    sanitize_json_outputs(resolved_output_root)
    sanitized = read_json(resolved_output_root / "summary.json")

    if verbose:
        print("Regime and ensemble showcase pipeline")
        for run in sanitized["runs"]:
            print(f"- {run['strategy']}: {run['pipeline_run_id']} ({run['status']})")
        print(f"output_root: {resolved_output_root.as_posix()}")

    return ExampleArtifacts(summary=sanitized)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the regime-aware and ensemble strategy showcase pipeline.")
    parser.add_argument("--output-root", help="Optional output root override.")
    parser.add_argument("--no-reset", action="store_true", help="Do not reset output directory before running.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_root = None if args.output_root is None else Path(args.output_root)
    run_example(output_root=output_root, reset_output=not args.no_reset)


if __name__ == "__main__":
    main()
