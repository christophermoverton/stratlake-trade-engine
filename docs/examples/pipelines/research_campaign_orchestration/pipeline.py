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

from _common import ensure_output_root, load_module_from_repo, read_json, sanitize_json_outputs, write_json


DEFAULT_OUTPUT_ROOT = (
    Path(__file__).resolve().parents[2] / "output" / "pipelines" / "research_campaign_orchestration"
)


@dataclass(frozen=True)
class ExampleArtifacts:
    summary: dict[str, Any]


def run_example(*, output_root: Path | None = None, verbose: bool = True, reset_output: bool = True) -> ExampleArtifacts:
    resolved_output_root = ensure_output_root(
        DEFAULT_OUTPUT_ROOT if output_root is None else Path(output_root),
        reset_output=reset_output,
    )
    config_path = Path(__file__).resolve().parent / "config.yml"
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    module = load_module_from_repo(
        str(payload.get("source_case_study", "docs/examples/real_world_scenario_sweep_case_study.py")),
        "m21_research_campaign_orchestration_source",
    )
    subdir = str(payload.get("output_subdir", "orchestrated_campaign"))
    source_output_root = resolved_output_root / subdir
    artifacts = module.run_example(output_root=source_output_root, verbose=False, reset_output=True)

    summary = {
        "pipeline": "research_campaign_orchestration",
        "purpose": "Canonical multi-run scenario orchestration with aggregation and cross-scenario comparison outputs.",
        "components": {
            "source_case_study": str(payload.get("source_case_study")),
            "orchestration": True,
            "scenario_matrix": True,
            "cross_run_grouping": True,
        },
        "run": {
            "orchestration_run_id": artifacts.summary["first_pass"]["orchestration_run_id"],
            "first_pass_status": artifacts.summary["first_pass"]["status"],
            "second_pass_status": artifacts.summary["second_pass"]["status"],
            "scenario_count": artifacts.summary["first_pass"]["scenario_count"],
        },
        "artifacts": {
            "source_output_root": subdir,
            "source_summary": f"{subdir}/summary.json",
            "summary": "summary.json",
        },
    }
    write_json(resolved_output_root / "summary.json", summary)
    sanitize_json_outputs(resolved_output_root)
    sanitized = read_json(resolved_output_root / "summary.json")

    if verbose:
        print("Research campaign orchestration pipeline")
        print(f"orchestration_run_id: {sanitized['run']['orchestration_run_id']}")
        print(f"scenario_count: {sanitized['run']['scenario_count']}")
        print(f"output_root: {resolved_output_root.as_posix()}")

    return ExampleArtifacts(summary=sanitized)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the canonical research campaign orchestration pipeline.")
    parser.add_argument("--output-root", help="Optional output root override.")
    parser.add_argument("--no-reset", action="store_true", help="Do not reset output directory before running.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_root = None if args.output_root is None else Path(args.output_root)
    run_example(output_root=output_root, reset_output=not args.no_reset)


if __name__ == "__main__":
    main()
