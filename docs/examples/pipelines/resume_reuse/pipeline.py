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


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "output" / "pipelines" / "resume_reuse"


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
        str(payload.get("source_case_study", "docs/examples/real_world_resume_workflow_case_study.py")),
        "m21_resume_reuse_source",
    )
    subdir = str(payload.get("output_subdir", "resume_reuse_flow"))
    source_output_root = resolved_output_root / subdir
    artifacts = module.run_example(output_root=source_output_root, verbose=False, reset_output=True)

    partial = artifacts.summary["resume_workflow"]["partial_run"]
    resumed = artifacts.summary["resume_workflow"]["resumed_run"]
    stable = artifacts.summary["resume_workflow"]["stable_run"]

    summary = {
        "pipeline": "resume_reuse",
        "purpose": "Canonical deterministic resume/retry/reuse semantics for orchestration workflows.",
        "components": {
            "checkpoint_resume": True,
            "retry_metadata": True,
            "reuse_policy": True,
        },
        "run": {
            "campaign_run_id": artifacts.summary["campaign"]["campaign_run_id"],
            "same_campaign_run_id_across_passes": artifacts.summary["campaign"]["same_campaign_run_id_across_passes"],
            "partial_state": partial["stage"]["state"],
            "resumed_state": resumed["stage"]["state"],
            "stable_state": stable["stage"]["state"],
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
        print("Resume/reuse pipeline")
        print(f"campaign_run_id: {sanitized['run']['campaign_run_id']}")
        print(
            "comparison stage states: "
            f"partial={sanitized['run']['partial_state']}, "
            f"resumed={sanitized['run']['resumed_state']}, "
            f"stable={sanitized['run']['stable_state']}"
        )
        print(f"output_root: {resolved_output_root.as_posix()}")

    return ExampleArtifacts(summary=sanitized)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the canonical resume/reuse pipeline.")
    parser.add_argument("--output-root", help="Optional output root override.")
    parser.add_argument("--no-reset", action="store_true", help="Do not reset output directory before running.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_root = None if args.output_root is None else Path(args.output_root)
    run_example(output_root=output_root, reset_output=not args.no_reset)


if __name__ == "__main__":
    main()
