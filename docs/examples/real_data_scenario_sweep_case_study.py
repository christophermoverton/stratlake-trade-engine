from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Iterator

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cli.run_research_campaign import run_research_campaign
from src.config.research_campaign import load_research_campaign_config, resolve_research_campaign_config


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "output" / "real_data_scenario_sweep_case_study"
SUMMARY_FILENAME = "summary.json"
BASE_CONFIG_PATH = (
    REPO_ROOT / "docs" / "examples" / "data" / "milestone_16_campaign_configs" / "real_world_campaign.yml"
)


@dataclass(frozen=True)
class ExampleArtifacts:
    summary: dict[str, Any]
    first_pass: dict[str, Any]
    second_pass: dict[str, Any] | None
    scenario_matrix: pd.DataFrame


@contextmanager
def example_environment() -> Iterator[None]:
    previous_features_root = os.environ.get("FEATURES_ROOT")
    previous_cwd = Path.cwd()
    os.environ["FEATURES_ROOT"] = str(REPO_ROOT / "data")
    os.chdir(REPO_ROOT)
    try:
        yield
    finally:
        os.chdir(previous_cwd)
        if previous_features_root is None:
            os.environ.pop("FEATURES_ROOT", None)
        else:
            os.environ["FEATURES_ROOT"] = previous_features_root


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8", newline="\n")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative_to_output(path: Path, output_root: Path) -> str:
    try:
        return path.resolve().relative_to(output_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _preview_rows(frame: pd.DataFrame, limit: int = 5) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    preferred_columns = [
        "rank",
        "scenario_id",
        "source",
        "sweep_top_k",
        "sweep_max_candidates",
        "sweep_redundancy_corr_cap",
        "ranking_metric",
        "ranking_value",
        "portfolio_sharpe_ratio",
        "alpha_ic_ir_max",
        "candidate_selected_count",
    ]
    columns = [column for column in preferred_columns if column in frame.columns]
    preview = frame.loc[:, columns].head(limit).copy(deep=True)
    return preview.to_dict(orient="records")


def _scenario_preflight_state_counts(orchestration_result: Any) -> dict[str, int]:
    counts = {
        "completed": 0,
        "reused": 0,
        "partial": 0,
        "failed": 0,
        "other": 0,
    }
    for scenario in orchestration_result.scenario_results:
        stage_statuses = scenario.result.campaign_summary.get("stage_statuses") or {}
        preflight_state = str(stage_statuses.get("preflight") or "")
        if preflight_state == "reused":
            counts["reused"] += 1
        elif preflight_state == "completed":
            counts["completed"] += 1
        elif preflight_state == "partial":
            counts["partial"] += 1
        elif preflight_state == "failed":
            counts["failed"] += 1
        else:
            counts["other"] += 1
    return counts


def _resolved_config(output_root: Path) -> Any:
    base = load_research_campaign_config(BASE_CONFIG_PATH).to_dict()
    artifacts_root = output_root / "artifacts"
    base["targets"]["alpha_catalog_path"] = (REPO_ROOT / "configs" / "alphas_2026_q1.yml").as_posix()
    base["targets"]["strategy_config_path"] = (REPO_ROOT / "configs" / "strategies.yml").as_posix()
    base["targets"]["portfolio_config_path"] = (REPO_ROOT / "configs" / "portfolios_alpha_2026_q1.yml").as_posix()
    base["candidate_selection"]["artifacts_root"] = (artifacts_root / "alpha").as_posix()
    base["candidate_selection"]["output"]["path"] = (artifacts_root / "candidate_selection").as_posix()
    base["candidate_selection"]["output"]["review_output_path"] = (
        artifacts_root / "reviews" / "candidate_review"
    ).as_posix()
    base["review"]["output"]["path"] = (artifacts_root / "reviews" / "research_review").as_posix()
    base["outputs"]["alpha_artifacts_root"] = (artifacts_root / "alpha").as_posix()
    base["outputs"]["candidate_selection_output_path"] = (artifacts_root / "candidate_selection").as_posix()
    base["outputs"]["campaign_artifacts_root"] = (artifacts_root / "research_campaigns").as_posix()
    base["outputs"]["portfolio_artifacts_root"] = (artifacts_root / "portfolios").as_posix()
    base["outputs"]["comparison_output_path"] = (artifacts_root / "campaign_comparisons").as_posix()
    base["outputs"]["review_output_path"] = (artifacts_root / "reviews" / "research_review").as_posix()
    base["portfolio"]["enabled"] = False
    base["review"]["filters"]["run_types"] = ["alpha_evaluation"]
    base["candidate_selection"]["execution"]["enable_review"] = False
    base["reuse_policy"] = {
        "enable_checkpoint_reuse": True,
        "reuse_prior_stages": [
            "preflight",
            "research",
            "comparison",
            "candidate_selection",
            "portfolio",
            "candidate_review",
            "review",
        ],
        "force_rerun_stages": [],
        "invalidate_downstream_after_stages": [],
    }
    base["scenarios"] = {
        "enabled": True,
        "max_scenarios": 12,
        "max_values_per_axis": 3,
        "matrix": [
            {
                "name": "top_k",
                "path": "comparison.top_k",
                "values": [3, 4],
            },
            {
                "name": "max_candidates",
                "path": "candidate_selection.max_candidates",
                "values": [2, 3],
            },
            {
                "name": "redundancy_corr_cap",
                "path": "candidate_selection.redundancy.max_pairwise_correlation",
                "values": [0.70, 0.80],
            },
        ],
        "include": [
            {
                "scenario_id": "sleeve_review_only",
                "description": "Focus review packet on alpha evaluation using sleeve metrics.",
                "overrides": {
                    "comparison": {"alpha_view": "sleeve"},
                    "review": {"filters": {"run_types": ["alpha_evaluation"]}},
                },
            }
        ],
    }
    return resolve_research_campaign_config(base)


def run_example(
    *,
    output_root: Path | None = None,
    reset_output: bool = True,
    checkpoint_demo: bool = False,
    verbose: bool = True,
) -> ExampleArtifacts:
    resolved_output_root = DEFAULT_OUTPUT_ROOT if output_root is None else Path(output_root)
    if reset_output and resolved_output_root.exists():
        shutil.rmtree(resolved_output_root)
    resolved_output_root.mkdir(parents=True, exist_ok=True)

    with example_environment():
        config = _resolved_config(resolved_output_root)
        first_pass = run_research_campaign(config)
        second_pass = run_research_campaign(config) if checkpoint_demo else None

    if first_pass.__class__.__name__ != "ResearchCampaignOrchestrationResult":
        raise RuntimeError("Expected a sweep orchestration result for first pass.")

    first_summary = _read_json(first_pass.orchestration_summary_path)
    first_manifest = _read_json(first_pass.orchestration_manifest_path)
    first_catalog = _read_json(first_pass.scenario_catalog_path)
    first_expansion_preflight = _read_json(first_pass.expansion_preflight_path)
    matrix_df = pd.read_csv(first_pass.scenario_matrix_csv_path)

    summary_payload: dict[str, Any] = {
        "workflow": [
            "load_real_world_campaign_base_config",
            "override_outputs_and_catalog_paths_under_case_study_root",
            "apply_scenario_sweep_matrix_and_include_config",
            "run_campaign_orchestration_for_all_expanded_scenarios",
            "collect_matrix_catalog_preflight_and_summary_artifacts",
        ],
        "config": {
            "base_config_path": BASE_CONFIG_PATH.as_posix(),
            "dataset": config.dataset_selection.dataset,
            "timeframe": config.dataset_selection.timeframe,
            "evaluation_horizon": config.dataset_selection.evaluation_horizon,
            "alpha_names": list(config.targets.alpha_names),
            "scenario_axes": [
                {"name": axis.name, "path": axis.path, "values": list(axis.values)}
                for axis in config.scenarios.matrix
            ],
            "include_ids": [item.scenario_id for item in config.scenarios.include],
        },
        "first_pass": {
            "orchestration_run_id": first_pass.orchestration_run_id,
            "status": first_summary.get("status"),
            "scenario_count": first_summary.get("scenario_count"),
            "scenario_status_counts": first_summary.get("scenario_status_counts"),
            "output_paths": {
                "orchestration_summary": _relative_to_output(first_pass.orchestration_summary_path, resolved_output_root),
                "orchestration_manifest": _relative_to_output(first_pass.orchestration_manifest_path, resolved_output_root),
                "scenario_catalog": _relative_to_output(first_pass.scenario_catalog_path, resolved_output_root),
                "scenario_matrix_csv": _relative_to_output(first_pass.scenario_matrix_csv_path, resolved_output_root),
                "scenario_matrix_summary": _relative_to_output(first_pass.scenario_matrix_summary_path, resolved_output_root),
                "expansion_preflight": _relative_to_output(first_pass.expansion_preflight_path, resolved_output_root),
            },
            "expansion_preflight": first_expansion_preflight,
            "matrix_preview": _preview_rows(matrix_df, limit=5),
            "selection_rule": (first_summary.get("scenario_matrix") or {}).get("selection_rule"),
            "metric_priority": (first_summary.get("scenario_matrix") or {}).get("metric_priority"),
            "catalog_preview": list(first_catalog.get("scenarios") or [])[:3],
            "manifest_artifact_files": list(first_manifest.get("artifact_files") or []),
        },
        "artifacts": {
            "output_root": resolved_output_root.as_posix(),
            "summary": SUMMARY_FILENAME,
        },
    }

    second_summary_payload: dict[str, Any] | None = None
    if second_pass is not None:
        second_summary = _read_json(second_pass.orchestration_summary_path)
        second_summary_payload = {
            "orchestration_run_id": second_pass.orchestration_run_id,
            "status": second_summary.get("status"),
            "scenario_count": second_summary.get("scenario_count"),
            "scenario_status_counts": second_summary.get("scenario_status_counts"),
            "scenario_preflight_state_counts": _scenario_preflight_state_counts(second_pass),
            "output_paths": {
                "orchestration_summary": _relative_to_output(second_pass.orchestration_summary_path, resolved_output_root),
                "orchestration_manifest": _relative_to_output(second_pass.orchestration_manifest_path, resolved_output_root),
            },
        }
        summary_payload["second_pass"] = second_summary_payload

    _write_json(resolved_output_root / SUMMARY_FILENAME, summary_payload)

    if verbose:
        print("Real-Data Scenario Sweep Case Study")
        print(f"Orchestration run id: {summary_payload['first_pass']['orchestration_run_id']}")
        print(f"Scenario count: {summary_payload['first_pass']['scenario_count']}")
        print(f"Output directory: {resolved_output_root.as_posix()}")
        if checkpoint_demo and second_summary_payload is not None:
            print(
                "Second-pass preflight state counts: "
                f"{second_summary_payload['scenario_preflight_state_counts']}"
            )

    return ExampleArtifacts(
        summary=summary_payload,
        first_pass=first_summary,
        second_pass=second_summary_payload,
        scenario_matrix=matrix_df,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a scenario-sweep campaign case study using repository features and existing alpha models."
    )
    parser.add_argument(
        "--output-root",
        help="Optional output root override. Defaults to docs/examples/output/real_data_scenario_sweep_case_study.",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Keep existing outputs instead of deleting the output root first.",
    )
    parser.add_argument(
        "--checkpoint-demo",
        action="store_true",
        help="Run a second pass over the same output root to demonstrate scenario-level checkpoint reuse.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_root = None if args.output_root is None else Path(args.output_root)
    run_example(
        output_root=output_root,
        reset_output=not args.no_reset,
        checkpoint_demo=args.checkpoint_demo,
    )


if __name__ == "__main__":
    main()