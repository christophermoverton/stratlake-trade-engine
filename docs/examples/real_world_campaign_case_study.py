from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Iterator

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cli.run_research_campaign import print_summary as print_campaign_summary
from src.cli.run_research_campaign import run_research_campaign
from src.config.research_campaign import load_research_campaign_config, resolve_research_campaign_config


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "output" / "real_world_campaign_case_study"
SUMMARY_FILENAME = "summary.json"
CAMPAIGN_CONFIG_PATH = (
    REPO_ROOT / "docs" / "examples" / "data" / "milestone_16_campaign_configs" / "real_world_campaign.yml"
)


@dataclass(frozen=True)
class ExampleArtifacts:
    summary: dict[str, Any]
    comparison_leaderboard: pd.DataFrame
    selected_candidates: pd.DataFrame
    rejected_candidates: pd.DataFrame
    portfolio_returns: pd.DataFrame
    review_leaderboard: pd.DataFrame


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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative_to_output(path: Path, output_root: Path) -> str:
    try:
        return path.relative_to(output_root).as_posix()
    except ValueError:
        return path.as_posix()


def _normalize_json_value(value: Any) -> Any:
    if value is None or isinstance(value, bool | str | int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, pd.Timestamp):
        timestamp = value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")
        return timestamp.isoformat().replace("+00:00", "Z")
    if isinstance(value, dict):
        return {key: _normalize_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_json_value(item) for item in value]
    if pd.isna(value):
        return None
    return value


def _preview(frame: pd.DataFrame, *, columns: list[str], limit: int) -> list[dict[str, Any]]:
    available = [column for column in columns if column in frame.columns]
    if not available:
        return []
    preview = frame.loc[:, available].head(limit).copy(deep=True)
    if "ts_utc" in preview.columns:
        preview["ts_utc"] = pd.to_datetime(preview["ts_utc"], utc=True, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return _normalize_json_value(preview.to_dict(orient="records"))


def _resolved_config(output_root: Path) -> dict[str, Any]:
    base = load_research_campaign_config(CAMPAIGN_CONFIG_PATH).to_dict()
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
    return base


def _write_summary(
    output_root: Path,
    *,
    result: Any,
    comparison_leaderboard: pd.DataFrame,
    selected_candidates: pd.DataFrame,
    rejected_candidates: pd.DataFrame,
    portfolio_returns: pd.DataFrame,
    review_leaderboard: pd.DataFrame,
) -> dict[str, Any]:
    campaign_summary = _read_json(result.campaign_summary_path)
    candidate_selection_summary = _read_json(result.candidate_selection_result.summary_json)
    candidate_selection_manifest = _read_json(result.candidate_selection_result.manifest_json)
    candidate_review_summary = _read_json(result.candidate_review_result.candidate_review_summary_json)
    candidate_review_manifest = _read_json(result.candidate_review_result.manifest_json)
    portfolio_manifest = _read_json(result.portfolio_result.experiment_dir / "manifest.json")
    portfolio_metrics = _read_json(result.portfolio_result.experiment_dir / "metrics.json")

    alpha_runs = []
    for alpha_result in sorted(result.alpha_results, key=lambda item: str(item.alpha_name)):
        evaluation_summary = alpha_result.evaluation.evaluation_result.summary
        sleeve_metrics = _read_json(Path(alpha_result.artifact_dir) / "sleeve_metrics.json")
        alpha_runs.append(
            {
                "alpha_name": alpha_result.alpha_name,
                "run_id": alpha_result.run_id,
                "mean_ic": float(evaluation_summary["mean_ic"]),
                "ic_ir": None if evaluation_summary["ic_ir"] is None else float(evaluation_summary["ic_ir"]),
                "mean_rank_ic": float(evaluation_summary["mean_rank_ic"]),
                "rank_ic_ir": (
                    None if evaluation_summary["rank_ic_ir"] is None else float(evaluation_summary["rank_ic_ir"])
                ),
                "n_periods": int(evaluation_summary["n_periods"]),
                "sleeve_total_return": float(sleeve_metrics.get("total_return", 0.0)),
                "sleeve_sharpe_ratio": float(sleeve_metrics.get("sharpe_ratio", 0.0)),
            }
        )

    summary = {
        "workflow": [
            "load_real_q1_2026_campaign_config",
            "run_alpha_evaluation_suite",
            "compare_alpha_runs",
            "run_candidate_selection",
            "construct_candidate_driven_portfolio",
            "run_candidate_review",
            "run_research_review",
            "write_stitched_case_study_summary",
        ],
        "config": {
            "source_config": CAMPAIGN_CONFIG_PATH.relative_to(REPO_ROOT).as_posix(),
            "dataset": result.config.dataset_selection.dataset,
            "timeframe": result.config.dataset_selection.timeframe,
            "evaluation_horizon": result.config.dataset_selection.evaluation_horizon,
            "mapping_name": result.config.dataset_selection.mapping_name,
            "alpha_names": list(result.config.targets.alpha_names),
            "portfolio_name": result.config.portfolio.portfolio_name,
            "candidate_selection_alpha_filter": result.config.candidate_selection.alpha_name,
        },
        "campaign": {
            "campaign_run_id": result.campaign_run_id,
            "status": campaign_summary["status"],
            "preflight_status": campaign_summary["preflight_status"],
            "stage_statuses": campaign_summary["stage_statuses"],
            "summary_path": _relative_to_output(result.campaign_summary_path, output_root),
            "manifest_path": _relative_to_output(result.campaign_manifest_path, output_root),
        },
        "alpha_runs": alpha_runs,
        "comparison": {
            "comparison_id": result.alpha_comparison_result.comparison_id,
            "view": result.alpha_comparison_result.view,
            "metric": result.alpha_comparison_result.metric,
            "sleeve_metric": result.alpha_comparison_result.sleeve_metric,
            "row_count": int(len(comparison_leaderboard)),
            "leaderboard_preview": _preview(
                comparison_leaderboard,
                columns=["rank", "alpha_name", "run_id", "ic_ir", "mean_ic", "sharpe_ratio"],
                limit=8,
            ),
        },
        "candidate_selection": {
            "run_id": result.candidate_selection_result.run_id,
            "primary_metric": result.candidate_selection_result.primary_metric,
            "counts": {
                "universe": result.candidate_selection_result.universe_count,
                "eligible": result.candidate_selection_result.eligible_count,
                "rejected": result.candidate_selection_result.rejected_count,
                "selected": result.candidate_selection_result.selected_count,
                "pruned_by_redundancy": result.candidate_selection_result.pruned_by_redundancy,
            },
            "stage_execution": result.candidate_selection_result.stage_execution,
            "selection_summary": candidate_selection_summary,
            "manifest_artifact_files": candidate_selection_manifest.get("artifact_files"),
            "selected_preview": _preview(
                selected_candidates,
                columns=[
                    "selection_rank",
                    "candidate_id",
                    "alpha_name",
                    "alpha_run_id",
                    "ic_ir",
                    "allocation_weight",
                ],
                limit=8,
            ),
            "rejected_preview": _preview(
                rejected_candidates,
                columns=[
                    "candidate_id",
                    "alpha_name",
                    "rejected_stage",
                    "rejection_reason",
                    "observed_correlation",
                ],
                limit=8,
            ),
        },
        "portfolio": {
            "portfolio_name": result.portfolio_result.portfolio_name,
            "run_id": result.portfolio_result.run_id,
            "component_count": result.portfolio_result.component_count,
            "allocator": result.portfolio_result.allocator_name,
            "metrics": {
                "total_return": result.portfolio_result.metrics.get("total_return"),
                "sharpe_ratio": result.portfolio_result.metrics.get("sharpe_ratio"),
                "max_drawdown": result.portfolio_result.metrics.get("max_drawdown"),
                "realized_volatility": result.portfolio_result.metrics.get("realized_volatility"),
            },
            "portfolio_manifest_files": portfolio_manifest.get("artifact_files"),
            "portfolio_metrics": portfolio_metrics,
            "returns_preview": _preview(
                portfolio_returns,
                columns=["ts", "portfolio_return", "portfolio_equity"],
                limit=8,
            ),
        },
        "review": {
            "review_id": result.review_result.review_id,
            "review_entry_count": len(result.review_result.entries),
            "leaderboard_preview": _preview(
                review_leaderboard,
                columns=["run_type", "entity_name", "run_id", "selected_metric_name", "selected_metric_value"],
                limit=8,
            ),
            "candidate_review": {
                "review_dir": _relative_to_output(result.candidate_review_result.review_dir, output_root),
                "candidate_selection_run_id": result.candidate_review_result.candidate_selection_run_id,
                "portfolio_run_id": result.candidate_review_result.portfolio_run_id,
                "total_candidates": result.candidate_review_result.total_candidates,
                "selected_candidates": result.candidate_review_result.selected_candidates,
                "rejected_candidates": result.candidate_review_result.rejected_candidates,
                "summary": candidate_review_summary,
                "manifest_artifact_files": candidate_review_manifest.get("artifact_files"),
            },
        },
        "artifacts": {
            "output_root": output_root.as_posix(),
            "campaign_dir": _relative_to_output(result.campaign_artifact_dir, output_root),
            "comparison_csv": _relative_to_output(result.alpha_comparison_result.csv_path, output_root),
            "candidate_selection_dir": _relative_to_output(result.candidate_selection_result.artifact_dir, output_root),
            "portfolio_dir": _relative_to_output(result.portfolio_result.experiment_dir, output_root),
            "review_dir": _relative_to_output(result.review_result.csv_path.parent, output_root),
            "summary": SUMMARY_FILENAME,
        },
    }

    normalized = _normalize_json_value(summary)
    (output_root / SUMMARY_FILENAME).write_text(
        json.dumps(normalized, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    return normalized


def print_summary(summary: dict[str, Any], output_root: Path) -> None:
    print("Real-World Campaign Case Study")
    print(
        "Campaign: "
        f"{summary['campaign']['campaign_run_id']} "
        f"(status={summary['campaign']['status']}, preflight={summary['campaign']['preflight_status']})"
    )
    print("Alpha runs:")
    print(pd.DataFrame(summary["alpha_runs"]).to_string(index=False))
    counts = summary["candidate_selection"]["counts"]
    print(
        "Candidate selection: "
        f"universe={counts['universe']}, eligible={counts['eligible']}, "
        f"rejected={counts['rejected']}, selected={counts['selected']}, "
        f"pruned_by_redundancy={counts['pruned_by_redundancy']}"
    )
    print(
        "Portfolio: "
        f"name={summary['portfolio']['portfolio_name']}, "
        f"run_id={summary['portfolio']['run_id']}, "
        f"total_return={summary['portfolio']['metrics']['total_return']:.2%}, "
        f"sharpe={summary['portfolio']['metrics']['sharpe_ratio']:.4f}"
    )
    print(
        "Review: "
        f"review_id={summary['review']['review_id']}, "
        f"candidate_review_dir={summary['review']['candidate_review']['review_dir']}"
    )
    print(f"Output directory: {output_root.as_posix()}")


def run_example(*, output_root: Path | None = None, verbose: bool = True) -> ExampleArtifacts:
    resolved_output_root = DEFAULT_OUTPUT_ROOT if output_root is None else Path(output_root)
    if resolved_output_root.exists():
        shutil.rmtree(resolved_output_root)
    resolved_output_root.mkdir(parents=True, exist_ok=True)

    with example_environment():
        config = resolve_research_campaign_config(_resolved_config(resolved_output_root))
        result = run_research_campaign(config)

    comparison_leaderboard = pd.read_csv(result.alpha_comparison_result.csv_path)
    selected_candidates = pd.read_csv(result.candidate_selection_result.selected_csv)
    rejected_candidates = pd.read_csv(result.candidate_selection_result.rejected_csv)
    portfolio_returns = pd.read_csv(result.portfolio_result.experiment_dir / "portfolio_returns.csv")
    review_leaderboard = pd.read_csv(result.review_result.csv_path)

    summary = _write_summary(
        resolved_output_root,
        result=result,
        comparison_leaderboard=comparison_leaderboard,
        selected_candidates=selected_candidates,
        rejected_candidates=rejected_candidates,
        portfolio_returns=portfolio_returns,
        review_leaderboard=review_leaderboard,
    )

    if verbose:
        print_campaign_summary(result)
        print()
        print_summary(summary, resolved_output_root)

    return ExampleArtifacts(
        summary=summary,
        comparison_leaderboard=comparison_leaderboard,
        selected_candidates=selected_candidates,
        rejected_candidates=rejected_candidates,
        portfolio_returns=portfolio_returns,
        review_leaderboard=review_leaderboard,
    )


def main() -> None:
    run_example()


if __name__ == "__main__":
    main()
