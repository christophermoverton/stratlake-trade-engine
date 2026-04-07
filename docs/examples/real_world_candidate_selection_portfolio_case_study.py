from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cli.compare_alpha import run_cli as compare_alpha_cli
from src.cli.run_alpha import run_cli as run_alpha_cli
from src.cli.run_candidate_selection import run_cli as run_candidate_selection_cli
from src.cli.run_portfolio import run_cli as run_portfolio_cli


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "output" / "real_world_candidate_selection_portfolio_case_study"
SUMMARY_FILENAME = "summary.json"
DATASET = "features_daily"
TIMEFRAME = "1D"
EVALUATION_HORIZON = 5
MAPPING_NAME = "top_bottom_quantile_q20"
ALPHA_CONFIG = Path("configs") / "alphas_2026_q1.yml"
ALPHA_NAMES = (
    "ml_cross_sectional_xgb_2026_q1",
    "ml_cross_sectional_lgbm_2026_q1",
    "rank_composite_momentum_2026_q1",
)
PORTFOLIO_NAME = "real_world_m15_candidate_driven"


@dataclass(frozen=True)
class ExampleArtifacts:
    summary: dict[str, Any]
    comparison_leaderboard: pd.DataFrame
    candidate_universe: pd.DataFrame
    selected_candidates: pd.DataFrame
    rejected_candidates: pd.DataFrame
    allocation_weights: pd.DataFrame
    portfolio_returns: pd.DataFrame


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
    available_columns = [column for column in columns if column in frame.columns]
    if not available_columns:
        return []
    preview = frame.loc[:, available_columns].head(limit).copy(deep=True)
    if "ts_utc" in preview.columns:
        preview["ts_utc"] = pd.to_datetime(preview["ts_utc"], utc=True, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return _normalize_json_value(preview.to_dict(orient="records"))


def _run_alpha_suite(alpha_artifacts_root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for alpha_name in ALPHA_NAMES:
        result = run_alpha_cli(
            [
                "--config",
                ALPHA_CONFIG.as_posix(),
                "--alpha-name",
                alpha_name,
                "--artifacts-root",
                alpha_artifacts_root.as_posix(),
                "--signal-policy",
                "top_bottom_quantile",
                "--signal-quantile",
                "0.2",
            ]
        )
        metrics = result.evaluation.evaluation_result.summary
        sleeve_metrics = _read_json(result.artifact_dir / "sleeve_metrics.json")
        runs.append(
            {
                "alpha_name": alpha_name,
                "run_id": result.run_id,
                "artifact_dir": result.artifact_dir.as_posix(),
                "mean_ic": float(metrics["mean_ic"]),
                "ic_ir": None if metrics["ic_ir"] is None else float(metrics["ic_ir"]),
                "mean_rank_ic": float(metrics["mean_rank_ic"]),
                "rank_ic_ir": None if metrics["rank_ic_ir"] is None else float(metrics["rank_ic_ir"]),
                "n_periods": int(metrics["n_periods"]),
                "sleeve_total_return": float(sleeve_metrics.get("total_return", 0.0)),
                "sleeve_sharpe_ratio": float(sleeve_metrics.get("sharpe_ratio", 0.0)),
            }
        )
    return runs


def _write_summary(
    output_root: Path,
    *,
    alpha_runs: list[dict[str, Any]],
    comparison_result: Any,
    candidate_result: Any,
    review_result: Any,
    portfolio_result: Any,
    candidate_universe: pd.DataFrame,
    selected_candidates: pd.DataFrame,
    rejected_candidates: pd.DataFrame,
    allocation_weights: pd.DataFrame,
) -> dict[str, Any]:
    comparison_leaderboard = pd.read_csv(comparison_result.csv_path)
    selection_summary = _read_json(candidate_result.summary_json)
    selection_manifest = _read_json(candidate_result.manifest_json)
    portfolio_manifest = _read_json(portfolio_result.experiment_dir / "manifest.json")
    portfolio_metrics = _read_json(portfolio_result.experiment_dir / "metrics.json")
    review_summary = _read_json(review_result.review_artifacts.candidate_review_summary_json)
    review_manifest = _read_json(review_result.review_artifacts.manifest_json)

    summary = {
        "workflow": [
            "run_builtin_alpha_models_on_features_daily",
            "compare_alpha_runs",
            "run_candidate_selection_governance",
            "apply_governed_allocation",
            "construct_candidate_driven_portfolio",
            "write_candidate_review_explainability",
        ],
        "alpha_runs": alpha_runs,
        "candidate_inputs": {
            "dataset": DATASET,
            "timeframe": TIMEFRAME,
            "evaluation_horizon": EVALUATION_HORIZON,
            "mapping_name": MAPPING_NAME,
            "comparison_id": comparison_result.comparison_id,
            "comparison_metric": comparison_result.metric,
            "comparison_view": comparison_result.view,
            "comparison_leaderboard_preview": _preview(
                comparison_leaderboard,
                columns=["rank", "alpha_name", "run_id", "ic_ir", "sharpe_ratio", "total_return"],
                limit=8,
            ),
            "considered_candidate_count": int(len(candidate_universe)),
        },
        "candidate_selection": {
            "run_id": candidate_result.run_id,
            "primary_metric": candidate_result.primary_metric,
            "counts": {
                "universe": candidate_result.universe_count,
                "eligible": candidate_result.eligible_count,
                "rejected": candidate_result.rejected_count,
                "selected": candidate_result.selected_count,
                "pruned_by_redundancy": candidate_result.pruned_by_redundancy,
            },
            "stage_execution": candidate_result.stage_execution,
            "selection_summary": selection_summary,
            "manifest_artifact_files": selection_manifest.get("artifact_files"),
            "selected_preview": _preview(
                selected_candidates,
                columns=["selection_rank", "candidate_id", "alpha_name", "ic_ir", "allocation_weight"],
                limit=8,
            ),
            "rejected_preview": _preview(
                rejected_candidates,
                columns=["candidate_id", "rejected_stage", "rejection_reason", "observed_correlation"],
                limit=8,
            ),
        },
        "allocation": {
            "enabled": candidate_result.allocation_enabled,
            "method": candidate_result.allocation_method,
            "summary": candidate_result.allocation_summary,
            "weights": _preview(
                allocation_weights,
                columns=["candidate_id", "allocation_weight", "allocation_method"],
                limit=12,
            ),
            "weight_sum": float(allocation_weights["allocation_weight"].sum()),
        },
        "portfolio": {
            "portfolio_name": portfolio_result.portfolio_name,
            "run_id": portfolio_result.run_id,
            "component_count": portfolio_result.component_count,
            "allocator": portfolio_result.allocator_name,
            "metrics": {
                "total_return": portfolio_result.metrics.get("total_return"),
                "sharpe_ratio": portfolio_result.metrics.get("sharpe_ratio"),
                "max_drawdown": portfolio_result.metrics.get("max_drawdown"),
                "realized_volatility": portfolio_result.metrics.get("realized_volatility"),
            },
            "portfolio_manifest_files": portfolio_manifest.get("artifact_files"),
            "portfolio_metrics": portfolio_metrics,
        },
        "review": {
            "review_dir": _relative_to_output(review_result.review_artifacts.review_dir, output_root),
            "candidate_selection_run_id": review_result.review_artifacts.candidate_selection_run_id,
            "portfolio_run_id": review_result.review_artifacts.portfolio_run_id,
            "total_candidates": review_result.review_artifacts.total_candidates,
            "selected_candidates": review_result.review_artifacts.selected_candidates,
            "rejected_candidates": review_result.review_artifacts.rejected_candidates,
            "manifest_artifact_files": review_manifest.get("artifact_files"),
            "review_summary": review_summary,
        },
        "artifacts": {
            "alpha_root": _relative_to_output(output_root / "artifacts" / "alpha", output_root),
            "comparison_csv": _relative_to_output(comparison_result.csv_path, output_root),
            "candidate_selection_dir": _relative_to_output(candidate_result.artifact_dir, output_root),
            "portfolio_dir": _relative_to_output(portfolio_result.experiment_dir, output_root),
            "review_dir": _relative_to_output(review_result.review_artifacts.review_dir, output_root),
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
    print("Real World Candidate Selection Portfolio Case Study")
    print("Alpha runs:")
    print(pd.DataFrame(summary["alpha_runs"]).to_string(index=False))
    counts = summary["candidate_selection"]["counts"]
    print(
        "Selection counts: "
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
    print(f"Output directory: {output_root.as_posix()}")


def run_example(*, output_root: Path | None = None, verbose: bool = True) -> ExampleArtifacts:
    resolved_output_root = DEFAULT_OUTPUT_ROOT if output_root is None else Path(output_root)
    if resolved_output_root.exists():
        shutil.rmtree(resolved_output_root)
    resolved_output_root.mkdir(parents=True, exist_ok=True)

    alpha_root = resolved_output_root / "artifacts" / "alpha"
    comparison_output = resolved_output_root / "artifacts" / "alpha_comparisons"
    candidate_output = resolved_output_root / "artifacts" / "candidate_selection"
    portfolio_output = resolved_output_root / "artifacts" / "portfolios"
    review_output = resolved_output_root / "artifacts" / "reviews"
    alpha_root.mkdir(parents=True, exist_ok=True)
    comparison_output.mkdir(parents=True, exist_ok=True)
    candidate_output.mkdir(parents=True, exist_ok=True)
    portfolio_output.mkdir(parents=True, exist_ok=True)
    review_output.mkdir(parents=True, exist_ok=True)

    alpha_runs = _run_alpha_suite(alpha_root)

    comparison_result = compare_alpha_cli(
        [
            "--from-registry",
            "--artifacts-root",
            alpha_root.as_posix(),
            "--dataset",
            DATASET,
            "--timeframe",
            TIMEFRAME,
            "--evaluation-horizon",
            str(EVALUATION_HORIZON),
            "--mapping-name",
            MAPPING_NAME,
            "--view",
            "combined",
            "--metric",
            "ic_ir",
            "--sleeve-metric",
            "sharpe_ratio",
            "--output-path",
            comparison_output.as_posix(),
        ]
    )

    candidate_result = run_candidate_selection_cli(
        [
            "--artifacts-root",
            alpha_root.as_posix(),
            "--output-path",
            candidate_output.as_posix(),
            "--dataset",
            DATASET,
            "--timeframe",
            TIMEFRAME,
            "--evaluation-horizon",
            str(EVALUATION_HORIZON),
            "--mapping-name",
            MAPPING_NAME,
            "--metric",
            "ic_ir",
            "--min-ic",
            "-1.0",
            "--min-rank-ic",
            "-1.0",
            "--min-ic-ir",
            "-1.0",
            "--min-rank-ic-ir",
            "-1.0",
            "--min-history-length",
            "10",
            "--max-pairwise-correlation",
            "0.75",
            "--min-overlap-observations",
            "10",
            "--allocation-method",
            "equal_weight",
            "--max-weight-per-candidate",
            "0.50",
            "--min-allocation-candidate-count",
            "2",
        ]
    )

    portfolio_result = run_portfolio_cli(
        [
            "--from-candidate-selection",
            candidate_result.artifact_dir.as_posix(),
            "--portfolio-name",
            PORTFOLIO_NAME,
            "--output-dir",
            portfolio_output.as_posix(),
            "--timeframe",
            TIMEFRAME,
        ]
    )

    review_result = run_candidate_selection_cli(
        [
            "--candidate-selection-path",
            candidate_result.artifact_dir.as_posix(),
            "--enable-review",
            "--portfolio-path",
            portfolio_result.experiment_dir.as_posix(),
            "--review-output-path",
            (review_output / candidate_result.run_id).as_posix(),
            "--review-top-n",
            "10",
        ]
    )

    candidate_universe = pd.read_csv(candidate_result.universe_csv)
    selected_candidates = pd.read_csv(candidate_result.selected_csv)
    rejected_candidates = pd.read_csv(candidate_result.rejected_csv)
    allocation_weights = pd.read_csv(candidate_result.allocation_csv)
    comparison_leaderboard = pd.read_csv(comparison_result.csv_path)
    portfolio_returns = pd.read_csv(portfolio_result.experiment_dir / "portfolio_returns.csv")

    summary = _write_summary(
        resolved_output_root,
        alpha_runs=alpha_runs,
        comparison_result=comparison_result,
        candidate_result=candidate_result,
        review_result=review_result,
        portfolio_result=portfolio_result,
        candidate_universe=candidate_universe,
        selected_candidates=selected_candidates,
        rejected_candidates=rejected_candidates,
        allocation_weights=allocation_weights,
    )

    if verbose:
        print_summary(summary, resolved_output_root)

    return ExampleArtifacts(
        summary=summary,
        comparison_leaderboard=comparison_leaderboard,
        candidate_universe=candidate_universe,
        selected_candidates=selected_candidates,
        rejected_candidates=rejected_candidates,
        allocation_weights=allocation_weights,
        portfolio_returns=portfolio_returns,
    )


def main() -> None:
    run_example()


if __name__ == "__main__":
    main()
