"""CLI entry point for candidate selection workflow."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import yaml

from src.config.runtime import resolve_runtime_config
from src.research.candidate_review import CandidateReviewArtifacts, review_candidate_selection
from src.research.candidate_selection import run_candidate_selection
from src.research.candidate_selection.loader import CandidateSelectionError
from src.research.candidate_selection.registry import candidate_selection_registry_path
from src.research.registry import load_registry
from src.research.strict_mode import ResearchStrictModeError, raise_research_validation_error
from src.pipeline.cli_adapter import build_pipeline_cli_result

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ALPHA_ARTIFACTS_ROOT = Path("artifacts") / "alpha"
DEFAULT_CANDIDATE_ARTIFACTS_ROOT = Path("artifacts") / "candidate_selection"
DEFAULT_PORTFOLIO_ARTIFACTS_ROOT = Path("artifacts") / "portfolios"


@dataclass(frozen=True)
class CandidateSelectionRunResult:
    """Structured result returned from candidate selection CLI run."""

    run_id: str
    universe_count: int
    eligible_count: int
    rejected_count: int
    selected_count: int
    pruned_by_redundancy: int
    primary_metric: str
    filters: dict[str, Any]
    thresholds: dict[str, Any]
    redundancy_thresholds: dict[str, Any]
    stage_execution: dict[str, bool]
    universe_csv: Path
    selected_csv: Path
    rejected_csv: Path
    eligibility_csv: Path
    correlation_csv: Path
    allocation_csv: Path | None
    summary_json: Path
    manifest_json: Path
    artifact_dir: Path
    allocation_method: str | None
    allocation_enabled: bool
    strict_mode: bool
    review_artifacts: CandidateReviewArtifacts | None = None
    registry_entry: dict[str, Any] | None = None
    allocation_constraints: dict[str, Any] = field(default_factory=dict)
    allocation_summary: dict[str, Any] = field(default_factory=dict)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for candidate selection."""

    parser = argparse.ArgumentParser(
        description="Run deterministic candidate selection (Universe -> Eligibility -> Redundancy -> Allocation) with optional review integration."
    )
    parser.add_argument("--config", help="Optional YAML/JSON config path for candidate selection inputs.")

    input_group = parser.add_argument_group("input filtering")
    input_group.add_argument(
        "--artifacts-root",
        help="Root directory of alpha evaluation artifacts. Defaults to artifacts/alpha.",
    )
    input_group.add_argument("--alpha-name", help="Optional alpha name filter.")
    input_group.add_argument("--dataset", help="Optional dataset filter.")
    input_group.add_argument("--timeframe", help="Optional timeframe filter.")
    input_group.add_argument(
        "--evaluation-horizon",
        type=int,
        help="Optional evaluation horizon (bars) filter.",
    )
    input_group.add_argument("--mapping-name", help="Optional signal mapping name filter.")
    input_group.add_argument(
        "--metric",
        choices=("ic_ir", "mean_ic", "mean_rank_ic", "rank_ic_ir"),
        help="Primary deterministic ranking metric.",
    )
    input_group.add_argument(
        "--max-candidates",
        type=int,
        help="Maximum number of post-filter candidates to keep. Defaults to all.",
    )

    gate_group = parser.add_argument_group("eligibility thresholds")
    gate_group.add_argument("--min-ic", type=float, dest="min_mean_ic", help="Minimum mean IC threshold.")
    gate_group.add_argument("--min-rank-ic", type=float, dest="min_mean_rank_ic", help="Minimum mean Rank IC threshold.")
    gate_group.add_argument("--min-ic-ir", type=float, dest="min_ic_ir", help="Minimum IC information ratio threshold.")
    gate_group.add_argument(
        "--min-rank-ic-ir",
        type=float,
        dest="min_rank_ic_ir",
        help="Minimum rank IC information ratio threshold.",
    )
    gate_group.add_argument(
        "--min-history-length",
        type=int,
        dest="min_history_length",
        help="Minimum evaluation history length threshold.",
    )
    gate_group.add_argument(
        "--min-coverage",
        type=float,
        dest="min_coverage",
        help="Reserved extension-point threshold for future candidate-coverage gating.",
    )

    redundancy_group = parser.add_argument_group("redundancy filtering")
    redundancy_group.add_argument(
        "--max-pairwise-correlation",
        type=float,
        dest="max_pairwise_correlation",
        help="Absolute pairwise sleeve-return correlation threshold in [0, 1].",
    )
    redundancy_group.add_argument(
        "--min-overlap-observations",
        type=int,
        dest="min_overlap_observations",
        help="Minimum overlapping timestamps required for pairwise correlation checks.",
    )

    allocation_group = parser.add_argument_group("allocation")
    allocation_group.add_argument(
        "--allocation-method",
        choices=("equal_weight", "max_sharpe", "risk_parity"),
        help="Deterministic allocation governance method.",
    )
    allocation_group.add_argument(
        "--max-weight-per-candidate",
        type=float,
        help="Maximum final allocation weight allowed per candidate.",
    )
    allocation_group.add_argument(
        "--min-allocation-candidate-count",
        type=int,
        help="Minimum number of candidates required after allocation constraints.",
    )
    allocation_group.add_argument(
        "--min-allocation-weight",
        type=float,
        help="Optional minimum allocation-weight threshold before renormalization.",
    )
    allocation_group.add_argument(
        "--allocation-weight-sum-tolerance",
        type=float,
        help="Tolerance used to validate post-allocation weight sum equals 1.",
    )
    allocation_group.add_argument(
        "--allocation-rounding-decimals",
        type=int,
        help="Deterministic decimal precision retained for allocation weights.",
    )

    execution_group = parser.add_argument_group("execution / behavior")
    execution_group.add_argument(
        "--strict",
        action="store_true",
        default=None,
        help="Enable strict mode validation for candidate selection outputs.",
    )
    execution_group.add_argument(
        "--skip-eligibility",
        action="store_true",
        default=None,
        help="Skip eligibility stage and pass all universe rows to ranking.",
    )
    execution_group.add_argument(
        "--skip-redundancy",
        action="store_true",
        default=None,
        help="Skip redundancy stage and keep ranked eligible rows unchanged.",
    )
    execution_group.add_argument(
        "--skip-allocation",
        action="store_true",
        default=None,
        help="Skip allocation governance stage.",
    )
    execution_group.add_argument(
        "--enable-review",
        action="store_true",
        default=None,
        help="Run candidate review stage after candidate selection completes.",
    )

    input_ref_group = parser.add_argument_group("input references")
    input_ref_group.add_argument(
        "--from-registry",
        action="store_true",
        default=None,
        help="Resolve candidate selection artifact path from the latest candidate selection registry entry.",
    )
    input_ref_group.add_argument(
        "--candidate-selection-run-id",
        help="Resolve candidate selection artifact path from a specific run id.",
    )
    input_ref_group.add_argument(
        "--candidate-selection-path",
        help="Use an explicit candidate selection artifact directory (for review/chaining workflows).",
    )

    output_group = parser.add_argument_group("output control")
    output_group.add_argument(
        "--output-path",
        help="Candidate selection artifact root override. Defaults to artifacts/candidate_selection.",
    )
    output_group.add_argument(
        "--registry-path",
        help="Candidate selection registry path override.",
    )
    output_group.add_argument(
        "--register-run",
        action="store_true",
        default=None,
        help="Upsert this run into candidate selection registry.",
    )

    review_group = parser.add_argument_group("review integration")
    review_group.add_argument(
        "--portfolio-run-id",
        help="Portfolio run id used by review stage (resolved under --portfolio-artifacts-root).",
    )
    review_group.add_argument(
        "--portfolio-path",
        help="Explicit portfolio artifact directory used by review stage.",
    )
    review_group.add_argument(
        "--portfolio-artifacts-root",
        help="Portfolio artifacts root when resolving --portfolio-run-id. Defaults to artifacts/portfolios.",
    )
    review_group.add_argument("--review-output-path", help="Optional output directory override for review artifacts.")
    review_group.add_argument(
        "--no-markdown-review",
        action="store_true",
        default=None,
        help="Disable candidate_review_report.md generation during review stage.",
    )
    review_group.add_argument("--review-top-n", type=int, help="Top-N rows to include in markdown review tables.")

    return parser.parse_args(argv)


def load_candidate_selection_config(path: str | Path) -> dict[str, Any]:
    """Load one candidate-selection YAML/JSON config mapping."""

    resolved = Path(path)
    with resolved.open("r", encoding="utf-8") as file_obj:
        if resolved.suffix.lower() == ".json":
            payload = json.load(file_obj)
        else:
            payload = yaml.safe_load(file_obj) or {}

    if not isinstance(payload, dict):
        raise ValueError("Candidate selection config must deserialize to a mapping.")

    nested = payload.get("candidate_selection")
    if nested is None:
        return payload
    if not isinstance(nested, dict):
        raise ValueError("candidate_selection config section must be a mapping.")
    return nested


def _to_flat_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}

    for key in (
        "artifacts_root",
        "alpha_name",
        "dataset",
        "timeframe",
        "evaluation_horizon",
        "mapping_name",
        "metric",
        "max_candidates",
        "output_path",
        "candidate_selection_run_id",
        "candidate_selection_path",
        "from_registry",
        "register_run",
        "registry_path",
        "portfolio_run_id",
        "portfolio_path",
        "portfolio_artifacts_root",
        "review_output_path",
        "review_top_n",
    ):
        if key in payload:
            flat[key] = payload[key]

    eligibility = payload.get("eligibility")
    if isinstance(eligibility, Mapping):
        mapping = {
            "min_ic": "min_mean_ic",
            "min_mean_ic": "min_mean_ic",
            "min_rank_ic": "min_mean_rank_ic",
            "min_mean_rank_ic": "min_mean_rank_ic",
            "min_ic_ir": "min_ic_ir",
            "min_rank_ic_ir": "min_rank_ic_ir",
            "min_history_length": "min_history_length",
            "min_coverage": "min_coverage",
        }
        for source, target in mapping.items():
            if source in eligibility:
                flat[target] = eligibility[source]

    redundancy = payload.get("redundancy")
    if isinstance(redundancy, Mapping):
        mapping = {
            "max_correlation": "max_pairwise_correlation",
            "max_pairwise_correlation": "max_pairwise_correlation",
            "min_overlap_observations": "min_overlap_observations",
        }
        for source, target in mapping.items():
            if source in redundancy:
                flat[target] = redundancy[source]

    allocation = payload.get("allocation")
    if isinstance(allocation, Mapping):
        mapping = {
            "method": "allocation_method",
            "allocation_method": "allocation_method",
            "max_weight": "max_weight_per_candidate",
            "max_weight_per_candidate": "max_weight_per_candidate",
            "min_candidates": "min_allocation_candidate_count",
            "min_allocation_candidate_count": "min_allocation_candidate_count",
            "min_weight": "min_allocation_weight",
            "min_allocation_weight": "min_allocation_weight",
            "weight_sum_tolerance": "allocation_weight_sum_tolerance",
            "allocation_weight_sum_tolerance": "allocation_weight_sum_tolerance",
            "rounding_decimals": "allocation_rounding_decimals",
            "allocation_rounding_decimals": "allocation_rounding_decimals",
        }
        for source, target in mapping.items():
            if source in allocation:
                flat[target] = allocation[source]

    execution = payload.get("execution")
    if isinstance(execution, Mapping):
        mapping = {
            "strict_mode": "strict_mode",
            "skip_eligibility": "skip_eligibility",
            "skip_redundancy": "skip_redundancy",
            "skip_allocation": "skip_allocation",
            "enable_review": "enable_review",
            "no_markdown_review": "no_markdown_review",
        }
        for source, target in mapping.items():
            if source in execution:
                flat[target] = execution[source]

    outputs = payload.get("output")
    if isinstance(outputs, Mapping):
        if "path" in outputs:
            flat["output_path"] = outputs["path"]
        if "registry_path" in outputs:
            flat["registry_path"] = outputs["registry_path"]

    return flat


def resolve_cli_config(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve deterministic candidate-selection config with precedence.

    Precedence: defaults < config file < CLI overrides.
    """

    defaults: dict[str, Any] = {
        "artifacts_root": DEFAULT_ALPHA_ARTIFACTS_ROOT,
        "output_path": DEFAULT_CANDIDATE_ARTIFACTS_ROOT,
        "portfolio_artifacts_root": DEFAULT_PORTFOLIO_ARTIFACTS_ROOT,
        "metric": "ic_ir",
        "allocation_method": "equal_weight",
        "allocation_weight_sum_tolerance": 1e-12,
        "allocation_rounding_decimals": 12,
        "strict_mode": False,
        "skip_eligibility": False,
        "skip_redundancy": False,
        "skip_allocation": False,
        "enable_review": False,
        "review_top_n": 10,
        "no_markdown_review": False,
        "register_run": False,
        "from_registry": False,
        "min_coverage": None,
    }
    resolved: dict[str, Any] = dict(defaults)

    if args.config is not None:
        config_payload = load_candidate_selection_config(args.config)
        flat_payload = _to_flat_config(config_payload)
        for key, value in flat_payload.items():
            if value is not None:
                resolved[key] = value

    cli_payload = {
        "artifacts_root": args.artifacts_root,
        "alpha_name": args.alpha_name,
        "dataset": args.dataset,
        "timeframe": args.timeframe,
        "evaluation_horizon": args.evaluation_horizon,
        "mapping_name": args.mapping_name,
        "metric": args.metric,
        "max_candidates": args.max_candidates,
        "output_path": args.output_path,
        "min_mean_ic": args.min_mean_ic,
        "min_mean_rank_ic": args.min_mean_rank_ic,
        "min_ic_ir": args.min_ic_ir,
        "min_rank_ic_ir": args.min_rank_ic_ir,
        "min_history_length": args.min_history_length,
        "min_coverage": args.min_coverage,
        "max_pairwise_correlation": args.max_pairwise_correlation,
        "min_overlap_observations": args.min_overlap_observations,
        "allocation_method": args.allocation_method,
        "max_weight_per_candidate": args.max_weight_per_candidate,
        "min_allocation_candidate_count": args.min_allocation_candidate_count,
        "min_allocation_weight": args.min_allocation_weight,
        "allocation_weight_sum_tolerance": args.allocation_weight_sum_tolerance,
        "allocation_rounding_decimals": args.allocation_rounding_decimals,
        "strict_mode": args.strict,
        "skip_eligibility": args.skip_eligibility,
        "skip_redundancy": args.skip_redundancy,
        "skip_allocation": args.skip_allocation,
        "enable_review": args.enable_review,
        "candidate_selection_run_id": args.candidate_selection_run_id,
        "candidate_selection_path": args.candidate_selection_path,
        "from_registry": args.from_registry,
        "register_run": args.register_run,
        "registry_path": args.registry_path,
        "portfolio_run_id": args.portfolio_run_id,
        "portfolio_path": args.portfolio_path,
        "portfolio_artifacts_root": args.portfolio_artifacts_root,
        "review_output_path": args.review_output_path,
        "review_top_n": args.review_top_n,
        "no_markdown_review": args.no_markdown_review,
    }
    for key, value in cli_payload.items():
        if value is not None:
            resolved[key] = value

    resolved["artifacts_root"] = Path(str(resolved["artifacts_root"]))
    resolved["output_path"] = Path(str(resolved["output_path"]))
    resolved["portfolio_artifacts_root"] = Path(str(resolved["portfolio_artifacts_root"]))
    if resolved.get("registry_path") is not None:
        resolved["registry_path"] = str(resolved["registry_path"])

    resolved["strict_mode"] = bool(resolved.get("strict_mode", False))
    resolved["skip_eligibility"] = bool(resolved.get("skip_eligibility", False))
    resolved["skip_redundancy"] = bool(resolved.get("skip_redundancy", False))
    resolved["skip_allocation"] = bool(resolved.get("skip_allocation", False))
    resolved["enable_review"] = bool(resolved.get("enable_review", False))
    resolved["register_run"] = bool(resolved.get("register_run", False))
    resolved["from_registry"] = bool(resolved.get("from_registry", False))
    resolved["no_markdown_review"] = bool(resolved.get("no_markdown_review", False))

    return resolved


def _resolve_candidate_selection_artifact_dir_from_registry(
    *,
    output_path: Path,
    registry_path: str | Path | None,
) -> Path:
    resolved_registry_path = candidate_selection_registry_path(output_path) if registry_path is None else Path(registry_path)
    entries = load_registry(resolved_registry_path)
    candidates = [
        entry
        for entry in entries
        if str(entry.get("run_type", "")) == "candidate_selection" and entry.get("artifact_path")
    ]
    if not candidates:
        raise ValueError(f"No candidate_selection entries found in registry: {resolved_registry_path}")
    latest = sorted(candidates, key=lambda entry: str(entry.get("timestamp") or ""))[-1]
    return Path(str(latest["artifact_path"]))


def _resolve_portfolio_artifact_dir(config: Mapping[str, Any]) -> Path:
    run_id = config.get("portfolio_run_id")
    path = config.get("portfolio_path")
    has_run_id = run_id is not None
    has_path = path is not None
    if has_run_id == has_path:
        raise ValueError("Review requires exactly one of --portfolio-run-id or --portfolio-path.")
    if has_path:
        return Path(str(path))
    return Path(config["portfolio_artifacts_root"]) / str(run_id)


def _resolve_existing_candidate_artifact_dir(config: Mapping[str, Any]) -> Path | None:
    has_run_id = config.get("candidate_selection_run_id") is not None
    has_path = config.get("candidate_selection_path") is not None
    has_registry = bool(config.get("from_registry", False))
    modes = int(has_run_id) + int(has_path) + int(has_registry)
    if modes == 0:
        return None
    if modes > 1:
        raise ValueError(
            "Provide only one of --candidate-selection-run-id, --candidate-selection-path, or --from-registry."
        )

    if has_run_id:
        return Path(config["output_path"]) / str(config["candidate_selection_run_id"])
    if has_path:
        return Path(str(config["candidate_selection_path"]))
    return _resolve_candidate_selection_artifact_dir_from_registry(
        output_path=Path(config["output_path"]),
        registry_path=config.get("registry_path"),
    )


def _result_from_existing_artifact_dir(
    artifact_dir: Path,
    *,
    strict_mode: bool,
    review_artifacts: CandidateReviewArtifacts | None,
) -> CandidateSelectionRunResult:
    if not artifact_dir.exists() or not artifact_dir.is_dir():
        raise ValueError(f"Candidate selection artifact dir does not exist: {artifact_dir}")

    manifest_path = artifact_dir / "manifest.json"
    summary_path = artifact_dir / "selection_summary.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    run_id = str(manifest.get("run_id") or summary.get("run_id") or artifact_dir.name)
    stage_execution = manifest.get("stage_execution")
    if not isinstance(stage_execution, dict):
        stage_execution = summary.get("stage_execution")
    if not isinstance(stage_execution, dict):
        stage_execution = {
            "universe": True,
            "eligibility": bool(manifest.get("eligibility_applied", True)),
            "redundancy": bool(manifest.get("redundancy_applied", True)),
            "allocation": bool(manifest.get("allocation_applied", False)),
        }

    return CandidateSelectionRunResult(
        run_id=run_id,
        universe_count=int(summary.get("universe_count", summary.get("total_candidates", 0))),
        eligible_count=int(summary.get("eligible_candidates", 0)),
        rejected_count=int(summary.get("rejected_count", summary.get("rejected_candidates", 0))),
        selected_count=int(summary.get("selected_count", summary.get("selected_candidates", 0))),
        pruned_by_redundancy=int(summary.get("pruned_by_redundancy") or 0),
        primary_metric=str(summary.get("primary_metric") or manifest.get("config_snapshot", {}).get("primary_metric") or "ic_ir"),
        filters=dict(manifest.get("config_snapshot", {}).get("filters", {})),
        thresholds=dict(summary.get("thresholds", {})),
        redundancy_thresholds=dict(summary.get("redundancy_thresholds", {})),
        stage_execution={k: bool(v) for k, v in stage_execution.items()},
        universe_csv=artifact_dir / "candidate_universe.csv",
        selected_csv=artifact_dir / "selected_candidates.csv",
        rejected_csv=artifact_dir / "rejected_candidates.csv",
        eligibility_csv=artifact_dir / "eligibility_filter_results.csv",
        correlation_csv=artifact_dir / "correlation_matrix.csv",
        allocation_csv=(artifact_dir / "allocation_weights.csv"),
        summary_json=summary_path,
        manifest_json=manifest_path,
        artifact_dir=artifact_dir,
        allocation_method=summary.get("allocation_method"),
        allocation_enabled=bool(manifest.get("allocation_applied", False)),
        strict_mode=bool(strict_mode),
        review_artifacts=review_artifacts,
        registry_entry=None,
        allocation_constraints=dict(summary.get("allocation_constraints", {})),
        allocation_summary=dict(summary.get("allocation_summary", {})),
    )


def run_cli(
    argv: Sequence[str] | None = None,
    *,
    state: Mapping[str, Any] | None = None,
    pipeline_context: Mapping[str, Any] | None = None,
) -> CandidateSelectionRunResult | dict[str, Any]:
    """Execute candidate selection CLI flow from parsed command-line arguments."""

    args = parse_args(argv)
    resolved = resolve_cli_config(args)

    runtime_config = resolve_runtime_config(
        {"runtime": {"strict_mode": {"enabled": bool(resolved.get("strict_mode", False))}}},
        cli_strict=bool(resolved.get("strict_mode", False)),
    )
    strict_enabled = bool(runtime_config.strict_mode.enabled)

    existing_candidate_dir = _resolve_existing_candidate_artifact_dir(resolved)

    if existing_candidate_dir is None:
        if resolved.get("min_coverage") is not None:
            raise ValueError("--min-coverage is reserved for a future stage and is not supported yet.")

        result = run_candidate_selection(
            artifacts_root=Path(resolved["artifacts_root"]),
            alpha_name=resolved.get("alpha_name"),
            dataset=resolved.get("dataset"),
            timeframe=resolved.get("timeframe"),
            evaluation_horizon=resolved.get("evaluation_horizon"),
            mapping_name=resolved.get("mapping_name"),
            primary_metric=str(resolved.get("metric") or "ic_ir"),
            max_candidate_count=resolved.get("max_candidates"),
            output_artifacts_root=Path(resolved["output_path"]),
            min_mean_ic=resolved.get("min_mean_ic"),
            min_mean_rank_ic=resolved.get("min_mean_rank_ic"),
            min_ic_ir=resolved.get("min_ic_ir"),
            min_rank_ic_ir=resolved.get("min_rank_ic_ir"),
            min_history_length=resolved.get("min_history_length"),
            max_pairwise_correlation=resolved.get("max_pairwise_correlation"),
            min_overlap_observations=resolved.get("min_overlap_observations"),
            skip_eligibility=bool(resolved.get("skip_eligibility", False)),
            skip_redundancy=bool(resolved.get("skip_redundancy", False)),
            allocation_method=str(resolved.get("allocation_method") or "equal_weight"),
            max_weight_per_candidate=resolved.get("max_weight_per_candidate"),
            min_allocation_candidate_count=resolved.get("min_allocation_candidate_count"),
            min_allocation_weight=resolved.get("min_allocation_weight"),
            allocation_weight_sum_tolerance=float(resolved.get("allocation_weight_sum_tolerance") or 1e-12),
            allocation_rounding_decimals=int(resolved.get("allocation_rounding_decimals") or 12),
            allocation_enabled=not bool(resolved.get("skip_allocation", False)),
            register_run=bool(resolved.get("register_run", False)),
            registry_path=resolved.get("registry_path"),
        )

        if strict_enabled and int(result.get("selected_count", 0)) <= 0:
            raise_research_validation_error(
                validator="candidate_selection",
                scope="candidate_selection",
                exc=ValueError("No candidates selected by configured pipeline."),
                strict_mode=True,
            )

        candidate_result = CandidateSelectionRunResult(
            run_id=result["run_id"],
            universe_count=result["universe_count"],
            eligible_count=result["eligible_count"],
            rejected_count=result["rejected_count"],
            selected_count=result["selected_count"],
            pruned_by_redundancy=result.get("redundancy_summary", {}).get("pruned_by_redundancy", 0),
            primary_metric=result["primary_metric"],
            filters=dict(result.get("filters", {})),
            thresholds=dict(result.get("thresholds", {})),
            redundancy_thresholds=dict(result.get("redundancy_thresholds", {})),
            stage_execution={k: bool(v) for k, v in dict(result.get("stage_execution", {})).items()},
            universe_csv=Path(result["universe_csv"]),
            selected_csv=Path(result["selected_csv"]),
            rejected_csv=Path(result["rejected_csv"]),
            eligibility_csv=Path(result["eligibility_csv"]),
            correlation_csv=Path(result["correlation_csv"]),
            allocation_csv=(None if result.get("allocation_csv") is None else Path(result["allocation_csv"])),
            summary_json=Path(result["summary_json"]),
            manifest_json=Path(result["manifest_json"]),
            artifact_dir=Path(result["artifact_dir"]),
            allocation_method=result.get("allocation_summary", {}).get("allocation_method"),
            allocation_enabled=bool(result.get("allocation_summary", {}).get("allocation_enabled", False)),
            strict_mode=strict_enabled,
            review_artifacts=None,
            registry_entry=result.get("registry_entry"),
            allocation_constraints=dict(result.get("allocation_constraints", {})),
            allocation_summary=dict(result.get("allocation_summary", {})),
        )
    else:
        candidate_result = _result_from_existing_artifact_dir(
            existing_candidate_dir,
            strict_mode=strict_enabled,
            review_artifacts=None,
        )

    if bool(resolved.get("enable_review", False)):
        portfolio_dir = _resolve_portfolio_artifact_dir(resolved)
        review_result = review_candidate_selection(
            candidate_selection_artifact_dir=candidate_result.artifact_dir,
            portfolio_artifact_dir=portfolio_dir,
            output_dir=(
                None
                if resolved.get("review_output_path") is None
                else Path(str(resolved["review_output_path"]))
            ),
            include_markdown_report=not bool(resolved.get("no_markdown_review", False)),
            top_n=max(1, int(resolved.get("review_top_n") or 10)),
        )
        candidate_result = CandidateSelectionRunResult(
            **{**candidate_result.__dict__, "review_artifacts": review_result}
        )

    if pipeline_context is not None:
        return build_pipeline_cli_result(
            identifier=candidate_result.run_id,
            name="candidate_selection",
            artifact_dir=candidate_result.artifact_dir,
            manifest_path=candidate_result.manifest_json,
            output_paths={
                "selected_candidates_csv": candidate_result.selected_csv,
                "rejected_candidates_csv": candidate_result.rejected_csv,
                "selection_summary_json": candidate_result.summary_json,
                "allocation_weights_csv": candidate_result.allocation_csv,
            },
            extra={
                "primary_metric": candidate_result.primary_metric,
                "selected_count": candidate_result.selected_count,
                "eligible_count": candidate_result.eligible_count,
                "rejected_count": candidate_result.rejected_count,
                "stage_execution": dict(candidate_result.stage_execution),
            },
            state_updates={
                "candidate_selection_artifact_dir": candidate_result.artifact_dir.as_posix(),
                "candidate_selection_manifest_path": candidate_result.manifest_json.as_posix(),
                "candidate_selection_run_id": candidate_result.run_id,
            },
        )
    return candidate_result


def print_candidate_selection_summary(result: CandidateSelectionRunResult) -> None:
    """Print concise candidate selection CLI summary."""

    rejected_by_eligibility = 0
    rejected_by_redundancy = 0
    try:
        rejected_rows = json.loads(result.summary_json.read_text(encoding="utf-8")).get("rejected_candidates_by_stage", {})
        rejected_by_eligibility = int(rejected_rows.get("eligibility", 0))
        rejected_by_redundancy = int(rejected_rows.get("redundancy", 0))
    except (OSError, ValueError, json.JSONDecodeError):
        rejected_by_redundancy = int(result.pruned_by_redundancy)

    print("Candidate Selection Summary")
    print("--------------------------")
    print(f"Run ID: {result.run_id}")
    print(f"Total candidates: {result.universe_count}")
    print(f"Eligible: {result.eligible_count}")
    print(f"Rejected (eligibility): {rejected_by_eligibility}")
    print(f"Rejected (redundancy): {rejected_by_redundancy}")
    print(f"Selected: {result.selected_count}")
    print()
    print(f"Primary metric: {result.primary_metric}")
    print(f"Allocation method: {result.allocation_method}")
    print(f"Allocation enabled: {result.allocation_enabled}")
    print(f"Strict mode: {result.strict_mode}")
    print(f"Stage execution: {result.stage_execution}")
    print()
    print("Artifacts written to:")
    print(result.artifact_dir.as_posix() + "/")
    if result.review_artifacts is not None:
        print()
        print("Review artifacts written to:")
        print(result.review_artifacts.review_dir.as_posix() + "/")


def main() -> None:
    try:
        result = run_cli()
        print_candidate_selection_summary(result)
    except (ValueError, OSError, CandidateSelectionError, ResearchStrictModeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
