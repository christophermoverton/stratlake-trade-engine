"""CLI entry point for candidate selection workflow."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import yaml

# Ensure src is in path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.research.candidate_selection import run_candidate_selection

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS_ROOT = Path("artifacts") / "candidate_selection"


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
    registry_entry: dict[str, Any] | None = None
    allocation_constraints: dict[str, Any] = field(default_factory=dict)
    allocation_summary: dict[str, Any] = field(default_factory=dict)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for candidate selection."""

    parser = argparse.ArgumentParser(
        description="Run a deterministic candidate selection workflow to select alpha candidates for allocation."
    )
    parser.add_argument("--config", help="Optional YAML/JSON config path for candidate selection inputs.")
    parser.add_argument(
        "--artifacts-root",
        help="Optional root directory of alpha evaluation artifacts. Defaults to artifacts/alpha.",
    )
    parser.add_argument("--alpha-name", help="Optional filter by alpha name.")
    parser.add_argument("--dataset", help="Optional filter by dataset.")
    parser.add_argument("--timeframe", help="Optional filter by timeframe.")
    parser.add_argument(
        "--evaluation-horizon",
        type=int,
        help="Optional filter by evaluation horizon (in bars).",
    )
    parser.add_argument("--mapping-name", help="Optional filter by signal mapping name.")
    parser.add_argument(
        "--metric",
        choices=("ic_ir", "mean_ic", "mean_rank_ic", "rank_ic_ir"),
        help="Primary ranking metric. Defaults to 'ic_ir'.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        help="Optional maximum number of candidates to select after eligibility gating. Defaults to all.",
    )
    parser.add_argument(
        "--output-path",
        help="Optional candidate selection artifacts root. Defaults to artifacts/candidate_selection.",
    )

    # Eligibility gate threshold arguments
    gate_group = parser.add_argument_group(
        "eligibility gates",
        "Configurable quality thresholds for candidate eligibility filtering. "
        "Omit any threshold to disable that gate check.",
    )
    gate_group.add_argument(
        "--min-ic",
        type=float,
        dest="min_mean_ic",
        help="Minimum mean IC threshold. Candidates below this value are rejected.",
    )
    gate_group.add_argument(
        "--min-rank-ic",
        type=float,
        dest="min_mean_rank_ic",
        help="Minimum mean Rank IC threshold.",
    )
    gate_group.add_argument(
        "--min-ic-ir",
        type=float,
        dest="min_ic_ir",
        help="Minimum IC information ratio threshold.",
    )
    gate_group.add_argument(
        "--min-rank-ic-ir",
        type=float,
        dest="min_rank_ic_ir",
        help="Minimum Rank IC information ratio threshold.",
    )
    gate_group.add_argument(
        "--min-history-length",
        type=int,
        dest="min_history_length",
        help="Minimum number of evaluation periods (observation count).",
    )
    gate_group.add_argument(
        "--require-mean-ic",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="require_mean_ic",
        help="Reject candidates whose mean_ic metric is missing.",
    )
    gate_group.add_argument(
        "--require-ic-ir",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="require_ic_ir",
        help="Reject candidates whose ic_ir metric is missing.",
    )

    redundancy_group = parser.add_argument_group(
        "redundancy filtering",
        "Optional deterministic cross-candidate redundancy pruning based on sleeve-return correlation.",
    )
    redundancy_group.add_argument(
        "--max-pairwise-correlation",
        type=float,
        dest="max_pairwise_correlation",
        help="Absolute pairwise sleeve-return correlation threshold in [0, 1]. If omitted, redundancy pruning is disabled.",
    )
    redundancy_group.add_argument(
        "--min-overlap-observations",
        type=int,
        dest="min_overlap_observations",
        help="Minimum overlapping timestamps required to evaluate pairwise correlation. Defaults to 10.",
    )

    allocation_group = parser.add_argument_group(
        "allocation governance",
        "Deterministic allocation rules and constraints for selected candidates.",
    )
    allocation_group.add_argument(
        "--allocation-method",
        choices=("equal_weight", "max_sharpe", "risk_parity"),
        help="Allocation governance method. Defaults to equal_weight.",
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
        help="Optional minimum weight threshold. Candidates below this are dropped then weights are renormalized.",
    )
    allocation_group.add_argument(
        "--allocation-weight-sum-tolerance",
        type=float,
        help="Tolerance for validating post-allocation weight-sum equals 1.0.",
    )
    allocation_group.add_argument(
        "--allocation-rounding-decimals",
        type=int,
        help="Number of deterministic decimals retained in final allocation weights.",
    )
    allocation_group.add_argument(
        "--disable-allocation",
        action="store_true",
        default=False,
        help="Disable allocation governance stage and keep Issue 1-3 behavior.",
    )

    parser.add_argument(
        "--register-run",
        action="store_true",
        default=False,
        help="Upsert this candidate-selection run into artifacts/candidate_selection/registry.jsonl.",
    )
    parser.add_argument(
        "--registry-path",
        help="Optional override for candidate-selection registry path.",
    )

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


def resolve_cli_config(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve deterministic candidate-selection config with precedence.

    Precedence: defaults < config file < CLI overrides.
    """

    defaults: dict[str, Any] = {
        "artifacts_root": Path("artifacts") / "alpha",
        "output_path": DEFAULT_ARTIFACTS_ROOT,
        "metric": "ic_ir",
        "allocation_method": "equal_weight",
        "allocation_enabled": True,
        "allocation_weight_sum_tolerance": 1e-12,
        "allocation_rounding_decimals": 12,
    }
    resolved: dict[str, Any] = dict(defaults)

    if args.config is not None:
        config_payload = load_candidate_selection_config(args.config)
        for key, value in config_payload.items():
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
        "require_mean_ic": args.require_mean_ic,
        "require_ic_ir": args.require_ic_ir,
        "max_pairwise_correlation": args.max_pairwise_correlation,
        "min_overlap_observations": args.min_overlap_observations,
        "allocation_method": args.allocation_method,
        "max_weight_per_candidate": args.max_weight_per_candidate,
        "min_allocation_candidate_count": args.min_allocation_candidate_count,
        "min_allocation_weight": args.min_allocation_weight,
        "allocation_weight_sum_tolerance": args.allocation_weight_sum_tolerance,
        "allocation_rounding_decimals": args.allocation_rounding_decimals,
        "register_run": args.register_run,
        "registry_path": args.registry_path,
    }
    for key, value in cli_payload.items():
        if value is not None:
            resolved[key] = value

    if args.disable_allocation:
        resolved["allocation_enabled"] = False

    resolved["artifacts_root"] = Path(resolved.get("artifacts_root") or (Path("artifacts") / "alpha"))
    resolved["output_path"] = Path(resolved.get("output_path") or DEFAULT_ARTIFACTS_ROOT)
    resolved["metric"] = str(resolved.get("metric") or "ic_ir")
    resolved["allocation_method"] = str(resolved.get("allocation_method") or "equal_weight")
    resolved["allocation_enabled"] = bool(resolved.get("allocation_enabled", True))
    resolved["register_run"] = bool(resolved.get("register_run", False))
    return resolved


def run_cli(argv: Sequence[str] | None = None) -> CandidateSelectionRunResult:
    """Execute the candidate selection CLI flow from parsed command-line arguments."""

    args = parse_args(argv)

    resolved = resolve_cli_config(args)
    artifacts_root = Path(resolved["artifacts_root"])
    output_path = Path(resolved["output_path"])

    result = run_candidate_selection(
        artifacts_root=artifacts_root,
        alpha_name=resolved.get("alpha_name"),
        dataset=resolved.get("dataset"),
        timeframe=resolved.get("timeframe"),
        evaluation_horizon=resolved.get("evaluation_horizon"),
        mapping_name=resolved.get("mapping_name"),
        primary_metric=resolved["metric"],
        max_candidate_count=resolved.get("max_candidates"),
        output_artifacts_root=output_path,
        min_mean_ic=resolved.get("min_mean_ic"),
        min_mean_rank_ic=resolved.get("min_mean_rank_ic"),
        min_ic_ir=resolved.get("min_ic_ir"),
        min_rank_ic_ir=resolved.get("min_rank_ic_ir"),
        min_history_length=resolved.get("min_history_length"),
        require_mean_ic=bool(resolved.get("require_mean_ic", False)),
        require_ic_ir=bool(resolved.get("require_ic_ir", False)),
        max_pairwise_correlation=resolved.get("max_pairwise_correlation"),
        min_overlap_observations=resolved.get("min_overlap_observations"),
        allocation_method=resolved.get("allocation_method", "equal_weight"),
        max_weight_per_candidate=resolved.get("max_weight_per_candidate"),
        min_allocation_candidate_count=resolved.get("min_allocation_candidate_count"),
        min_allocation_weight=resolved.get("min_allocation_weight"),
        allocation_weight_sum_tolerance=resolved.get("allocation_weight_sum_tolerance", 1e-12),
        allocation_rounding_decimals=resolved.get("allocation_rounding_decimals", 12),
        allocation_enabled=bool(resolved.get("allocation_enabled", True)),
        register_run=bool(resolved.get("register_run", False)),
        registry_path=resolved.get("registry_path"),
    )

    return CandidateSelectionRunResult(
        run_id=result["run_id"],
        universe_count=result["universe_count"],
        eligible_count=result["eligible_count"],
        rejected_count=result["rejected_count"],
        selected_count=result["selected_count"],
        pruned_by_redundancy=result.get("redundancy_summary", {}).get("pruned_by_redundancy", 0),
        primary_metric=result["primary_metric"],
        filters=result["filters"],
        thresholds=result["thresholds"],
        redundancy_thresholds=result.get("redundancy_thresholds", {}),
        universe_csv=Path(result["universe_csv"]),
        selected_csv=Path(result["selected_csv"]),
        rejected_csv=Path(result["rejected_csv"]),
        eligibility_csv=Path(result["eligibility_csv"]),
        correlation_csv=Path(result["correlation_csv"]),
        allocation_csv=None if result.get("allocation_csv") is None else Path(result["allocation_csv"]),
        summary_json=Path(result["summary_json"]),
        manifest_json=Path(result["manifest_json"]),
        artifact_dir=Path(result["artifact_dir"]),
        allocation_method=result.get("allocation_summary", {}).get("allocation_method"),
        allocation_enabled=bool(result.get("allocation_summary", {}).get("allocation_enabled", False)),
        registry_entry=result.get("registry_entry"),
        allocation_constraints=dict(result.get("allocation_constraints", {})),
        allocation_summary=dict(result.get("allocation_summary", {})),
    )


def print_candidate_selection_summary(result: CandidateSelectionRunResult) -> None:
    """Print a concise CLI summary of candidate selection results."""

    print(f"Candidate Selection Run: {result.run_id}")
    print(f"  Universe count:  {result.universe_count}")
    print(f"  Eligible count:  {result.eligible_count}")
    print(f"  Rejected count:  {result.rejected_count}")
    print(f"  Pruned (redundancy): {result.pruned_by_redundancy}")
    print(f"  Selected count:  {result.selected_count}")
    print(f"  Primary metric:  {result.primary_metric}")
    print()
    print(f"  Universe CSV:    {result.universe_csv}")
    print(f"  Eligibility CSV: {result.eligibility_csv}")
    print(f"  Selected CSV:    {result.selected_csv}")
    print(f"  Rejected CSV:    {result.rejected_csv}")
    print(f"  Correlation CSV: {result.correlation_csv}")
    if result.allocation_csv is not None:
        print(f"  Allocation CSV:  {result.allocation_csv}")
    print(f"  Summary JSON:    {result.summary_json}")
    print(f"  Manifest JSON:   {result.manifest_json}")
    print(f"  Artifact dir:    {result.artifact_dir}")
    if isinstance(result.registry_entry, dict):
        print(f"  Registry entry:  {result.registry_entry.get('run_id')}")
    print()
    if result.filters.get("alpha_name"):
        print(f"  Filter alpha_name: {result.filters['alpha_name']}")
    if result.filters.get("dataset"):
        print(f"  Filter dataset: {result.filters['dataset']}")
    if result.filters.get("timeframe"):
        print(f"  Filter timeframe: {result.filters['timeframe']}")
    if result.filters.get("evaluation_horizon") is not None:
        print(f"  Filter evaluation_horizon: {result.filters['evaluation_horizon']}")
    if result.filters.get("mapping_name"):
        print(f"  Filter mapping_name: {result.filters['mapping_name']}")
    print()
    active_thresholds = {k: v for k, v in result.thresholds.items() if v is not None and v is not False}
    if active_thresholds:
        print("  Active eligibility thresholds:")
        for k, v in sorted(active_thresholds.items()):
            print(f"    {k}: {v}")

    active_redundancy_thresholds = {
        k: v for k, v in result.redundancy_thresholds.items() if v is not None and v is not False
    }
    if active_redundancy_thresholds:
        print("  Active redundancy thresholds:")
        for k, v in sorted(active_redundancy_thresholds.items()):
            print(f"    {k}: {v}")

    print()
    if result.allocation_enabled:
        print(f"  Allocation method: {result.allocation_method}")
        print(f"  Allocated candidates: {result.allocation_summary.get('allocated_candidates')}")
        print(f"  Constraint-adjusted candidates: {result.allocation_summary.get('constraint_adjusted_candidates')}")
        if result.allocation_constraints:
            print("  Allocation constraints:")
            for k, v in sorted(result.allocation_constraints.items()):
                if v is not None:
                    print(f"    {k}: {v}")
    else:
        print("  Allocation governance: disabled")


if __name__ == "__main__":
    result = run_cli()
    print_candidate_selection_summary(result)
