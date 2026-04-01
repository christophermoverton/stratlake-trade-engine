from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.research.registry import load_registry
from src.research.review import compare_research_runs


FIXTURE_ROOT = Path(__file__).resolve().parent / "data" / "milestone_13_review_inputs"
ALPHA_ARTIFACTS_ROOT = FIXTURE_ROOT / "alpha"
STRATEGY_ARTIFACTS_ROOT = FIXTURE_ROOT / "strategies"
PORTFOLIO_ARTIFACTS_ROOT = FIXTURE_ROOT / "portfolios"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "output" / "milestone_13_review_promotion_workflow"
SUMMARY_FILENAME = "summary.json"


@dataclass(frozen=True)
class ExampleArtifacts:
    summary: dict[str, Any]
    leaderboard: pd.DataFrame
    review_summary: dict[str, Any]
    manifest: dict[str, Any]
    promotion_gates: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _drop_output_path(payload: dict[str, Any]) -> dict[str, Any]:
    review_config = payload.get("review_config")
    if not isinstance(review_config, dict):
        return payload
    output = review_config.get("output")
    if not isinstance(output, dict):
        return payload
    normalized_output = dict(output)
    normalized_output.pop("path", None)
    normalized_review_config = dict(review_config)
    normalized_review_config["output"] = normalized_output
    normalized_payload = dict(payload)
    normalized_payload["review_config"] = normalized_review_config
    return normalized_payload


def _normalize_review_artifacts(result: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    review_summary = _drop_output_path(_read_json(result.json_path))
    manifest = _drop_output_path(_read_json(result.manifest_path))
    result.json_path.write_text(json.dumps(review_summary, indent=2, sort_keys=True), encoding="utf-8")
    result.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return review_summary, manifest


def _collect_input_artifacts(*, entries: list[Any]) -> list[dict[str, Any]]:
    roots = {
        "alpha_evaluation": ALPHA_ARTIFACTS_ROOT,
        "strategy": STRATEGY_ARTIFACTS_ROOT,
        "portfolio": PORTFOLIO_ARTIFACTS_ROOT,
    }
    artifacts: list[dict[str, Any]] = []
    for entry in entries:
        root = roots[entry.run_type]
        artifact_dir = root / entry.artifact_path
        manifest = _read_json(artifact_dir / "manifest.json")
        artifacts.append(
            {
                "run_type": entry.run_type,
                "entity_name": entry.entity_name,
                "run_id": entry.run_id,
                "review_status": entry.promotion_status,
                "artifact_path": entry.artifact_path,
                "artifact_files": manifest["artifact_files"],
                "promotion_gate_summary": manifest.get("promotion_gate_summary"),
            }
        )
    return artifacts


def _write_summary(
    output_root: Path,
    *,
    result: Any,
    leaderboard: pd.DataFrame,
    review_summary: dict[str, Any],
    manifest: dict[str, Any],
    promotion_gates: dict[str, Any],
) -> dict[str, Any]:
    alpha_registry = load_registry(ALPHA_ARTIFACTS_ROOT / "registry.jsonl")
    strategy_registry = load_registry(STRATEGY_ARTIFACTS_ROOT / "registry.jsonl")
    portfolio_registry = load_registry(PORTFOLIO_ARTIFACTS_ROOT / "registry.jsonl")

    summary = {
        "workflow": [
            "load_completed_artifacts",
            "collect_registry_entries",
            "rank_latest_runs",
            "write_review_outputs",
            "evaluate_review_promotion_gates",
        ],
        "fixtures": {
            "alpha_registry_rows": len(alpha_registry),
            "strategy_registry_rows": len(strategy_registry),
            "portfolio_registry_rows": len(portfolio_registry),
        },
        "review": {
            "review_id": result.review_id,
            "filters": result.filters,
            "selected_run_ids": [entry.run_id for entry in result.entries],
            "statuses": {
                entry.run_id: entry.promotion_status
                for entry in result.entries
            },
            "leaderboard_rows": int(len(leaderboard)),
            "leaderboard_columns": leaderboard.columns.tolist(),
            "selected_metrics": manifest["selected_metrics"],
        },
        "input_artifacts": _collect_input_artifacts(entries=result.entries),
        "output_artifacts": {
            "artifact_files": manifest["artifact_files"],
            "leaderboard_path": manifest["leaderboard_path"],
            "review_summary_path": manifest["review_summary_path"],
            "promotion_gate_path": "promotion_gates.json",
        },
        "review_promotion": {
            "promotion_status": promotion_gates["promotion_status"],
            "evaluation_status": promotion_gates["evaluation_status"],
            "passed_gate_count": promotion_gates["passed_gate_count"],
            "gate_count": promotion_gates["gate_count"],
            "gate_results": [
                {
                    "gate_id": item["gate_id"],
                    "status": item["status"],
                    "actual_value": item["actual_value"],
                    "threshold": item["threshold"],
                }
                for item in promotion_gates["results"]
            ],
        },
        "review_summary_counts": review_summary["counts_by_run_type"],
    }

    (output_root / SUMMARY_FILENAME).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def print_summary(summary: dict[str, Any], output_root: Path) -> None:
    print("Milestone 13 Review And Promotion Workflow Example")
    print(f"Review id: {summary['review']['review_id']}")
    print(f"Selected runs: {', '.join(summary['review']['selected_run_ids'])}")
    print(f"Review promotion: {summary['review_promotion']['promotion_status']}")
    print()
    print("Input artifacts:")
    print(pd.DataFrame(summary["input_artifacts"]).to_string(index=False))
    print()
    print("Review gate results:")
    print(pd.DataFrame(summary["review_promotion"]["gate_results"]).to_string(index=False))
    print()
    print(f"Output directory: {output_root.as_posix()}")


def run_example(*, output_root: Path | None = None, verbose: bool = True) -> ExampleArtifacts:
    resolved_output_root = DEFAULT_OUTPUT_ROOT if output_root is None else Path(output_root)
    resolved_output_root.mkdir(parents=True, exist_ok=True)

    result = compare_research_runs(
        run_types=["alpha_evaluation", "strategy", "portfolio"],
        top_k_per_type=1,
        strategy_artifacts_root=STRATEGY_ARTIFACTS_ROOT,
        portfolio_artifacts_root=PORTFOLIO_ARTIFACTS_ROOT,
        alpha_artifacts_root=ALPHA_ARTIFACTS_ROOT,
        output_path=resolved_output_root,
        review_config={"output": {"emit_plots": False}},
        promotion_gate_config={
            "status_on_pass": "review_ready",
            "status_on_fail": "needs_work",
            "gates": [
                {
                    "gate_id": "review_has_three_rows",
                    "source": "metrics",
                    "metric_path": "reviewed_entry_count",
                    "comparator": "gte",
                    "threshold": 3,
                },
                {
                    "gate_id": "at_least_one_promoted_candidate",
                    "source": "metrics",
                    "metric_path": "promoted_entry_count",
                    "comparator": "gte",
                    "threshold": 1,
                },
                {
                    "gate_id": "blocked_rows_capped",
                    "source": "metrics",
                    "metric_path": "blocked_entry_count",
                    "comparator": "lte",
                    "threshold": 1,
                },
            ],
        },
    )

    leaderboard = pd.read_csv(result.csv_path)
    review_summary, manifest = _normalize_review_artifacts(result)
    assert result.promotion_gate_path is not None
    promotion_gates = _read_json(result.promotion_gate_path)
    summary = _write_summary(
        resolved_output_root,
        result=result,
        leaderboard=leaderboard,
        review_summary=review_summary,
        manifest=manifest,
        promotion_gates=promotion_gates,
    )

    if verbose:
        print_summary(summary, resolved_output_root)

    return ExampleArtifacts(
        summary=summary,
        leaderboard=leaderboard,
        review_summary=review_summary,
        manifest=manifest,
        promotion_gates=promotion_gates,
    )


def main() -> None:
    run_example()


if __name__ == "__main__":
    main()
