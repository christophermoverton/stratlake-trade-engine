from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from typing import Any, Iterator
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cli.run_research_campaign import print_summary as print_campaign_summary
from src.cli.run_research_campaign import run_research_campaign
from src.config.research_campaign import load_research_campaign_config, resolve_research_campaign_config


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "output" / "real_world_resume_workflow_case_study"
SUMMARY_FILENAME = "summary.json"
CONFIG_PATH = (
    REPO_ROOT / "docs" / "examples" / "data" / "milestone_17_campaign_configs" / "resume_reuse_campaign.yml"
)


@dataclass(frozen=True)
class ExampleArtifacts:
    summary: dict[str, Any]
    partial_summary: dict[str, Any]
    resumed_summary: dict[str, Any]
    stable_summary: dict[str, Any]


@contextmanager
def example_environment(features_root: Path) -> Iterator[None]:
    previous_features_root = os.environ.get("FEATURES_ROOT")
    previous_cwd = Path.cwd()
    os.environ["FEATURES_ROOT"] = str(features_root)
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


def _write_feature_fixture(features_root: Path) -> None:
    dataset_root = features_root / "curated" / "features_daily" / "symbol=AAA" / "year=2025"
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "part-0.parquet").write_text("fixture", encoding="utf-8")


def _resolved_config(output_root: Path) -> Any:
    base = load_research_campaign_config(CONFIG_PATH).to_dict()
    artifacts_root = output_root / "artifacts"
    base["candidate_selection"]["output"]["path"] = (artifacts_root / "candidate_selection").as_posix()
    base["candidate_selection"]["output"]["review_output_path"] = (
        artifacts_root / "reviews" / "candidate_review"
    ).as_posix()
    base["review"]["output"]["path"] = (artifacts_root / "reviews" / "research_review").as_posix()
    base["outputs"]["alpha_artifacts_root"] = (artifacts_root / "alpha").as_posix()
    base["outputs"]["candidate_selection_output_path"] = (artifacts_root / "candidate_selection").as_posix()
    base["outputs"]["comparison_output_path"] = (artifacts_root / "campaign_comparisons").as_posix()
    base["outputs"]["campaign_artifacts_root"] = (artifacts_root / "research_campaigns").as_posix()
    base["outputs"]["portfolio_artifacts_root"] = (artifacts_root / "portfolios").as_posix()
    base["outputs"]["review_output_path"] = (artifacts_root / "reviews" / "research_review").as_posix()
    return resolve_research_campaign_config(base)


def _campaign_dir(output_root: Path) -> Path:
    campaign_root = output_root / "artifacts" / "research_campaigns"
    directories = sorted(path for path in campaign_root.iterdir() if path.is_dir())
    if len(directories) != 1:
        raise RuntimeError(f"Expected exactly one campaign directory under {campaign_root.as_posix()}.")
    return directories[0]


def _stage_summary_excerpt(summary: dict[str, Any], manifest: dict[str, Any], stage_name: str) -> dict[str, Any]:
    stage = next(stage for stage in summary["stages"] if stage["stage_name"] == stage_name)
    return {
        "campaign_run_id": summary["campaign_run_id"],
        "status": summary["status"],
        "stage_statuses": summary["stage_statuses"],
        "retry_stage_names": summary["final_outcomes"]["retry_stage_names"],
        "partial_stage_names": summary["final_outcomes"]["partial_stage_names"],
        "reused_stage_names": summary["final_outcomes"]["reused_stage_names"],
        "resumable_stage_names": summary["final_outcomes"]["resumable_stage_names"],
        "stage_state_counts": summary["final_outcomes"]["stage_state_counts"],
        "failure_summaries": summary["final_outcomes"]["failures"],
        "stage": {
            "stage_name": stage_name,
            "state": stage["state"],
            "details": stage["details"],
            "execution_metadata": stage["execution_metadata"],
        },
        "manifest": {
            "status": manifest["status"],
            "retry_stage_names": manifest["retry_stage_names"],
            "partial_stage_names": manifest["partial_stage_names"],
            "reused_stage_names": manifest["reused_stage_names"],
            "resumable_stage_names": manifest["resumable_stage_names"],
            "stage_execution": manifest["stage_execution"][stage_name],
            "stage_status": manifest["stage_statuses"][stage_name],
        },
    }


def _print_snapshot(label: str, summary: dict[str, Any], output_root: Path) -> None:
    stage_statuses = summary["stage_statuses"]
    state_counts = summary["final_outcomes"]["stage_state_counts"]
    print(label)
    print("Research Campaign Summary")
    print("-------------------------")
    print(
        f"Campaign: {summary['campaign_run_id']} | "
        f"status={summary['status']} | "
        f"dir={(output_root / 'artifacts' / 'research_campaigns' / summary['campaign_run_id']).as_posix()}"
    )
    print(
        "Stage States: "
        + " | ".join(
            f"{state}={count}"
            for state, count in state_counts.items()
            if count
        )
    )
    print(
        "Stage Details: "
        + " | ".join(f"{stage_name}={stage_statuses[stage_name]}" for stage_name in stage_statuses)
    )
    print()


def _relative_to_output(path: Path, output_root: Path) -> str:
    try:
        return path.relative_to(output_root).as_posix()
    except ValueError:
        return path.as_posix()


def _build_stage_stubs(output_root: Path) -> tuple[dict[str, int], dict[str, Any]]:
    attempts = {"comparison": 0}
    alpha_metrics = {
        "resume_alpha_quality": {"mean_ic": 0.21, "ic_ir": 1.48, "n_periods": 18, "sharpe_ratio": 1.61, "total_return": 0.031},
        "resume_alpha_diversified": {"mean_ic": 0.13, "ic_ir": 0.94, "n_periods": 18, "sharpe_ratio": 0.88, "total_return": 0.018},
    }

    def write_text(path: Path, contents: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
        return path

    def fake_alpha_cli(argv: list[str]) -> SimpleNamespace:
        alpha_name = argv[argv.index("--alpha-name") + 1]
        artifact_dir = output_root / "artifacts" / "alpha" / f"{alpha_name}_alpha_eval_resume_demo"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            artifact_dir / "manifest.json",
            {
                "run_id": f"{alpha_name}_alpha_eval_resume_demo",
                "alpha_name": alpha_name,
                "artifact_files": ["manifest.json", "sleeve_metrics.json"],
                "sleeve": {"metric_summary": {"sharpe_ratio": alpha_metrics[alpha_name]["sharpe_ratio"], "total_return": alpha_metrics[alpha_name]["total_return"]}},
            },
        )
        _write_json(artifact_dir / "sleeve_metrics.json", {"sharpe_ratio": alpha_metrics[alpha_name]["sharpe_ratio"], "total_return": alpha_metrics[alpha_name]["total_return"]})
        return SimpleNamespace(
            alpha_name=alpha_name,
            run_id=f"{alpha_name}_alpha_eval_resume_demo",
            artifact_dir=artifact_dir,
            evaluation=SimpleNamespace(
                evaluation_result=SimpleNamespace(summary=alpha_metrics[alpha_name]),
                manifest={
                    "sleeve": {
                        "metric_summary": {
                            "sharpe_ratio": alpha_metrics[alpha_name]["sharpe_ratio"],
                            "total_return": alpha_metrics[alpha_name]["total_return"],
                        }
                    }
                },
            ),
        )

    def fake_strategy_cli(argv: list[str]) -> SimpleNamespace:
        strategy_name = argv[argv.index("--strategy") + 1]
        experiment_dir = output_root / "artifacts" / "strategies" / f"{strategy_name}_strategy_resume_demo"
        experiment_dir.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(
            strategy_name=strategy_name,
            run_id=f"{strategy_name}_strategy_resume_demo",
            experiment_dir=experiment_dir,
            metrics={"cumulative_return": 0.11, "sharpe_ratio": 0.93, "max_drawdown": -0.04},
        )

    def fake_alpha_compare(argv: list[str]) -> SimpleNamespace:
        attempts["comparison"] += 1
        if attempts["comparison"] == 1:
            raise KeyboardInterrupt("operator requested checkpoint stop after research")
        output_root_path = Path(argv[argv.index("--output-path") + 1])
        csv_path = write_text(
            output_root_path / "alpha" / "leaderboard.csv",
            "rank,alpha_name,run_id,ic_ir,mean_ic,sharpe_ratio\n"
            "1,resume_alpha_quality,resume_alpha_quality_alpha_eval_resume_demo,1.48,0.21,1.61\n"
            "2,resume_alpha_diversified,resume_alpha_diversified_alpha_eval_resume_demo,0.94,0.13,0.88\n",
        )
        json_path = _write_json(
            output_root_path / "alpha" / "summary.json",
            {
                "comparison_id": "alpha_resume_comparison_demo",
                "metric": "ic_ir+sharpe_ratio",
                "view": "combined",
                "row_count": 2,
            },
        )
        return SimpleNamespace(
            comparison_id="alpha_resume_comparison_demo",
            csv_path=csv_path,
            json_path=json_path,
            metric="ic_ir+sharpe_ratio",
            sleeve_metric="sharpe_ratio",
            view="combined",
        )

    def fake_strategy_compare(argv: list[str]) -> SimpleNamespace:
        output_root_path = Path(argv[argv.index("--output-path") + 1])
        csv_path = write_text(
            output_root_path / "strategy" / "leaderboard.csv",
            "rank,strategy_name,run_id,sharpe_ratio\n1,resume_strategy_momentum,resume_strategy_momentum_strategy_resume_demo,0.93\n",
        )
        json_path = _write_json(
            output_root_path / "strategy" / "summary.json",
            {"comparison_id": "strategy_resume_comparison_demo", "metric": "sharpe_ratio", "row_count": 1},
        )
        return SimpleNamespace(comparison_id="strategy_resume_comparison_demo", csv_path=csv_path, json_path=json_path)

    def fake_candidate_selection(argv: list[str]) -> SimpleNamespace:
        artifact_dir = output_root / "artifacts" / "candidate_selection" / "candidate_selection_resume_demo"
        selected_csv = write_text(
            artifact_dir / "selected_candidates.csv",
            "selection_rank,candidate_id,alpha_name,alpha_run_id,ic_ir,allocation_weight\n"
            "1,resume_alpha_quality_alpha_eval_resume_demo,resume_alpha_quality,resume_alpha_quality_alpha_eval_resume_demo,1.48,0.6\n"
            "2,resume_alpha_diversified_alpha_eval_resume_demo,resume_alpha_diversified,resume_alpha_diversified_alpha_eval_resume_demo,0.94,0.4\n",
        )
        rejected_csv = write_text(
            artifact_dir / "rejected_candidates.csv",
            "candidate_id,alpha_name,rejected_stage,rejection_reason,observed_correlation\n",
        )
        summary_json = _write_json(
            artifact_dir / "selection_summary.json",
            {
                "run_id": "candidate_selection_resume_demo",
                "primary_metric": "ic_ir",
                "selected_candidates": 2,
                "rejected_candidates": 0,
                "pruned_by_redundancy": 0,
                "stage_execution": {"universe": True, "eligibility": True, "redundancy": True, "allocation": True},
            },
        )
        manifest_json = _write_json(
            artifact_dir / "manifest.json",
            {
                "run_id": "candidate_selection_resume_demo",
                "artifact_files": ["manifest.json", "rejected_candidates.csv", "selected_candidates.csv", "selection_summary.json"],
            },
        )
        return SimpleNamespace(
            run_id="candidate_selection_resume_demo",
            artifact_dir=artifact_dir,
            summary_json=summary_json,
            manifest_json=manifest_json,
            selected_csv=selected_csv,
            rejected_csv=rejected_csv,
            primary_metric="ic_ir",
            universe_count=2,
            eligible_count=2,
            selected_count=2,
            rejected_count=0,
            pruned_by_redundancy=0,
            stage_execution={"universe": True, "eligibility": True, "redundancy": True, "allocation": True},
        )

    def fake_portfolio(argv: list[str]) -> SimpleNamespace:
        experiment_dir = output_root / "artifacts" / "portfolios" / "resume_candidate_portfolio_portfolio_resume_demo"
        experiment_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            experiment_dir / "manifest.json",
            {
                "run_id": "resume_candidate_portfolio_portfolio_resume_demo",
                "artifact_files": ["manifest.json", "metrics.json", "portfolio_returns.csv"],
            },
        )
        _write_json(
            experiment_dir / "metrics.json",
            {"total_return": 0.024, "sharpe_ratio": 1.18, "max_drawdown": 0.012, "realized_volatility": 0.084},
        )
        write_text(
            experiment_dir / "portfolio_returns.csv",
            "ts,portfolio_return,portfolio_equity\n2025-02-03,0.0000,1.0000\n2025-02-04,0.0030,1.0030\n2025-02-05,-0.0010,1.0020\n",
        )
        return SimpleNamespace(
            run_id="resume_candidate_portfolio_portfolio_resume_demo",
            experiment_dir=experiment_dir,
            portfolio_name="resume_candidate_portfolio",
            allocator_name="equal_weight",
            component_count=2,
            metrics={"total_return": 0.024, "sharpe_ratio": 1.18, "max_drawdown": 0.012, "realized_volatility": 0.084},
        )

    def fake_candidate_review(argv: list[str]) -> SimpleNamespace:
        review_dir = output_root / "artifacts" / "reviews" / "candidate_review"
        summary_json = _write_json(
            review_dir / "candidate_review_summary.json",
            {
                "candidate_selection_run_id": "candidate_selection_resume_demo",
                "portfolio_run_id": "resume_candidate_portfolio_portfolio_resume_demo",
                "counts": {"total_candidates": 2, "selected_candidates": 2, "rejected_candidates": 0},
            },
        )
        manifest_json = _write_json(
            review_dir / "manifest.json",
            {
                "run_id": "candidate_review_resume_demo",
                "artifact_files": ["candidate_review_summary.json", "manifest.json"],
            },
        )
        return SimpleNamespace(
            review_dir=review_dir,
            candidate_review_summary_json=summary_json,
            manifest_json=manifest_json,
            candidate_selection_run_id="candidate_selection_resume_demo",
            portfolio_run_id="resume_candidate_portfolio_portfolio_resume_demo",
            total_candidates=2,
            selected_candidates=2,
            rejected_candidates=0,
        )

    def fake_review(argv: list[str]) -> SimpleNamespace:
        review_dir = output_root / "artifacts" / "reviews" / "research_review"
        csv_path = write_text(
            review_dir / "leaderboard.csv",
            "run_type,entity_name,run_id,selected_metric_name,selected_metric_value\n"
            "alpha_evaluation,resume_alpha_quality,resume_alpha_quality_alpha_eval_resume_demo,ic_ir,1.48\n"
            "portfolio,resume_candidate_portfolio,resume_candidate_portfolio_portfolio_resume_demo,sharpe_ratio,1.18\n",
        )
        json_path = _write_json(review_dir / "summary.json", {"review_id": "resume_review_demo", "entry_count": 2})
        manifest_path = _write_json(
            review_dir / "manifest.json",
            {"review_id": "resume_review_demo", "artifact_files": ["leaderboard.csv", "manifest.json", "summary.json", "promotion_gates.json"]},
        )
        promotion_gate_path = _write_json(
            review_dir / "promotion_gates.json",
            {
                "evaluation_status": "passed",
                "promotion_status": "approved",
                "gate_count": 2,
                "passed_gate_count": 2,
                "failed_gate_count": 0,
                "missing_gate_count": 0,
            },
        )
        return SimpleNamespace(
            review_id="resume_review_demo",
            csv_path=csv_path,
            json_path=json_path,
            manifest_path=manifest_path,
            promotion_gate_path=promotion_gate_path,
            entries=[SimpleNamespace(run_type="alpha_evaluation"), SimpleNamespace(run_type="portfolio")],
        )

    patches = {
        "src.cli.run_research_campaign.run_alpha_cli.run_cli": fake_alpha_cli,
        "src.cli.run_research_campaign.run_strategy_cli.run_cli": fake_strategy_cli,
        "src.cli.run_research_campaign.compare_alpha_cli.run_cli": fake_alpha_compare,
        "src.cli.run_research_campaign.compare_strategies_cli.run_cli": fake_strategy_compare,
        "src.cli.run_research_campaign.run_candidate_selection_cli.run_cli": fake_candidate_selection,
        "src.cli.run_research_campaign.run_portfolio_cli.run_cli": fake_portfolio,
        "src.cli.run_research_campaign.review_candidate_selection_cli.run_cli": fake_candidate_review,
        "src.cli.run_research_campaign.compare_research_cli.run_cli": fake_review,
    }
    return attempts, patches


def _write_snapshots(output_root: Path, prefix: str, *, summary: dict[str, Any], manifest: dict[str, Any], checkpoint: dict[str, Any]) -> dict[str, str]:
    snapshot_root = output_root / "snapshots"
    return {
        "summary": _relative_to_output(_write_json(snapshot_root / f"{prefix}_summary.json", summary), output_root),
        "manifest": _relative_to_output(_write_json(snapshot_root / f"{prefix}_manifest.json", manifest), output_root),
        "checkpoint": _relative_to_output(_write_json(snapshot_root / f"{prefix}_checkpoint.json", checkpoint), output_root),
    }


def run_example(*, output_root: Path | None = None, verbose: bool = True, reset_output: bool = True) -> ExampleArtifacts:
    resolved_output_root = DEFAULT_OUTPUT_ROOT if output_root is None else Path(output_root)
    if reset_output and resolved_output_root.exists():
        shutil.rmtree(resolved_output_root)
    resolved_output_root.mkdir(parents=True, exist_ok=True)

    features_root = resolved_output_root / "workspace"
    _write_feature_fixture(features_root)
    config = _resolved_config(resolved_output_root)
    attempts, patchers = _build_stage_stubs(resolved_output_root)

    with example_environment(features_root):
        with ExitStackCompat(patchers):
            try:
                run_research_campaign(config)
            except KeyboardInterrupt:
                pass
            else:
                raise RuntimeError("Expected the first pass to be interrupted during comparison.")

            campaign_dir = _campaign_dir(resolved_output_root)
            partial_summary = _read_json(campaign_dir / "summary.json")
            partial_manifest = _read_json(campaign_dir / "manifest.json")
            partial_checkpoint = _read_json(campaign_dir / "checkpoint.json")

            resumed_result = run_research_campaign(config)
            resumed_summary = _read_json(resumed_result.campaign_summary_path)
            resumed_manifest = _read_json(resumed_result.campaign_manifest_path)
            resumed_checkpoint = _read_json(resumed_result.campaign_checkpoint_path)

            stable_result = run_research_campaign(config)
            stable_summary = _read_json(stable_result.campaign_summary_path)
            stable_manifest = _read_json(stable_result.campaign_manifest_path)
            stable_checkpoint = _read_json(stable_result.campaign_checkpoint_path)

    snapshots = {
        "partial": _write_snapshots(
            resolved_output_root,
            "partial",
            summary=partial_summary,
            manifest=partial_manifest,
            checkpoint=partial_checkpoint,
        ),
        "resumed": _write_snapshots(
            resolved_output_root,
            "resumed",
            summary=resumed_summary,
            manifest=resumed_manifest,
            checkpoint=resumed_checkpoint,
        ),
        "stable": _write_snapshots(
            resolved_output_root,
            "stable",
            summary=stable_summary,
            manifest=stable_manifest,
            checkpoint=stable_checkpoint,
        ),
    }

    summary = {
        "workflow": [
            "run_campaign_until_partial_comparison_interrupt",
            "resume_campaign_from_checkpoint",
            "rerun_campaign_to_confirm_full_stage_reuse",
            "persist_campaign_summary_manifest_checkpoint_snapshots",
        ],
        "config": {
            "source_config": CONFIG_PATH.relative_to(REPO_ROOT).as_posix(),
            "dataset": config.dataset_selection.dataset,
            "timeframe": config.dataset_selection.timeframe,
            "evaluation_horizon": config.dataset_selection.evaluation_horizon,
            "mapping_name": config.dataset_selection.mapping_name,
            "alpha_names": list(config.targets.alpha_names),
            "strategy_names": list(config.targets.strategy_names),
            "portfolio_name": config.portfolio.portfolio_name,
        },
        "campaign": {
            "campaign_run_id": stable_summary["campaign_run_id"],
            "same_campaign_run_id_across_passes": (
                partial_summary["campaign_run_id"] == resumed_summary["campaign_run_id"] == stable_summary["campaign_run_id"]
            ),
            "summary_path": _relative_to_output(stable_result.campaign_summary_path, resolved_output_root),
            "manifest_path": _relative_to_output(stable_result.campaign_manifest_path, resolved_output_root),
            "checkpoint_path": _relative_to_output(stable_result.campaign_checkpoint_path, resolved_output_root),
        },
        "resume_workflow": {
            "interrupt_stage": "comparison",
            "attempt_counts": dict(attempts),
            "partial_run": _stage_summary_excerpt(partial_summary, partial_manifest, "comparison"),
            "resumed_run": _stage_summary_excerpt(resumed_summary, resumed_manifest, "comparison"),
            "stable_run": _stage_summary_excerpt(stable_summary, stable_manifest, "comparison"),
        },
        "artifacts": {
            "output_root": resolved_output_root.as_posix(),
            "campaign_dir": _relative_to_output(stable_result.campaign_artifact_dir, resolved_output_root),
            "snapshots": snapshots,
            "summary": SUMMARY_FILENAME,
        },
    }
    _write_json(resolved_output_root / SUMMARY_FILENAME, summary)

    if verbose:
        _print_snapshot("Interrupted pass", partial_summary, resolved_output_root)
        print("Resumed pass")
        print_campaign_summary(resumed_result)
        print()
        print("Stable reuse pass")
        print_campaign_summary(stable_result)
        print()
        print("Real-World Resume Workflow Case Study")
        print(f"Campaign: {summary['campaign']['campaign_run_id']}")
        print(
            "Comparison stage states: "
            f"partial={summary['resume_workflow']['partial_run']['stage']['state']}, "
            f"resumed={summary['resume_workflow']['resumed_run']['stage']['state']}, "
            f"stable={summary['resume_workflow']['stable_run']['stage']['state']}"
        )
        print(
            "Retry stage names: "
            f"partial={summary['resume_workflow']['partial_run']['retry_stage_names']}, "
            f"resumed={summary['resume_workflow']['resumed_run']['retry_stage_names']}, "
            f"stable={summary['resume_workflow']['stable_run']['retry_stage_names']}"
        )
        print(
            "Reused stage names (stable): "
            f"{summary['resume_workflow']['stable_run']['reused_stage_names']}"
        )
        print(f"Output directory: {resolved_output_root.as_posix()}")

    return ExampleArtifacts(
        summary=summary,
        partial_summary=partial_summary,
        resumed_summary=resumed_summary,
        stable_summary=stable_summary,
    )


class ExitStackCompat:
    def __init__(self, patchers: dict[str, Any]) -> None:
        self._patchers = [patch(target, replacement) for target, replacement in patchers.items()]

    def __enter__(self) -> "ExitStackCompat":
        for item in self._patchers:
            item.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for item in reversed(self._patchers):
            item.stop()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Milestone 17 resume workflow case study.")
    parser.add_argument(
        "--output-root",
        help="Optional output root override. Defaults to docs/examples/output/real_world_resume_workflow_case_study.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_root = None if args.output_root is None else Path(args.output_root)
    run_example(output_root=output_root)


if __name__ == "__main__":
    main()
