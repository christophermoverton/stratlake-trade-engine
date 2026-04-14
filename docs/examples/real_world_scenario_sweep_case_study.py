from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cli.run_research_campaign import print_summary as print_campaign_summary
from src.cli.run_research_campaign import run_research_campaign
from src.config.research_campaign import resolve_research_campaign_config


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "output" / "real_world_scenario_sweep_case_study"
SUMMARY_FILENAME = "summary.json"


@dataclass(frozen=True)
class ExampleArtifacts:
    summary: dict[str, Any]
    first_pass: dict[str, Any]
    second_pass: dict[str, Any]


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


def _relativize_text(text: str, *, output_root: Path, repo_root: Path) -> str:
    normalized = text.replace("\\", "/")
    output_root_text = output_root.resolve().as_posix()
    repo_root_text = repo_root.resolve().as_posix()
    if normalized == repo_root_text:
        return "."
    if normalized.startswith(output_root_text + "/"):
        return normalized[len(output_root_text) + 1 :]
    if normalized == output_root_text:
        return "."
    if output_root_text in normalized:
        return normalized.replace(output_root_text + "/", "").replace(output_root_text, ".")
    if repo_root_text in normalized:
        return normalized.replace(repo_root_text, ".")
    return normalized


def _relativize_payload(payload: Any, *, output_root: Path, repo_root: Path) -> Any:
    if isinstance(payload, dict):
        return {
            key: _relativize_payload(value, output_root=output_root, repo_root=repo_root)
            for key, value in payload.items()
        }
    if isinstance(payload, list):
        return [_relativize_payload(value, output_root=output_root, repo_root=repo_root) for value in payload]
    if isinstance(payload, str):
        return _relativize_text(payload, output_root=output_root, repo_root=repo_root)
    return payload


def _sanitize_json_outputs(output_root: Path) -> None:
    for path in sorted(output_root.rglob("*.json")):
        payload = _read_json(path)
        sanitized = _relativize_payload(payload, output_root=output_root, repo_root=REPO_ROOT)
        _write_json(path, sanitized)


def _resolved_config(output_root: Path) -> Any:
    artifacts_root = output_root / "artifacts"
    config_payload = {
        "dataset_selection": {
            "dataset": "features_daily",
            "timeframe": "1D",
            "evaluation_horizon": 5,
            "mapping_name": "top_bottom_quantile_q20",
        },
        "targets": {
            "alpha_names": ["ml_cross_sectional_xgb_2026_q1", "ml_cross_sectional_lgbm_2026_q1"],
            "strategy_names": ["momentum_v1"],
            "portfolio_names": ["candidate_portfolio"],
            "alpha_catalog_path": "configs/alphas_2026_q1.yml",
            "strategy_config_path": "configs/strategies.yml",
            "portfolio_config_path": "configs/portfolios.yml",
        },
        "comparison": {
            "enabled": True,
            "alpha_view": "combined",
            "top_k": 5,
        },
        "candidate_selection": {
            "enabled": True,
            "metric": "ic_ir",
            "max_candidates": 2,
            "artifacts_root": (artifacts_root / "alpha").as_posix(),
            "output": {
                "path": (artifacts_root / "candidate_selection").as_posix(),
                "review_output_path": (artifacts_root / "reviews" / "candidate_review").as_posix(),
            },
        },
        "portfolio": {
            "enabled": True,
            "portfolio_name": "candidate_portfolio",
            "timeframe": "1D",
            "from_candidate_selection": True,
        },
        "review": {
            "filters": {
                "run_types": ["alpha_evaluation", "portfolio"],
            },
            "output": {
                "path": (artifacts_root / "reviews" / "research_review").as_posix(),
            },
        },
        "milestone_reporting": {
            "enabled": True,
        },
        "outputs": {
            "alpha_artifacts_root": (artifacts_root / "alpha").as_posix(),
            "candidate_selection_output_path": (artifacts_root / "candidate_selection").as_posix(),
            "campaign_artifacts_root": (artifacts_root / "research_campaigns").as_posix(),
            "portfolio_artifacts_root": (artifacts_root / "portfolios").as_posix(),
            "comparison_output_path": (artifacts_root / "campaign_comparisons").as_posix(),
            "review_output_path": (artifacts_root / "reviews" / "research_review").as_posix(),
        },
        "scenarios": {
            "enabled": True,
            "max_scenarios": 16,
            "max_values_per_axis": 4,
            "matrix": [
                {
                    "name": "timeframe",
                    "path": "dataset_selection.timeframe",
                    "values": ["1D", "4H"],
                },
                {
                    "name": "top_k",
                    "path": "comparison.top_k",
                    "values": [5, 10],
                },
                {
                    "name": "max_candidates",
                    "path": "candidate_selection.max_candidates",
                    "values": [2, 3],
                },
            ],
            "include": [
                {
                    "scenario_id": "review_only_sleeve",
                    "description": "Review-focused pass using sleeve view and alpha-only review.",
                    "overrides": {
                        "comparison": {"alpha_view": "sleeve"},
                        "review": {"filters": {"run_types": ["alpha_evaluation"]}},
                    },
                }
            ],
        },
    }
    return resolve_research_campaign_config(config_payload)


def _parse_flag(argv: list[str], flag: str, *, default: str | None = None) -> str | None:
    if flag not in argv:
        return default
    index = argv.index(flag)
    if index + 1 >= len(argv):
        return default
    return argv[index + 1]


def _float_from_flag(argv: list[str], flag: str, *, default: float) -> float:
    value = _parse_flag(argv, flag)
    if value is None:
        return default
    return float(value)


def _int_from_flag(argv: list[str], flag: str, *, default: int) -> int:
    value = _parse_flag(argv, flag)
    if value is None:
        return default
    return int(value)


def _timeframe_score(timeframe: str) -> float:
    return 0.22 if timeframe.upper() == "4H" else 0.12


def _build_stage_stubs(output_root: Path) -> dict[str, Any]:
    alpha_counter = {"value": 0}

    def write_text(path: Path, contents: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8", newline="\n")
        return path

    def fake_alpha_cli(argv: list[str]) -> SimpleNamespace:
        alpha_name = _parse_flag(argv, "--alpha-name", default="unknown_alpha")
        artifacts_root = Path(_parse_flag(argv, "--artifacts-root", default=(output_root / "artifacts" / "alpha").as_posix()))
        timeframe = (_parse_flag(argv, "--timeframe") or _parse_flag(argv, "--dataset") or "1D")
        alpha_counter["value"] += 1
        run_id = f"{alpha_name}_alpha_eval_sweep_{alpha_counter['value']:04d}"
        artifact_dir = artifacts_root / run_id
        artifact_dir.mkdir(parents=True, exist_ok=True)

        tf_bonus = _timeframe_score(str(timeframe))
        base_ic = 0.35 if alpha_name.endswith("fast") else 0.27
        mean_ic = round(base_ic + tf_bonus, 6)
        ic_ir = round(mean_ic * 4.0, 6)
        sharpe = round(0.8 + mean_ic * 2.2, 6)
        total_return = round(0.02 + mean_ic * 0.06, 6)

        _write_json(
            artifact_dir / "manifest.json",
            {
                "run_id": run_id,
                "alpha_name": alpha_name,
                "artifact_files": ["manifest.json", "sleeve_metrics.json"],
                "sleeve": {
                    "metric_summary": {
                        "sharpe_ratio": sharpe,
                        "total_return": total_return,
                    }
                },
            },
        )
        _write_json(
            artifact_dir / "sleeve_metrics.json",
            {
                "sharpe_ratio": sharpe,
                "total_return": total_return,
            },
        )

        return SimpleNamespace(
            alpha_name=alpha_name,
            run_id=run_id,
            artifact_dir=artifact_dir,
            evaluation=SimpleNamespace(
                evaluation_result=SimpleNamespace(
                    summary={
                        "mean_ic": mean_ic,
                        "ic_ir": ic_ir,
                        "n_periods": 24,
                    }
                ),
                manifest={
                    "sleeve": {
                        "metric_summary": {
                            "sharpe_ratio": sharpe,
                            "total_return": total_return,
                        }
                    }
                },
            ),
        )

    def fake_strategy_cli(argv: list[str]) -> SimpleNamespace:
        strategy_name = _parse_flag(argv, "--strategy", default="unknown_strategy")
        run_id = f"{strategy_name}_strategy_sweep_demo"
        experiment_dir = output_root / "artifacts" / "strategies" / run_id
        experiment_dir.mkdir(parents=True, exist_ok=True)
        metrics = {
            "cumulative_return": 0.09,
            "sharpe_ratio": 0.78,
            "max_drawdown": -0.04,
        }
        return SimpleNamespace(
            strategy_name=strategy_name,
            run_id=run_id,
            experiment_dir=experiment_dir,
            metrics=metrics,
        )

    def fake_alpha_compare(argv: list[str]) -> SimpleNamespace:
        output_path = Path(_parse_flag(argv, "--output-path", default=(output_root / "artifacts" / "campaign_comparisons" / "alpha").as_posix()))
        timeframe = (_parse_flag(argv, "--timeframe", default="1D") or "1D").upper()
        view = _parse_flag(argv, "--view", default="combined") or "combined"
        ic_ir_fast = round(1.35 + _timeframe_score(timeframe) * 3.0 + (0.05 if view == "sleeve" else 0.0), 6)
        ic_ir_stable = round(ic_ir_fast - 0.18, 6)

        csv_path = write_text(
            output_path / "leaderboard.csv",
            "rank,alpha_name,run_id,ic_ir,mean_ic,sharpe_ratio\n"
            f"1,ml_cross_sectional_xgb_2026_q1,ml_cross_sectional_xgb_2026_q1_alpha_eval_sweep,{ic_ir_fast:.6f},0.420000,1.320000\n"
            f"2,ml_cross_sectional_lgbm_2026_q1,ml_cross_sectional_lgbm_2026_q1_alpha_eval_sweep,{ic_ir_stable:.6f},0.350000,1.140000\n",
        )
        json_path = _write_json(
            output_path / "summary.json",
            {
                "comparison_id": f"alpha_comparison_{timeframe.lower()}_{view}",
                "metric": "ic_ir",
                "view": view,
                "row_count": 2,
            },
        )
        return SimpleNamespace(
            comparison_id=f"alpha_comparison_{timeframe.lower()}_{view}",
            csv_path=csv_path,
            json_path=json_path,
            metric="ic_ir",
            sleeve_metric="sharpe_ratio",
            view=view,
        )

    def fake_strategy_compare(argv: list[str]) -> SimpleNamespace:
        output_path = Path(_parse_flag(argv, "--output-path", default=(output_root / "artifacts" / "campaign_comparisons" / "strategy").as_posix()))
        top_k = _int_from_flag(argv, "--top-k", default=5)
        sharpe = round(0.70 + (0.08 if top_k >= 10 else 0.03), 6)
        cumulative_return = round(0.075 + (0.02 if top_k >= 10 else 0.01), 6)

        csv_path = write_text(
            output_path / "leaderboard.csv",
            "rank,strategy_name,run_id,sharpe_ratio,cumulative_return\n"
            f"1,sweep_strategy_momentum,sweep_strategy_momentum_strategy_sweep_demo,{sharpe:.6f},{cumulative_return:.6f}\n",
        )
        json_path = _write_json(
            output_path / "summary.json",
            {
                "comparison_id": f"strategy_comparison_topk_{top_k}",
                "metric": "sharpe_ratio",
                "row_count": 1,
            },
        )
        return SimpleNamespace(
            comparison_id=f"strategy_comparison_topk_{top_k}",
            csv_path=csv_path,
            json_path=json_path,
        )

    def fake_candidate_selection(argv: list[str]) -> SimpleNamespace:
        output_path = Path(_parse_flag(argv, "--output-path", default=(output_root / "artifacts" / "candidate_selection").as_posix()))
        timeframe = (_parse_flag(argv, "--timeframe", default="1D") or "1D").upper()
        max_candidates = _int_from_flag(argv, "--max-candidates", default=2)
        run_id = f"candidate_selection_{timeframe.lower()}_{max_candidates}"
        artifact_dir = output_path / run_id
        artifact_dir.mkdir(parents=True, exist_ok=True)

        selected_count = max_candidates
        first_weight = round(0.55 if selected_count == 2 else 0.40, 6)
        second_weight = round(1.0 - first_weight, 6) if selected_count == 2 else 0.30

        selected_rows = [
            "selection_rank,candidate_id,alpha_name,alpha_run_id,ic_ir,allocation_weight",
            f"1,ml_cross_sectional_xgb_2026_q1_run,ml_cross_sectional_xgb_2026_q1,ml_cross_sectional_xgb_2026_q1_alpha_eval_sweep,1.58,{first_weight:.6f}",
            f"2,ml_cross_sectional_lgbm_2026_q1_run,ml_cross_sectional_lgbm_2026_q1,ml_cross_sectional_lgbm_2026_q1_alpha_eval_sweep,1.32,{second_weight:.6f}",
        ]
        if selected_count >= 3:
            selected_rows.append(
                "3,rank_composite_momentum_2026_q1_run,rank_composite_momentum_2026_q1,rank_composite_momentum_2026_q1_alpha_eval_sweep,1.10,0.300000"
            )
        selected_csv = write_text(artifact_dir / "selected_candidates.csv", "\n".join(selected_rows) + "\n")
        rejected_csv = write_text(
            artifact_dir / "rejected_candidates.csv",
            "candidate_id,alpha_name,rejected_stage,rejection_reason,observed_correlation\n",
        )
        summary_json = _write_json(
            artifact_dir / "selection_summary.json",
            {
                "run_id": run_id,
                "primary_metric": "ic_ir",
                "selected_candidates": selected_count,
                "rejected_candidates": 0,
                "pruned_by_redundancy": 0,
                "timeframe": timeframe,
            },
        )
        manifest_json = _write_json(
            artifact_dir / "manifest.json",
            {
                "run_id": run_id,
                "artifact_files": [
                    "manifest.json",
                    "rejected_candidates.csv",
                    "selected_candidates.csv",
                    "selection_summary.json",
                ],
            },
        )

        return SimpleNamespace(
            run_id=run_id,
            artifact_dir=artifact_dir,
            summary_json=summary_json,
            manifest_json=manifest_json,
            selected_csv=selected_csv,
            rejected_csv=rejected_csv,
            primary_metric="ic_ir",
            universe_count=4,
            eligible_count=4,
            selected_count=selected_count,
            rejected_count=0,
            pruned_by_redundancy=0,
            stage_execution={"universe": True, "eligibility": True, "redundancy": True, "allocation": True},
        )

    def fake_portfolio(argv: list[str]) -> SimpleNamespace:
        timeframe = (_parse_flag(argv, "--timeframe", default="1D") or "1D").upper()
        from_candidate = _parse_flag(argv, "--from-candidate-selection", default="") or ""
        max_candidates = 3 if from_candidate.endswith("_3") else 2

        run_id = f"candidate_portfolio_{timeframe.lower()}_{max_candidates}"
        experiment_dir = output_root / "artifacts" / "portfolios" / run_id
        experiment_dir.mkdir(parents=True, exist_ok=True)

        sharpe = round(1.05 + _timeframe_score(timeframe) + (0.07 if max_candidates == 3 else 0.03), 6)
        total_return = round(0.018 + _timeframe_score(timeframe) * 0.06 + (0.006 if max_candidates == 3 else 0.002), 6)

        _write_json(
            experiment_dir / "manifest.json",
            {
                "run_id": run_id,
                "artifact_files": ["manifest.json", "metrics.json", "portfolio_returns.csv"],
            },
        )
        _write_json(
            experiment_dir / "metrics.json",
            {
                "total_return": total_return,
                "sharpe_ratio": sharpe,
                "max_drawdown": 0.021,
                "realized_volatility": 0.093,
            },
        )
        write_text(
            experiment_dir / "portfolio_returns.csv",
            "ts,portfolio_return,portfolio_equity\n"
            "2026-01-02,0.0000,1.0000\n"
            "2026-01-03,0.0022,1.0022\n"
            "2026-01-04,-0.0005,1.0017\n",
        )

        return SimpleNamespace(
            run_id=run_id,
            experiment_dir=experiment_dir,
            portfolio_name="candidate_portfolio",
            allocator_name="equal_weight",
            component_count=max_candidates,
            metrics={
                "total_return": total_return,
                "sharpe_ratio": sharpe,
                "max_drawdown": 0.021,
                "realized_volatility": 0.093,
            },
        )

    def fake_candidate_review(argv: list[str]) -> SimpleNamespace:
        review_dir = Path(_parse_flag(argv, "--output-path", default=(output_root / "artifacts" / "reviews" / "candidate_review").as_posix()))
        summary_json = _write_json(
            review_dir / "candidate_review_summary.json",
            {
                "candidate_selection_run_id": "candidate_selection_demo",
                "portfolio_run_id": "candidate_portfolio_demo",
                "counts": {"total_candidates": 3, "selected_candidates": 3, "rejected_candidates": 0},
            },
        )
        manifest_json = _write_json(
            review_dir / "manifest.json",
            {
                "run_id": "candidate_review_sweep_demo",
                "artifact_files": ["candidate_review_summary.json", "manifest.json"],
            },
        )
        return SimpleNamespace(
            review_dir=review_dir,
            candidate_review_summary_json=summary_json,
            manifest_json=manifest_json,
            candidate_selection_run_id="candidate_selection_demo",
            portfolio_run_id="candidate_portfolio_demo",
            total_candidates=3,
            selected_candidates=3,
            rejected_candidates=0,
        )

    def fake_review(argv: list[str]) -> SimpleNamespace:
        review_dir = Path(_parse_flag(argv, "--output-path", default=(output_root / "artifacts" / "reviews" / "research_review").as_posix()))
        csv_path = write_text(
            review_dir / "leaderboard.csv",
            "run_type,entity_name,run_id,selected_metric_name,selected_metric_value\n"
            "alpha_evaluation,ml_cross_sectional_xgb_2026_q1,ml_cross_sectional_xgb_2026_q1_alpha_eval_sweep,ic_ir,1.58\n"
            "portfolio,candidate_portfolio,candidate_portfolio_demo,sharpe_ratio,1.42\n",
        )
        json_path = _write_json(review_dir / "summary.json", {"review_id": "review_sweep_demo", "entry_count": 2})
        manifest_path = _write_json(
            review_dir / "manifest.json",
            {
                "review_id": "review_sweep_demo",
                "artifact_files": ["leaderboard.csv", "manifest.json", "summary.json", "promotion_gates.json"],
            },
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
            review_id="review_sweep_demo",
            csv_path=csv_path,
            json_path=json_path,
            manifest_path=manifest_path,
            promotion_gate_path=promotion_gate_path,
            entries=[SimpleNamespace(run_type="alpha_evaluation"), SimpleNamespace(run_type="portfolio")],
        )

    return {
        "src.cli.run_research_campaign.run_alpha_cli.run_cli": fake_alpha_cli,
        "src.cli.run_research_campaign.run_strategy_cli.run_cli": fake_strategy_cli,
        "src.cli.run_research_campaign.compare_alpha_cli.run_cli": fake_alpha_compare,
        "src.cli.run_research_campaign.compare_strategies_cli.run_cli": fake_strategy_compare,
        "src.cli.run_research_campaign.run_candidate_selection_cli.run_cli": fake_candidate_selection,
        "src.cli.run_research_campaign.run_portfolio_cli.run_cli": fake_portfolio,
        "src.cli.run_research_campaign.review_candidate_selection_cli.run_cli": fake_candidate_review,
        "src.cli.run_research_campaign.compare_research_cli.run_cli": fake_review,
    }


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


def _top_leaderboard_entry(orchestration_summary: dict[str, Any]) -> dict[str, Any] | None:
    matrix = orchestration_summary.get("scenario_matrix") or {}
    leaderboard = list(matrix.get("leaderboard") or [])
    return None if not leaderboard else dict(leaderboard[0])


def _scenario_catalog_preview(orchestration_catalog: dict[str, Any], limit: int = 3) -> list[dict[str, Any]]:
    preview: list[dict[str, Any]] = []
    for entry in list(orchestration_catalog.get("scenarios") or [])[:limit]:
        preview.append(
            {
                "scenario_id": entry.get("scenario_id"),
                "source": entry.get("source"),
                "sweep_values": entry.get("sweep_values"),
                "fingerprint": entry.get("fingerprint"),
            }
        )
    return preview


def _scenario_reuse_counts(orchestration_result: Any) -> dict[str, int]:
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


def run_example(*, output_root: Path | None = None, verbose: bool = True, reset_output: bool = True) -> ExampleArtifacts:
    resolved_output_root = DEFAULT_OUTPUT_ROOT if output_root is None else Path(output_root)
    if reset_output and resolved_output_root.exists():
        shutil.rmtree(resolved_output_root)
    resolved_output_root.mkdir(parents=True, exist_ok=True)

    config = _resolved_config(resolved_output_root)
    patchers = _build_stage_stubs(resolved_output_root)

    with ExitStackCompat(patchers):
        first = run_research_campaign(config)
        second = run_research_campaign(config)

    if first.__class__.__name__ != "ResearchCampaignOrchestrationResult":
        raise RuntimeError("Expected a sweep orchestration result for first pass.")
    if second.__class__.__name__ != "ResearchCampaignOrchestrationResult":
        raise RuntimeError("Expected a sweep orchestration result for second pass.")

    first_summary = _read_json(first.orchestration_summary_path)
    first_manifest = _read_json(first.orchestration_manifest_path)
    second_summary = _read_json(second.orchestration_summary_path)
    second_manifest = _read_json(second.orchestration_manifest_path)
    first_catalog = _read_json(first.scenario_catalog_path)
    first_expansion_preflight = _read_json(first.expansion_preflight_path)

    top_entry = _top_leaderboard_entry(first_summary)
    summary_payload = {
        "workflow": [
            "resolve_sweep_enabled_campaign_config",
            "run_sweep_orchestration_first_pass",
            "collect_scenario_catalog_matrix_and_preflight_outputs",
            "rerun_same_orchestration_to_demonstrate_checkpoint_reuse",
            "write_case_study_summary",
        ],
        "config": {
            "dataset": config.dataset_selection.dataset,
            "timeframe": config.dataset_selection.timeframe,
            "evaluation_horizon": config.dataset_selection.evaluation_horizon,
            "alpha_names": list(config.targets.alpha_names),
            "strategy_names": list(config.targets.strategy_names),
            "portfolio_name": config.portfolio.portfolio_name,
            "scenarios": {
                "enabled": config.scenarios.enabled,
                "max_scenarios": config.scenarios.max_scenarios,
                "max_values_per_axis": config.scenarios.max_values_per_axis,
                "matrix_axes": [
                    {
                        "name": axis.name,
                        "path": axis.path,
                        "value_count": len(axis.values),
                        "values": list(axis.values),
                    }
                    for axis in config.scenarios.matrix
                ],
                "include_ids": [scenario.scenario_id for scenario in config.scenarios.include],
            },
        },
        "first_pass": {
            "orchestration_run_id": first.orchestration_run_id,
            "status": first_summary.get("status"),
            "scenario_count": first_summary.get("scenario_count"),
            "scenario_status_counts": first_summary.get("scenario_status_counts"),
            "output_paths": {
                "orchestration_summary": _relative_to_output(first.orchestration_summary_path, resolved_output_root),
                "orchestration_manifest": _relative_to_output(first.orchestration_manifest_path, resolved_output_root),
                "scenario_catalog": _relative_to_output(first.scenario_catalog_path, resolved_output_root),
                "scenario_matrix_csv": _relative_to_output(first.scenario_matrix_csv_path, resolved_output_root),
                "scenario_matrix_summary": _relative_to_output(first.scenario_matrix_summary_path, resolved_output_root),
                "expansion_preflight": _relative_to_output(first.expansion_preflight_path, resolved_output_root),
            },
            "expansion_preflight": first_expansion_preflight,
            "scenario_catalog_preview": _scenario_catalog_preview(first_catalog, limit=4),
            "scenario_matrix": {
                "selection_rule": (first_summary.get("scenario_matrix") or {}).get("selection_rule"),
                "metric_priority": (first_summary.get("scenario_matrix") or {}).get("metric_priority"),
                "row_count": (first_summary.get("scenario_matrix") or {}).get("row_count"),
                "leaderboard_top": top_entry,
            },
            "trace_top_scenario": {
                "scenario_id": None if top_entry is None else top_entry.get("scenario_id"),
                "campaign_summary_path": None if top_entry is None else top_entry.get("campaign_summary_path"),
                "catalog_match": None
                if top_entry is None
                else next(
                    (
                        entry
                        for entry in list(first_catalog.get("scenarios") or [])
                        if entry.get("scenario_id") == top_entry.get("scenario_id")
                    ),
                    None,
                ),
            },
        },
        "second_pass": {
            "orchestration_run_id": second.orchestration_run_id,
            "status": second_summary.get("status"),
            "scenario_count": second_summary.get("scenario_count"),
            "scenario_status_counts": second_summary.get("scenario_status_counts"),
            "scenario_preflight_state_counts": _scenario_reuse_counts(second),
            "output_paths": {
                "orchestration_summary": _relative_to_output(second.orchestration_summary_path, resolved_output_root),
                "orchestration_manifest": _relative_to_output(second.orchestration_manifest_path, resolved_output_root),
            },
        },
        "artifact_contract": {
            "first_pass_manifest_artifact_files": list(first_manifest.get("artifact_files") or []),
            "second_pass_manifest_artifact_files": list(second_manifest.get("artifact_files") or []),
        },
        "artifacts": {
            "output_root": resolved_output_root.as_posix(),
            "summary": SUMMARY_FILENAME,
        },
    }

    _write_json(resolved_output_root / SUMMARY_FILENAME, summary_payload)
    _sanitize_json_outputs(resolved_output_root)

    summary_payload = _read_json(resolved_output_root / SUMMARY_FILENAME)
    first_summary = _read_json(first.orchestration_summary_path)
    second_summary = _read_json(second.orchestration_summary_path)

    if verbose:
        print("First pass")
        print_campaign_summary(first)
        print()
        print("Second pass")
        print_campaign_summary(second)
        print()
        print("Real-World Scenario Sweep Case Study")
        print(f"Orchestration run id: {summary_payload['first_pass']['orchestration_run_id']}")
        print(f"Scenario count: {summary_payload['first_pass']['scenario_count']}")
        top = summary_payload["first_pass"]["scenario_matrix"]["leaderboard_top"]
        if isinstance(top, dict):
            print(
                "Top scenario: "
                f"{top.get('scenario_id')} | "
                f"metric={top.get('ranking_metric')} | "
                f"value={top.get('ranking_value')}"
            )
        print(
            "Second-pass preflight state counts: "
            f"{summary_payload['second_pass']['scenario_preflight_state_counts']}"
        )
        print(f"Output directory: {resolved_output_root.as_posix()}")

    return ExampleArtifacts(
        summary=summary_payload,
        first_pass=first_summary,
        second_pass=second_summary,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Milestone 19 scenario sweep case study.")
    parser.add_argument(
        "--output-root",
        help="Optional output root override. Defaults to docs/examples/output/real_world_scenario_sweep_case_study.",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Keep existing outputs instead of deleting the output root first.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_root = None if args.output_root is None else Path(args.output_root)
    run_example(output_root=output_root, reset_output=not args.no_reset)


if __name__ == "__main__":
    main()
