from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.execution.regime_aware_candidate_selection import run_regime_aware_candidate_selection  # noqa: E402
from src.execution.regime_benchmark import run_regime_benchmark_pack  # noqa: E402
from src.execution.regime_policy_stress_tests import run_regime_policy_stress_tests  # noqa: E402
from src.execution.regime_promotion_gates import run_regime_promotion_gates  # noqa: E402
from src.execution.regime_review_pack import generate_regime_review_pack  # noqa: E402


CASE_STUDY_NAME = "full_year_regime_policy_benchmark_case_study"
CASE_STUDY_MODE = "fixture_backed"
SUMMARY_FILENAME = "summary.json"
MANIFEST_FILENAME = "manifest.json"
FINAL_INTERPRETATION_FILENAME = "final_interpretation.md"
POLICY_VARIANT_COMPARISON_FILENAME = "policy_variant_comparison.csv"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "output" / CASE_STUDY_NAME

EVALUATION_WINDOW = {
    "start": "2026-01-01",
    "end": "2026-12-31",
    "timeframe": "1D",
}

CONCEPTUAL_POLICY_ROLES = (
    "static_baseline",
    "taxonomy_only_policy",
    "calibrated_taxonomy_policy",
    "gmm_confidence_policy",
    "hybrid_calibrated_gmm_policy",
)

ROLE_FROM_VARIANT_TYPE = {
    "static_baseline": "static_baseline",
    "taxonomy_only": "taxonomy_only_policy",
    "calibrated_taxonomy": "calibrated_taxonomy_policy",
    "gmm_classifier": "gmm_confidence_policy",
    "policy_optimized": "hybrid_calibrated_gmm_policy",
}

STRESS_POLICY_BY_ROLE = {
    "static_baseline": "static_baseline",
    "taxonomy_only_policy": None,
    "calibrated_taxonomy_policy": None,
    "gmm_confidence_policy": "gmm_calibrated_overlay",
    "hybrid_calibrated_gmm_policy": "policy_optimized",
}

M26_PROMOTION_GATES_CONFIG = REPO_ROOT / "configs" / "regime_promotion_gates" / "m26_regime_policy_gates.yml"
M26_CANDIDATE_SELECTION_CONFIG = REPO_ROOT / "configs" / "candidate_selection" / "m26_regime_aware_candidate_selection.yml"
M26_CANDIDATE_METRICS_FIXTURE = (
    REPO_ROOT / "configs" / "candidate_selection" / "fixtures" / "regime_candidate_metrics.csv"
)
M26_STRESS_CONFIG = REPO_ROOT / "configs" / "regime_stress_tests" / "m26_adaptive_policy_stress.yml"
M26_STRESS_POLICY_FIXTURE = REPO_ROOT / "configs" / "regime_stress_tests" / "fixtures" / "policy_candidates.csv"
M25_POLICY_CONFIG = REPO_ROOT / "configs" / "regime_policies" / "m25_policy_default.yml"


@dataclass(frozen=True)
class CaseStudyArtifacts:
    output_root: Path
    summary_path: Path
    manifest_path: Path
    final_interpretation_path: Path
    summary: dict[str, Any]


def _normalize_json_value(value: Any) -> Any:
    if value is None or isinstance(value, bool | str | int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {key: _normalize_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_json_value(item) for item in value]
    if pd.isna(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_normalize_json_value(payload), indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")


def _relative_to_output(path: Path, output_root: Path) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(output_root.resolve())
    except ValueError:
        return resolved.as_posix()
    if str(relative) == ".":
        return "."
    return relative.as_posix()


def _relative_to_repo(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _portable_path(path: Path, *, output_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(output_root.resolve()).as_posix()
    except ValueError:
        pass
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _relative_config_path(path: Path, *, output_root: Path) -> str:
    return _portable_path(path, output_root=output_root)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path.as_posix()}.")
    return payload


def _write_feature_fixture(features_root: Path) -> None:
    dataset_root = features_root / "curated" / "features_daily"
    timestamps = pd.date_range(
        EVALUATION_WINDOW["start"],
        EVALUATION_WINDOW["end"],
        freq="D",
        tz="UTC",
    )
    symbols = ("AAA", "BBB", "CCC")

    for symbol_index, symbol in enumerate(symbols):
        rows: list[dict[str, Any]] = []
        for index, ts_utc in enumerate(timestamps):
            drift = index * (0.07 + (symbol_index * 0.013))
            seasonal = ((index % 28) - 14) * (0.045 + (symbol_index * 0.005))
            close = max(1.0, 100.0 + (symbol_index * 8.0) + drift + seasonal)
            rows.append(
                {
                    "symbol": symbol,
                    "ts_utc": ts_utc,
                    "timeframe": "1D",
                    "date": ts_utc.strftime("%Y-%m-%d"),
                    "close": close,
                }
            )
        frame = pd.DataFrame(rows)
        output_dir = dataset_root / f"symbol={symbol}" / "year=2026"
        output_dir.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(output_dir / "part-0.parquet", index=False)


def _write_benchmark_fixture_config(inputs_root: Path, benchmark_output_root: Path) -> Path:
    inputs_root.mkdir(parents=True, exist_ok=True)
    fixture_features_root = inputs_root / "fixture_features"
    _write_feature_fixture(fixture_features_root)

    policy_copy = inputs_root / "m25_policy_default.yml"
    policy_copy.write_text(M25_POLICY_CONFIG.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")

    config = {
        "benchmark_name": "m26_full_year_regime_policy_fixture",
        "dataset": "features_daily",
        "timeframe": EVALUATION_WINDOW["timeframe"],
        "start": EVALUATION_WINDOW["start"],
        "end": EVALUATION_WINDOW["end"],
        "features_root": fixture_features_root.as_posix(),
        "output_root": benchmark_output_root.as_posix(),
        "variants": [
            {"name": "static_baseline", "regime_source": "static"},
            {"name": "taxonomy_only_policy", "regime_source": "taxonomy"},
            {
                "name": "calibrated_taxonomy_policy",
                "regime_source": "taxonomy",
                "calibration_profile": "baseline",
            },
            {"name": "gmm_confidence_policy", "regime_source": "gmm"},
            {
                "name": "hybrid_calibrated_gmm_policy",
                "regime_source": "policy",
                "calibration_profile": "baseline",
                "policy_config": policy_copy.name,
            },
        ],
        "metrics": {
            "include_transition_metrics": True,
            "include_stability_metrics": True,
            "include_policy_metrics": True,
            "include_conditional_performance": True,
            "high_entropy_threshold": 0.75,
        },
        "gmm": {
            "n_components": 3,
            "random_state": 42,
            "min_observations": 30,
            "feature_columns": [
                "volatility_metric",
                "trend_metric",
                "drawdown_metric",
                "stress_correlation_metric",
                "stress_dispersion_metric",
            ],
        },
    }

    config_path = inputs_root / "m26_full_year_regime_policy_fixture.yml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8", newline="\n")
    return config_path


def _build_stage_summaries(
    *,
    output_root: Path,
    benchmark_result: Any,
    promotion_result: Any,
    review_result: Any,
    candidate_result: Any,
    stress_result: Any,
) -> dict[str, dict[str, Any]]:
    benchmark_summary = _read_json(benchmark_result.output_path("benchmark_summary_json"))
    promotion_summary = _read_json(promotion_result.output_path("decision_summary_json"))
    review_summary = _read_json(review_result.output_path("review_summary_json"))
    candidate_summary = _read_json(candidate_result.output_path("selection_summary_json"))
    scenario_summary = _read_json(stress_result.output_path("scenario_summary_json"))
    policy_stress_summary = _read_json(stress_result.output_path("policy_stress_summary_json"))

    benchmark_stage = {
        "benchmark_run_id": benchmark_result.run_id,
        "benchmark_name": benchmark_summary.get("benchmark_name"),
        "variant_count": benchmark_summary.get("variant_count"),
        "variant_names": benchmark_summary.get("variant_names", []),
        "warning_count": benchmark_summary.get("warning_count", 0),
        "warnings": benchmark_summary.get("warnings", []),
        "source_artifact_paths": benchmark_summary.get("source_artifact_paths", []),
    }
    promotion_stage = {
        "benchmark_run_id": promotion_summary.get("benchmark_run_id"),
        "gate_config_name": promotion_summary.get("gate_config_name"),
        "decision_policy": promotion_summary.get("decision_policy"),
        "decision_counts": promotion_summary.get("decision_counts", {}),
        "variant_decisions": promotion_summary.get("variant_decisions", []),
    }
    review_stage = {
        "review_run_id": review_result.run_id,
        "review_name": review_summary.get("review_name"),
        "source_benchmark_run_id": review_summary.get("source_benchmark_run_id"),
        "source_gate_config_name": review_summary.get("source_gate_config_name"),
        "decision_counts": review_summary.get("decision_counts", {}),
        "leaderboard_path": _relative_to_output(review_result.output_path("leaderboard_csv"), output_root),
    }
    candidate_stage = {
        "selection_run_id": candidate_result.run_id,
        "selection_name": candidate_summary.get("selection_name"),
        "source_review_run_id": candidate_summary.get("source_review_run_id"),
        "source_benchmark_run_id": candidate_summary.get("source_benchmark_run_id"),
        "selected_count": candidate_summary.get("selected_count"),
        "selected_count_by_category": candidate_summary.get("selected_count_by_category", {}),
        "warning_count": candidate_summary.get("warning_count", 0),
        "limitations": candidate_summary.get("limitations", []),
    }
    stress_stage = {
        "stress_run_id": stress_result.run_id,
        "stress_test_name": scenario_summary.get("stress_test_name"),
        "scenario_count": scenario_summary.get("scenario_count"),
        "policy_count": scenario_summary.get("policy_count"),
        "baseline_policy": scenario_summary.get("baseline_policy"),
        "worst_scenario_type": scenario_summary.get("worst_scenario_type"),
        "most_resilient_policy": policy_stress_summary.get("most_resilient_policy"),
        "most_resilient_adaptive_policy": policy_stress_summary.get("most_resilient_adaptive_policy"),
        "baseline_rank": policy_stress_summary.get("baseline_rank"),
        "baseline_included_in_leaderboard": policy_stress_summary.get("baseline_included_in_leaderboard"),
        "adaptive_policy_count": policy_stress_summary.get("adaptive_policy_count"),
        "limitations": scenario_summary.get("limitations", []),
    }

    return {
        "benchmark": benchmark_stage,
        "promotion_gates": promotion_stage,
        "review": review_stage,
        "candidate_selection": candidate_stage,
        "stress": stress_stage,
    }


def _collect_role_rows(benchmark_matrix: pd.DataFrame) -> dict[str, pd.Series]:
    rows_by_role: dict[str, pd.Series] = {}
    for _, row in benchmark_matrix.sort_values("variant_name", kind="stable").iterrows():
        variant_type = str(row.get("variant_type", ""))
        role = ROLE_FROM_VARIANT_TYPE.get(variant_type)
        if role is None:
            continue
        rows_by_role[role] = row
    return rows_by_role


def _build_policy_variant_comparison(
    *,
    output_root: Path,
    benchmark_result: Any,
    promotion_summary: dict[str, Any],
    leaderboard: pd.DataFrame,
    candidate_selection: pd.DataFrame,
    candidate_summary: dict[str, Any],
    stress_matrix: pd.DataFrame,
    stress_result: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    benchmark_matrix = pd.read_csv(benchmark_result.output_path("benchmark_matrix_csv"))
    role_rows = _collect_role_rows(benchmark_matrix)

    gate_decisions = {
        str(item.get("variant_name")): item
        for item in promotion_summary.get("variant_decisions", [])
        if isinstance(item, dict)
    }
    leaderboard_by_variant = {
        str(row["variant_name"]): row
        for _, row in leaderboard.iterrows()
        if "variant_name" in row
    }

    selected_categories = sorted(
        {
            category
            for category in candidate_selection.get("selection_category", pd.Series(dtype="string"))
            .dropna()
            .astype("string")
            .tolist()
            for category in str(category).split("|")
            if category
        }
    )
    selected_categories_text = "|".join(selected_categories)

    stress_group = stress_matrix.groupby("policy_name", sort=True) if not stress_matrix.empty else None
    static_row = role_rows.get("static_baseline")
    static_return = _safe_float(static_row.get("adaptive_vs_static_return_delta") if static_row is not None else None)
    static_sharpe = _safe_float(static_row.get("adaptive_vs_static_sharpe_delta") if static_row is not None else None)
    static_drawdown = _safe_float(static_row.get("adaptive_vs_static_max_drawdown_delta") if static_row is not None else None)

    records: list[dict[str, Any]] = []
    for role in CONCEPTUAL_POLICY_ROLES:
        row = role_rows.get(role)
        if row is None:
            records.append(
                {
                    "policy_role": role,
                    "policy_name": None,
                    "regime_source": None,
                    "calibration_profile": None,
                    "classifier_model": None,
                    "benchmark_decision": "missing",
                    "review_rank": None,
                    "accepted": False,
                    "warning_count": None,
                    "failed_gate_count": None,
                    "stress_pass_rate": None,
                    "mean_stress_sharpe": None,
                    "mean_stress_max_drawdown": None,
                    "adaptive_vs_static_return_delta": None,
                    "adaptive_vs_static_sharpe_delta": None,
                    "adaptive_vs_static_drawdown_delta": None,
                    "selected_candidate_roles": selected_categories_text,
                    "primary_interpretation": "No mapped benchmark variant found for conceptual role.",
                    "source_artifact_path": None,
                }
            )
            continue

        variant_name = str(row.get("variant_name"))
        decision_row = gate_decisions.get(variant_name, {})
        leaderboard_row = leaderboard_by_variant.get(variant_name)
        decision = str(decision_row.get("decision", "missing"))
        accepted = decision in {"accepted", "accepted_with_warnings"}

        stress_policy = STRESS_POLICY_BY_ROLE.get(role)
        stress_pass_rate = None
        mean_stress_sharpe = None
        mean_stress_max_drawdown = None
        if stress_group is not None and stress_policy is not None and stress_policy in stress_group.groups:
            policy_rows = stress_group.get_group(stress_policy)
            if "stress_passed" in policy_rows.columns and len(policy_rows) > 0:
                stress_pass_rate = _safe_float(policy_rows["stress_passed"].astype("float64").mean())
            if "stress_sharpe" in policy_rows.columns and len(policy_rows) > 0:
                mean_stress_sharpe = _safe_float(policy_rows["stress_sharpe"].astype("float64").mean())
            if "stress_max_drawdown" in policy_rows.columns and len(policy_rows) > 0:
                mean_stress_max_drawdown = _safe_float(policy_rows["stress_max_drawdown"].astype("float64").mean())

        relative_source_artifact = None
        source_artifact = str(row.get("source_artifact_path") or "")
        if source_artifact:
            relative_source_artifact = _portable_path(
                benchmark_result.artifact_dir / source_artifact,
                output_root=output_root,
            )

        return_delta = _safe_float(row.get("adaptive_vs_static_return_delta"))
        sharpe_delta = _safe_float(row.get("adaptive_vs_static_sharpe_delta"))
        drawdown_delta = _safe_float(row.get("adaptive_vs_static_max_drawdown_delta"))

        records.append(
            {
                "policy_role": role,
                "policy_name": row.get("policy_name") or variant_name,
                "regime_source": row.get("regime_source"),
                "calibration_profile": row.get("calibration_profile"),
                "classifier_model": row.get("classifier_model"),
                "benchmark_decision": decision,
                "review_rank": None if leaderboard_row is None else int(leaderboard_row.get("rank", 0) or 0),
                "accepted": bool(accepted),
                "warning_count": decision_row.get("warning_count"),
                "failed_gate_count": decision_row.get("failed_gate_count"),
                "stress_pass_rate": stress_pass_rate,
                "mean_stress_sharpe": mean_stress_sharpe,
                "mean_stress_max_drawdown": mean_stress_max_drawdown,
                "adaptive_vs_static_return_delta": return_delta,
                "adaptive_vs_static_sharpe_delta": sharpe_delta,
                "adaptive_vs_static_drawdown_delta": drawdown_delta,
                "selected_candidate_roles": selected_categories_text,
                "primary_interpretation": (
                    "Accepted under governance with positive adaptive uplift."
                    if accepted and (sharpe_delta is not None and sharpe_delta > 0.0)
                    else "Governance decision is warning/review/reject or uplift is limited."
                ),
                "source_artifact_path": relative_source_artifact,
            }
        )

    comparison = pd.DataFrame(records)
    comparison = comparison.loc[:, [
        "policy_role",
        "policy_name",
        "regime_source",
        "calibration_profile",
        "classifier_model",
        "benchmark_decision",
        "review_rank",
        "accepted",
        "warning_count",
        "failed_gate_count",
        "stress_pass_rate",
        "mean_stress_sharpe",
        "mean_stress_max_drawdown",
        "adaptive_vs_static_return_delta",
        "adaptive_vs_static_sharpe_delta",
        "adaptive_vs_static_drawdown_delta",
        "selected_candidate_roles",
        "primary_interpretation",
        "source_artifact_path",
    ]].copy(deep=True)
    comparison["policy_role"] = pd.Categorical(
        comparison["policy_role"],
        categories=list(CONCEPTUAL_POLICY_ROLES),
        ordered=True,
    )
    comparison = comparison.sort_values("policy_role", kind="stable").reset_index(drop=True)
    comparison["policy_role"] = comparison["policy_role"].astype("string")

    best_return_delta = None
    best_sharpe_delta = None
    best_drawdown_delta = None
    deltas = comparison.loc[
        comparison["policy_role"].ne("static_baseline"),
        [
            "adaptive_vs_static_return_delta",
            "adaptive_vs_static_sharpe_delta",
            "adaptive_vs_static_drawdown_delta",
        ],
    ]
    if not deltas.empty:
        if deltas["adaptive_vs_static_return_delta"].notna().any():
            best_return_delta = _safe_float(deltas["adaptive_vs_static_return_delta"].max())
        if deltas["adaptive_vs_static_sharpe_delta"].notna().any():
            best_sharpe_delta = _safe_float(deltas["adaptive_vs_static_sharpe_delta"].max())
        if deltas["adaptive_vs_static_drawdown_delta"].notna().any():
            best_drawdown_delta = _safe_float(deltas["adaptive_vs_static_drawdown_delta"].max())

    return comparison, {
        "best_return_delta": best_return_delta if best_return_delta is not None else static_return,
        "best_sharpe_delta": best_sharpe_delta if best_sharpe_delta is not None else static_sharpe,
        "best_drawdown_delta": best_drawdown_delta if best_drawdown_delta is not None else static_drawdown,
    }


def _build_summary(
    *,
    output_root: Path,
    stage_summaries: dict[str, dict[str, Any]],
    review_leaderboard: pd.DataFrame,
    comparison: pd.DataFrame,
    static_baseline_comparison: dict[str, Any],
    candidate_summary: dict[str, Any],
    stress_summary: dict[str, Any],
    limitations: list[str],
) -> dict[str, Any]:
    top_policy: dict[str, Any]
    if review_leaderboard.empty:
        top_policy = {
            "name": None,
            "decision": None,
            "primary_reason": "No leaderboard rows available.",
        }
    else:
        row = review_leaderboard.sort_values("rank", kind="stable").iloc[0]
        top_policy = {
            "name": row.get("variant_name"),
            "decision": row.get("decision"),
            "primary_reason": row.get("primary_reason"),
        }

    summary = {
        "case_study_name": CASE_STUDY_NAME,
        "mode": CASE_STUDY_MODE,
        "evaluation_window": dict(EVALUATION_WINDOW),
        "policy_variants": list(CONCEPTUAL_POLICY_ROLES),
        "source_runs": {
            "benchmark_run_id": stage_summaries["benchmark"].get("benchmark_run_id"),
            "promotion_gate_run_id": stage_summaries["promotion_gates"].get("benchmark_run_id"),
            "promotion_gate_source_benchmark_run_id": stage_summaries["promotion_gates"].get("benchmark_run_id"),
            "promotion_gate_config_name": stage_summaries["promotion_gates"].get("gate_config_name"),
            "review_run_id": stage_summaries["review"].get("review_run_id"),
            "candidate_selection_run_id": stage_summaries["candidate_selection"].get("selection_run_id"),
            "stress_run_id": stage_summaries["stress"].get("stress_run_id"),
        },
        "top_policy": top_policy,
        "best_adaptive_policy_under_stress": stage_summaries["stress"].get("most_resilient_adaptive_policy"),
        "static_baseline_comparison": static_baseline_comparison,
        "decision_counts": stage_summaries["review"].get("decision_counts", {}),
        "candidate_selection": {
            "selected_count": candidate_summary.get("selected_count", 0),
            "selected_count_by_category": candidate_summary.get("selected_count_by_category", {}),
        },
        "stress_summary": {
            "most_resilient_policy": stress_summary.get("most_resilient_policy"),
            "most_resilient_adaptive_policy": stress_summary.get("most_resilient_adaptive_policy"),
            "baseline_rank": stress_summary.get("baseline_rank"),
            "worst_scenario_type": stress_summary.get("worst_scenario_type"),
        },
        "policy_variant_mapping": comparison.loc[:, [
            "policy_role",
            "policy_name",
            "regime_source",
            "calibration_profile",
            "classifier_model",
            "source_artifact_path",
        ]].to_dict(orient="records"),
        "evidence_separation": {
            "observed_benchmark_evidence": [
                "benchmark_summary.json",
                "promotion_gate_summary.json",
                "review_summary.json",
            ],
            "candidate_selection_evidence": ["candidate_selection_summary.json"],
            "deterministic_synthetic_stress_evidence": ["stress_summary.json"],
        },
        "limitations": limitations,
    }
    return _normalize_json_value(summary)


def _build_manifest(
    *,
    output_root: Path,
    inputs_root: Path,
    workflow_outputs_root: Path,
    summary: dict[str, Any],
    stage_summaries: dict[str, dict[str, Any]],
    generated_files: list[Path],
    source_artifacts: dict[str, Path],
    source_configs: dict[str, Path],
    policy_variant_count: int,
    warning_count: int,
) -> dict[str, Any]:
    manifest = {
        "artifact_type": CASE_STUDY_NAME,
        "schema_version": "1.0.0",
        "case_study_name": CASE_STUDY_NAME,
        "mode": CASE_STUDY_MODE,
        "generated_at_policy": "deterministic_no_timestamp",
        "output_root": ".",
        "generated_files": sorted(_relative_to_output(path, output_root) for path in generated_files),
        "source_configs": {
            key: _relative_config_path(path, output_root=output_root)
            for key, path in sorted(source_configs.items())
        },
        "source_artifacts": {
            key: _portable_path(path, output_root=output_root)
            for key, path in sorted(source_artifacts.items())
        },
        "source_runs": dict(summary.get("source_runs", {})),
        "row_counts": {
            "policy_variant_comparison": policy_variant_count,
            "benchmark_variants": stage_summaries["benchmark"].get("variant_count", 0),
            "review_decisions": sum(int(v) for v in stage_summaries["review"].get("decision_counts", {}).values()),
            "stress_scenarios": stage_summaries["stress"].get("scenario_count", 0),
        },
        "policy_variant_count": policy_variant_count,
        "warning_count": warning_count,
        "limitations": list(summary.get("limitations", [])),
        "inputs_root": _relative_to_output(inputs_root, output_root),
        "workflow_outputs_root": _relative_to_output(workflow_outputs_root, output_root),
    }
    return _normalize_json_value(manifest)


def _render_final_interpretation(
    *,
    summary: dict[str, Any],
    comparison: pd.DataFrame,
    stage_summaries: dict[str, dict[str, Any]],
    output_root: Path,
) -> str:
    decisions = summary.get("decision_counts", {})
    accepted_total = int(decisions.get("accepted", 0)) + int(decisions.get("accepted_with_warnings", 0))
    rejected_total = int(decisions.get("rejected", 0))

    lines = [
        "# Full-Year Regime Policy Benchmark Case Study Interpretation",
        "",
        "## Workflow Steps",
        "1. Generate deterministic full-year fixture market features.",
        "2. Run regime benchmark-pack policy variants.",
        "3. Apply promotion gates.",
        "4. Build review pack and ranked governance decisions.",
        "5. Run regime-aware candidate selection from review evidence.",
        "6. Run deterministic adaptive policy stress tests.",
        "7. Stitch benchmark/governance/candidate/stress outputs into this case-study layer.",
        "",
        "## Policy Variants",
        f"- {', '.join(summary.get('policy_variants', []))}",
        "",
        "## Observed Benchmark Evidence",
        "- Uses deterministic fixture-backed inputs for a 2026 full-year daily window.",
        "- Benchmark and governance evidence comes from saved benchmark, gate, and review artifacts.",
        f"- Top reviewed policy: {summary.get('top_policy', {}).get('name')} ({summary.get('top_policy', {}).get('decision')}).",
        "",
        "## Promotion And Review Governance Evidence",
        f"- Accepted or accepted_with_warnings: {accepted_total}.",
        f"- Rejected: {rejected_total}.",
        f"- Primary top-policy reason: {summary.get('top_policy', {}).get('primary_reason')}.",
        (
            "- Promotion-gate source benchmark run: "
            f"{summary.get('source_runs', {}).get('promotion_gate_source_benchmark_run_id')}"
            f" (config={summary.get('source_runs', {}).get('promotion_gate_config_name')})."
        ),
        "",
        "## Candidate-Selection Evidence",
        f"- Selected candidates: {summary.get('candidate_selection', {}).get('selected_count')}.",
        (
            "- Selected by category: "
            f"{summary.get('candidate_selection', {}).get('selected_count_by_category', {})}."
        ),
        "",
        "## Deterministic Synthetic Stress Evidence",
        "- Stress scenarios are deterministic transforms designed for robustness checks.",
        (
            "- Most resilient policy under stress: "
            f"{summary.get('stress_summary', {}).get('most_resilient_policy')}."
        ),
        (
            "- Most resilient adaptive policy under stress: "
            f"{summary.get('stress_summary', {}).get('most_resilient_adaptive_policy')}."
        ),
        (
            "- Baseline rank under stress: "
            f"{summary.get('stress_summary', {}).get('baseline_rank')}"
            f"; worst scenario type: {summary.get('stress_summary', {}).get('worst_scenario_type')}."
        ),
        "",
        "## Static Baseline Comparison",
        (
            "- Best adaptive vs static deltas: "
            f"return={summary.get('static_baseline_comparison', {}).get('best_return_delta')}, "
            f"sharpe={summary.get('static_baseline_comparison', {}).get('best_sharpe_delta')}, "
            f"drawdown={summary.get('static_baseline_comparison', {}).get('best_drawdown_delta')}."
        ),
        "",
        "## Limitations",
    ]
    for limitation in summary.get("limitations", []):
        lines.append(f"- {limitation}")

    lines.extend(
        [
            "",
            "## Reproduction Command",
            "```bash",
            "python docs/examples/full_year_regime_policy_benchmark_case_study.py",
            "```",
            "",
            "## Artifact Index",
            "- summary.json",
            "- manifest.json",
            "- benchmark_summary.json",
            "- promotion_gate_summary.json",
            "- review_summary.json",
            "- candidate_selection_summary.json",
            "- stress_summary.json",
            "- policy_variant_comparison.csv",
            "- final_interpretation.md",
            "- evidence_index.json",
            "",
            "## Caution",
            "- This example is not production or trading readiness evidence.",
            "- Deterministic stress transforms are not empirical market simulations.",
            "",
            "## Output Root",
            f"- {_relative_to_output(output_root, output_root)}",
        ]
    )

    _ = comparison
    _ = stage_summaries
    return "\n".join(lines)


def run_case_study(*, output_root: Path | None = None, verbose: bool = True) -> CaseStudyArtifacts:
    resolved_output_root = DEFAULT_OUTPUT_ROOT if output_root is None else Path(output_root)
    if resolved_output_root.exists():
        shutil.rmtree(resolved_output_root)
    resolved_output_root.mkdir(parents=True, exist_ok=True)

    inputs_root = resolved_output_root / "inputs"
    workflow_outputs_root = resolved_output_root / "workflow_outputs"
    benchmark_output_root = workflow_outputs_root / "benchmark"
    promotion_output_root = workflow_outputs_root / "promotion_gates"
    review_output_root = workflow_outputs_root / "review"
    candidate_output_root = workflow_outputs_root / "candidate_selection"
    stress_output_root = workflow_outputs_root / "stress_tests"

    benchmark_config_path = _write_benchmark_fixture_config(inputs_root, benchmark_output_root)

    benchmark_result = run_regime_benchmark_pack(
        config_path=benchmark_config_path,
        output_root=benchmark_output_root,
    )
    promotion_result = run_regime_promotion_gates(
        benchmark_path=benchmark_result.artifact_dir,
        config_path=M26_PROMOTION_GATES_CONFIG,
        output_dir=promotion_output_root,
    )
    review_result = generate_regime_review_pack(
        benchmark_path=benchmark_result.artifact_dir,
        promotion_gates_path=promotion_result.artifact_dir,
        output_root=review_output_root,
        review_name="m26_full_year_regime_policy_case_study_review",
    )
    candidate_result = run_regime_aware_candidate_selection(
        config_path=M26_CANDIDATE_SELECTION_CONFIG,
        source_review_pack=review_result.artifact_dir,
        candidate_metrics_path=M26_CANDIDATE_METRICS_FIXTURE,
        output_root=candidate_output_root,
    )
    stress_result = run_regime_policy_stress_tests(
        config_path=M26_STRESS_CONFIG,
        source_review_pack=review_result.artifact_dir,
        policy_metrics_path=M26_STRESS_POLICY_FIXTURE,
        output_root=stress_output_root,
    )

    stage_summaries = _build_stage_summaries(
        output_root=resolved_output_root,
        benchmark_result=benchmark_result,
        promotion_result=promotion_result,
        review_result=review_result,
        candidate_result=candidate_result,
        stress_result=stress_result,
    )

    review_leaderboard = pd.read_csv(review_result.output_path("leaderboard_csv"))
    candidate_selection = pd.read_csv(candidate_result.output_path("candidate_selection_csv"))
    stress_matrix = pd.read_csv(stress_result.output_path("stress_matrix_csv"))

    promotion_summary = _read_json(promotion_result.output_path("decision_summary_json"))
    candidate_summary = _read_json(candidate_result.output_path("selection_summary_json"))
    stress_summary = _read_json(stress_result.output_path("policy_stress_summary_json"))

    comparison, static_baseline_comparison = _build_policy_variant_comparison(
        output_root=resolved_output_root,
        benchmark_result=benchmark_result,
        promotion_summary=promotion_summary,
        leaderboard=review_leaderboard,
        candidate_selection=candidate_selection,
        candidate_summary=candidate_summary,
        stress_matrix=stress_matrix,
        stress_result=stress_result,
    )

    limitations = [
        "This case study uses deterministic fixture-backed benchmark inputs to mirror a full-year workflow when complete real-data coverage is unavailable.",
        "Stress evidence uses deterministic synthetic transforms and should not be interpreted as empirical market simulation results.",
        "Outputs are intended for research governance and reproducibility, not production trading readiness.",
    ]
    limitations.extend(
        item
        for item in stage_summaries["candidate_selection"].get("limitations", [])
        if isinstance(item, str)
    )
    limitations.extend(
        item
        for item in stage_summaries["stress"].get("limitations", [])
        if isinstance(item, str)
    )
    limitations = list(dict.fromkeys(limitations))

    summary = _build_summary(
        output_root=resolved_output_root,
        stage_summaries=stage_summaries,
        review_leaderboard=review_leaderboard,
        comparison=comparison,
        static_baseline_comparison=static_baseline_comparison,
        candidate_summary=candidate_summary,
        stress_summary=stress_summary,
        limitations=limitations,
    )

    benchmark_summary_path = resolved_output_root / "benchmark_summary.json"
    promotion_summary_path = resolved_output_root / "promotion_gate_summary.json"
    review_summary_path = resolved_output_root / "review_summary.json"
    candidate_summary_path = resolved_output_root / "candidate_selection_summary.json"
    stress_summary_path = resolved_output_root / "stress_summary.json"
    comparison_path = resolved_output_root / POLICY_VARIANT_COMPARISON_FILENAME
    summary_path = resolved_output_root / SUMMARY_FILENAME
    interpretation_path = resolved_output_root / FINAL_INTERPRETATION_FILENAME
    manifest_path = resolved_output_root / MANIFEST_FILENAME
    evidence_index_path = resolved_output_root / "evidence_index.json"
    workflow_outputs_path = resolved_output_root / "workflow_outputs.json"

    _write_json(benchmark_summary_path, stage_summaries["benchmark"])
    _write_json(promotion_summary_path, stage_summaries["promotion_gates"])
    _write_json(review_summary_path, stage_summaries["review"])
    _write_json(candidate_summary_path, stage_summaries["candidate_selection"])
    _write_json(stress_summary_path, stage_summaries["stress"])

    comparison.to_csv(comparison_path, index=False)
    _write_json(summary_path, summary)

    interpretation = _render_final_interpretation(
        summary=summary,
        comparison=comparison,
        stage_summaries=stage_summaries,
        output_root=resolved_output_root,
    )
    _write_text(interpretation_path, interpretation)

    source_artifacts = {
        "benchmark_manifest": benchmark_result.output_path("manifest_json"),
        "benchmark_matrix": benchmark_result.output_path("benchmark_matrix_csv"),
        "promotion_gate_manifest": promotion_result.output_path("manifest_json"),
        "promotion_gate_decision_summary": promotion_result.output_path("decision_summary_json"),
        "review_manifest": review_result.output_path("manifest_json"),
        "review_leaderboard": review_result.output_path("leaderboard_csv"),
        "candidate_selection_manifest": candidate_result.output_path("manifest_json"),
        "candidate_selection_table": candidate_result.output_path("candidate_selection_csv"),
        "stress_manifest": stress_result.output_path("manifest_json"),
        "stress_matrix": stress_result.output_path("stress_matrix_csv"),
    }
    source_configs = {
        "benchmark_fixture_config": benchmark_config_path,
        "promotion_gates_config": M26_PROMOTION_GATES_CONFIG,
        "candidate_selection_config": M26_CANDIDATE_SELECTION_CONFIG,
        "candidate_metrics_fixture": M26_CANDIDATE_METRICS_FIXTURE,
        "stress_test_config": M26_STRESS_CONFIG,
        "stress_policy_fixture": M26_STRESS_POLICY_FIXTURE,
    }

    benchmark_artifacts = {
        "benchmark_summary": _portable_path(benchmark_summary_path, output_root=resolved_output_root),
        "benchmark_matrix": _portable_path(source_artifacts["benchmark_matrix"], output_root=resolved_output_root),
        "benchmark_manifest": _portable_path(source_artifacts["benchmark_manifest"], output_root=resolved_output_root),
    }
    promotion_gate_artifacts = {
        "promotion_gate_summary": _portable_path(promotion_summary_path, output_root=resolved_output_root),
        "decision_summary": _portable_path(source_artifacts["promotion_gate_decision_summary"], output_root=resolved_output_root),
        "manifest": _portable_path(source_artifacts["promotion_gate_manifest"], output_root=resolved_output_root),
    }
    review_pack_artifacts = {
        "review_summary": _portable_path(review_summary_path, output_root=resolved_output_root),
        "leaderboard": _portable_path(source_artifacts["review_leaderboard"], output_root=resolved_output_root),
        "manifest": _portable_path(source_artifacts["review_manifest"], output_root=resolved_output_root),
    }
    candidate_selection_artifacts = {
        "candidate_selection_summary": _portable_path(candidate_summary_path, output_root=resolved_output_root),
        "candidate_selection_table": _portable_path(source_artifacts["candidate_selection_table"], output_root=resolved_output_root),
        "manifest": _portable_path(source_artifacts["candidate_selection_manifest"], output_root=resolved_output_root),
    }
    stress_test_artifacts = {
        "stress_summary": _portable_path(stress_summary_path, output_root=resolved_output_root),
        "stress_matrix": _portable_path(source_artifacts["stress_matrix"], output_root=resolved_output_root),
        "manifest": _portable_path(source_artifacts["stress_manifest"], output_root=resolved_output_root),
        "evidence_type": "deterministic_synthetic_stress",
    }

    evidence_index = {
        "source_artifacts": {
            "benchmark": benchmark_artifacts,
            "promotion_gates": promotion_gate_artifacts,
            "review": review_pack_artifacts,
            "candidate_selection": candidate_selection_artifacts,
            "stress": stress_test_artifacts,
        },
        "benchmark_artifacts": benchmark_artifacts,
        "promotion_gate_artifacts": promotion_gate_artifacts,
        "review_pack_artifacts": review_pack_artifacts,
        "candidate_selection_artifacts": candidate_selection_artifacts,
        "stress_test_artifacts": stress_test_artifacts,
        "generated_case_study_artifacts": {
            "summary": SUMMARY_FILENAME,
            "manifest": MANIFEST_FILENAME,
            "policy_variant_comparison": POLICY_VARIANT_COMPARISON_FILENAME,
            "final_interpretation": FINAL_INTERPRETATION_FILENAME,
            "workflow_outputs": "workflow_outputs.json",
        },
    }
    _write_json(evidence_index_path, evidence_index)

    workflow_outputs = {
        "benchmark_artifact_dir": _portable_path(benchmark_result.artifact_dir, output_root=resolved_output_root),
        "promotion_gate_artifact_dir": _portable_path(promotion_result.artifact_dir, output_root=resolved_output_root),
        "review_artifact_dir": _portable_path(review_result.artifact_dir, output_root=resolved_output_root),
        "candidate_selection_artifact_dir": _portable_path(candidate_result.artifact_dir, output_root=resolved_output_root),
        "stress_artifact_dir": _portable_path(stress_result.artifact_dir, output_root=resolved_output_root),
        "source_runs": summary["source_runs"],
    }
    _write_json(workflow_outputs_path, workflow_outputs)

    warning_count = int(len(summary.get("limitations", [])))
    manifest = _build_manifest(
        output_root=resolved_output_root,
        inputs_root=inputs_root,
        workflow_outputs_root=workflow_outputs_root,
        summary=summary,
        stage_summaries=stage_summaries,
        generated_files=[
            summary_path,
            manifest_path,
            benchmark_summary_path,
            promotion_summary_path,
            review_summary_path,
            candidate_summary_path,
            stress_summary_path,
            comparison_path,
            interpretation_path,
            evidence_index_path,
            workflow_outputs_path,
        ],
        source_artifacts=source_artifacts,
        source_configs=source_configs,
        policy_variant_count=len(comparison),
        warning_count=warning_count,
    )
    _write_json(manifest_path, manifest)

    if verbose:
        print(f"Case Study Output Root: {resolved_output_root.as_posix()}")
        print(f"Summary: {summary_path.as_posix()}")
        print(f"Final Interpretation: {interpretation_path.as_posix()}")
        print(f"Benchmark Run ID: {summary['source_runs']['benchmark_run_id']}")
        print(f"Review Run ID: {summary['source_runs']['review_run_id']}")
        print(f"Stress Run ID: {summary['source_runs']['stress_run_id']}")
        print("Status: success (fixture_backed)")

    return CaseStudyArtifacts(
        output_root=resolved_output_root,
        summary_path=summary_path,
        manifest_path=manifest_path,
        final_interpretation_path=interpretation_path,
        summary=summary,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Milestone 26 full-year regime policy benchmark case study "
            "using deterministic fixture-backed inputs."
        )
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT.as_posix(),
        help="Output root for stitched case-study artifacts.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable concise status printing.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> CaseStudyArtifacts:
    args = parse_args(argv)
    return run_case_study(output_root=Path(args.output_root), verbose=not args.quiet)


if __name__ == "__main__":
    main()
