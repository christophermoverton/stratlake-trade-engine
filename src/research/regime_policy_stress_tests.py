from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.config.regime_policy_stress_tests import RegimePolicyStressScenarioConfig, RegimePolicyStressTestConfig
from src.research.market_simulation.policy_stress_integration import (
    MARKET_SIMULATION_STRESS_LEADERBOARD_FILENAME,
    MARKET_SIMULATION_STRESS_SUMMARY_FILENAME,
    run_market_simulation_policy_stress_integration,
)
from src.research.registry import canonicalize_value, serialize_canonical_json


STRESS_MATRIX_COLUMNS = (
    "scenario_id",
    "scenario_name",
    "scenario_type",
    "policy_name",
    "baseline_policy",
    "policy_type",
    "regime_source",
    "classifier_model",
    "calibration_profile",
    "transition_window_bars",
    "stress_intensity",
    "mean_confidence_under_stress",
    "mean_entropy_under_stress",
    "disagreement_rate",
    "fallback_activation_count",
    "fallback_activation_rate",
    "policy_state_change_count",
    "policy_turnover",
    "stress_return",
    "stress_sharpe",
    "stress_max_drawdown",
    "adaptive_vs_static_return_delta",
    "adaptive_vs_static_sharpe_delta",
    "adaptive_vs_static_drawdown_delta",
    "stress_passed",
    "primary_failure_reason",
)
STRESS_LEADERBOARD_COLUMNS = (
    "rank",
    "policy_name",
    "policy_type",
    "scenario_count",
    "stress_pass_rate",
    "mean_stress_return",
    "mean_stress_sharpe",
    "mean_stress_max_drawdown",
    "mean_policy_turnover",
    "mean_fallback_activation_rate",
    "mean_adaptive_vs_static_return_delta",
    "mean_adaptive_vs_static_sharpe_delta",
    "mean_adaptive_vs_static_drawdown_delta",
    "worst_scenario_type",
    "primary_strength",
    "primary_weakness",
)
SCENARIO_RESULT_COLUMNS = (
    "scenario_id",
    "scenario_name",
    "scenario_type",
    "policy_name",
    "stress_passed",
    "primary_failure_reason",
    "stress_return",
    "stress_sharpe",
    "stress_max_drawdown",
    "policy_turnover",
    "policy_state_change_count",
    "fallback_activation_rate",
)
FALLBACK_USAGE_COLUMNS = (
    "scenario_id",
    "scenario_name",
    "scenario_type",
    "policy_name",
    "fallback_activation_count",
    "fallback_activation_rate",
    "mean_confidence_under_stress",
    "mean_entropy_under_stress",
    "selected_fallback_behavior",
    "warning",
)
TURNOVER_COLUMNS = (
    "scenario_id",
    "scenario_name",
    "scenario_type",
    "policy_name",
    "policy_turnover",
    "policy_state_change_count",
    "max_policy_turnover",
    "max_state_change_count",
    "turnover_gate_passed",
    "state_change_gate_passed",
)
ADAPTIVE_STATIC_COLUMNS = (
    "scenario_id",
    "scenario_name",
    "scenario_type",
    "policy_name",
    "baseline_policy",
    "adaptive_vs_static_return_delta",
    "adaptive_vs_static_sharpe_delta",
    "adaptive_vs_static_drawdown_delta",
    "comparison_reason",
)
REQUIRED_POLICY_FIELDS = (
    "policy_name",
    "policy_type",
    "is_baseline",
    "regime_source",
    "classifier_model",
    "calibration_profile",
)


@dataclass(frozen=True)
class RegimePolicyStressTestRunResult:
    stress_run_id: str
    stress_test_name: str
    output_dir: Path
    stress_matrix_csv_path: Path
    stress_matrix_json_path: Path
    stress_leaderboard_csv_path: Path
    scenario_summary_path: Path
    policy_stress_summary_path: Path
    scenario_catalog_path: Path
    scenario_results_csv_path: Path
    fallback_usage_csv_path: Path
    policy_turnover_csv_path: Path
    adaptive_vs_static_comparison_csv_path: Path
    market_simulation_stress_summary_path: Path | None
    market_simulation_stress_leaderboard_path: Path | None
    config_path: Path
    manifest_path: Path
    scenario_summary: dict[str, Any]
    policy_stress_summary: dict[str, Any]
    manifest: dict[str, Any]


def run_regime_policy_stress_tests(config: RegimePolicyStressTestConfig) -> RegimePolicyStressTestRunResult:
    source_review_metadata, source_review_warnings = _load_source_review_pack(config.source_review_pack)
    if config.source_policy_candidates is None:
        raise ValueError(
            "Regime policy stress tests currently require source_policy_candidates.policy_metrics_path for file-backed execution."
        )

    policy_rows = _load_rows(Path(config.source_policy_candidates.policy_metrics_path))
    if not policy_rows:
        raise ValueError("Regime policy stress tests require at least one policy candidate row.")

    missing_required = [
        field
        for field in REQUIRED_POLICY_FIELDS
        if any(_missing(row.get(field)) for row in policy_rows)
    ]
    if missing_required:
        raise ValueError(
            f"Policy candidate metrics input is missing required fields: {missing_required}."
        )

    scenario_catalog = _build_scenario_catalog(config.scenarios)
    policies, policy_inventory, policy_warnings = _load_policy_inventory(
        policy_rows,
        baseline_policy=config.source_policy_candidates.baseline_policy,
        configured_candidates=config.source_policy_candidates.candidate_policy_names,
    )

    stress_run_id = _build_stress_run_id(
        config=config,
        source_review_run_id=str(source_review_metadata.get("review_run_id") or "unknown"),
        policy_names=[item["policy_name"] for item in policies],
        scenario_ids=[scenario["scenario_id"] for scenario in scenario_catalog],
    )
    output_dir = Path(config.output_root).resolve() / stress_run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_policy_name = config.source_policy_candidates.baseline_policy
    policy_by_name = {item["policy_name"]: item for item in policies}
    if baseline_policy_name not in policy_by_name:
        raise ValueError(
            f"Baseline policy {baseline_policy_name!r} was not present in policy candidate metrics."
        )
    baseline_policy = policy_by_name[baseline_policy_name]

    evaluation_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    warnings.extend(source_review_warnings)
    warnings.extend(policy_warnings)
    for scenario in scenario_catalog:
        baseline_metrics = _resolve_stressed_metrics_for_policy(
            baseline_policy,
            scenario=scenario,
            transition_window_bars=config.regime_context.transition_window_bars,
        )
        for policy in policies:
            stressed = _resolve_stressed_metrics_for_policy(
                policy,
                scenario=scenario,
                transition_window_bars=config.regime_context.transition_window_bars,
            )
            row = _build_matrix_row(
                policy=policy,
                scenario=scenario,
                stressed=stressed,
                baseline_metrics=baseline_metrics,
                baseline_policy=baseline_policy_name,
                transition_window_bars=config.regime_context.transition_window_bars,
            )
            gate_result = _evaluate_stress_gates(row, config=config)
            row["stress_passed"] = gate_result["stress_passed"]
            row["primary_failure_reason"] = gate_result["primary_failure_reason"]
            warnings.extend(gate_result["warnings"])
            evaluation_rows.append(canonicalize_value(row))

    evaluation_rows = sorted(
        evaluation_rows,
        key=lambda row: (str(row["scenario_id"]), str(row["policy_name"])),
    )

    stress_matrix_json = {
        "stress_run_id": stress_run_id,
        "stress_test_name": config.stress_test_name,
        "row_count": len(evaluation_rows),
        "columns": list(STRESS_MATRIX_COLUMNS),
        "rows": evaluation_rows,
    }

    leaderboard_rows = _build_stress_leaderboard(evaluation_rows)
    scenario_summary = _build_scenario_summary(
        stress_run_id=stress_run_id,
        config=config,
        rows=evaluation_rows,
        baseline_policy=baseline_policy_name,
        leaderboard_rows=leaderboard_rows,
        warnings=warnings,
        source_review_metadata=source_review_metadata,
    )
    policy_stress_summary = _build_policy_stress_summary(
        stress_run_id=stress_run_id,
        config=config,
        rows=evaluation_rows,
        baseline_policy=baseline_policy_name,
        leaderboard_rows=leaderboard_rows,
        source_review_metadata=source_review_metadata,
    )
    scenario_results = _build_scenario_results(evaluation_rows)
    fallback_usage = _build_fallback_usage_rows(evaluation_rows)
    turnover_rows = _build_turnover_rows(evaluation_rows, config=config)
    adaptive_rows = _build_adaptive_vs_static_rows(evaluation_rows, baseline_policy=baseline_policy_name)

    paths = {
        "stress_matrix_csv": output_dir / "stress_matrix.csv",
        "stress_matrix_json": output_dir / "stress_matrix.json",
        "stress_leaderboard_csv": output_dir / "stress_leaderboard.csv",
        "scenario_summary_json": output_dir / "scenario_summary.json",
        "policy_stress_summary_json": output_dir / "policy_stress_summary.json",
        "scenario_catalog_json": output_dir / "scenario_catalog.json",
        "scenario_results_csv": output_dir / "scenario_results.csv",
        "fallback_usage_csv": output_dir / "fallback_usage.csv",
        "policy_turnover_csv": output_dir / "policy_turnover_under_stress.csv",
        "adaptive_vs_static_csv": output_dir / "adaptive_vs_static_stress_comparison.csv",
        "config_json": output_dir / "config.json",
        "manifest_json": output_dir / "manifest.json",
        "source_review_pack_json": output_dir / "source_regime_review_pack.json",
        "policy_inventory_json": output_dir / "policy_input_inventory.json",
    }

    market_simulation_stress_result = run_market_simulation_policy_stress_integration(
        config.market_simulation_stress,
        output_dir=output_dir,
    )
    if market_simulation_stress_result is not None:
        if config.market_simulation_stress.include_in_policy_stress_summary:
            policy_stress_summary["market_simulation_stress"] = market_simulation_stress_result.summary
        scenario_summary["market_simulation_stress"] = {
            "market_simulation_enabled": True,
            "market_simulation_mode": market_simulation_stress_result.summary.get("market_simulation_mode"),
            "summary_path": _rel(market_simulation_stress_result.summary_path),
            "leaderboard_path": _rel(market_simulation_stress_result.leaderboard_path),
            "source_market_simulation_run_id": market_simulation_stress_result.summary.get(
                "source_market_simulation_run_id"
            ),
        }

    _write_csv(paths["stress_matrix_csv"], evaluation_rows, STRESS_MATRIX_COLUMNS)
    _write_json(paths["stress_matrix_json"], stress_matrix_json)
    _write_csv(paths["stress_leaderboard_csv"], leaderboard_rows, STRESS_LEADERBOARD_COLUMNS)
    _write_json(paths["scenario_summary_json"], scenario_summary)
    _write_json(paths["policy_stress_summary_json"], policy_stress_summary)
    _write_json(paths["scenario_catalog_json"], {"rows": scenario_catalog, "row_count": len(scenario_catalog)})
    _write_csv(paths["scenario_results_csv"], scenario_results, SCENARIO_RESULT_COLUMNS)
    _write_csv(paths["fallback_usage_csv"], fallback_usage, FALLBACK_USAGE_COLUMNS)
    _write_csv(paths["policy_turnover_csv"], turnover_rows, TURNOVER_COLUMNS)
    _write_csv(paths["adaptive_vs_static_csv"], adaptive_rows, ADAPTIVE_STATIC_COLUMNS)
    _write_json(paths["config_json"], config.to_dict())
    _write_json(paths["source_review_pack_json"], source_review_metadata)
    _write_json(paths["policy_inventory_json"], policy_inventory)

    generated_files = [
        "stress_matrix.csv",
        "stress_matrix.json",
        "stress_leaderboard.csv",
        "scenario_summary.json",
        "policy_stress_summary.json",
        "scenario_catalog.json",
        "scenario_results.csv",
        "fallback_usage.csv",
        "policy_turnover_under_stress.csv",
        "adaptive_vs_static_stress_comparison.csv",
        "config.json",
        "manifest.json",
        "source_regime_review_pack.json",
        "policy_input_inventory.json",
    ]
    if market_simulation_stress_result is not None:
        generated_files.extend(
            [
                MARKET_SIMULATION_STRESS_SUMMARY_FILENAME,
                MARKET_SIMULATION_STRESS_LEADERBOARD_FILENAME,
            ]
        )

    manifest = _build_manifest(
        stress_run_id=stress_run_id,
        config=config,
        output_dir=output_dir,
        baseline_policy=baseline_policy_name,
        rows=evaluation_rows,
        warnings=warnings,
        generated_files=generated_files,
        source_review_metadata=source_review_metadata,
        source_policy_candidates_path=Path(config.source_policy_candidates.policy_metrics_path),
        scenario_catalog=scenario_catalog,
        market_simulation_stress_summary=None
        if market_simulation_stress_result is None
        else market_simulation_stress_result.summary,
    )
    _write_json(paths["manifest_json"], manifest)

    return RegimePolicyStressTestRunResult(
        stress_run_id=stress_run_id,
        stress_test_name=config.stress_test_name,
        output_dir=output_dir,
        stress_matrix_csv_path=paths["stress_matrix_csv"],
        stress_matrix_json_path=paths["stress_matrix_json"],
        stress_leaderboard_csv_path=paths["stress_leaderboard_csv"],
        scenario_summary_path=paths["scenario_summary_json"],
        policy_stress_summary_path=paths["policy_stress_summary_json"],
        scenario_catalog_path=paths["scenario_catalog_json"],
        scenario_results_csv_path=paths["scenario_results_csv"],
        fallback_usage_csv_path=paths["fallback_usage_csv"],
        policy_turnover_csv_path=paths["policy_turnover_csv"],
        adaptive_vs_static_comparison_csv_path=paths["adaptive_vs_static_csv"],
        market_simulation_stress_summary_path=None
        if market_simulation_stress_result is None
        else market_simulation_stress_result.summary_path,
        market_simulation_stress_leaderboard_path=None
        if market_simulation_stress_result is None
        else market_simulation_stress_result.leaderboard_path,
        config_path=paths["config_json"],
        manifest_path=paths["manifest_json"],
        scenario_summary=scenario_summary,
        policy_stress_summary=policy_stress_summary,
        manifest=manifest,
    )


def _build_scenario_catalog(scenarios: tuple[RegimePolicyStressScenarioConfig, ...]) -> list[dict[str, Any]]:
    rows = [scenario.to_dict() for scenario in scenarios]
    rows.sort(key=lambda row: (str(row["scenario_id"]), str(row["name"])))
    return rows


def _load_source_review_pack(path_value: str | None) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    if path_value is None:
        return {}, ["No source review pack was provided; provenance is limited to policy input artifacts."]
    root = Path(path_value).resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Source regime review pack does not exist: {root.as_posix()}")

    manifest_path = root / "manifest.json"
    summary_path = root / "review_summary.json"
    if not manifest_path.exists() and not summary_path.exists():
        raise FileNotFoundError(
            f"Source regime review pack must include manifest.json or review_summary.json: {root.as_posix()}"
        )

    metadata: dict[str, Any] = {"source_review_pack": _rel(root)}
    if summary_path.exists():
        metadata.update(_load_json(summary_path))
    else:
        warnings.append("Source review pack review_summary.json was missing.")
    if manifest_path.exists():
        metadata.update(_load_json(manifest_path))
    else:
        warnings.append("Source review pack manifest.json was missing.")
    return canonicalize_value(metadata), warnings


def _load_policy_inventory(
    rows: list[dict[str, Any]],
    *,
    baseline_policy: str,
    configured_candidates: tuple[str, ...],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    warnings: list[str] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("policy_name")), []).append(row)

    if baseline_policy not in grouped:
        raise ValueError(f"Baseline policy {baseline_policy!r} was not found in policy metrics input.")

    selected_policy_names = sorted(grouped)
    if configured_candidates:
        allowed = set(configured_candidates) | {baseline_policy}
        selected_policy_names = sorted(name for name in selected_policy_names if name in allowed)
        missing = sorted(name for name in configured_candidates if name not in grouped)
        if missing:
            warnings.append(
                f"Configured candidate_policy_names were not present in policy metrics input: {missing}."
            )
    policies: list[dict[str, Any]] = []

    optional_metrics = (
        "base_return",
        "base_sharpe",
        "base_max_drawdown",
        "base_policy_turnover",
        "base_state_change_count",
        "base_fallback_activation_count",
        "base_confidence",
        "base_entropy",
    )

    for name in selected_policy_names:
        policy_rows = grouped[name]
        base_rows = [row for row in policy_rows if _missing(row.get("scenario_name"))]
        scenario_rows = [row for row in policy_rows if not _missing(row.get("scenario_name"))]
        if not base_rows:
            base_rows = [policy_rows[0]]
            warnings.append(
                f"Policy {name!r} had no base row; first row was used as baseline metrics source."
            )
        base = base_rows[0]
        policy = {
            "policy_name": str(base.get("policy_name")),
            "policy_type": str(base.get("policy_type")),
            "is_baseline": _as_bool(base.get("is_baseline")),
            "regime_source": str(base.get("regime_source")),
            "classifier_model": str(base.get("classifier_model")),
            "calibration_profile": str(base.get("calibration_profile")),
            "base_return": _float_or_none(base.get("base_return")),
            "base_sharpe": _float_or_none(base.get("base_sharpe")),
            "base_max_drawdown": _float_or_none(base.get("base_max_drawdown")),
            "base_policy_turnover": _float_or_none(base.get("base_policy_turnover")),
            "base_state_change_count": _float_or_none(base.get("base_state_change_count")),
            "base_fallback_activation_count": _float_or_none(base.get("base_fallback_activation_count")),
            "base_confidence": _float_or_none(base.get("base_confidence")),
            "base_entropy": _float_or_none(base.get("base_entropy")),
            "observation_count": _float_or_none(base.get("observation_count")) or 100.0,
            "scenario_rows": _index_scenario_rows(scenario_rows),
        }
        for metric in optional_metrics:
            if policy[metric] is None:
                warnings.append(
                    f"Optional policy metric {metric!r} was missing for policy {name!r}; deterministic derivation defaults were used."
                )
        policies.append(canonicalize_value(policy))

    policies = sorted(policies, key=lambda item: str(item["policy_name"]))
    inventory = {
        "input_mode": "file_backed",
        "policy_count": len(policies),
        "policy_names": [item["policy_name"] for item in policies],
        "baseline_policy": baseline_policy,
        "row_count": len(rows),
        "columns": sorted({str(key) for row in rows for key in row}),
        "warnings": sorted(set(warnings)),
    }
    return policies, canonicalize_value(inventory), sorted(set(warnings))


def _index_scenario_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        scenario_name = str(row.get("scenario_name"))
        if scenario_name in indexed:
            continue
        indexed[scenario_name] = dict(row)
    return indexed


def _resolve_stressed_metrics_for_policy(
    policy: Mapping[str, Any],
    *,
    scenario: Mapping[str, Any],
    transition_window_bars: int,
) -> dict[str, float | None]:
    scenario_name = str(scenario["name"])
    precomputed = policy.get("scenario_rows", {}).get(scenario_name) if isinstance(policy.get("scenario_rows"), Mapping) else None
    if isinstance(precomputed, Mapping):
        return _precomputed_metrics(policy, precomputed)
    return _derived_metrics(policy, scenario=scenario, transition_window_bars=transition_window_bars)


def _precomputed_metrics(policy: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, float | None]:
    confidence = _float_or_none(row.get("mean_confidence_under_stress"))
    entropy = _float_or_none(row.get("mean_entropy_under_stress"))
    fallback_count = _float_or_none(row.get("fallback_activation_count"))
    observation_count = _float_or_none(policy.get("observation_count")) or 100.0
    fallback_rate = _float_or_none(row.get("fallback_activation_rate"))
    if fallback_rate is None and fallback_count is not None and observation_count > 0:
        fallback_rate = min(max(fallback_count / observation_count, 0.0), 1.0)
    return {
        "stress_return": _float_or_none(row.get("stress_return")),
        "stress_sharpe": _float_or_none(row.get("stress_sharpe")),
        "stress_max_drawdown": _float_or_none(row.get("stress_max_drawdown")),
        "policy_turnover": _float_or_none(row.get("policy_turnover")),
        "policy_state_change_count": _float_or_none(row.get("policy_state_change_count")),
        "fallback_activation_count": fallback_count,
        "fallback_activation_rate": fallback_rate,
        "mean_confidence_under_stress": confidence,
        "mean_entropy_under_stress": entropy,
        "disagreement_rate": _float_or_none(row.get("disagreement_rate")),
    }


def _derived_metrics(
    policy: Mapping[str, Any],
    *,
    scenario: Mapping[str, Any],
    transition_window_bars: int,
) -> dict[str, float | None]:
    intensity = _float_or_none(scenario.get("stress_intensity")) or 1.0
    scenario_type = str(scenario["type"])
    params = dict(scenario)
    base_return = _fallback(policy.get("base_return"), 0.05)
    base_sharpe = _fallback(policy.get("base_sharpe"), 1.0)
    base_drawdown = _fallback(policy.get("base_max_drawdown"), -0.10)
    base_turnover = _fallback(policy.get("base_policy_turnover"), 0.20)
    base_state_changes = _fallback(policy.get("base_state_change_count"), 12.0)
    base_fallback_count = _fallback(policy.get("base_fallback_activation_count"), 5.0)
    base_confidence = _fallback(policy.get("base_confidence"), 0.65)
    base_entropy = _fallback(policy.get("base_entropy"), 0.90)
    observation_count = _fallback(policy.get("observation_count"), 100.0)

    stress_return = base_return
    stress_sharpe = base_sharpe
    stress_drawdown = base_drawdown
    turnover = base_turnover
    state_changes = base_state_changes
    fallback_count = base_fallback_count
    confidence = base_confidence
    entropy = base_entropy
    disagreement_rate = None

    is_defensive = "defensive" in str(policy.get("policy_type", "")).lower() or "fallback" in str(policy.get("policy_name", "")).lower()

    if scenario_type == "transition_shock":
        shock_length = _fallback(params.get("shock_length_bars"), 10.0)
        stress_return -= 0.03 * intensity
        stress_sharpe -= 0.25 * intensity
        stress_drawdown -= 0.04 * intensity
        turnover += 0.12 * intensity
        state_changes += (transition_window_bars + shock_length * 0.3) * intensity
        fallback_count += 2.0 * intensity
    elif scenario_type == "regime_whipsaw":
        cycles = _fallback(params.get("cycles"), 4.0)
        cycle_length = _fallback(params.get("cycle_length_bars"), 3.0)
        stress_return -= 0.02 * intensity
        stress_sharpe -= 0.35 * intensity
        stress_drawdown -= 0.03 * intensity
        turnover += (0.16 + 0.01 * cycles + 0.005 * cycle_length) * intensity
        state_changes += (cycles * 2.0 + cycle_length) * intensity
        fallback_count += 1.0 * intensity
    elif scenario_type == "high_vol_persistence":
        persistence = _fallback(params.get("persistence_length_bars"), 30.0)
        stress_return -= 0.015 * intensity
        stress_sharpe -= 0.20 * intensity
        stress_drawdown -= 0.06 * intensity
        turnover += 0.05 * intensity
        state_changes += 4.0 * intensity
        fallback_count += 2.0 * intensity
        if is_defensive:
            stress_drawdown += 0.03 * intensity
            stress_sharpe += 0.06 * intensity
            stress_return += 0.01 * intensity
        if persistence > 30.0:
            stress_drawdown -= 0.01 * intensity
    elif scenario_type == "classifier_uncertainty":
        confidence_multiplier = _fallback(params.get("confidence_multiplier"), 0.70)
        entropy_multiplier = _fallback(params.get("entropy_multiplier"), 1.25)
        confidence = base_confidence * confidence_multiplier
        entropy = base_entropy * entropy_multiplier
        stress_return -= 0.01 * intensity
        stress_sharpe -= 0.15 * intensity
        turnover += 0.08 * intensity
        state_changes += 5.0 * intensity
        fallback_count += 3.0 * intensity
    elif scenario_type == "taxonomy_ml_disagreement":
        disagreement_rate = _fallback(params.get("disagreement_rate"), 0.25)
        confidence -= 0.08 * intensity
        entropy += 0.20 * intensity
        stress_return -= 0.012 * intensity
        stress_sharpe -= 0.12 * intensity
        stress_drawdown -= 0.02 * intensity
        turnover += 0.10 * intensity
        state_changes += 7.0 * intensity
        fallback_count += 5.0 * intensity
    elif scenario_type == "confidence_collapse":
        confidence_override = _fallback(params.get("confidence_floor_override"), 0.25)
        confidence = min(confidence, confidence_override)
        entropy += 0.35 * intensity
        stress_return -= 0.025 * intensity
        stress_sharpe -= 0.18 * intensity
        stress_drawdown -= 0.05 * intensity
        turnover += 0.09 * intensity
        state_changes += 8.0 * intensity
        fallback_count += 8.0 * intensity

    if confidence is not None:
        confidence = min(max(confidence, 0.0), 1.0)
    if entropy is not None:
        entropy = max(entropy, 0.0)

    turnover = min(max(turnover, 0.0), 1.0)
    state_changes = max(state_changes, 0.0)
    fallback_count = max(fallback_count, 0.0)
    fallback_rate = min(max(fallback_count / observation_count, 0.0), 1.0) if observation_count > 0 else None

    return {
        "stress_return": _round(stress_return),
        "stress_sharpe": _round(stress_sharpe),
        "stress_max_drawdown": _round(stress_drawdown),
        "policy_turnover": _round(turnover),
        "policy_state_change_count": _round(state_changes),
        "fallback_activation_count": _round(fallback_count),
        "fallback_activation_rate": _round(fallback_rate) if fallback_rate is not None else None,
        "mean_confidence_under_stress": _round(confidence) if confidence is not None else None,
        "mean_entropy_under_stress": _round(entropy) if entropy is not None else None,
        "disagreement_rate": _round(disagreement_rate) if disagreement_rate is not None else None,
    }


def _build_matrix_row(
    *,
    policy: Mapping[str, Any],
    scenario: Mapping[str, Any],
    stressed: Mapping[str, float | None],
    baseline_metrics: Mapping[str, float | None],
    baseline_policy: str,
    transition_window_bars: int,
) -> dict[str, Any]:
    return {
        "scenario_id": scenario["scenario_id"],
        "scenario_name": scenario["name"],
        "scenario_type": scenario["type"],
        "policy_name": policy["policy_name"],
        "baseline_policy": baseline_policy,
        "policy_type": policy["policy_type"],
        "regime_source": policy["regime_source"],
        "classifier_model": policy["classifier_model"],
        "calibration_profile": policy["calibration_profile"],
        "transition_window_bars": transition_window_bars,
        "stress_intensity": _float_or_none(scenario.get("stress_intensity")) or 1.0,
        "mean_confidence_under_stress": stressed.get("mean_confidence_under_stress"),
        "mean_entropy_under_stress": stressed.get("mean_entropy_under_stress"),
        "disagreement_rate": stressed.get("disagreement_rate"),
        "fallback_activation_count": stressed.get("fallback_activation_count"),
        "fallback_activation_rate": stressed.get("fallback_activation_rate"),
        "policy_state_change_count": stressed.get("policy_state_change_count"),
        "policy_turnover": stressed.get("policy_turnover"),
        "stress_return": stressed.get("stress_return"),
        "stress_sharpe": stressed.get("stress_sharpe"),
        "stress_max_drawdown": stressed.get("stress_max_drawdown"),
        "adaptive_vs_static_return_delta": _delta(stressed.get("stress_return"), baseline_metrics.get("stress_return")),
        "adaptive_vs_static_sharpe_delta": _delta(stressed.get("stress_sharpe"), baseline_metrics.get("stress_sharpe")),
        "adaptive_vs_static_drawdown_delta": _delta(
            stressed.get("stress_max_drawdown"), baseline_metrics.get("stress_max_drawdown")
        ),
        "stress_passed": True,
        "primary_failure_reason": "",
    }


def _evaluate_stress_gates(
    row: Mapping[str, Any],
    *,
    config: RegimePolicyStressTestConfig,
) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []

    checks = [
        (
            "max_policy_turnover",
            config.stress_gates.max_policy_turnover,
            row.get("policy_turnover"),
            lambda value, threshold: value <= threshold,
            "policy_turnover exceeds max_policy_turnover",
        ),
        (
            "max_stress_drawdown",
            config.stress_gates.max_stress_drawdown,
            row.get("stress_max_drawdown"),
            lambda value, threshold: value >= threshold,
            "stress_max_drawdown breaches max_stress_drawdown",
        ),
        (
            "min_adaptive_vs_static_drawdown_delta",
            config.stress_gates.min_adaptive_vs_static_drawdown_delta,
            row.get("adaptive_vs_static_drawdown_delta"),
            lambda value, threshold: value >= threshold,
            "adaptive_vs_static_drawdown_delta is below configured minimum",
        ),
        (
            "max_fallback_activation_rate",
            config.stress_gates.max_fallback_activation_rate,
            row.get("fallback_activation_rate"),
            lambda value, threshold: value <= threshold,
            "fallback_activation_rate exceeds configured maximum",
        ),
        (
            "max_state_change_count",
            config.stress_gates.max_state_change_count,
            row.get("policy_state_change_count"),
            lambda value, threshold: value <= threshold,
            "policy_state_change_count exceeds configured maximum",
        ),
    ]

    for gate_name, threshold, value, predicate, failure_message in checks:
        if threshold is None:
            continue
        numeric = _float_or_none(value)
        if numeric is None:
            failures.append(f"missing_required_metric:{gate_name}")
            warnings.append(
                f"Required gate metric for {gate_name} was missing on scenario {row.get('scenario_name')} policy {row.get('policy_name')}."
            )
            continue
        if not predicate(numeric, threshold):
            failures.append(f"{gate_name}:{failure_message}")

    return {
        "stress_passed": not failures,
        "primary_failure_reason": failures[0] if failures else "",
        "warnings": warnings,
    }


def _build_stress_leaderboard(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_policy: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_policy.setdefault(str(row["policy_name"]), []).append(row)

    leaderboard: list[dict[str, Any]] = []
    for policy_name, policy_rows in sorted(by_policy.items()):
        policy_type = str(policy_rows[0].get("policy_type") or "unknown")
        scenario_count = len(policy_rows)
        passed_count = sum(1 for row in policy_rows if bool(row.get("stress_passed")))
        stress_pass_rate = passed_count / scenario_count if scenario_count > 0 else 0.0
        mean_return = _mean(policy_rows, "stress_return")
        mean_sharpe = _mean(policy_rows, "stress_sharpe")
        mean_drawdown = _mean(policy_rows, "stress_max_drawdown")
        mean_turnover = _mean(policy_rows, "policy_turnover")
        mean_fallback_rate = _mean(policy_rows, "fallback_activation_rate")
        mean_return_delta = _mean(policy_rows, "adaptive_vs_static_return_delta")
        mean_sharpe_delta = _mean(policy_rows, "adaptive_vs_static_sharpe_delta")
        mean_drawdown_delta = _mean(policy_rows, "adaptive_vs_static_drawdown_delta")
        worst_scenario_type = _worst_scenario_type(policy_rows)

        primary_strength = "Stable stress-pass behavior across scenarios"
        if mean_drawdown_delta is not None and mean_drawdown_delta > 0:
            primary_strength = "Lower drawdown than static baseline under stress"
        elif mean_sharpe_delta is not None and mean_sharpe_delta > 0:
            primary_strength = "Higher Sharpe than static baseline under stress"

        primary_weakness = "No dominant weakness observed"
        if mean_turnover is not None and mean_turnover > 0.5:
            primary_weakness = "Elevated policy turnover under stress"
        elif mean_fallback_rate is not None and mean_fallback_rate > 0.5:
            primary_weakness = "High fallback activation during stressed periods"
        elif mean_drawdown is not None and mean_drawdown < -0.18:
            primary_weakness = "Severe drawdown during adverse stress scenarios"

        leaderboard.append(
            canonicalize_value(
                {
                    "rank": None,
                    "policy_name": policy_name,
                    "policy_type": policy_type,
                    "scenario_count": scenario_count,
                    "stress_pass_rate": _round(stress_pass_rate),
                    "mean_stress_return": _round(mean_return),
                    "mean_stress_sharpe": _round(mean_sharpe),
                    "mean_stress_max_drawdown": _round(mean_drawdown),
                    "mean_policy_turnover": _round(mean_turnover),
                    "mean_fallback_activation_rate": _round(mean_fallback_rate),
                    "mean_adaptive_vs_static_return_delta": _round(mean_return_delta),
                    "mean_adaptive_vs_static_sharpe_delta": _round(mean_sharpe_delta),
                    "mean_adaptive_vs_static_drawdown_delta": _round(mean_drawdown_delta),
                    "worst_scenario_type": worst_scenario_type,
                    "primary_strength": primary_strength,
                    "primary_weakness": primary_weakness,
                }
            )
        )

    leaderboard.sort(
        key=lambda row: (
            -float(row.get("stress_pass_rate") or 0.0),
            -(float(row.get("mean_adaptive_vs_static_drawdown_delta") or -1e9)),
            -(float(row.get("mean_stress_sharpe") or -1e9)),
            str(row.get("policy_name")),
        )
    )
    for index, row in enumerate(leaderboard, start=1):
        row["rank"] = index
    return leaderboard


def _build_scenario_summary(
    *,
    stress_run_id: str,
    config: RegimePolicyStressTestConfig,
    rows: list[dict[str, Any]],
    baseline_policy: str,
    leaderboard_rows: list[dict[str, Any]],
    warnings: list[str],
    source_review_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    scenario_types = sorted({str(row["scenario_type"]) for row in rows})
    policy_names = sorted({str(row["policy_name"]) for row in rows})
    by_scenario_id: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_scenario_id.setdefault(str(row["scenario_id"]), []).append(row)

    scenario_pass_counts = {
        scenario_id: sum(1 for row in group if bool(row.get("stress_passed")))
        for scenario_id, group in sorted(by_scenario_id.items())
    }
    scenario_failure_counts = {
        scenario_id: sum(1 for row in group if not bool(row.get("stress_passed")))
        for scenario_id, group in sorted(by_scenario_id.items())
    }

    worst_scenario_type = _worst_scenario_type(rows)
    baseline_rank = None
    for item in leaderboard_rows:
        if str(item.get("policy_name")) == baseline_policy:
            baseline_rank = int(item.get("rank")) if item.get("rank") is not None else None
            break

    adaptive_policy_count = len({
        str(row["policy_name"])
        for row in rows
        if str(row.get("policy_name")) != baseline_policy
    })

    limitations = sorted(set(warnings + [
        "Derived stress transforms are deterministic approximations, not market simulations.",
        "Deterministic stress transforms are fixture-oriented approximations and not full market simulations.",
        "Registry-backed candidate sourcing is reserved for a future expansion.",
    ]))

    return canonicalize_value(
        {
            "stress_run_id": stress_run_id,
            "stress_test_name": config.stress_test_name,
            "scenario_count": len(by_scenario_id),
            "policy_count": len(policy_names),
            "baseline_policy": baseline_policy,
            "scenario_types": scenario_types,
            "generated_scenarios": sorted(by_scenario_id),
            "scenario_pass_counts": scenario_pass_counts,
            "scenario_failure_counts": scenario_failure_counts,
            "worst_scenario_type": worst_scenario_type,
            "baseline_included_in_leaderboard": baseline_rank is not None,
            "baseline_rank": baseline_rank,
            "adaptive_policy_count": adaptive_policy_count,
            "provenance": {
                "source_review_run_id": source_review_metadata.get("review_run_id"),
                "source_benchmark_run_id": source_review_metadata.get("source_benchmark_run_id"),
                "source_candidate_selection": _path_or_none(config.source_candidate_selection),
            },
            "limitations": limitations,
        }
    )


def _build_policy_stress_summary(
    *,
    stress_run_id: str,
    config: RegimePolicyStressTestConfig,
    rows: list[dict[str, Any]],
    baseline_policy: str,
    leaderboard_rows: list[dict[str, Any]],
    source_review_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    by_policy: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_policy.setdefault(str(row["policy_name"]), []).append(row)

    policy_results: list[dict[str, Any]] = []
    for policy_name, policy_rows in sorted(by_policy.items()):
        scenario_count = len(policy_rows)
        stress_pass_rate = sum(1 for row in policy_rows if bool(row.get("stress_passed"))) / scenario_count
        mean_drawdown_delta = _mean(policy_rows, "adaptive_vs_static_drawdown_delta")
        mean_turnover = _mean(policy_rows, "policy_turnover")
        primary_strength = "Balanced behavior across stress scenarios"
        if mean_drawdown_delta is not None and mean_drawdown_delta > 0.0:
            primary_strength = "Lower drawdown during stressed regimes"

        primary_weakness = "No severe weakness identified"
        if mean_turnover is not None and mean_turnover > 0.45:
            primary_weakness = "Turnover increases under unstable regime evidence"
        elif _mean(policy_rows, "mean_entropy_under_stress") not in {None, 0.0} and (
            _mean(policy_rows, "mean_entropy_under_stress") or 0.0
        ) > 1.15:
            primary_weakness = "Entropy sensitivity during classifier uncertainty"

        policy_results.append(
            canonicalize_value(
                {
                    "policy_name": policy_name,
                    "stress_pass_rate": _round(stress_pass_rate),
                    "mean_adaptive_vs_static_drawdown_delta": _round(mean_drawdown_delta),
                    "mean_policy_turnover": _round(mean_turnover),
                    "primary_strength": primary_strength,
                    "primary_weakness": primary_weakness,
                }
            )
        )

    most_resilient = None
    if policy_results:
        most_resilient = sorted(
            policy_results,
            key=lambda item: (
                -float(item.get("stress_pass_rate") or 0.0),
                -(float(item.get("mean_adaptive_vs_static_drawdown_delta") or -1e9)),
                str(item.get("policy_name")),
            ),
        )[0]["policy_name"]

    highest_turnover_policy = None
    if policy_results:
        highest_turnover_policy = sorted(
            policy_results,
            key=lambda item: (
                -(float(item.get("mean_policy_turnover") or 0.0)),
                str(item.get("policy_name")),
            ),
        )[0]["policy_name"]

    most_resilient_adaptive_policy = None
    adaptive_only = [
        item for item in policy_results if str(item.get("policy_name")) != baseline_policy
    ]
    if adaptive_only:
        most_resilient_adaptive_policy = sorted(
            adaptive_only,
            key=lambda item: (
                -float(item.get("stress_pass_rate") or 0.0),
                -(float(item.get("mean_adaptive_vs_static_drawdown_delta") or -1e9)),
                str(item.get("policy_name")),
            ),
        )[0]["policy_name"]

    baseline_rank = None
    for item in leaderboard_rows:
        if str(item.get("policy_name")) == baseline_policy:
            baseline_rank = int(item.get("rank")) if item.get("rank") is not None else None
            break

    return canonicalize_value(
        {
            "stress_run_id": stress_run_id,
            "source_review_run_id": source_review_metadata.get("review_run_id"),
            "scenario_count": len({str(row["scenario_id"]) for row in rows}),
            "policy_count": len(policy_results),
            "baseline_policy": baseline_policy,
            "baseline_rank": baseline_rank,
            "adaptive_policy_count": len(adaptive_only),
            "baseline_included_in_leaderboard": baseline_rank is not None,
            "summary": {
                "stress_passed_count": sum(1 for row in rows if bool(row.get("stress_passed"))),
                "stress_failed_count": sum(1 for row in rows if not bool(row.get("stress_passed"))),
            },
            "policy_results": policy_results,
            "most_resilient_policy": most_resilient,
            "most_resilient_adaptive_policy": most_resilient_adaptive_policy,
            "highest_turnover_policy": highest_turnover_policy,
            "worst_scenario_type": _worst_scenario_type(rows),
            "provenance": {
                "source_review_run_id": source_review_metadata.get("review_run_id"),
                "source_benchmark_run_id": source_review_metadata.get("source_benchmark_run_id"),
                "source_candidate_selection": _path_or_none(config.source_candidate_selection),
            },
        }
    )


def _build_scenario_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {column: row.get(column) for column in SCENARIO_RESULT_COLUMNS}
        for row in rows
    ]


def _build_fallback_usage_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for row in rows:
        warning = ""
        if _float_or_none(row.get("mean_confidence_under_stress")) is None:
            warning = "mean_confidence_under_stress unavailable"
        payload.append(
            canonicalize_value(
                {
                    "scenario_id": row.get("scenario_id"),
                    "scenario_name": row.get("scenario_name"),
                    "scenario_type": row.get("scenario_type"),
                    "policy_name": row.get("policy_name"),
                    "fallback_activation_count": row.get("fallback_activation_count"),
                    "fallback_activation_rate": row.get("fallback_activation_rate"),
                    "mean_confidence_under_stress": row.get("mean_confidence_under_stress"),
                    "mean_entropy_under_stress": row.get("mean_entropy_under_stress"),
                    "selected_fallback_behavior": "baseline_fallback"
                    if (_float_or_none(row.get("fallback_activation_count")) or 0.0) > 0.0
                    else "none",
                    "warning": warning,
                }
            )
        )
    return payload


def _build_turnover_rows(
    rows: list[dict[str, Any]],
    *,
    config: RegimePolicyStressTestConfig,
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for row in rows:
        turnover = _float_or_none(row.get("policy_turnover"))
        state_changes = _float_or_none(row.get("policy_state_change_count"))
        max_turnover = config.stress_gates.max_policy_turnover
        max_state_changes = config.stress_gates.max_state_change_count
        payload.append(
            canonicalize_value(
                {
                    "scenario_id": row.get("scenario_id"),
                    "scenario_name": row.get("scenario_name"),
                    "scenario_type": row.get("scenario_type"),
                    "policy_name": row.get("policy_name"),
                    "policy_turnover": turnover,
                    "policy_state_change_count": state_changes,
                    "max_policy_turnover": max_turnover,
                    "max_state_change_count": max_state_changes,
                    "turnover_gate_passed": True
                    if max_turnover is None
                    else (turnover is not None and turnover <= max_turnover),
                    "state_change_gate_passed": True
                    if max_state_changes is None
                    else (state_changes is not None and state_changes <= max_state_changes),
                }
            )
        )
    return payload


def _build_adaptive_vs_static_rows(rows: list[dict[str, Any]], *, baseline_policy: str) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for row in rows:
        dd_delta = _float_or_none(row.get("adaptive_vs_static_drawdown_delta"))
        comparison_reason = "No comparison available"
        if dd_delta is not None:
            comparison_reason = (
                "Adaptive policy reduced drawdown vs static baseline"
                if dd_delta > 0
                else "Adaptive policy worsened drawdown vs static baseline"
                if dd_delta < 0
                else "Adaptive and static drawdown were unchanged"
            )
        payload.append(
            canonicalize_value(
                {
                    "scenario_id": row.get("scenario_id"),
                    "scenario_name": row.get("scenario_name"),
                    "scenario_type": row.get("scenario_type"),
                    "policy_name": row.get("policy_name"),
                    "baseline_policy": baseline_policy,
                    "adaptive_vs_static_return_delta": row.get("adaptive_vs_static_return_delta"),
                    "adaptive_vs_static_sharpe_delta": row.get("adaptive_vs_static_sharpe_delta"),
                    "adaptive_vs_static_drawdown_delta": row.get("adaptive_vs_static_drawdown_delta"),
                    "comparison_reason": comparison_reason,
                }
            )
        )
    return payload


def _build_manifest(
    *,
    stress_run_id: str,
    config: RegimePolicyStressTestConfig,
    output_dir: Path,
    baseline_policy: str,
    rows: list[dict[str, Any]],
    warnings: list[str],
    generated_files: list[str],
    source_review_metadata: Mapping[str, Any],
    source_policy_candidates_path: Path,
    scenario_catalog: list[dict[str, Any]],
    market_simulation_stress_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    pass_count = sum(1 for row in rows if bool(row.get("stress_passed")))
    fail_count = len(rows) - pass_count
    file_inventory = {
        name: {
            "path": _rel(output_dir / name),
            "row_count": _row_count(output_dir / name),
        }
        for name in generated_files
    }
    row_counts = {
        "stress_matrix": len(rows),
        "scenario_results": len(rows),
        "fallback_usage": len(rows),
        "policy_turnover_under_stress": len(rows),
        "adaptive_vs_static_stress_comparison": len(rows),
        "stress_leaderboard": len({str(row['policy_name']) for row in rows}),
    }
    source_paths = {
        "source_review_pack": _path_or_none(config.source_review_pack),
        "policy_metrics_path": _rel(source_policy_candidates_path),
        "source_candidate_selection": _path_or_none(config.source_candidate_selection),
    }
    if market_simulation_stress_summary is not None:
        row_counts["market_simulation_stress_leaderboard"] = int(
            market_simulation_stress_summary.get("leaderboard_row_count", 0) or 0
        )
        source_paths["market_simulation_metrics_dir"] = _path_or_none(
            config.market_simulation_stress.simulation_metrics_dir
        )
        source_paths["market_simulation_config_path"] = _path_or_none(
            config.market_simulation_stress.config_path
        )

    payload = {
            "artifact_type": "regime_policy_stress_tests",
            "schema_version": "1.0",
            "stress_run_id": stress_run_id,
            "stress_test_name": config.stress_test_name,
            "source_review_run_id": source_review_metadata.get("review_run_id"),
            "source_benchmark_run_id": source_review_metadata.get("source_benchmark_run_id"),
            "baseline_policy": baseline_policy,
            "source_paths": source_paths,
            "generated_files": generated_files,
            "row_counts": row_counts,
            "scenario_count": len(scenario_catalog),
            "policy_count": len({str(row['policy_name']) for row in rows}),
            "stress_pass_count": pass_count,
            "stress_fail_count": fail_count,
            "warning_count": len(sorted(set(warnings))),
            "limitations": sorted(
                set(
                    warnings
                    + [
                        "Derived stress transforms are deterministic approximations, not market simulations.",
                        "Deterministic stress transforms are not full market simulations.",
                        "Inputs are file-backed in this issue implementation.",
                    ]
                )
            ),
            "file_inventory": file_inventory,
        }
    if market_simulation_stress_summary is not None:
        payload["market_simulation_stress"] = market_simulation_stress_summary
    return canonicalize_value(payload)


def _build_stress_run_id(
    *,
    config: RegimePolicyStressTestConfig,
    source_review_run_id: str,
    policy_names: list[str],
    scenario_ids: list[str],
) -> str:
    payload = {
        "config": config.to_dict(),
        "source_review_run_id": source_review_run_id,
        "policy_names": sorted(policy_names),
        "scenario_ids": sorted(scenario_ids),
    }
    digest = hashlib.sha256(serialize_canonical_json(payload).encode("utf-8")).hexdigest()[:12]
    return f"{config.stress_test_name}_{digest}"


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Policy metrics input does not exist: {path.as_posix()}")
    if path.suffix.lower() == ".json":
        payload = _load_json(path)
        rows = payload.get("rows") if isinstance(payload, Mapping) else None
        if not isinstance(rows, list):
            raise ValueError(f"Policy metrics JSON must contain a rows array: {path.as_posix()}")
        return [dict(row) for row in rows if isinstance(row, Mapping)]
    frame = pd.read_csv(path, keep_default_na=True)
    return [
        {str(key): (None if _missing(value) else value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path.as_posix()}")
    return payload


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(canonicalize_value(payload), indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )


def _row_count(path: Path) -> int | None:
    if not path.exists():
        return None
    if path.suffix.lower() == ".csv":
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            return 0
        return max(text.count("\n") - 1, 0)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
            return len(payload["rows"])
    return None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    if isinstance(value, int):
        return bool(value)
    return False


def _delta(value: Any, baseline: Any) -> float | None:
    numeric = _float_or_none(value)
    baseline_numeric = _float_or_none(baseline)
    if numeric is None or baseline_numeric is None:
        return None
    return _round(numeric - baseline_numeric)


def _mean(rows: list[Mapping[str, Any]], key: str) -> float | None:
    values = [float(value) for value in (_float_or_none(row.get(key)) for row in rows) if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _worst_scenario_type(rows: list[Mapping[str, Any]]) -> str | None:
    by_type: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_type.setdefault(str(row.get("scenario_type")), []).append(row)
    if not by_type:
        return None
    ranked = sorted(
        by_type.items(),
        key=lambda item: (
            sum(1 for row in item[1] if bool(row.get("stress_passed"))) / len(item[1]),
            _mean(item[1], "stress_max_drawdown") or 0.0,
            item[0],
        ),
    )
    return ranked[0][0]


def _fallback(value: Any, default: float) -> float:
    numeric = _float_or_none(value)
    return default if numeric is None else numeric


def _float_or_none(value: Any) -> float | None:
    if _missing(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return float(number)


def _round(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 12)


def _missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _path_or_none(path_value: str | None) -> str | None:
    if path_value is None:
        return None
    return Path(path_value).as_posix()


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


__all__ = [
    "RegimePolicyStressTestRunResult",
    "run_regime_policy_stress_tests",
]
