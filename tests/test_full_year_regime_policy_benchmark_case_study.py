from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import shutil
import sys

import pandas as pd
import pytest

from tests.test_market_simulation_policy_stress_integration import _write_metrics_dir


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = REPO_ROOT / "docs" / "examples" / "full_year_regime_policy_benchmark_case_study.py"

EXPECTED_POLICY_ROLES = [
    "static_baseline",
    "taxonomy_only_policy",
    "calibrated_taxonomy_policy",
    "gmm_confidence_policy",
    "hybrid_calibrated_gmm_policy",
]

EXPECTED_OUTPUT_FILES = [
    "summary.json",
    "manifest.json",
    "benchmark_summary.json",
    "promotion_gate_summary.json",
    "review_summary.json",
    "candidate_selection_summary.json",
    "stress_summary.json",
    "market_simulation_stress_summary.json",
    "market_simulation_stress_leaderboard.csv",
    "policy_variant_comparison.csv",
    "final_interpretation.md",
    "evidence_index.json",
    "workflow_outputs.json",
]


def _load_example_module():
    spec = spec_from_file_location("full_year_regime_policy_benchmark_case_study", EXAMPLE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_canonical_m27_case_study_metrics(tmp_path: Path) -> Path:
    source_metrics_dir = _write_metrics_dir(tmp_path)
    output_root = tmp_path / "m27_market_simulation_case_study"
    target_metrics_dir = (
        output_root
        / "source_simulation_artifacts"
        / "sim_fixture_001"
        / "market_simulations"
        / "simulation_metrics"
    )
    target_metrics_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_metrics_dir, target_metrics_dir)
    return output_root


def _run_case_study_with_market_metrics(module, tmp_path: Path, name: str):
    module.M27_CASE_STUDY_OUTPUT_ROOT = _write_canonical_m27_case_study_metrics(tmp_path)
    return module.run_case_study(output_root=tmp_path / name, verbose=False)


def test_full_year_regime_policy_case_study_runs_and_writes_expected_outputs(tmp_path: Path) -> None:
    module = _load_example_module()

    artifacts = _run_case_study_with_market_metrics(module, tmp_path, "case_study")

    for name in EXPECTED_OUTPUT_FILES:
        path = artifacts.output_root / name
        assert path.exists(), f"Expected artifact missing: {path}"

    summary = _read_json(artifacts.summary_path)
    assert summary["mode"] == "fixture_backed"
    assert summary["policy_variants"] == EXPECTED_POLICY_ROLES
    assert set(summary["source_runs"]) == {
        "benchmark_run_id",
        "promotion_gate_run_id",
        "promotion_gate_source_benchmark_run_id",
        "promotion_gate_config_name",
        "review_run_id",
        "candidate_selection_run_id",
        "stress_run_id",
        "market_simulation_run_id",
    }
    assert summary["source_runs"]["promotion_gate_run_id"] == summary["source_runs"]["promotion_gate_source_benchmark_run_id"]


def test_full_year_regime_policy_case_study_manifest_and_interpretation_have_key_sections(tmp_path: Path) -> None:
    module = _load_example_module()

    artifacts = _run_case_study_with_market_metrics(module, tmp_path, "case_study_sections")

    manifest = _read_json(artifacts.manifest_path)
    interpretation = artifacts.final_interpretation_path.read_text(encoding="utf-8")

    assert manifest["artifact_type"] == "full_year_regime_policy_benchmark_case_study"
    assert "generated_files" in manifest and manifest["generated_files"]
    assert "source_artifacts" in manifest and manifest["source_artifacts"]
    assert set(EXPECTED_OUTPUT_FILES).issubset(set(manifest["generated_files"]))

    workflow_outputs = _read_json(artifacts.output_root / "workflow_outputs.json")
    assert {
        "benchmark_artifact_dir",
        "promotion_gate_artifact_dir",
        "review_artifact_dir",
        "candidate_selection_artifact_dir",
        "stress_artifact_dir",
        "source_runs",
    }.issubset(set(workflow_outputs))

    assert "## Observed Benchmark Evidence" in interpretation
    assert "## Promotion And Review Governance Evidence" in interpretation
    assert "## Candidate-Selection Evidence" in interpretation
    assert "## Deterministic Synthetic Stress Evidence" in interpretation
    assert "## Market Simulation Stress Evidence" in interpretation
    assert "## Reproduction Command" in interpretation


def test_full_year_regime_policy_case_study_comparison_and_stress_fields(tmp_path: Path) -> None:
    module = _load_example_module()

    artifacts = _run_case_study_with_market_metrics(module, tmp_path, "case_study_comparison")

    comparison = pd.read_csv(artifacts.output_root / "policy_variant_comparison.csv")
    stress_summary = _read_json(artifacts.output_root / "stress_summary.json")
    market_simulation_summary = _read_json(artifacts.output_root / "market_simulation_stress_summary.json")

    assert comparison["policy_role"].tolist() == EXPECTED_POLICY_ROLES
    assert {
        "most_resilient_policy",
        "most_resilient_adaptive_policy",
        "baseline_rank",
        "worst_scenario_type",
    }.issubset(set(stress_summary))
    if stress_summary.get("adaptive_policy_count", 0) > 0:
        assert stress_summary.get("most_resilient_adaptive_policy") is not None
    assert market_simulation_summary["market_simulation_enabled"] is True
    assert market_simulation_summary["market_simulation_available"] is True
    assert market_simulation_summary["regime_only_monte_carlo_note"] == (
        "Monte Carlo paths are regime-only and do not fabricate return or policy metrics."
    )

    comparison_source_paths = comparison["source_artifact_path"].dropna().astype("string").tolist()
    for value in comparison_source_paths:
        assert not Path(value).is_absolute()


def test_full_year_regime_policy_case_study_evidence_index_has_all_workflow_families(tmp_path: Path) -> None:
    module = _load_example_module()

    artifacts = _run_case_study_with_market_metrics(module, tmp_path, "case_study_evidence")

    evidence_index = _read_json(artifacts.output_root / "evidence_index.json")
    assert "source_artifacts" in evidence_index
    assert {
        "benchmark",
        "promotion_gates",
        "review",
        "candidate_selection",
        "stress",
        "market_simulation_stress",
    }.issubset(set(evidence_index["source_artifacts"]))
    assert "generated_case_study_artifacts" in evidence_index
    assert {
        "summary",
        "manifest",
        "policy_variant_comparison",
        "final_interpretation",
        "workflow_outputs",
        "market_simulation_stress_summary",
        "market_simulation_stress_leaderboard",
    }.issubset(set(evidence_index["generated_case_study_artifacts"]))


def test_full_year_regime_policy_case_study_is_deterministic_and_uses_relative_paths(tmp_path: Path) -> None:
    module = _load_example_module()

    output_root = tmp_path / "stable"
    module.M27_CASE_STUDY_OUTPUT_ROOT = _write_canonical_m27_case_study_metrics(tmp_path)
    first = module.run_case_study(output_root=output_root, verbose=False)
    second = module.run_case_study(output_root=output_root, verbose=False)

    first_summary = _read_json(first.summary_path)
    second_summary = _read_json(second.summary_path)
    first_manifest = _read_json(first.manifest_path)
    second_manifest = _read_json(second.manifest_path)

    assert first_summary == second_summary
    assert first_manifest == second_manifest
    assert (first.output_root / "policy_variant_comparison.csv").read_text(encoding="utf-8") == (
        second.output_root / "policy_variant_comparison.csv"
    ).read_text(encoding="utf-8")

    temp_root_text = tmp_path.as_posix()
    assert temp_root_text not in first.summary_path.read_text(encoding="utf-8")
    assert temp_root_text not in first.manifest_path.read_text(encoding="utf-8")
    assert temp_root_text not in (first.output_root / "policy_variant_comparison.csv").read_text(encoding="utf-8")

    for path_value in first_manifest["generated_files"]:
        assert not Path(path_value).is_absolute()
    for path_value in first_manifest["source_artifacts"].values():
        assert not Path(path_value).is_absolute()
    for path_value in first_manifest["source_configs"].values():
        assert not Path(path_value).is_absolute()


def test_portable_path_prefers_repo_relative_before_absolute(tmp_path: Path) -> None:
    module = _load_example_module()

    repo_file = REPO_ROOT / "README.md"
    result = module._portable_path(repo_file, output_root=tmp_path / "portable")

    assert not Path(result).is_absolute()
    assert result == "README.md"


def test_full_year_regime_policy_case_study_succeeds_without_m27_artifacts(tmp_path: Path) -> None:
    module = _load_example_module()
    module.M27_CASE_STUDY_OUTPUT_ROOT = tmp_path / "missing_m27_case_study"

    artifacts = module.run_case_study(output_root=tmp_path / "case_study_without_m27", verbose=False)

    summary = _read_json(artifacts.summary_path)
    market_summary = summary["market_simulation_stress_summary"]
    assert market_summary["market_simulation_enabled"] is False
    assert market_summary["market_simulation_available"] is False
    assert market_summary["market_simulation_mode"] == "not_available"
    assert market_summary["reason"] == "M27 market simulation metrics artifacts were not found."
    assert market_summary["path_metric_row_count"] == 0
    assert market_summary["summary_row_count"] == 0
    assert market_summary["leaderboard_row_count"] == 0
    assert market_summary["source_artifact_paths"] == {}
    assert market_summary["regime_only_monte_carlo_note"] == (
        "Monte Carlo paths are regime-only unless return or policy replay artifacts are explicitly available."
    )
    assert (artifacts.output_root / "market_simulation_stress_summary.json").exists()
    empty_leaderboard = pd.read_csv(artifacts.output_root / "market_simulation_stress_leaderboard.csv")
    assert empty_leaderboard.empty


def test_full_year_regime_policy_case_study_required_market_simulation_fails_when_absent(
    tmp_path: Path,
) -> None:
    module = _load_example_module()
    module.M27_CASE_STUDY_OUTPUT_ROOT = tmp_path / "missing_required_m27_case_study"

    with pytest.raises(FileNotFoundError, match="required but were not found"):
        module.run_case_study(
            output_root=tmp_path / "case_study_required_m27",
            verbose=False,
            require_market_simulation_stress=True,
        )
