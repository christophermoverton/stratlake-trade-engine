from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys

import pandas as pd


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
    "policy_variant_comparison.csv",
    "final_interpretation.md",
    "evidence_index.json",
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


def test_full_year_regime_policy_case_study_runs_and_writes_expected_outputs(tmp_path: Path) -> None:
    module = _load_example_module()

    artifacts = module.run_case_study(output_root=tmp_path / "case_study", verbose=False)

    for name in EXPECTED_OUTPUT_FILES:
        path = artifacts.output_root / name
        assert path.exists(), f"Expected artifact missing: {path}"

    summary = _read_json(artifacts.summary_path)
    assert summary["mode"] == "fixture_backed"
    assert summary["policy_variants"] == EXPECTED_POLICY_ROLES
    assert set(summary["source_runs"]) == {
        "benchmark_run_id",
        "promotion_gate_run_id",
        "review_run_id",
        "candidate_selection_run_id",
        "stress_run_id",
    }


def test_full_year_regime_policy_case_study_manifest_and_interpretation_have_key_sections(tmp_path: Path) -> None:
    module = _load_example_module()

    artifacts = module.run_case_study(output_root=tmp_path / "case_study_sections", verbose=False)

    manifest = _read_json(artifacts.manifest_path)
    interpretation = artifacts.final_interpretation_path.read_text(encoding="utf-8")

    assert manifest["artifact_type"] == "full_year_regime_policy_benchmark_case_study"
    assert "generated_files" in manifest and manifest["generated_files"]
    assert "source_artifacts" in manifest and manifest["source_artifacts"]
    assert set(EXPECTED_OUTPUT_FILES).issubset(set(manifest["generated_files"]))

    assert "## Observed Benchmark Evidence" in interpretation
    assert "## Promotion And Review Governance Evidence" in interpretation
    assert "## Candidate-Selection Evidence" in interpretation
    assert "## Deterministic Synthetic Stress Evidence" in interpretation
    assert "## Reproduction Command" in interpretation


def test_full_year_regime_policy_case_study_comparison_and_stress_fields(tmp_path: Path) -> None:
    module = _load_example_module()

    artifacts = module.run_case_study(output_root=tmp_path / "case_study_comparison", verbose=False)

    comparison = pd.read_csv(artifacts.output_root / "policy_variant_comparison.csv")
    stress_summary = _read_json(artifacts.output_root / "stress_summary.json")

    assert comparison["policy_role"].tolist() == EXPECTED_POLICY_ROLES
    assert {
        "most_resilient_policy",
        "most_resilient_adaptive_policy",
        "baseline_rank",
        "worst_scenario_type",
    }.issubset(set(stress_summary))
    if stress_summary.get("adaptive_policy_count", 0) > 0:
        assert stress_summary.get("most_resilient_adaptive_policy") is not None


def test_full_year_regime_policy_case_study_is_deterministic_and_uses_relative_paths(tmp_path: Path) -> None:
    module = _load_example_module()

    output_root = tmp_path / "stable"
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

    for path_value in first_manifest["generated_files"]:
        assert not Path(path_value).is_absolute()
    for path_value in first_manifest["source_artifacts"].values():
        assert not Path(path_value).is_absolute()
    for path_value in first_manifest["source_configs"].values():
        assert not Path(path_value).is_absolute()
