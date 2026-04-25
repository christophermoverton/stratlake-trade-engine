from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = REPO_ROOT / "docs" / "examples" / "full_year_regime_calibration_case_study.py"


def _load_example_module():
    spec = spec_from_file_location("full_year_regime_calibration_case_study", EXAMPLE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_full_year_regime_calibration_case_study_runs_and_writes_expected_outputs(tmp_path: Path) -> None:
    module = _load_example_module()

    artifacts = module.run_case_study(
        output_root=tmp_path / "full_year_case_study",
        features_root=REPO_ROOT / "data",
        verbose=False,
    )

    expected_paths = [
        artifacts.output_root,
        artifacts.data_coverage_path,
        artifacts.summary_path,
        artifacts.interpretation_path,
        artifacts.artifact_manifest_path,
        artifacts.output_root / "regime_bundle" / "regime_labels.csv",
        artifacts.output_root / "static_portfolio_bundle" / "regime_attribution_report.md",
        artifacts.output_root / "calibrated_portfolio_bundle" / "regime_attribution_report.md",
        artifacts.output_root / "calibration_profiles" / "baseline" / "regime_calibration.json",
        artifacts.output_root / "gmm_confidence" / "regime_gmm_manifest.json",
        artifacts.output_root / "adaptive_policy" / "adaptive_vs_static_comparison.csv",
        artifacts.output_root / "adaptive_policy" / "regime_policy_summary.json",
    ]
    for path in expected_paths:
        assert path.exists(), f"Expected artifact missing: {path}"

    summary = artifacts.summary
    assert {
        "case_study",
        "input_path",
        "data_coverage",
        "static_baseline",
        "classification",
        "raw_regime_evaluation",
        "calibration_profile_evaluation",
        "gmm_confidence",
        "calibrated_regime_evaluation",
        "adaptive_policy",
        "artifact_tree",
    }.issubset(summary)

    coverage = json.loads(artifacts.data_coverage_path.read_text(encoding="utf-8"))
    assert coverage["requested_window"]["start_date"] == "2025-01-01"
    assert coverage["requested_window"]["end_date"] == "2025-12-31"
    assert coverage["data_source"]["real_data_used"] is True
    assert coverage["data_source"]["mock_or_synthetic_data"] is False
    assert coverage["coverage_status"]["passed"] is True


def test_full_year_regime_calibration_case_study_is_deterministic_and_portable(tmp_path: Path) -> None:
    module = _load_example_module()

    first = module.run_case_study(
        output_root=tmp_path / "first",
        features_root=REPO_ROOT / "data",
        verbose=False,
    )
    second = module.run_case_study(
        output_root=tmp_path / "second",
        features_root=REPO_ROOT / "data",
        verbose=False,
    )

    assert first.summary == second.summary
    assert first.summary_path.read_text(encoding="utf-8") == second.summary_path.read_text(encoding="utf-8")
    assert first.data_coverage_path.read_text(encoding="utf-8") == second.data_coverage_path.read_text(encoding="utf-8")

    tmp_root_text = tmp_path.as_posix()
    for path in (
        first.summary_path,
        first.data_coverage_path,
        first.interpretation_path,
        first.artifact_manifest_path,
        first.output_root / "gmm_confidence" / "regime_gmm_manifest.json",
        first.output_root / "adaptive_policy" / "adaptive_policy_manifest.json",
    ):
        text = path.read_text(encoding="utf-8")
        assert tmp_root_text not in text

    manifest = json.loads(first.artifact_manifest_path.read_text(encoding="utf-8"))
    for relative_path in manifest["file_inventory"]:
        assert not Path(relative_path).is_absolute()


def test_full_year_regime_calibration_case_study_fails_fast_when_real_2025_data_is_missing(tmp_path: Path) -> None:
    module = _load_example_module()
    empty_root = tmp_path / "missing_data_root"
    empty_root.mkdir(parents=True, exist_ok=True)

    with pytest.raises(module.RealDataCoverageError):
        module.run_case_study(
            output_root=tmp_path / "missing_case",
            features_root=empty_root,
            verbose=False,
        )

    coverage_path = tmp_path / "missing_case" / "data_coverage_summary.json"
    assert coverage_path.exists()
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    assert coverage["coverage_status"]["passed"] is False
    assert coverage["data_source"]["mock_or_synthetic_data"] is False


def test_full_year_regime_calibration_case_study_docs_are_linked() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    doc = (REPO_ROOT / "docs" / "examples" / "full_year_regime_calibration_case_study.md").read_text(encoding="utf-8")

    assert "docs/examples/full_year_regime_calibration_case_study.py" in readme
    assert "docs/examples/full_year_regime_calibration_case_study.md" in readme
    assert "data_coverage_summary.json" in doc
    assert "adaptive_vs_static_comparison.csv" in doc
