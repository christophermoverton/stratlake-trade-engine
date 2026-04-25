from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys

import pandas as pd
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


def _real_2025_data_available(module, tmp_path: Path) -> tuple[bool, str]:
    probe_root = tmp_path / "coverage_probe"
    probe_root.mkdir(parents=True, exist_ok=True)
    try:
        module.validate_real_2025_data_coverage(
            output_root=probe_root,
            features_root=REPO_ROOT / "data",
        )
    except module.RealDataCoverageError:
        coverage_path = probe_root / "data_coverage_summary.json"
        if coverage_path.exists():
            coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
            reasons = coverage["coverage_status"]["failure_reasons"]
            return False, "; ".join(reasons) if reasons else "real 2025 coverage unavailable"
        return False, "real 2025 coverage unavailable"
    return True, ""


def _run_or_skip_real_e2e(module, tmp_path: Path):
    available, reason = _real_2025_data_available(module, tmp_path)
    if not available:
        pytest.skip(f"real full-year 2025 features_daily coverage not available: {reason}")
    return module.run_case_study(
        output_root=tmp_path / "full_year_case_study",
        features_root=REPO_ROOT / "data",
        verbose=False,
    )


def _assert_only_relative_inventory_paths(value: object) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "path" and isinstance(item, str):
                assert not Path(item).is_absolute(), f"Expected relative artifact path, found {item!r}"
            else:
                _assert_only_relative_inventory_paths(item)
    elif isinstance(value, list):
        for item in value:
            _assert_only_relative_inventory_paths(item)


def test_discover_2025_symbols_accepts_arbitrary_parquet_filenames(tmp_path: Path) -> None:
    module = _load_example_module()
    dataset_root = tmp_path / "curated" / "features_daily"

    alpha_dir = dataset_root / "symbol=AAA" / "year=2025"
    alpha_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"symbol": ["AAA"], "date": ["2025-01-02"]}).to_parquet(alpha_dir / "chunk-0007.parquet", index=False)

    beta_dir = dataset_root / "symbol=BBB" / "year=2025" / "month=03"
    beta_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"symbol": ["BBB"], "date": ["2025-03-03"]}).to_parquet(beta_dir / "custom-name.parquet", index=False)

    detected = module._discover_2025_symbols(dataset_root)
    assert detected == ["AAA", "BBB"]


def test_discover_2025_symbols_ignores_empty_2025_directories(tmp_path: Path) -> None:
    module = _load_example_module()
    dataset_root = tmp_path / "curated" / "features_daily"

    (dataset_root / "symbol=AAA" / "year=2025").mkdir(parents=True, exist_ok=True)
    non_2025_dir = dataset_root / "symbol=BBB" / "year=2024"
    non_2025_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"symbol": ["BBB"], "date": ["2024-12-31"]}).to_parquet(non_2025_dir / "part-0.parquet", index=False)

    detected = module._discover_2025_symbols(dataset_root)
    assert detected == []


def test_full_year_regime_calibration_case_study_runs_and_writes_expected_outputs(tmp_path: Path) -> None:
    module = _load_example_module()
    artifacts = _run_or_skip_real_e2e(module, tmp_path)

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
    assert coverage["data_source"]["type"] == "downloaded_real_data"
    assert coverage["data_source"]["downloaded_real_data"] is True
    assert coverage["data_source"]["real_data_fixture"] is False
    assert coverage["data_source"]["real_data_used"] is True
    assert coverage["data_source"]["mock_or_synthetic_data"] is False
    assert coverage["data_source"]["allow_real_data_fixture"] is False
    assert coverage["data_source"]["fixture_mode_enabled"] is False
    assert coverage["coverage_status"]["passed"] is True


def test_full_year_regime_calibration_case_study_is_deterministic_and_portable(tmp_path: Path) -> None:
    module = _load_example_module()
    available, reason = _real_2025_data_available(module, tmp_path)
    if not available:
        pytest.skip(f"real full-year 2025 features_daily coverage not available: {reason}")

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
    portable_paths = (
        first.summary_path,
        first.data_coverage_path,
        first.interpretation_path,
        first.artifact_manifest_path,
        first.output_root / "gmm_confidence" / "regime_gmm_manifest.json",
        first.output_root / "adaptive_policy" / "adaptive_policy_manifest.json",
        first.output_root / "adaptive_policy" / "regime_policy_summary.json",
    )
    for path in portable_paths:
        text = path.read_text(encoding="utf-8")
        assert tmp_root_text not in text

    manifest = json.loads(first.artifact_manifest_path.read_text(encoding="utf-8"))
    for relative_path in manifest["file_inventory"]:
        assert not Path(relative_path).is_absolute()
    _assert_only_relative_inventory_paths(manifest)

    summary = json.loads(first.summary_path.read_text(encoding="utf-8"))
    coverage = json.loads(first.data_coverage_path.read_text(encoding="utf-8"))
    gmm_manifest = json.loads((first.output_root / "gmm_confidence" / "regime_gmm_manifest.json").read_text(encoding="utf-8"))
    adaptive_manifest = json.loads((first.output_root / "adaptive_policy" / "adaptive_policy_manifest.json").read_text(encoding="utf-8"))
    _assert_only_relative_inventory_paths(summary)
    _assert_only_relative_inventory_paths(coverage)
    _assert_only_relative_inventory_paths(gmm_manifest)
    _assert_only_relative_inventory_paths(adaptive_manifest)


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
    assert coverage["coverage_status"]["failure_reasons"]
    assert coverage["data_source"]["type"] == "downloaded_real_data"
    assert coverage["data_source"]["downloaded_real_data"] is True
    assert coverage["data_source"]["real_data_fixture"] is False
    assert coverage["data_source"]["real_data_used"] is True
    assert coverage["data_source"]["mock_or_synthetic_data"] is False
    assert coverage["data_source"]["fixture_mode_enabled"] is False


def test_full_year_regime_calibration_case_study_real_data_probe_reports_local_status(tmp_path: Path) -> None:
    module = _load_example_module()
    available, reason = _real_2025_data_available(module, tmp_path)

    if available:
        assert reason == ""
    else:
        assert reason


def test_full_year_regime_calibration_case_study_docs_are_linked() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    doc = (REPO_ROOT / "docs" / "examples" / "full_year_regime_calibration_case_study.md").read_text(encoding="utf-8")

    assert "docs/examples/full_year_regime_calibration_case_study.py" in readme
    assert "docs/examples/full_year_regime_calibration_case_study.md" in readme
    assert "data_coverage_summary.json" in doc
    assert "adaptive_vs_static_comparison.csv" in doc
    assert "downloaded real 2025" in doc.lower()
    assert "skip" in doc.lower()
    assert "arbitrary parquet" in doc.lower()
