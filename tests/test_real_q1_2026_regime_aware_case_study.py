from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = REPO_ROOT / "docs" / "examples" / "real_q1_2026_regime_aware_case_study.py"


def _load_example_module():
    spec = spec_from_file_location("real_q1_2026_regime_aware_case_study", EXAMPLE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_real_q1_2026_regime_aware_case_study_runs_and_writes_expected_outputs(tmp_path: Path) -> None:
    module = _load_example_module()

    artifacts = module.run_case_study(output_root=tmp_path / "real_q1_case_study", verbose=False)

    expected_paths = [
        artifacts.summary_path,
        artifacts.output_root / "final_interpretation.md",
        artifacts.regime_dir / "regime_labels.csv",
        artifacts.strategy_dir / "regime_attribution_report.md",
        artifacts.portfolio_dir / "regime_attribution_report.md",
        artifacts.output_root / "notebook_review" / "strategy_inventory.md",
        artifacts.output_root / "notebook_review" / "portfolio_inventory.md",
        artifacts.output_root / "notebook_review" / "alpha_comparison_summary.md",
    ]
    for path in expected_paths:
        assert path.exists(), f"Expected artifact missing: {path}"

    assert artifacts.summary["strategy"]["surface"] == "included"
    assert artifacts.summary["portfolio"]["surface"] == "included"
    assert artifacts.summary["alpha"]["surface"] == "included"
    assert len(artifacts.summary["alpha"]["runs"]) == 3


def test_real_q1_2026_regime_aware_case_study_is_deterministic_and_portable(tmp_path: Path) -> None:
    module = _load_example_module()

    first = module.run_case_study(output_root=tmp_path / "first", verbose=False)
    second = module.run_case_study(output_root=tmp_path / "second", verbose=False)

    assert first.summary == second.summary
    assert first.summary_path.read_text(encoding="utf-8") == second.summary_path.read_text(encoding="utf-8")

    tmp_root_text = tmp_path.as_posix()
    for path in (
        first.summary_path,
        first.output_root / "final_interpretation.md",
        first.strategy_dir / "regime_attribution_report.md",
        first.portfolio_dir / "regime_attribution_report.md",
    ):
        assert tmp_root_text not in path.read_text(encoding="utf-8")


def test_real_q1_2026_regime_aware_case_study_bundles_load_via_notebook_helpers(tmp_path: Path) -> None:
    module = _load_example_module()

    artifacts = module.run_case_study(output_root=tmp_path / "bundle_load", verbose=False)
    strategy_bundle = module.load_regime_review_bundle(artifacts.strategy_dir)
    portfolio_bundle = module.load_regime_review_bundle(artifacts.portfolio_dir)
    alpha_bundle = module.load_regime_review_bundle(artifacts.alpha_bundle_dirs[module.ALPHA_NAMES[0]])

    assert {"regime", "conditional", "transition", "attribution"}.issubset(set(strategy_bundle.available_sections()))
    assert {"regime", "conditional", "transition", "attribution"}.issubset(set(portfolio_bundle.available_sections()))
    assert {"regime", "conditional", "transition", "attribution", "comparison"}.issubset(
        set(alpha_bundle.available_sections())
    )


def test_real_q1_2026_regime_aware_case_study_docs_are_linked() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    doc = (REPO_ROOT / "docs" / "examples" / "real_q1_2026_regime_aware_case_study.md").read_text(encoding="utf-8")

    assert "docs/examples/real_q1_2026_regime_aware_case_study.py" in readme
    assert "docs/examples/real_q1_2026_regime_aware_case_study.md" in readme
    assert "load_regime_review_bundle" in doc
    assert "docs/examples/output/real_q1_2026_regime_aware_case_study/" in doc
