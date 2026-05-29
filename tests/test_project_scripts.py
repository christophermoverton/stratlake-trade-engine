from __future__ import annotations

from importlib import import_module
from pathlib import Path
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"

EXPECTED_SCRIPT_TARGETS = {
    "stratlake-init-notebook": "src.cli.init_notebook_workspace:main",
    "stratlake-run-strategy": "src.cli.run_strategy:main",
    "stratlake-run-alpha": "src.cli.run_alpha:main",
    "stratlake-run-alpha-evaluation": "src.cli.run_alpha_evaluation:main",
    "stratlake-run-portfolio": "src.cli.run_portfolio:main",
    "stratlake-run-pipeline": "src.cli.run_pipeline:main",
    "stratlake-run-research-campaign": "src.cli.run_research_campaign:main",
    "stratlake-run-benchmark-pack": "src.cli.run_benchmark_pack:main",
    "stratlake-run-candidate-selection": "src.cli.run_candidate_selection:main",
    "stratlake-review-candidate-selection": "src.cli.review_candidate_selection:main",
    "stratlake-compare-strategies": "src.cli.compare_strategies:main",
    "stratlake-compare-alpha": "src.cli.compare_alpha:main",
    "stratlake-validate-config": "src.cli.validate_config:main",
    "stratlake-doctor": "src.cli.stratlake_doctor:main",
    "stratlake-notebook-doctor": "src.cli.notebook_doctor:main",
    "stratlake-explain-config": "src.cli.explain_config:main",
    "stratlake-catalog-index": "src.cli.catalog_index:main",
    "stratlake-query-catalog": "src.cli.query_catalog:main",
    "stratlake-explore-catalog-evidence": "src.cli.explore_catalog_evidence:main",
    "stratlake-export-catalog-lineage": "src.cli.export_catalog_lineage:main",
    "stratlake-build-evidence-review": "src.cli.build_evidence_review:main",
    "stratlake-run-promotion-governance-report": "src.cli.run_promotion_governance_report:main",
}


def test_pyproject_declares_expected_project_scripts() -> None:
    pyproject = _load_pyproject()
    scripts = pyproject["project"]["scripts"]

    for script_name, target in EXPECTED_SCRIPT_TARGETS.items():
        assert scripts.get(script_name) == target


def test_project_script_targets_are_importable_callables() -> None:
    pyproject = _load_pyproject()
    scripts = pyproject["project"]["scripts"]

    for target in scripts.values():
        module_name, symbol_name = target.split(":", 1)
        module = import_module(module_name)
        symbol = getattr(module, symbol_name)
        assert callable(symbol)


def _load_pyproject() -> dict[str, object]:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
