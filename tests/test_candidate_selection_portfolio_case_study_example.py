from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = REPO_ROOT / "docs" / "examples" / "candidate_selection_portfolio_case_study.py"


def _load_example_module():
    spec = spec_from_file_location("candidate_selection_portfolio_case_study", EXAMPLE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_candidate_selection_portfolio_case_study_is_deterministic(tmp_path: Path) -> None:
    module = _load_example_module()

    first = module.run_example(output_root=tmp_path / "run_one", verbose=False)
    second = module.run_example(output_root=tmp_path / "run_two", verbose=False)

    assert first.summary == second.summary
    pd.testing.assert_frame_equal(first.comparison_leaderboard, second.comparison_leaderboard, check_dtype=True, check_exact=True)
    pd.testing.assert_frame_equal(first.candidate_universe, second.candidate_universe, check_dtype=True, check_exact=True)
    pd.testing.assert_frame_equal(first.selected_candidates, second.selected_candidates, check_dtype=True, check_exact=True)
    pd.testing.assert_frame_equal(first.rejected_candidates, second.rejected_candidates, check_dtype=True, check_exact=True)
    pd.testing.assert_frame_equal(first.allocation_weights, second.allocation_weights, check_dtype=True, check_exact=True)
    pd.testing.assert_frame_equal(first.portfolio_returns, second.portfolio_returns, check_dtype=True, check_exact=True)

    assert first.summary["candidate_inputs"]["considered_candidate_count"] >= 3
    assert first.summary["candidate_selection"]["counts"]["universe"] >= 3
    assert first.summary["candidate_selection"]["counts"]["selected"] >= 2
    assert first.summary["candidate_selection"]["counts"]["pruned_by_redundancy"] >= 1

    assert first.summary["allocation"]["enabled"] is True
    assert first.summary["allocation"]["method"] == "equal_weight"
    assert first.summary["allocation"]["weight_sum"] == 1.0

    assert first.summary["portfolio"]["portfolio_name"] == "milestone_15_candidate_driven_case_study"
    assert first.summary["portfolio"]["component_count"] == first.summary["candidate_selection"]["counts"]["selected"]

    assert first.summary["review"]["selected_candidates"] == first.summary["candidate_selection"]["counts"]["selected"]
    assert "candidate_review_summary.json" in first.summary["review"]["manifest_artifact_files"]
    assert (tmp_path / "run_one" / "summary.json").exists()
