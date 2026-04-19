from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = REPO_ROOT / "docs" / "examples" / "real_alpha_workflow.py"


def _load_example_module():
    spec = spec_from_file_location("real_alpha_workflow", EXAMPLE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_real_alpha_workflow_example_is_deterministic(tmp_path: Path) -> None:
    module = _load_example_module()

    first = module.run_example(output_root=tmp_path / "run_one", verbose=False)
    second = module.run_example(output_root=tmp_path / "run_two", verbose=False)

    assert first.summary == second.summary
    pd.testing.assert_frame_equal(first.dataset, second.dataset, check_dtype=True, check_exact=True)
    pd.testing.assert_frame_equal(first.signals, second.signals, check_dtype=True, check_exact=True)
    pd.testing.assert_frame_equal(first.sleeve_returns, second.sleeve_returns, check_dtype=True, check_exact=True)
    assert first.portfolio_returns is not None
    assert second.portfolio_returns is not None
    pd.testing.assert_frame_equal(first.portfolio_returns, second.portfolio_returns, check_dtype=True, check_exact=True)
    pd.testing.assert_frame_equal(first.review_leaderboard, second.review_leaderboard, check_dtype=True, check_exact=True)

    assert first.summary["alpha"]["alpha_name"] == "rank_composite_momentum"
    assert first.summary["alpha"]["resolved_config"]["dataset"] == "features_daily"
    assert first.summary["alpha"]["signal_mapping"]["policy"] == "top_bottom_quantile"
    assert first.summary["portfolio"]["included"] is True
    assert first.summary["review"]["alpha_entry"]["linked_portfolio_count"] == 1
    assert first.summary["review"]["portfolio_entry"]["entity_name"] == "rank_composite_momentum_sleeve_portfolio"

    assert (tmp_path / "run_one" / "summary.json").exists()
    assert (
        tmp_path
        / "run_one"
        / "workspace"
        / "artifacts"
        / "alpha"
        / first.summary["alpha"]["run_id"]
        / "signal_semantics.json"
    ).exists()
    assert len(first.summary["review"]["selected_run_ids"]) == 2
