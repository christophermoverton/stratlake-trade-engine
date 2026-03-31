from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = REPO_ROOT / "docs" / "examples" / "milestone_12_alpha_portfolio_workflow.py"


def _load_example_module():
    spec = spec_from_file_location("milestone_12_alpha_portfolio_workflow", EXAMPLE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_milestone_12_example_workflow_is_deterministic(tmp_path: Path) -> None:
    module = _load_example_module()

    first = module.run_example(output_root=tmp_path / "run_one", verbose=False)
    second = module.run_example(output_root=tmp_path / "run_two", verbose=False)

    assert first.summary == second.summary
    pd.testing.assert_frame_equal(first.prediction_frame, second.prediction_frame, check_dtype=True, check_exact=True)
    pd.testing.assert_frame_equal(first.cross_section, second.cross_section, check_dtype=True, check_exact=True)
    pd.testing.assert_frame_equal(first.single_symbol_backtest, second.single_symbol_backtest, check_dtype=True, check_exact=True)
    pd.testing.assert_frame_equal(first.returns_wide, second.returns_wide, check_dtype=True, check_exact=True)
    pd.testing.assert_frame_equal(first.baseline_portfolio, second.baseline_portfolio, check_dtype=True, check_exact=True)
    pd.testing.assert_frame_equal(first.targeted_portfolio, second.targeted_portfolio, check_dtype=True, check_exact=True)

    assert (tmp_path / "run_one" / "summary.json").exists()
    assert (tmp_path / "run_one" / "artifacts" / "baseline" / "manifest.json").exists()
    assert (tmp_path / "run_one" / "artifacts" / "targeted" / "manifest.json").exists()
    assert first.summary["portfolio"]["baseline_targeting_enabled"] is False
    assert first.summary["portfolio"]["targeted"]["targeting"]["scaling_factor"] is not None
