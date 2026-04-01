from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = REPO_ROOT / "docs" / "examples" / "milestone_13_review_promotion_workflow.py"
COMMITTED_OUTPUT_ROOT = REPO_ROOT / "docs" / "examples" / "output" / "milestone_13_review_promotion_workflow"


def _load_example_module():
    spec = spec_from_file_location("milestone_13_review_promotion_workflow", EXAMPLE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_milestone_13_review_promotion_example_is_deterministic_and_matches_committed_outputs(tmp_path: Path) -> None:
    module = _load_example_module()

    first = module.run_example(output_root=tmp_path / "run_one", verbose=False)
    second = module.run_example(output_root=tmp_path / "run_two", verbose=False)

    assert first.summary == second.summary
    pd.testing.assert_frame_equal(first.leaderboard, second.leaderboard, check_dtype=True, check_exact=True)
    assert first.review_summary == second.review_summary
    assert first.manifest == second.manifest
    assert first.promotion_gates == second.promotion_gates

    assert first.summary["review"]["selected_run_ids"] == [
        "alpha_demo_eligible",
        "strategy_demo_promoted",
        "portfolio_demo_blocked",
    ]
    assert first.summary["review_promotion"]["promotion_status"] == "review_ready"
    assert first.summary["review_promotion"]["passed_gate_count"] == 3
    assert first.manifest["artifact_files"] == [
        "leaderboard.csv",
        "manifest.json",
        "promotion_gates.json",
        "review_summary.json",
    ]
    assert first.review_summary["counts_by_run_type"] == {
        "alpha_evaluation": 1,
        "portfolio": 1,
        "strategy": 1,
    }

    for relative_path in [
        Path("leaderboard.csv"),
        Path("review_summary.json"),
        Path("manifest.json"),
        Path("promotion_gates.json"),
        Path("summary.json"),
    ]:
        generated = (tmp_path / "run_one" / relative_path).read_bytes()
        committed = (COMMITTED_OUTPUT_ROOT / relative_path).read_bytes()
        assert generated == committed

    assert _read_json(tmp_path / "run_one" / "summary.json") == _read_json(COMMITTED_OUTPUT_ROOT / "summary.json")
