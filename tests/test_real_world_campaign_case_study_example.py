from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = REPO_ROOT / "docs" / "examples" / "real_world_campaign_case_study.py"


def _load_example_module():
    spec = spec_from_file_location("real_world_campaign_case_study", EXAMPLE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _has_campaign_input_data() -> bool:
    curated_root = REPO_ROOT / "data" / "curated"
    return curated_root.exists() and any(curated_root.rglob("*.parquet"))


def test_real_world_campaign_case_study_checkpoint_demo_reuses_second_run(tmp_path: Path) -> None:
    if not _has_campaign_input_data():
        pytest.skip("Campaign example requires curated parquet inputs not present in CI.")

    module = _load_example_module()

    try:
        artifacts = module.run_checkpoint_demo(output_root=tmp_path / "campaign_checkpoint_demo", verbose=False)
    except Exception as exc:
        if "Dataset `features_daily` has no parquet files" in str(exc):
            pytest.skip("Campaign example requires curated parquet inputs not present in CI.")
        raise

    checkpoint_demo = artifacts.summary["checkpoint_demo"]
    assert checkpoint_demo["same_campaign_run_id"] is True
    assert checkpoint_demo["fingerprints_stable"] is True
    assert checkpoint_demo["first_run"]["reused_stage_count"] == 0
    assert checkpoint_demo["second_run"]["reused_stage_count"] == 7
    assert checkpoint_demo["first_run"]["stage_statuses"] == {
        "preflight": "completed",
        "research": "completed",
        "comparison": "completed",
        "candidate_selection": "completed",
        "portfolio": "completed",
        "candidate_review": "completed",
        "review": "completed",
    }
    assert checkpoint_demo["second_run"]["stage_statuses"] == {
        "preflight": "reused",
        "research": "reused",
        "comparison": "reused",
        "candidate_selection": "reused",
        "portfolio": "reused",
        "candidate_review": "reused",
        "review": "reused",
    }
    assert artifacts.summary["campaign"]["reused_stage_count"] == 7
    assert (tmp_path / "campaign_checkpoint_demo" / "summary.json").exists()
