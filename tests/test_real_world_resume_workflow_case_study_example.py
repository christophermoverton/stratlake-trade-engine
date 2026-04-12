from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = REPO_ROOT / "docs" / "examples" / "real_world_resume_workflow_case_study.py"


def _load_example_module():
    spec = spec_from_file_location("real_world_resume_workflow_case_study", EXAMPLE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_real_world_resume_workflow_case_study_tracks_partial_resume_and_reuse(tmp_path: Path) -> None:
    module = _load_example_module()

    artifacts = module.run_example(output_root=tmp_path / "resume_workflow_case_study", verbose=False)

    summary = artifacts.summary
    partial = summary["resume_workflow"]["partial_run"]
    resumed = summary["resume_workflow"]["resumed_run"]
    stable = summary["resume_workflow"]["stable_run"]

    assert summary["campaign"]["same_campaign_run_id_across_passes"] is True
    assert summary["resume_workflow"]["attempt_counts"]["comparison"] == 2

    assert partial["status"] == "partial"
    assert partial["partial_stage_names"] == ["comparison"]
    assert partial["resumable_stage_names"] == [
        "comparison",
        "candidate_selection",
        "portfolio",
        "candidate_review",
        "review",
    ]
    assert partial["stage"]["state"] == "partial"
    assert partial["stage"]["execution_metadata"]["failure"]["kind"] == "interrupted"
    assert partial["stage"]["execution_metadata"]["retry"]["attempted"] is False

    assert resumed["status"] == "completed"
    assert resumed["retry_stage_names"] == ["comparison"]
    assert resumed["reused_stage_names"] == ["preflight", "research"]
    assert resumed["stage"]["state"] == "completed"
    assert resumed["stage"]["execution_metadata"]["retry"]["attempted"] is True
    assert resumed["stage"]["execution_metadata"]["retry"]["previous_state"] == "partial"
    assert resumed["stage"]["execution_metadata"]["reuse"]["reused"] is False

    assert stable["status"] == "completed"
    assert stable["retry_stage_names"] == ["comparison"]
    assert stable["reused_stage_names"] == [
        "preflight",
        "research",
        "comparison",
        "candidate_selection",
        "portfolio",
        "candidate_review",
        "review",
    ]
    assert stable["stage"]["state"] == "reused"
    assert stable["stage"]["execution_metadata"]["retry"]["attempted"] is True
    assert stable["stage"]["execution_metadata"]["retry"]["previous_state"] == "partial"
    assert stable["stage"]["execution_metadata"]["reuse"]["reused"] is True

    output_root = tmp_path / "resume_workflow_case_study"
    assert (output_root / "summary.json").exists()
    assert (output_root / "snapshots" / "partial_summary.json").exists()
    assert (output_root / "snapshots" / "resumed_manifest.json").exists()
    assert (output_root / "snapshots" / "stable_checkpoint.json").exists()

    output_root_text = output_root.resolve().as_posix()
    repo_root_text = REPO_ROOT.resolve().as_posix()
    for path in output_root.rglob("*.json"):
        contents = path.read_text(encoding="utf-8").replace("\\", "/")
        assert output_root_text not in contents
        assert repo_root_text not in contents
