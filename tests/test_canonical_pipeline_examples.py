from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_example_module(path: Path):
    spec = spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("relative_path", "summary_key"),
    [
        ("docs/examples/pipelines/baseline_reference/pipeline.py", "runs"),
        ("docs/examples/pipelines/strategy_archetype_showcase/pipeline.py", "runs"),
        ("docs/examples/pipelines/regime_ensemble_showcase/pipeline.py", "runs"),
        ("docs/examples/pipelines/long_short_risk_controls/pipeline.py", "run"),
        ("docs/examples/pipelines/robustness_scenario_sweep/pipeline.py", "run"),
        ("docs/examples/pipelines/declarative_builder/pipeline.py", "equivalence"),
    ],
)
def test_canonical_execution_pipelines_run(relative_path: str, summary_key: str, tmp_path: Path) -> None:
    module = _load_example_module(REPO_ROOT / relative_path)
    artifacts = module.run_example(output_root=tmp_path / Path(relative_path).stem, verbose=False, reset_output=True)
    assert summary_key in artifacts.summary
    assert (tmp_path / Path(relative_path).stem / "summary.json").exists()


def test_campaign_orchestration_pipeline_runs(tmp_path: Path) -> None:
    module = _load_example_module(
        REPO_ROOT / "docs/examples/pipelines/research_campaign_orchestration/pipeline.py"
    )
    artifacts = module.run_example(output_root=tmp_path / "campaign_orchestration", verbose=False, reset_output=True)

    assert artifacts.summary["run"]["first_pass_status"] == "completed"
    assert artifacts.summary["run"]["second_pass_status"] == "completed"
    assert (tmp_path / "campaign_orchestration" / "summary.json").exists()


def test_resume_reuse_pipeline_runs(tmp_path: Path) -> None:
    module = _load_example_module(REPO_ROOT / "docs/examples/pipelines/resume_reuse/pipeline.py")
    artifacts = module.run_example(output_root=tmp_path / "resume_reuse", verbose=False, reset_output=True)

    assert artifacts.summary["run"]["partial_state"] == "partial"
    assert artifacts.summary["run"]["resumed_state"] == "completed"
    assert artifacts.summary["run"]["stable_state"] == "reused"
    assert (tmp_path / "resume_reuse" / "summary.json").exists()
