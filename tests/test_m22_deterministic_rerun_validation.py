from __future__ import annotations

from pathlib import Path

from src.artifacts.safety import read_run_status
from src.validation.deterministic_rerun import run_deterministic_rerun_validation


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_selected_canonical_reruns_are_stable(tmp_path: Path) -> None:
    report = run_deterministic_rerun_validation(
        repo_root=REPO_ROOT,
        output_root=tmp_path,
        targets=(
            ("docs/examples/pipelines/baseline_reference/pipeline.py", "runs"),
            ("docs/examples/pipelines/robustness_scenario_sweep/pipeline.py", "run"),
        ),
    )

    assert report["status"] == "passed", report
    assert report["target_count"] == 2
    assert report["pass_count"] == 2


def test_deterministic_rerun_validation_repeated_runs_do_not_corrupt_prior_outputs(
    tmp_path: Path,
) -> None:
    targets = (("docs/examples/pipelines/baseline_reference/pipeline.py", "runs"),)

    first = run_deterministic_rerun_validation(
        repo_root=REPO_ROOT,
        output_root=tmp_path,
        targets=targets,
    )
    first_status = read_run_status(tmp_path)
    second = run_deterministic_rerun_validation(
        repo_root=REPO_ROOT,
        output_root=tmp_path,
        targets=targets,
    )
    second_status = read_run_status(tmp_path)

    assert first["status"] == "passed"
    assert second["status"] == "passed"
    assert first["targets"] == second["targets"]
    assert first_status["status"] == "completed"
    assert second_status["status"] == "completed"
