from __future__ import annotations

from pathlib import Path

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
