from __future__ import annotations

import contextlib
import io
import json
import runpy
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = REPO_ROOT / "docs" / "examples" / "robustness_report_example.py"
OUTPUT_DIR = REPO_ROOT / "docs" / "examples" / "output" / "robustness_report_example"
EXPECTED_ARTIFACTS = {
    "leakage_validation.json",
    "manifest.json",
    "multiple_testing_summary.json",
    "purged_split_plan.json",
    "purged_split_summary.csv",
    "robustness_findings.json",
    "robustness_report.md",
    "robustness_summary.json",
    "sample_size_validation.json",
    "sensitivity_summary.csv",
    "walk_forward_efficiency.csv",
}


def test_robustness_report_example_runs_and_is_deterministic() -> None:
    first_summary = _run_example()
    first_snapshot = _artifact_bytes()

    second_summary = _run_example()
    second_snapshot = _artifact_bytes()

    assert first_summary == second_summary
    assert first_snapshot == second_snapshot
    assert set(first_snapshot) == EXPECTED_ARTIFACTS
    assert first_summary["output_dir"] == "docs/examples/output/robustness_report_example"
    assert first_summary["governance_available"] is True
    assert first_summary["temporal_validation_status"] == "pass"


def test_robustness_report_example_outputs_are_path_portable() -> None:
    _run_example()
    repo_root_text = REPO_ROOT.as_posix()
    repo_root_windows = str(REPO_ROOT)

    for path in sorted(OUTPUT_DIR.iterdir(), key=lambda item: item.name):
        text = path.read_text(encoding="utf-8")
        assert repo_root_text not in text
        assert repo_root_windows not in text
        assert "file://" not in text
        assert "C:\\" not in text


def _run_example() -> dict[str, object]:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        runpy.run_path(str(EXAMPLE), run_name="__main__")
    return json.loads(buffer.getvalue())


def _artifact_bytes() -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(OUTPUT_DIR.iterdir(), key=lambda item: item.name)}
