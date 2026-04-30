from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from docs.examples.m27_market_simulation_case_study import DEFAULT_OUTPUT_ROOT, run_example


OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
EXPECTED_FILES = (
    "simulation_summary.json",
    "leaderboard.csv",
    "case_study_report.md",
    "manifest.json",
)
SUMMARY_KEYS = {
    "case_study_name",
    "milestone",
    "source_simulation_run_id",
    "scenario_count",
    "path_metric_row_count",
    "summary_row_count",
    "leaderboard_row_count",
    "simulation_types",
    "best_ranked_scenario",
    "worst_ranked_scenario",
    "policy_failure_rate",
    "regime_only_monte_carlo_note",
    "limitations",
}
LEADERBOARD_COLUMNS = {
    "rank",
    "simulation_run_id",
    "scenario_id",
    "scenario_name",
    "simulation_type",
    "policy_name",
    "path_count",
    "tail_quantile",
    "policy_failure_rate",
    "mean_stress_score",
    "ranking_metric",
    "ranking_value",
    "decision_label",
}


@pytest.fixture()
def case_study_outputs() -> Path:
    run_example()
    return OUTPUT_ROOT


def test_case_study_script_runs_from_repo_root() -> None:
    completed = subprocess.run(
        [sys.executable, "docs/examples/m27_market_simulation_case_study.py"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "M27 market simulation case study completed:" in completed.stdout


def test_expected_output_files_are_created(case_study_outputs: Path) -> None:
    for filename in EXPECTED_FILES:
        assert (case_study_outputs / filename).exists()


def test_simulation_summary_has_required_keys(case_study_outputs: Path) -> None:
    summary = json.loads((case_study_outputs / "simulation_summary.json").read_text(encoding="utf-8"))

    assert SUMMARY_KEYS <= set(summary)
    assert summary["case_study_name"] == "m27_market_simulation_case_study"
    assert summary["milestone"] == "M27"
    assert summary["scenario_count"] == 4
    assert set(summary["simulation_types"]) == {
        "historical_episode_replay",
        "regime_block_bootstrap",
        "regime_transition_monte_carlo",
        "shock_overlay",
    }
    assert summary["leaderboard_row_count"] >= 1
    assert "regime-only" in summary["regime_only_monte_carlo_note"]


def test_leaderboard_has_expected_columns_and_rows(case_study_outputs: Path) -> None:
    with (case_study_outputs / "leaderboard.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows
    assert LEADERBOARD_COLUMNS <= set(rows[0])
    assert [int(row["rank"]) for row in rows] == sorted(int(row["rank"]) for row in rows)


def test_case_study_report_contains_key_sections(case_study_outputs: Path) -> None:
    report = (case_study_outputs / "case_study_report.md").read_text(encoding="utf-8")

    for heading in (
        "## Research Question",
        "## Scenario Methods",
        "## Artifact Flow",
        "## Generated Outputs",
        "## Leaderboard Interpretation",
        "## Historical Replay Interpretation",
        "## Block Bootstrap Interpretation",
        "## Monte Carlo Interpretation",
        "## Shock Overlay Interpretation",
        "## Simulation-Aware Metrics Interpretation",
        "## Limitations",
        "## Recommended Follow-Ups",
    ):
        assert heading in report
    assert "not forecast" in report or "do not forecast" in report


def test_manifest_uses_relative_paths_only(case_study_outputs: Path) -> None:
    manifest = json.loads((case_study_outputs / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["case_study_name"] == "m27_market_simulation_case_study"
    assert manifest["schema_version"] == "1.0"
    assert manifest["source_config_path"] == "configs/regime_stress_tests/m27_market_simulation_case_study.yml"
    _assert_relative_paths(manifest)


def test_repeated_runs_are_deterministic() -> None:
    run_example()
    first = {filename: (OUTPUT_ROOT / filename).read_text(encoding="utf-8") for filename in EXPECTED_FILES}

    run_example()
    second = {filename: (OUTPUT_ROOT / filename).read_text(encoding="utf-8") for filename in EXPECTED_FILES}

    assert first == second


def _assert_relative_paths(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if "path" in str(key).lower() and isinstance(item, str):
                assert not Path(item).is_absolute()
                assert ":\\" not in item
            _assert_relative_paths(item)
    elif isinstance(value, list):
        for item in value:
            _assert_relative_paths(item)
