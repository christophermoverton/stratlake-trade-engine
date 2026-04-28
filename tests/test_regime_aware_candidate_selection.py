from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.cli.run_regime_aware_candidate_selection import run_cli
from src.config.regime_aware_candidate_selection import RegimeAwareCandidateSelectionConfig
from src.research.regime_aware_candidate_selection import run_regime_aware_candidate_selection


def _write_review_pack(tmp_path: Path) -> Path:
    review = tmp_path / "review_pack"
    review.mkdir()
    (review / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_type": "regime_review_pack",
                "review_run_id": "review_001",
                "source_benchmark_run_id": "benchmark_001",
            },
            indent=2,
        ),
        encoding="utf-8",
        newline="\n",
    )
    return review


def _candidate_rows() -> list[dict[str, object]]:
    return [
        {
            "candidate_id": "alpha_global",
            "candidate_name": "Alpha Global",
            "artifact_type": "alpha",
            "source_run_id": "a1",
            "global_score": 0.90,
            "regime_id": "trend",
            "regime_label": "Trend",
            "observation_count": 40,
            "regime_score": 0.70,
            "transition_score": 0.60,
            "transition_window_drawdown": -0.03,
            "defensive_score": 0.40,
            "high_vol_drawdown": -0.15,
            "correlation_to_selected": 0.30,
            "redundancy_group": "carry",
            "regime_confidence_observed": 0.40,
        },
        {
            "candidate_id": "alpha_redundant",
            "candidate_name": "Alpha Redundant",
            "artifact_type": "alpha",
            "source_run_id": "a2",
            "global_score": 0.82,
            "regime_id": "trend",
            "regime_label": "Trend",
            "observation_count": 42,
            "regime_score": 0.69,
            "transition_score": 0.57,
            "transition_window_drawdown": -0.04,
            "defensive_score": 0.39,
            "high_vol_drawdown": -0.16,
            "correlation_to_selected": 0.31,
            "redundancy_group": "carry",
            "regime_confidence_observed": 0.70,
        },
        {
            "candidate_id": "strategy_chop",
            "candidate_name": "Strategy Chop",
            "artifact_type": "strategy",
            "source_run_id": "s1",
            "global_score": 0.58,
            "regime_id": "chop",
            "regime_label": "Chop",
            "observation_count": 50,
            "regime_score": 0.88,
            "transition_score": 0.62,
            "transition_window_drawdown": -0.05,
            "defensive_score": 0.61,
            "high_vol_drawdown": -0.08,
            "correlation_to_selected": 0.40,
            "redundancy_group": "chop",
            "regime_confidence_observed": 0.72,
        },
        {
            "candidate_id": "portfolio_transition",
            "candidate_name": "Portfolio Transition",
            "artifact_type": "portfolio",
            "source_run_id": "p1",
            "global_score": 0.66,
            "regime_id": "transition",
            "regime_label": "Transition",
            "observation_count": 30,
            "regime_score": 0.66,
            "transition_score": 0.91,
            "transition_window_drawdown": -0.02,
            "defensive_score": 0.72,
            "high_vol_drawdown": -0.05,
            "correlation_to_selected": 0.22,
            "redundancy_group": "transition",
            "regime_confidence_observed": 0.65,
        },
        {
            "candidate_id": "raw_fallback",
            "candidate_name": "Raw Fallback",
            "artifact_type": "alpha",
            "source_run_id": "a3",
            "total_return": 0.24,
            "sharpe": 1.6,
            "max_drawdown": -0.08,
            "regime_id": "chop",
            "regime_label": "Chop",
            "observation_count": 35,
            "regime_return": 0.18,
            "regime_sharpe": 1.7,
            "regime_max_drawdown": -0.06,
            "regime_ic": 0.05,
            "regime_rank_ic": 0.05,
            "transition_window_return": 0.04,
            "transition_window_drawdown": -0.04,
            "high_vol_drawdown": -0.07,
            "volatility": 0.10,
            "correlation_to_selected": 0.25,
            "redundancy_group": "raw",
            "regime_confidence_observed": 0.68,
        },
    ]


def _write_candidates_csv(tmp_path: Path) -> Path:
    path = tmp_path / "candidates.csv"
    pd.DataFrame(_candidate_rows()).to_csv(path, index=False)
    return path


def _write_candidates_json(tmp_path: Path) -> Path:
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps({"rows": _candidate_rows()}, indent=2), encoding="utf-8", newline="\n")
    return path


def _config(tmp_path: Path, review: Path, candidates: Path) -> RegimeAwareCandidateSelectionConfig:
    return RegimeAwareCandidateSelectionConfig.from_mapping(
        {
            "selection_name": "test_regime_selection",
            "source_review_pack": review.as_posix(),
            "source_candidate_universe": {"candidate_metrics_path": candidates.as_posix()},
            "selection_categories": {
                "global_performer": {"enabled": True, "max_candidates": 3, "min_global_score": 0.60},
                "regime_specialist": {
                    "enabled": True,
                    "max_candidates_per_regime": 1,
                    "min_regime_score": 0.65,
                    "min_regime_observations": 20,
                },
                "transition_resilient": {"enabled": True, "max_candidates": 2, "min_transition_score": 0.55},
                "defensive_fallback": {"enabled": True, "max_candidates": 2, "max_high_vol_drawdown": -0.12},
            },
            "redundancy": {"enabled": True, "max_pairwise_correlation": 0.85},
            "allocation_hints": {
                "default_category_budget": {
                    "global_performer": 0.45,
                    "regime_specialist": 0.30,
                    "transition_resilient": 0.15,
                    "defensive_fallback": 0.10,
                }
            },
            "output_root": (tmp_path / "candidate_selection").as_posix(),
        }
    )


def test_regime_aware_candidate_selection_generates_artifacts(tmp_path: Path) -> None:
    result = run_regime_aware_candidate_selection(
        _config(tmp_path, _write_review_pack(tmp_path), _write_candidates_csv(tmp_path))
    )

    selection = pd.read_csv(result.candidate_selection_csv_path)
    regime_scores = pd.read_csv(result.regime_candidate_scores_csv_path)
    redundancy = pd.read_csv(result.redundancy_report_path)
    summary = json.loads(result.selection_summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    allocation = json.loads(result.allocation_hints_path.read_text(encoding="utf-8"))

    assert set(selection["candidate_id"]) == {row["candidate_id"] for row in _candidate_rows()}
    assert "regime_score" in regime_scores.columns
    assert "pruned" in redundancy["redundancy_action"].tolist()
    assert summary["source_review_run_id"] == "review_001"
    assert summary["multi_category_selection_enabled"] is True
    assert summary["multi_category_candidate_count"] > 0
    assert summary["regime_candidate_score_count"] == len(regime_scores)
    assert summary["regime_specialist_selected_count"] > 0
    assert summary["redundancy_pruned_count"] > 0
    assert manifest["artifact_type"] == "regime_aware_candidate_selection"
    assert manifest["multi_category_selection_enabled"] is True
    assert allocation["candidate_weight_hints"]


def test_regime_aware_candidate_selection_loads_json_candidate_metrics(tmp_path: Path) -> None:
    result = run_regime_aware_candidate_selection(
        _config(tmp_path, _write_review_pack(tmp_path), _write_candidates_json(tmp_path))
    )

    assert result.selection_summary["candidate_count"] == len(_candidate_rows())


def test_regime_aware_candidate_selection_missing_required_candidate_field_fails(tmp_path: Path) -> None:
    candidates = tmp_path / "bad.csv"
    pd.DataFrame([{"candidate_name": "No ID", "artifact_type": "alpha"}]).to_csv(candidates, index=False)

    with pytest.raises(ValueError, match="required fields"):
        run_regime_aware_candidate_selection(_config(tmp_path, _write_review_pack(tmp_path), candidates))


def test_regime_aware_candidate_selection_missing_optional_metrics_warns(tmp_path: Path) -> None:
    candidates = tmp_path / "minimal.csv"
    pd.DataFrame(
        [{"candidate_id": "a", "candidate_name": "A", "artifact_type": "alpha", "total_return": 0.1}]
    ).to_csv(candidates, index=False)

    result = run_regime_aware_candidate_selection(_config(tmp_path, _write_review_pack(tmp_path), candidates))

    assert result.selection_summary["warning_count"] > 0
    assert any("Optional candidate metric" in item for item in result.selection_summary["limitations"])


def test_regime_aware_candidate_selection_category_limits_and_ranking_are_deterministic(tmp_path: Path) -> None:
    config = _config(tmp_path, _write_review_pack(tmp_path), _write_candidates_csv(tmp_path))

    first = run_regime_aware_candidate_selection(config)
    second = run_regime_aware_candidate_selection(config)

    assert first.selection_run_id == second.selection_run_id
    assert first.candidate_selection_csv_path.read_text(encoding="utf-8") == second.candidate_selection_csv_path.read_text(encoding="utf-8")
    assignments = pd.read_csv(first.category_assignments_path)
    selected_global = assignments[(assignments["category"] == "global_performer") & assignments["selected"]]
    assert len(selected_global) <= 3


def test_regime_aware_candidate_selection_reports_multi_category_roles(tmp_path: Path) -> None:
    result = run_regime_aware_candidate_selection(
        _config(tmp_path, _write_review_pack(tmp_path), _write_candidates_csv(tmp_path))
    )

    selection = pd.read_csv(result.candidate_selection_csv_path)
    allocation = json.loads(result.allocation_hints_path.read_text(encoding="utf-8"))
    summary = result.selection_summary
    multi_category_ids = {item["candidate_id"] for item in summary["multi_category_candidates"]}

    assert multi_category_ids
    assert any("|" in value for value in selection["selection_category"].dropna().astype(str))
    assert any(
        hint["candidate_id"] in multi_category_ids and hint["category"] == "global_performer"
        for hint in allocation["candidate_weight_hints"]
    )
    assert any(
        hint["candidate_id"] in multi_category_ids and hint["category"] != "global_performer"
        for hint in allocation["candidate_weight_hints"]
    )


def test_regime_aware_candidate_selection_reports_advisory_low_confidence(tmp_path: Path) -> None:
    result = run_regime_aware_candidate_selection(
        _config(tmp_path, _write_review_pack(tmp_path), _write_candidates_csv(tmp_path))
    )

    selection = pd.read_csv(result.candidate_selection_csv_path)
    low_confidence = result.selection_summary["low_confidence_selected_candidates"]

    assert result.selection_summary["low_confidence_selected_count"] == len(low_confidence)
    assert low_confidence == [
        {
            "candidate_id": "alpha_global",
            "regime_confidence_observed": 0.4,
            "regime_confidence_required": 0.55,
            "selection_categories": ["global_performer", "regime_specialist"],
        }
    ]
    reason = selection.loc[selection["candidate_id"] == "alpha_global", "selection_reason"].iloc[0]
    assert "Advisory warning" in reason


def test_regime_aware_candidate_selection_no_redundancy_data_warning(tmp_path: Path) -> None:
    rows = _candidate_rows()
    for row in rows:
        row.pop("redundancy_group", None)
        row.pop("correlation_to_selected", None)
    candidates = tmp_path / "no_redundancy.csv"
    pd.DataFrame(rows).to_csv(candidates, index=False)

    result = run_regime_aware_candidate_selection(_config(tmp_path, _write_review_pack(tmp_path), candidates))

    assert any("No redundancy_group" in item for item in result.selection_summary["limitations"])


def test_regime_aware_candidate_selection_cli_smoke(tmp_path: Path) -> None:
    review = _write_review_pack(tmp_path)
    candidates = _write_candidates_csv(tmp_path)
    config_path = tmp_path / "config.yml"
    payload = _config(tmp_path, review, candidates).to_dict()
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8", newline="\n")

    result = run_cli(["--config", config_path.as_posix()])

    assert result.candidate_selection_csv_path.exists()


def test_regime_aware_candidate_selection_cli_missing_candidate_metrics_fails(tmp_path: Path) -> None:
    review = _write_review_pack(tmp_path)
    config_path = tmp_path / "config.yml"
    payload = _config(tmp_path, review, tmp_path / "missing.csv").to_dict()
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8", newline="\n")

    with pytest.raises(FileNotFoundError):
        run_cli(["--config", config_path.as_posix()])
