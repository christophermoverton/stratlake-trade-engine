from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.cli.review_candidate_selection import run_cli
from src.research.candidate_review import review_candidate_selection


def _write_candidate_selection_artifacts(root: Path) -> Path:
    run_dir = root / "candidate_selection_test"
    run_dir.mkdir(parents=True, exist_ok=True)

    universe = pd.DataFrame(
        [
            {
                "selection_rank": 1,
                "candidate_id": "cand_a",
                "alpha_name": "AlphaA",
                "alpha_run_id": "alpha_run_a",
                "sleeve_run_id": "sleeve_run_a",
                "mapping_name": "rank_long_short",
                "dataset": "features_daily",
                "timeframe": "1D",
                "evaluation_horizon": 5,
                "mean_ic": 0.08,
                "ic_ir": 1.6,
                "mean_rank_ic": 0.09,
                "rank_ic_ir": 1.7,
                "n_periods": 120,
                "sharpe_ratio": 1.0,
                "annualized_return": 0.10,
                "total_return": 0.08,
                "max_drawdown": -0.03,
                "average_turnover": 0.05,
                "promotion_status": "eligible",
                "review_status": "candidate",
                "artifact_path": "artifacts/alpha/alpha_run_a",
            },
            {
                "selection_rank": 2,
                "candidate_id": "cand_b",
                "alpha_name": "AlphaB",
                "alpha_run_id": "alpha_run_b",
                "sleeve_run_id": "sleeve_run_b",
                "mapping_name": "rank_long_short",
                "dataset": "features_daily",
                "timeframe": "1D",
                "evaluation_horizon": 5,
                "mean_ic": 0.01,
                "ic_ir": 0.2,
                "mean_rank_ic": 0.01,
                "rank_ic_ir": 0.3,
                "n_periods": 120,
                "sharpe_ratio": 0.2,
                "annualized_return": 0.01,
                "total_return": 0.01,
                "max_drawdown": -0.10,
                "average_turnover": 0.08,
                "promotion_status": "blocked",
                "review_status": "candidate",
                "artifact_path": "artifacts/alpha/alpha_run_b",
            },
            {
                "selection_rank": 3,
                "candidate_id": "cand_c",
                "alpha_name": "AlphaC",
                "alpha_run_id": "alpha_run_c",
                "sleeve_run_id": "sleeve_run_c",
                "mapping_name": "rank_long_short",
                "dataset": "features_daily",
                "timeframe": "1D",
                "evaluation_horizon": 5,
                "mean_ic": 0.07,
                "ic_ir": 1.3,
                "mean_rank_ic": 0.08,
                "rank_ic_ir": 1.4,
                "n_periods": 120,
                "sharpe_ratio": 0.7,
                "annualized_return": 0.07,
                "total_return": 0.05,
                "max_drawdown": -0.06,
                "average_turnover": 0.06,
                "promotion_status": "eligible",
                "review_status": "candidate",
                "artifact_path": "artifacts/alpha/alpha_run_c",
            },
        ]
    )
    universe.to_csv(run_dir / "candidate_universe.csv", index=False)

    eligibility = pd.DataFrame(
        [
            {
                "candidate_id": "cand_a",
                "alpha_name": "AlphaA",
                "sleeve_run_id": "sleeve_run_a",
                "dataset": "features_daily",
                "timeframe": "1D",
                "evaluation_horizon": 5,
                "is_eligible": True,
                "failed_checks": "",
                "mean_ic": 0.08,
                "mean_rank_ic": 0.09,
                "ic_ir": 1.6,
                "rank_ic_ir": 1.7,
                "history_length": 120,
                "threshold_min_mean_ic": 0.03,
                "threshold_min_mean_rank_ic": 0.03,
                "threshold_min_ic_ir": 0.5,
                "threshold_min_rank_ic_ir": 0.5,
                "threshold_min_history_length": 60,
            },
            {
                "candidate_id": "cand_b",
                "alpha_name": "AlphaB",
                "sleeve_run_id": "sleeve_run_b",
                "dataset": "features_daily",
                "timeframe": "1D",
                "evaluation_horizon": 5,
                "is_eligible": False,
                "failed_checks": "min_ic_ir",
                "mean_ic": 0.01,
                "mean_rank_ic": 0.01,
                "ic_ir": 0.2,
                "rank_ic_ir": 0.3,
                "history_length": 120,
                "threshold_min_mean_ic": 0.03,
                "threshold_min_mean_rank_ic": 0.03,
                "threshold_min_ic_ir": 0.5,
                "threshold_min_rank_ic_ir": 0.5,
                "threshold_min_history_length": 60,
            },
            {
                "candidate_id": "cand_c",
                "alpha_name": "AlphaC",
                "sleeve_run_id": "sleeve_run_c",
                "dataset": "features_daily",
                "timeframe": "1D",
                "evaluation_horizon": 5,
                "is_eligible": True,
                "failed_checks": "",
                "mean_ic": 0.07,
                "mean_rank_ic": 0.08,
                "ic_ir": 1.3,
                "rank_ic_ir": 1.4,
                "history_length": 120,
                "threshold_min_mean_ic": 0.03,
                "threshold_min_mean_rank_ic": 0.03,
                "threshold_min_ic_ir": 0.5,
                "threshold_min_rank_ic_ir": 0.5,
                "threshold_min_history_length": 60,
            },
        ]
    )
    eligibility.to_csv(run_dir / "eligibility_filter_results.csv", index=False)

    selected = universe.loc[universe["candidate_id"] == "cand_a"].copy()
    selected["allocation_weight"] = 1.0
    selected["allocation_method"] = "equal_weight"
    selected["pre_constraint_weight"] = 1.0
    selected["constraint_adjusted_flag"] = False
    selected.to_csv(run_dir / "selected_candidates.csv", index=False)

    rejected = pd.DataFrame(
        [
            {
                **universe.loc[universe["candidate_id"] == "cand_b"].iloc[0].to_dict(),
                "rejected_stage": "eligibility_gate",
                "rejection_reason": "eligibility_failed",
                "failed_checks": "min_ic_ir",
                "rejected_against_candidate_id": None,
                "observed_correlation": None,
                "configured_max_correlation": None,
                "overlap_observations": None,
            },
            {
                **universe.loc[universe["candidate_id"] == "cand_c"].iloc[0].to_dict(),
                "rejected_stage": "redundancy_filter",
                "rejection_reason": "correlation_above_threshold",
                "failed_checks": "",
                "rejected_against_candidate_id": "cand_a",
                "observed_correlation": 0.92,
                "configured_max_correlation": 0.80,
                "overlap_observations": 120,
            },
        ]
    )
    rejected.to_csv(run_dir / "rejected_candidates.csv", index=False)

    pd.DataFrame(
        [
            {
                "candidate_id": "cand_a",
                "alpha_name": "AlphaA",
                "sleeve_run_id": "sleeve_run_a",
                "allocation_weight": 1.0,
                "allocation_method": "equal_weight",
                "pre_constraint_weight": 1.0,
                "constraint_adjusted_flag": False,
            }
        ]
    ).to_csv(run_dir / "allocation_weights.csv", index=False)

    pd.DataFrame(
        [
            {"candidate_id": "cand_a", "cand_a": 1.0, "cand_c": 0.92},
            {"candidate_id": "cand_c", "cand_a": 0.92, "cand_c": 1.0},
        ]
    ).to_csv(run_dir / "correlation_matrix.csv", index=False)

    (run_dir / "selection_summary.json").write_text(
        json.dumps(
            {
                "run_id": "candidate_selection_test",
                "total_candidates": 3,
                "selected_candidates": 1,
                "rejected_candidates": 2,
                "allocation_summary": {
                    "allocation_enabled": True,
                    "allocation_method": "equal_weight",
                    "allocated_candidates": 1,
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "candidate_selection_test",
                "run_type": "candidate_selection",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return run_dir


def _write_portfolio_artifacts(root: Path) -> Path:
    run_dir = root / "portfolio_test"
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "components.json").write_text(
        json.dumps(
            {
                "components": [
                    {
                        "strategy_name": "AlphaA__cand_a",
                        "run_id": "sleeve_run_a",
                        "artifact_type": "alpha_sleeve",
                        "provenance": {
                            "candidate_id": "cand_a",
                            "alpha_name": "AlphaA",
                            "alpha_run_id": "alpha_run_a",
                            "sleeve_run_id": "sleeve_run_a",
                            "candidate_selection_run_id": "candidate_selection_test",
                        },
                    }
                ]
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    pd.DataFrame(
        [
            {
                "ts_utc": "2026-01-01T00:00:00Z",
                "strategy_return__AlphaA__cand_a": 0.01,
                "weight__AlphaA__cand_a": 1.0,
                "portfolio_return": 0.01,
            },
            {
                "ts_utc": "2026-01-02T00:00:00Z",
                "strategy_return__AlphaA__cand_a": -0.02,
                "weight__AlphaA__cand_a": 1.0,
                "portfolio_return": -0.02,
            },
            {
                "ts_utc": "2026-01-03T00:00:00Z",
                "strategy_return__AlphaA__cand_a": 0.03,
                "weight__AlphaA__cand_a": 1.0,
                "portfolio_return": 0.03,
            },
        ]
    ).to_csv(run_dir / "portfolio_returns.csv", index=False)

    (run_dir / "metrics.json").write_text(
        json.dumps({"total_return": 0.0194, "sharpe_ratio": 1.1}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": "portfolio_test", "run_type": "portfolio"}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return run_dir


def test_candidate_review_generates_required_artifacts(tmp_path: Path) -> None:
    candidate_dir = _write_candidate_selection_artifacts(tmp_path / "artifacts" / "candidate_selection")
    portfolio_dir = _write_portfolio_artifacts(tmp_path / "artifacts" / "portfolios")

    result = review_candidate_selection(
        candidate_selection_artifact_dir=candidate_dir,
        portfolio_artifact_dir=portfolio_dir,
    )

    assert result.candidate_decisions_csv.exists()
    assert result.candidate_summary_csv.exists()
    assert result.candidate_contributions_csv.exists()
    assert result.diversification_summary_json.exists()
    assert result.candidate_review_summary_json.exists()
    assert result.candidate_review_report_md is not None and result.candidate_review_report_md.exists()
    assert result.total_candidates == 3
    assert result.selected_candidates == 1
    assert result.rejected_candidates == 2


def test_candidate_decision_status_and_reasons_are_correct(tmp_path: Path) -> None:
    candidate_dir = _write_candidate_selection_artifacts(tmp_path / "artifacts" / "candidate_selection")
    portfolio_dir = _write_portfolio_artifacts(tmp_path / "artifacts" / "portfolios")

    result = review_candidate_selection(
        candidate_selection_artifact_dir=candidate_dir,
        portfolio_artifact_dir=portfolio_dir,
        include_markdown_report=False,
    )

    decisions = pd.read_csv(result.candidate_decisions_csv)
    by_id = {row["candidate_id"]: row for _, row in decisions.iterrows()}

    assert by_id["cand_a"]["selection_status"] == "selected"
    assert by_id["cand_b"]["selection_status"] == "rejected_eligibility"
    assert by_id["cand_c"]["selection_status"] == "rejected_redundancy"
    assert "min_ic_ir" in str(by_id["cand_b"]["rejection_reasons"])


def test_candidate_contribution_values_are_computed(tmp_path: Path) -> None:
    candidate_dir = _write_candidate_selection_artifacts(tmp_path / "artifacts" / "candidate_selection")
    portfolio_dir = _write_portfolio_artifacts(tmp_path / "artifacts" / "portfolios")

    result = review_candidate_selection(
        candidate_selection_artifact_dir=candidate_dir,
        portfolio_artifact_dir=portfolio_dir,
        include_markdown_report=False,
    )

    contributions = pd.read_csv(result.candidate_contributions_csv)
    assert len(contributions) == 1
    row = contributions.iloc[0]
    assert row["candidate_id"] == "cand_a"
    assert abs(float(row["return_contribution"]) - 0.02) < 1e-12
    assert abs(float(row["average_weight"]) - 1.0) < 1e-12
    assert abs(float(row["return_contribution_share"]) - 1.0) < 1e-12


def test_review_outputs_are_deterministic(tmp_path: Path) -> None:
    candidate_dir = _write_candidate_selection_artifacts(tmp_path / "artifacts" / "candidate_selection")
    portfolio_dir = _write_portfolio_artifacts(tmp_path / "artifacts" / "portfolios")

    first = review_candidate_selection(
        candidate_selection_artifact_dir=candidate_dir,
        portfolio_artifact_dir=portfolio_dir,
        include_markdown_report=False,
    )
    second = review_candidate_selection(
        candidate_selection_artifact_dir=candidate_dir,
        portfolio_artifact_dir=portfolio_dir,
        include_markdown_report=False,
    )

    assert first.candidate_decisions_csv.read_bytes() == second.candidate_decisions_csv.read_bytes()
    assert first.candidate_summary_csv.read_bytes() == second.candidate_summary_csv.read_bytes()
    assert first.candidate_contributions_csv.read_bytes() == second.candidate_contributions_csv.read_bytes()
    assert first.diversification_summary_json.read_bytes() == second.diversification_summary_json.read_bytes()
    assert first.candidate_review_summary_json.read_bytes() == second.candidate_review_summary_json.read_bytes()


def test_cli_smoke_review_candidate_selection(tmp_path: Path) -> None:
    candidate_dir = _write_candidate_selection_artifacts(tmp_path / "artifacts" / "candidate_selection")
    portfolio_dir = _write_portfolio_artifacts(tmp_path / "artifacts" / "portfolios")

    result = run_cli(
        [
            "--candidate-selection-path",
            str(candidate_dir),
            "--portfolio-path",
            str(portfolio_dir),
            "--no-markdown-report",
        ]
    )

    assert result.review_dir == candidate_dir / "review"
    assert result.candidate_review_summary_json.exists()
