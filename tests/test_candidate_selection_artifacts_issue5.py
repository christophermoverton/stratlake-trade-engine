from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.research.candidate_selection import run_candidate_selection
from src.research.candidate_selection.artifact_contract import CandidateArtifactConsistencyError
from src.research.candidate_selection.eligibility import EligibilityResult
from src.research.candidate_selection.persistence import (
    build_candidate_selection_run_id,
    write_candidate_selection_run_artifacts,
)
from src.research.candidate_selection.schema import CandidateRecord


def _write_alpha_registry(alpha_root: Path) -> None:
    run_a = alpha_root / "run_a"
    run_b = alpha_root / "run_b"
    run_a.mkdir(parents=True, exist_ok=True)
    run_b.mkdir(parents=True, exist_ok=True)

    entries = [
        {
            "run_type": "alpha_evaluation",
            "alpha_name": "AlphaA",
            "run_id": "run_a",
            "artifact_path": str(run_a),
            "dataset": "features_daily",
            "timeframe": "daily",
            "evaluation_horizon": 5,
            "config": {"signal_mapping": {"policy": "rank_long_short"}},
            "metrics_summary": {
                "mean_ic": 0.08,
                "ic_ir": 1.8,
                "mean_rank_ic": 0.09,
                "rank_ic_ir": 1.9,
                "n_periods": 100,
            },
            "promotion_status": "eligible",
            "review_status": "candidate",
        },
        {
            "run_type": "alpha_evaluation",
            "alpha_name": "AlphaB",
            "run_id": "run_b",
            "artifact_path": str(run_b),
            "dataset": "features_daily",
            "timeframe": "daily",
            "evaluation_horizon": 5,
            "config": {"signal_mapping": {"policy": "rank_long_short"}},
            "metrics_summary": {
                "mean_ic": 0.04,
                "ic_ir": 1.0,
                "mean_rank_ic": 0.05,
                "rank_ic_ir": 1.1,
                "n_periods": 100,
            },
            "promotion_status": "eligible",
            "review_status": "candidate",
        },
    ]

    registry_path = alpha_root / "registry.jsonl"
    with registry_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")


def test_run_id_changes_when_thresholds_change() -> None:
    base_filters = {"dataset": "features_daily", "timeframe": "daily"}
    base_ids = ["run_a", "run_b"]

    run_id_a = build_candidate_selection_run_id(
        filters=base_filters,
        candidate_ids=base_ids,
        primary_metric="ic_ir",
        thresholds={"min_ic_ir": 0.5},
        redundancy_thresholds={"max_pairwise_correlation": None},
        allocation_config={"allocation_enabled": True, "allocation_method": "equal_weight"},
    )
    run_id_b = build_candidate_selection_run_id(
        filters=base_filters,
        candidate_ids=base_ids,
        primary_metric="ic_ir",
        thresholds={"min_ic_ir": 1.5},
        redundancy_thresholds={"max_pairwise_correlation": None},
        allocation_config={"allocation_enabled": True, "allocation_method": "equal_weight"},
    )

    assert run_id_a != run_id_b


def test_end_to_end_artifacts_manifest_and_determinism() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        alpha_root = root / "alpha"
        output_root = root / "candidate_selection"
        alpha_root.mkdir(parents=True, exist_ok=True)
        _write_alpha_registry(alpha_root)

        first = run_candidate_selection(
            artifacts_root=alpha_root,
            output_artifacts_root=output_root,
            max_pairwise_correlation=None,
            max_candidate_count=1,
            allocation_enabled=True,
            allocation_method="equal_weight",
        )
        second = run_candidate_selection(
            artifacts_root=alpha_root,
            output_artifacts_root=output_root,
            max_pairwise_correlation=None,
            max_candidate_count=1,
            allocation_enabled=True,
            allocation_method="equal_weight",
        )

        run_dir = output_root / first["run_id"]
        required = [
            "candidate_universe.csv",
            "eligibility_filter_results.csv",
            "correlation_matrix.csv",
            "selected_candidates.csv",
            "rejected_candidates.csv",
            "allocation_weights.csv",
            "selection_summary.json",
            "manifest.json",
        ]
        for name in required:
            assert run_dir.joinpath(name).exists(), f"missing artifact: {name}"

        manifest_first = run_dir.joinpath("manifest.json").read_bytes()
        summary_first = run_dir.joinpath("selection_summary.json").read_bytes()

        assert first["run_id"] == second["run_id"]
        assert manifest_first == run_dir.joinpath("manifest.json").read_bytes()
        assert summary_first == run_dir.joinpath("selection_summary.json").read_bytes()

        manifest = json.loads(run_dir.joinpath("manifest.json").read_text(encoding="utf-8"))
        assert manifest["row_counts"]["candidate_universe"] == 2
        assert manifest["row_counts"]["selected_candidates"] == 1
        assert manifest["row_counts"]["rejected_candidates"] == 1
        assert manifest["candidate_statistics"]["selected_candidates"] == 1


def test_registry_upsert_no_duplicates() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        alpha_root = root / "alpha"
        output_root = root / "candidate_selection"
        alpha_root.mkdir(parents=True, exist_ok=True)
        _write_alpha_registry(alpha_root)

        run_candidate_selection(
            artifacts_root=alpha_root,
            output_artifacts_root=output_root,
            max_pairwise_correlation=None,
            register_run=True,
        )
        run_candidate_selection(
            artifacts_root=alpha_root,
            output_artifacts_root=output_root,
            max_pairwise_correlation=None,
            register_run=True,
        )

        registry_path = output_root / "registry.jsonl"
        lines = [line for line in registry_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(lines) == 1


def test_validation_fails_on_partition_mismatch() -> None:
    candidate_a = CandidateRecord(
        candidate_id="a",
        alpha_name="A",
        alpha_run_id="a",
        sleeve_run_id=None,
        mapping_name="rank_long_short",
        dataset="features_daily",
        timeframe="daily",
        evaluation_horizon=5,
        mean_ic=0.1,
        ic_ir=2.0,
        mean_rank_ic=0.1,
        rank_ic_ir=2.0,
        n_periods=100,
        sharpe_ratio=None,
        annualized_return=None,
        total_return=None,
        max_drawdown=None,
        average_turnover=None,
        selection_rank=1,
        promotion_status="eligible",
        review_status="candidate",
        artifact_path="artifacts/alpha/a",
    )
    candidate_b = CandidateRecord(
        candidate_id="b",
        alpha_name="B",
        alpha_run_id="b",
        sleeve_run_id=None,
        mapping_name="rank_long_short",
        dataset="features_daily",
        timeframe="daily",
        evaluation_horizon=5,
        mean_ic=0.05,
        ic_ir=1.0,
        mean_rank_ic=0.05,
        rank_ic_ir=1.0,
        n_periods=100,
        sharpe_ratio=None,
        annualized_return=None,
        total_return=None,
        max_drawdown=None,
        average_turnover=None,
        selection_rank=2,
        promotion_status="eligible",
        review_status="candidate",
        artifact_path="artifacts/alpha/b",
    )

    eligibility_results = [
        EligibilityResult(
            candidate_id="a",
            alpha_name="A",
            sleeve_run_id=None,
            dataset="features_daily",
            timeframe="daily",
            evaluation_horizon=5,
            is_eligible=True,
            failed_checks="",
            mean_ic=0.1,
            mean_rank_ic=0.1,
            ic_ir=2.0,
            rank_ic_ir=2.0,
            history_length=100,
            threshold_min_mean_ic=None,
            threshold_min_mean_rank_ic=None,
            threshold_min_ic_ir=None,
            threshold_min_rank_ic_ir=None,
            threshold_min_history_length=None,
        ),
        EligibilityResult(
            candidate_id="b",
            alpha_name="B",
            sleeve_run_id=None,
            dataset="features_daily",
            timeframe="daily",
            evaluation_horizon=5,
            is_eligible=True,
            failed_checks="",
            mean_ic=0.05,
            mean_rank_ic=0.05,
            ic_ir=1.0,
            rank_ic_ir=1.0,
            history_length=100,
            threshold_min_mean_ic=None,
            threshold_min_mean_rank_ic=None,
            threshold_min_ic_ir=None,
            threshold_min_rank_ic_ir=None,
            threshold_min_history_length=None,
        ),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(CandidateArtifactConsistencyError):
            write_candidate_selection_run_artifacts(
                run_id="candidate_selection_test",
                artifacts_root=tmpdir,
                universe=[candidate_a, candidate_b],
                eligibility_results=eligibility_results,
                selected_candidates=[candidate_a],
                rejected_rows=[],
                correlation_matrix=pd.DataFrame(),
                allocation_decisions=[],
                allocation_enabled=False,
                allocation_method=None,
                allocation_constraints={},
                allocation_summary={"allocation_enabled": False},
                filters={"dataset": "features_daily"},
                primary_metric="ic_ir",
                thresholds={},
                redundancy_thresholds={},
                redundancy_summary={},
                provenance={
                    "dataset": "features_daily",
                    "timeframe": "daily",
                    "evaluation_horizon": 5,
                },
                allocation_weight_sum_tolerance=1e-12,
            )
