"""Tests for candidate allocation governance stage (Milestone 15 Issue 4)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.research.candidate_selection import CandidateRecord, run_candidate_selection
from src.research.candidate_selection.allocation import (
    AllocationError,
    allocate_candidates,
    resolve_allocation_constraints,
)


def _candidate(candidate_id: str, rank: int) -> CandidateRecord:
    return CandidateRecord(
        candidate_id=candidate_id,
        alpha_name=f"Alpha{candidate_id[-1].upper()}",
        alpha_run_id=candidate_id,
        sleeve_run_id=f"sleeve_{candidate_id}",
        mapping_name="rank_long_short",
        dataset="features_daily",
        timeframe="daily",
        evaluation_horizon=5,
        mean_ic=0.1,
        ic_ir=1.0,
        mean_rank_ic=0.1,
        rank_ic_ir=1.0,
        n_periods=50,
        sharpe_ratio=None,
        annualized_return=None,
        total_return=None,
        max_drawdown=None,
        average_turnover=None,
        selection_rank=rank,
        promotion_status="eligible",
        review_status="candidate",
        artifact_path=".",
    )


class TestAllocationUnit:
    def test_equal_weight_allocation_sums_to_one(self) -> None:
        candidates = [_candidate("run_a", 1), _candidate("run_b", 2), _candidate("run_c", 3)]

        decisions, summary = allocate_candidates(candidates, method="equal_weight")

        assert len(decisions) == 3
        assert summary["allocation_method"] == "equal_weight"
        assert summary["allocated_candidates"] == 3
        assert sum(item.allocation_weight for item in decisions) == pytest.approx(1.0, abs=1e-12)
        assert all(item.allocation_weight == pytest.approx(1.0 / 3.0, abs=1e-12) for item in decisions)

    def test_max_weight_constraint_is_enforced(self) -> None:
        candidates = [_candidate("run_a", 1), _candidate("run_b", 2), _candidate("run_c", 3)]
        constraints = resolve_allocation_constraints(max_weight_per_candidate=0.4)

        decisions, _ = allocate_candidates(candidates, method="equal_weight", constraints=constraints)

        assert all(item.allocation_weight <= 0.4 + 1e-12 for item in decisions)
        assert sum(item.allocation_weight for item in decisions) == pytest.approx(1.0, abs=1e-12)

    def test_min_candidate_constraint_is_enforced(self) -> None:
        candidates = [_candidate("run_a", 1)]
        constraints = resolve_allocation_constraints(min_candidate_count=2)

        with pytest.raises(AllocationError, match="requires at least"):
            allocate_candidates(candidates, constraints=constraints)

    def test_deterministic_repeated_runs_identical(self) -> None:
        candidates = [_candidate("run_a", 1), _candidate("run_b", 2), _candidate("run_c", 3)]
        constraints = resolve_allocation_constraints(max_weight_per_candidate=0.5)

        first, _ = allocate_candidates(candidates, constraints=constraints)
        second, _ = allocate_candidates(candidates, constraints=constraints)

        assert [item.candidate_id for item in first] == [item.candidate_id for item in second]
        assert [item.allocation_weight for item in first] == [item.allocation_weight for item in second]

    def test_stable_output_order_follows_selection_rank(self) -> None:
        candidates = [_candidate("run_b", 2), _candidate("run_a", 1), _candidate("run_c", 3)]

        decisions, _ = allocate_candidates(candidates)

        assert [item.candidate_id for item in decisions] == ["run_a", "run_b", "run_c"]


class TestAllocationIntegration:
    def _build_workspace(self, tmpdir: str) -> tuple[Path, Path]:
        root = Path(tmpdir)
        alpha_root = root / "alpha"
        output_root = root / "output"
        alpha_root.mkdir(parents=True, exist_ok=True)

        entries = []
        for index, run_id in enumerate(("run_a", "run_b", "run_c"), start=1):
            run_dir = alpha_root / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    ("2026-01-01T00:00:00Z", 0.01 * index),
                    ("2026-01-02T00:00:00Z", -0.01 * index),
                    ("2026-01-03T00:00:00Z", 0.02 * index),
                ],
                columns=["ts_utc", "sleeve_return"],
            ).to_csv(run_dir / "sleeve_returns.csv", index=False, lineterminator="\n")

            entries.append(
                {
                    "run_type": "alpha_evaluation",
                    "alpha_name": f"Alpha{index}",
                    "run_id": run_id,
                    "artifact_path": str(run_dir),
                    "dataset": "features_daily",
                    "timeframe": "daily",
                    "evaluation_horizon": 5,
                    "config": {"signal_mapping": {"policy": "rank_long_short"}},
                    "metrics_summary": {
                        "mean_ic": 0.1 - (index * 0.01),
                        "ic_ir": 2.0 - (index * 0.2),
                        "mean_rank_ic": 0.1 - (index * 0.01),
                        "rank_ic_ir": 2.0 - (index * 0.2),
                        "n_periods": 50,
                    },
                    "promotion_status": "eligible",
                    "review_status": "candidate",
                }
            )

        registry_path = alpha_root / "registry.jsonl"
        with registry_path.open("w", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(entry, sort_keys=True) + "\n")

        return alpha_root, output_root

    def test_end_to_end_allocation_artifacts_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            alpha_root, output_root = self._build_workspace(tmpdir)
            result = run_candidate_selection(
                artifacts_root=alpha_root,
                output_artifacts_root=output_root,
                allocation_method="equal_weight",
                max_weight_per_candidate=0.5,
                max_pairwise_correlation=None,
            )

            assert result["allocation_csv"] is not None
            allocation_csv = Path(str(result["allocation_csv"]))
            assert allocation_csv.exists()

            allocation_df = pd.read_csv(allocation_csv)
            assert list(allocation_df.columns) == [
                "candidate_id",
                "alpha_name",
                "sleeve_run_id",
                "allocation_weight",
                "allocation_method",
                "pre_constraint_weight",
                "constraint_adjusted_flag",
            ]
            assert allocation_df["allocation_weight"].sum() == pytest.approx(1.0, abs=1e-12)

            selected_df = pd.read_csv(result["selected_csv"])
            assert "allocation_weight" in selected_df.columns
            assert "allocation_method" in selected_df.columns

            summary = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))
            assert summary["allocation_method"] == "equal_weight"
            assert summary["allocated_candidates"] == 3
            assert summary["weight_max"] <= 0.5 + 1e-12

            manifest = json.loads(Path(result["manifest_json"]).read_text(encoding="utf-8"))
            assert manifest["allocation_applied"] is True
            assert "allocation_weights_csv" in manifest

    def test_allocation_disabled_preserves_prior_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            alpha_root, output_root = self._build_workspace(tmpdir)
            result = run_candidate_selection(
                artifacts_root=alpha_root,
                output_artifacts_root=output_root,
                allocation_enabled=False,
            )

            assert result["allocation_csv"] is None
            selected_df = pd.read_csv(result["selected_csv"])
            assert "allocation_weight" not in selected_df.columns
            assert result["selected_count"] == 3
