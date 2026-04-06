"""Tests for candidate redundancy filtering stage (Milestone 15 Issue 3)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd

from src.research.candidate_selection import CandidateRecord, run_candidate_selection
from src.research.candidate_selection.redundancy import (
    apply_redundancy_filter,
    compute_redundancy_matrix,
    resolve_redundancy_thresholds,
)


def _candidate(
    candidate_id: str,
    *,
    artifact_path: str,
    selection_rank: int,
    ic_ir: float,
    mean_ic: float,
    alpha_name: str | None = None,
) -> CandidateRecord:
    return CandidateRecord(
        candidate_id=candidate_id,
        alpha_name=alpha_name or candidate_id,
        alpha_run_id=candidate_id,
        sleeve_run_id=f"sleeve_{candidate_id}",
        mapping_name="rank_long_short",
        dataset="features_daily",
        timeframe="daily",
        evaluation_horizon=5,
        mean_ic=mean_ic,
        ic_ir=ic_ir,
        mean_rank_ic=mean_ic,
        rank_ic_ir=ic_ir,
        n_periods=32,
        sharpe_ratio=None,
        annualized_return=None,
        total_return=None,
        max_drawdown=None,
        average_turnover=None,
        selection_rank=selection_rank,
        promotion_status="eligible",
        review_status="candidate",
        artifact_path=artifact_path,
    )


def _write_sleeve_returns(run_dir: Path, rows: list[tuple[str, float]]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["ts_utc", "sleeve_return"]).to_csv(
        run_dir / "sleeve_returns.csv",
        index=False,
        lineterminator="\n",
    )


def _write_registry(alpha_root: Path, entries: list[dict[str, object]]) -> None:
    registry_path = alpha_root / "registry.jsonl"
    with registry_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")


class TestRedundancyUnit:
    def test_correlation_matrix_is_symmetric_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            d1 = root / "a"
            d2 = root / "b"
            _write_sleeve_returns(
                d1,
                [
                    ("2026-01-01T00:00:00Z", 0.01),
                    ("2026-01-02T00:00:00Z", 0.02),
                    ("2026-01-03T00:00:00Z", -0.01),
                    ("2026-01-04T00:00:00Z", 0.03),
                ],
            )
            _write_sleeve_returns(
                d2,
                [
                    ("2026-01-01T00:00:00Z", 0.02),
                    ("2026-01-02T00:00:00Z", 0.04),
                    ("2026-01-03T00:00:00Z", -0.02),
                    ("2026-01-04T00:00:00Z", 0.06),
                ],
            )

            ordered = [
                _candidate("run_a", artifact_path=str(d1), selection_rank=1, ic_ir=2.0, mean_ic=0.2),
                _candidate("run_b", artifact_path=str(d2), selection_rank=2, ic_ir=1.9, mean_ic=0.19),
            ]

            corr1, overlap1, _, _ = compute_redundancy_matrix(ordered, min_overlap_observations=3)
            corr2, overlap2, _, _ = compute_redundancy_matrix(ordered, min_overlap_observations=3)

            assert list(corr1.index) == ["run_a", "run_b"]
            assert list(corr1.columns) == ["run_a", "run_b"]
            assert float(corr1.loc["run_a", "run_b"]) == float(corr1.loc["run_b", "run_a"])
            assert int(overlap1.loc["run_a", "run_b"]) == int(overlap1.loc["run_b", "run_a"])
            pd.testing.assert_frame_equal(corr1, corr2)
            pd.testing.assert_frame_equal(overlap1, overlap2)

    def test_prunes_when_absolute_correlation_exceeds_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            d1 = root / "a"
            d2 = root / "b"
            _write_sleeve_returns(
                d1,
                [
                    ("2026-01-01T00:00:00Z", 0.01),
                    ("2026-01-02T00:00:00Z", 0.02),
                    ("2026-01-03T00:00:00Z", -0.01),
                    ("2026-01-04T00:00:00Z", 0.03),
                ],
            )
            _write_sleeve_returns(
                d2,
                [
                    ("2026-01-01T00:00:00Z", 0.02),
                    ("2026-01-02T00:00:00Z", 0.04),
                    ("2026-01-03T00:00:00Z", -0.02),
                    ("2026-01-04T00:00:00Z", 0.06),
                ],
            )

            ranked = [
                _candidate("run_a", artifact_path=str(d1), selection_rank=1, ic_ir=2.0, mean_ic=0.2),
                _candidate("run_b", artifact_path=str(d2), selection_rank=2, ic_ir=1.9, mean_ic=0.19),
            ]
            thresholds = resolve_redundancy_thresholds(
                max_pairwise_correlation=0.95,
                min_overlap_observations=3,
            )

            kept, rejected, _, _, summary = apply_redundancy_filter(ranked, thresholds=thresholds)

            assert [c.candidate_id for c in kept] == ["run_a"]
            assert len(rejected) == 1
            assert rejected[0].candidate_id == "run_b"
            assert rejected[0].rejected_against_candidate_id == "run_a"
            assert summary["pruned_by_redundancy"] == 1

    def test_insufficient_overlap_yields_undefined_not_pruned(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            d1 = root / "a"
            d2 = root / "b"
            _write_sleeve_returns(
                d1,
                [
                    ("2026-01-01T00:00:00Z", 0.01),
                    ("2026-01-02T00:00:00Z", 0.02),
                    ("2026-01-03T00:00:00Z", -0.01),
                ],
            )
            _write_sleeve_returns(
                d2,
                [
                    ("2026-01-04T00:00:00Z", 0.01),
                    ("2026-01-05T00:00:00Z", 0.02),
                    ("2026-01-06T00:00:00Z", -0.01),
                ],
            )

            ranked = [
                _candidate("run_a", artifact_path=str(d1), selection_rank=1, ic_ir=2.0, mean_ic=0.2),
                _candidate("run_b", artifact_path=str(d2), selection_rank=2, ic_ir=1.9, mean_ic=0.19),
            ]
            thresholds = resolve_redundancy_thresholds(
                max_pairwise_correlation=0.2,
                min_overlap_observations=2,
            )

            kept, rejected, corr, _, summary = apply_redundancy_filter(ranked, thresholds=thresholds)
            assert [c.candidate_id for c in kept] == ["run_a", "run_b"]
            assert rejected == []
            assert pd.isna(float(corr.loc["run_a", "run_b"]))
            assert summary["undefined_pairwise_correlation"]["insufficient_overlap"] == 1

    def test_zero_variance_pair_is_undefined_not_pruned(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            d1 = root / "a"
            d2 = root / "b"
            _write_sleeve_returns(
                d1,
                [
                    ("2026-01-01T00:00:00Z", 0.01),
                    ("2026-01-02T00:00:00Z", 0.01),
                    ("2026-01-03T00:00:00Z", 0.01),
                ],
            )
            _write_sleeve_returns(
                d2,
                [
                    ("2026-01-01T00:00:00Z", 0.03),
                    ("2026-01-02T00:00:00Z", 0.02),
                    ("2026-01-03T00:00:00Z", 0.01),
                ],
            )

            ranked = [
                _candidate("run_a", artifact_path=str(d1), selection_rank=1, ic_ir=2.0, mean_ic=0.2),
                _candidate("run_b", artifact_path=str(d2), selection_rank=2, ic_ir=1.9, mean_ic=0.19),
            ]
            thresholds = resolve_redundancy_thresholds(
                max_pairwise_correlation=0.2,
                min_overlap_observations=3,
            )

            kept, rejected, corr, _, summary = apply_redundancy_filter(ranked, thresholds=thresholds)
            assert [c.candidate_id for c in kept] == ["run_a", "run_b"]
            assert rejected == []
            assert pd.isna(float(corr.loc["run_a", "run_b"]))
            assert summary["undefined_pairwise_correlation"]["zero_variance"] == 1

    def test_disabled_filter_leaves_ranked_set_unchanged(self) -> None:
        ranked = [
            _candidate("run_a", artifact_path=".", selection_rank=1, ic_ir=2.0, mean_ic=0.2),
            _candidate("run_b", artifact_path=".", selection_rank=2, ic_ir=1.9, mean_ic=0.19),
        ]
        thresholds = resolve_redundancy_thresholds(max_pairwise_correlation=None)

        kept, rejected, corr, _, summary = apply_redundancy_filter(ranked, thresholds=thresholds)

        assert [c.candidate_id for c in kept] == ["run_a", "run_b"]
        assert rejected == []
        assert corr.empty
        assert summary["redundancy_enabled"] is False


class TestRedundancyIntegration:
    def _build_workspace(self, tmpdir: str) -> tuple[Path, Path]:
        root = Path(tmpdir)
        alpha_root = root / "alpha"
        output_root = root / "output"
        alpha_root.mkdir(parents=True, exist_ok=True)

        run_a = alpha_root / "run_a"
        run_b = alpha_root / "run_b"
        run_c = alpha_root / "run_c"

        # run_a and run_b are highly correlated, run_c is less related.
        _write_sleeve_returns(
            run_a,
            [
                ("2026-01-01T00:00:00Z", 0.01),
                ("2026-01-02T00:00:00Z", 0.02),
                ("2026-01-03T00:00:00Z", -0.01),
                ("2026-01-04T00:00:00Z", 0.03),
                ("2026-01-05T00:00:00Z", 0.00),
            ],
        )
        _write_sleeve_returns(
            run_b,
            [
                ("2026-01-01T00:00:00Z", 0.02),
                ("2026-01-02T00:00:00Z", 0.04),
                ("2026-01-03T00:00:00Z", -0.02),
                ("2026-01-04T00:00:00Z", 0.06),
                ("2026-01-05T00:00:00Z", 0.00),
            ],
        )
        _write_sleeve_returns(
            run_c,
            [
                ("2026-01-01T00:00:00Z", 0.01),
                ("2026-01-02T00:00:00Z", -0.02),
                ("2026-01-03T00:00:00Z", 0.015),
                ("2026-01-04T00:00:00Z", -0.005),
                ("2026-01-05T00:00:00Z", 0.01),
            ],
        )

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
                    "mean_ic": 0.10,
                    "ic_ir": 2.0,
                    "mean_rank_ic": 0.10,
                    "rank_ic_ir": 2.0,
                    "n_periods": 50,
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
                    "mean_ic": 0.09,
                    "ic_ir": 1.9,
                    "mean_rank_ic": 0.09,
                    "rank_ic_ir": 1.9,
                    "n_periods": 50,
                },
                "promotion_status": "eligible",
                "review_status": "candidate",
            },
            {
                "run_type": "alpha_evaluation",
                "alpha_name": "AlphaC",
                "run_id": "run_c",
                "artifact_path": str(run_c),
                "dataset": "features_daily",
                "timeframe": "daily",
                "evaluation_horizon": 5,
                "config": {"signal_mapping": {"policy": "rank_long_short"}},
                "metrics_summary": {
                    "mean_ic": 0.04,
                    "ic_ir": 1.0,
                    "mean_rank_ic": 0.04,
                    "rank_ic_ir": 1.0,
                    "n_periods": 50,
                },
                "promotion_status": "eligible",
                "review_status": "candidate",
            },
        ]
        _write_registry(alpha_root, entries)
        return alpha_root, output_root

    def test_end_to_end_redundancy_artifacts_and_rejections(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            alpha_root, output_root = self._build_workspace(tmpdir)
            result = run_candidate_selection(
                artifacts_root=alpha_root,
                output_artifacts_root=output_root,
                max_pairwise_correlation=0.95,
                min_overlap_observations=3,
            )

            assert result["eligible_count"] == 3
            assert result["selected_count"] == 2
            assert result["rejected_count"] == 1
            assert Path(result["correlation_csv"]).exists()

            corr = pd.read_csv(result["correlation_csv"])
            assert list(corr.columns) == ["candidate_id", "run_a", "run_b", "run_c"]

            rejected = pd.read_csv(result["rejected_csv"])
            redundancy_rejected = rejected[rejected["rejected_stage"] == "redundancy_filter"]
            assert len(redundancy_rejected) == 1
            assert str(redundancy_rejected.iloc[0]["candidate_id"]) == "run_b"
            assert str(redundancy_rejected.iloc[0]["rejected_against_candidate_id"]) == "run_a"

            summary = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))
            assert summary["pruned_by_redundancy"] == 1
            assert summary["retained_after_redundancy"] == 2

    def test_deterministic_artifacts_across_repeated_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            alpha_root, output_root = self._build_workspace(tmpdir)
            first = run_candidate_selection(
                artifacts_root=alpha_root,
                output_artifacts_root=output_root,
                max_pairwise_correlation=0.95,
                min_overlap_observations=3,
            )
            second = run_candidate_selection(
                artifacts_root=alpha_root,
                output_artifacts_root=output_root,
                max_pairwise_correlation=0.95,
                min_overlap_observations=3,
            )

            assert first["run_id"] == second["run_id"]
            assert Path(first["selected_csv"]).read_bytes() == Path(second["selected_csv"]).read_bytes()
            assert Path(first["rejected_csv"]).read_bytes() == Path(second["rejected_csv"]).read_bytes()
            assert Path(first["correlation_csv"]).read_bytes() == Path(second["correlation_csv"]).read_bytes()

    def test_disabled_redundancy_preserves_issue_2_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            alpha_root, output_root = self._build_workspace(tmpdir)
            result = run_candidate_selection(
                artifacts_root=alpha_root,
                output_artifacts_root=output_root,
                max_pairwise_correlation=None,
            )

            assert result["eligible_count"] == 3
            assert result["selected_count"] == 3
            assert result["rejected_count"] == 0
            assert result["redundancy_summary"]["redundancy_enabled"] is False
