"""Tests for eligibility gate framework — Milestone 15 Issue 2.

Coverage:
    Unit tests:
        - TestGateChecks: individual gate check behavior
        - TestEligibilityThresholds: threshold config serialization / factory
        - TestEligibilityResult: EligibilityResult schema / serialization
        - TestEvaluateEligibility: evaluate_eligibility() pipeline stage
        - TestFilterByEligibility: filter_by_eligibility() split logic
        - TestSummarizeEligibility: aggregation / summary dict
    Integration tests:
        - TestEligibilityArtifactPersistence: artifact writing and content
        - TestEndToEndWithGating: full run_candidate_selection with gating
    Determinism tests:
        - TestDeterminismGating: repeated runs produce identical outputs
    Regression tests:
        - TestRegressionNoGating: Issue 1 behavior undisturbed when no thresholds set
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.research.candidate_selection import (
    CandidateRecord,
    EligibilityResult,
    EligibilityThresholds,
    evaluate_eligibility,
    filter_by_eligibility,
    resolve_eligibility_thresholds,
    summarize_eligibility,
    write_eligibility_artifacts,
)
from src.research.candidate_selection.gates import (
    HISTORY_LENGTH_BELOW_THRESHOLD,
    IC_IR_BELOW_THRESHOLD,
    MEAN_IC_BELOW_THRESHOLD,
    MEAN_RANK_IC_BELOW_THRESHOLD,
    MISSING_REQUIRED_METRIC,
    NON_FINITE_METRIC,
    RANK_IC_IR_BELOW_THRESHOLD,
    evaluate_candidate_gates,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_candidate(
    candidate_id: str = "c1",
    alpha_name: str = "Alpha1",
    mean_ic: float | None = 0.05,
    ic_ir: float | None = 1.5,
    mean_rank_ic: float | None = 0.06,
    rank_ic_ir: float | None = 1.6,
    n_periods: int | None = 100,
    **kwargs,
) -> CandidateRecord:
    defaults = dict(
        alpha_run_id=candidate_id,
        sleeve_run_id=None,
        mapping_name=None,
        dataset="features_daily",
        timeframe="daily",
        evaluation_horizon=5,
        sharpe_ratio=None,
        annualized_return=None,
        total_return=None,
        max_drawdown=None,
        average_turnover=None,
        selection_rank=0,
        promotion_status="eligible",
        review_status="candidate",
        artifact_path="artifacts/alpha/c1/",
    )
    defaults.update(kwargs)
    return CandidateRecord(
        candidate_id=candidate_id,
        alpha_name=alpha_name,
        mean_ic=mean_ic,
        ic_ir=ic_ir,
        mean_rank_ic=mean_rank_ic,
        rank_ic_ir=rank_ic_ir,
        n_periods=n_periods,
        **defaults,
    )


def _default_thresholds(**overrides) -> EligibilityThresholds:
    """Return thresholds with all gates disabled by default."""
    return resolve_eligibility_thresholds(**overrides)


def _passing_candidates() -> list[CandidateRecord]:
    return [
        _make_candidate("c1", "Alpha1", mean_ic=0.05, ic_ir=1.5, mean_rank_ic=0.06, rank_ic_ir=1.6, n_periods=100),
        _make_candidate("c2", "Alpha2", mean_ic=0.04, ic_ir=1.2, mean_rank_ic=0.05, rank_ic_ir=1.3, n_periods=80),
        _make_candidate("c3", "Alpha3", mean_ic=0.03, ic_ir=0.8, mean_rank_ic=0.04, rank_ic_ir=0.9, n_periods=60),
    ]


# ---------------------------------------------------------------------------
# Unit Tests: Gate Checks
# ---------------------------------------------------------------------------


class TestGateChecks:
    """Test individual gate check behavior in evaluate_candidate_gates()."""

    def test_passes_all_disabled_thresholds(self):
        candidate = _make_candidate()
        thresholds = _default_thresholds()
        failed = evaluate_candidate_gates(candidate, thresholds)
        assert failed == []

    def test_mean_ic_below_threshold(self):
        candidate = _make_candidate(mean_ic=0.01)
        thresholds = _default_thresholds(min_mean_ic=0.03)
        failed = evaluate_candidate_gates(candidate, thresholds)
        assert MEAN_IC_BELOW_THRESHOLD in failed

    def test_mean_ic_at_threshold_passes(self):
        candidate = _make_candidate(mean_ic=0.03)
        thresholds = _default_thresholds(min_mean_ic=0.03)
        failed = evaluate_candidate_gates(candidate, thresholds)
        assert MEAN_IC_BELOW_THRESHOLD not in failed

    def test_mean_rank_ic_below_threshold(self):
        candidate = _make_candidate(mean_rank_ic=0.01)
        thresholds = _default_thresholds(min_mean_rank_ic=0.04)
        failed = evaluate_candidate_gates(candidate, thresholds)
        assert MEAN_RANK_IC_BELOW_THRESHOLD in failed

    def test_ic_ir_below_threshold(self):
        candidate = _make_candidate(ic_ir=0.5)
        thresholds = _default_thresholds(min_ic_ir=1.0)
        failed = evaluate_candidate_gates(candidate, thresholds)
        assert IC_IR_BELOW_THRESHOLD in failed

    def test_ic_ir_at_threshold_passes(self):
        candidate = _make_candidate(ic_ir=1.0)
        thresholds = _default_thresholds(min_ic_ir=1.0)
        failed = evaluate_candidate_gates(candidate, thresholds)
        assert IC_IR_BELOW_THRESHOLD not in failed

    def test_rank_ic_ir_below_threshold(self):
        candidate = _make_candidate(rank_ic_ir=0.4)
        thresholds = _default_thresholds(min_rank_ic_ir=0.8)
        failed = evaluate_candidate_gates(candidate, thresholds)
        assert RANK_IC_IR_BELOW_THRESHOLD in failed

    def test_history_length_below_threshold(self):
        candidate = _make_candidate(n_periods=30)
        thresholds = _default_thresholds(min_history_length=50)
        failed = evaluate_candidate_gates(candidate, thresholds)
        assert HISTORY_LENGTH_BELOW_THRESHOLD in failed

    def test_history_length_at_threshold_passes(self):
        candidate = _make_candidate(n_periods=50)
        thresholds = _default_thresholds(min_history_length=50)
        failed = evaluate_candidate_gates(candidate, thresholds)
        assert HISTORY_LENGTH_BELOW_THRESHOLD not in failed

    def test_missing_metric_not_penalised_by_threshold(self):
        """A None metric is NOT penalised by its threshold gate unless require_* is set."""
        candidate = _make_candidate(mean_ic=None)
        thresholds = _default_thresholds(min_mean_ic=0.05)
        failed = evaluate_candidate_gates(candidate, thresholds)
        # Should not fail because mean_ic is None and require_mean_ic is False
        assert MEAN_IC_BELOW_THRESHOLD not in failed
        assert f"{MISSING_REQUIRED_METRIC}:mean_ic" not in failed

    def test_require_mean_ic_fails_when_none(self):
        candidate = _make_candidate(mean_ic=None)
        thresholds = _default_thresholds(require_mean_ic=True)
        failed = evaluate_candidate_gates(candidate, thresholds)
        assert f"{MISSING_REQUIRED_METRIC}:mean_ic" in failed

    def test_require_ic_ir_fails_when_none(self):
        candidate = _make_candidate(ic_ir=None)
        thresholds = _default_thresholds(require_ic_ir=True)
        failed = evaluate_candidate_gates(candidate, thresholds)
        assert f"{MISSING_REQUIRED_METRIC}:ic_ir" in failed

    def test_require_mean_ic_passes_when_present(self):
        candidate = _make_candidate(mean_ic=0.05)
        thresholds = _default_thresholds(require_mean_ic=True)
        failed = evaluate_candidate_gates(candidate, thresholds)
        assert f"{MISSING_REQUIRED_METRIC}:mean_ic" not in failed

    def test_non_finite_nan_fails(self):
        candidate = _make_candidate(ic_ir=float("nan"))
        thresholds = _default_thresholds()
        failed = evaluate_candidate_gates(candidate, thresholds)
        assert f"{NON_FINITE_METRIC}:ic_ir" in failed

    def test_non_finite_inf_fails(self):
        candidate = _make_candidate(mean_ic=math.inf)
        thresholds = _default_thresholds()
        failed = evaluate_candidate_gates(candidate, thresholds)
        assert f"{NON_FINITE_METRIC}:mean_ic" in failed

    def test_non_finite_neg_inf_fails(self):
        candidate = _make_candidate(rank_ic_ir=-math.inf)
        thresholds = _default_thresholds()
        failed = evaluate_candidate_gates(candidate, thresholds)
        assert f"{NON_FINITE_METRIC}:rank_ic_ir" in failed

    def test_multiple_failures_collected(self):
        candidate = _make_candidate(mean_ic=0.01, ic_ir=0.3, n_periods=10)
        thresholds = _default_thresholds(min_mean_ic=0.03, min_ic_ir=1.0, min_history_length=50)
        failed = evaluate_candidate_gates(candidate, thresholds)
        assert MEAN_IC_BELOW_THRESHOLD in failed
        assert IC_IR_BELOW_THRESHOLD in failed
        assert HISTORY_LENGTH_BELOW_THRESHOLD in failed
        assert len(failed) == 3

    def test_failure_ordering_deterministic(self):
        """Failures must always appear in the same canonical order."""
        candidate = _make_candidate(mean_ic=0.01, ic_ir=0.3, n_periods=10)
        thresholds = _default_thresholds(min_mean_ic=0.03, min_ic_ir=1.0, min_history_length=50)
        failed_1 = evaluate_candidate_gates(candidate, thresholds)
        failed_2 = evaluate_candidate_gates(candidate, thresholds)
        assert failed_1 == failed_2
        # Check canonical order: mean_ic first, then ic_ir, then history
        assert failed_1.index(MEAN_IC_BELOW_THRESHOLD) < failed_1.index(IC_IR_BELOW_THRESHOLD)
        assert failed_1.index(IC_IR_BELOW_THRESHOLD) < failed_1.index(HISTORY_LENGTH_BELOW_THRESHOLD)

    def test_disabled_threshold_never_fails(self):
        """When all thresholds disabled, no failures should ever be produced."""
        candidate = _make_candidate(mean_ic=-99.0, ic_ir=-99.0, n_periods=1)
        thresholds = _default_thresholds()
        failed = evaluate_candidate_gates(candidate, thresholds)
        assert failed == []


# ---------------------------------------------------------------------------
# Unit Tests: EligibilityThresholds
# ---------------------------------------------------------------------------


class TestEligibilityThresholds:
    """Test threshold configuration object."""

    def test_default_thresholds_all_disabled(self):
        t = EligibilityThresholds()
        assert t.min_mean_ic is None
        assert t.min_mean_rank_ic is None
        assert t.min_ic_ir is None
        assert t.min_rank_ic_ir is None
        assert t.min_history_length is None
        assert t.require_mean_ic is False
        assert t.require_ic_ir is False

    def test_to_dict_roundtrip(self):
        t = EligibilityThresholds(min_mean_ic=0.03, min_ic_ir=1.0, min_history_length=50)
        d = t.to_dict()
        assert d["min_mean_ic"] == 0.03
        assert d["min_ic_ir"] == 1.0
        assert d["min_history_length"] == 50
        assert d["require_mean_ic"] is False
        assert d["require_ic_ir"] is False

    def test_resolve_factory_matches_dataclass(self):
        resolved = resolve_eligibility_thresholds(min_mean_ic=0.02, require_ic_ir=True)
        assert resolved.min_mean_ic == 0.02
        assert resolved.require_ic_ir is True
        assert resolved.min_ic_ir is None

    def test_frozen(self):
        t = EligibilityThresholds()
        with pytest.raises((AttributeError, TypeError)):
            t.min_mean_ic = 0.05  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Unit Tests: EligibilityResult
# ---------------------------------------------------------------------------


class TestEligibilityResult:
    """Test EligibilityResult schema."""

    def _make_result(self, is_eligible: bool = True, failed_checks: str = "") -> EligibilityResult:
        return EligibilityResult(
            candidate_id="c1",
            alpha_name="Alpha1",
            sleeve_run_id="sleeve_1",
            dataset="features_daily",
            timeframe="daily",
            evaluation_horizon=5,
            is_eligible=is_eligible,
            failed_checks=failed_checks,
            mean_ic=0.05,
            mean_rank_ic=0.06,
            ic_ir=1.5,
            rank_ic_ir=1.6,
            history_length=100,
            threshold_min_mean_ic=0.03,
            threshold_min_mean_rank_ic=None,
            threshold_min_ic_ir=0.5,
            threshold_min_rank_ic_ir=None,
            threshold_min_history_length=50,
        )

    def test_eligible_result(self):
        r = self._make_result(is_eligible=True, failed_checks="")
        assert r.is_eligible is True
        assert r.failed_checks == ""

    def test_rejected_result_with_checks(self):
        r = self._make_result(
            is_eligible=False,
            failed_checks="ic_ir_below_threshold|mean_ic_below_threshold",
        )
        assert r.is_eligible is False
        assert "ic_ir_below_threshold" in r.failed_checks

    def test_to_dict(self):
        r = self._make_result()
        d = r.to_dict()
        assert d["candidate_id"] == "c1"
        assert d["is_eligible"] is True
        assert d["threshold_min_mean_ic"] == 0.03

    def test_csv_columns_complete(self):
        cols = EligibilityResult.csv_columns()
        assert "candidate_id" in cols
        assert "is_eligible" in cols
        assert "failed_checks" in cols
        assert "mean_ic" in cols
        assert "threshold_min_mean_ic" in cols

    def test_frozen(self):
        r = self._make_result()
        with pytest.raises((AttributeError, TypeError)):
            r.is_eligible = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Unit Tests: evaluate_eligibility()
# ---------------------------------------------------------------------------


class TestEvaluateEligibility:
    """Test evaluate_eligibility() pipeline stage."""

    def test_all_pass_no_thresholds(self):
        candidates = _passing_candidates()
        thresholds = _default_thresholds()
        results = evaluate_eligibility(candidates, thresholds)
        assert len(results) == 3
        assert all(r.is_eligible for r in results)
        assert all(r.failed_checks == "" for r in results)

    def test_some_fail_with_threshold(self):
        candidates = _passing_candidates()
        # Only c1 (ic_ir=1.5) and c2 (ic_ir=1.2) pass min_ic_ir=1.0; c3 (ic_ir=0.8) fails
        thresholds = _default_thresholds(min_ic_ir=1.0)
        results = evaluate_eligibility(candidates, thresholds)
        by_id = {r.candidate_id: r for r in results}
        assert by_id["c1"].is_eligible is True
        assert by_id["c2"].is_eligible is True
        assert by_id["c3"].is_eligible is False
        assert IC_IR_BELOW_THRESHOLD in by_id["c3"].failed_checks

    def test_all_fail_strict_threshold(self):
        candidates = _passing_candidates()
        thresholds = _default_thresholds(min_ic_ir=99.0)
        results = evaluate_eligibility(candidates, thresholds)
        assert all(not r.is_eligible for r in results)

    def test_order_preserved(self):
        candidates = _passing_candidates()
        thresholds = _default_thresholds()
        results = evaluate_eligibility(candidates, thresholds)
        assert [r.candidate_id for r in results] == [c.candidate_id for c in candidates]

    def test_threshold_snapshot_recorded(self):
        candidates = _passing_candidates()
        thresholds = _default_thresholds(min_mean_ic=0.02, min_ic_ir=0.5)
        results = evaluate_eligibility(candidates, thresholds)
        for r in results:
            assert r.threshold_min_mean_ic == 0.02
            assert r.threshold_min_ic_ir == 0.5

    def test_empty_candidates(self):
        results = evaluate_eligibility([], _default_thresholds())
        assert results == []

    def test_history_length_mirrored(self):
        c = _make_candidate(n_periods=75)
        results = evaluate_eligibility([c], _default_thresholds())
        assert results[0].history_length == 75

    def test_none_n_periods_mirrored(self):
        c = _make_candidate(n_periods=None)
        results = evaluate_eligibility([c], _default_thresholds())
        assert results[0].history_length is None


# ---------------------------------------------------------------------------
# Unit Tests: filter_by_eligibility()
# ---------------------------------------------------------------------------


class TestFilterByEligibility:
    """Test filter_by_eligibility() split logic."""

    def test_all_eligible(self):
        candidates = _passing_candidates()
        thresholds = _default_thresholds()
        results = evaluate_eligibility(candidates, thresholds)
        eligible, rejected = filter_by_eligibility(candidates, results)
        assert len(eligible) == 3
        assert len(rejected) == 0

    def test_some_rejected(self):
        candidates = _passing_candidates()
        thresholds = _default_thresholds(min_ic_ir=1.0)
        results = evaluate_eligibility(candidates, thresholds)
        eligible, rejected = filter_by_eligibility(candidates, results)
        assert len(eligible) == 2
        assert len(rejected) == 1
        assert rejected[0].candidate_id == "c3"

    def test_all_rejected(self):
        candidates = _passing_candidates()
        thresholds = _default_thresholds(min_ic_ir=99.0)
        results = evaluate_eligibility(candidates, thresholds)
        eligible, rejected = filter_by_eligibility(candidates, results)
        assert len(eligible) == 0
        assert len(rejected) == 3

    def test_order_preserved_in_eligible(self):
        candidates = _passing_candidates()
        thresholds = _default_thresholds(min_ic_ir=1.0)
        results = evaluate_eligibility(candidates, thresholds)
        eligible, _ = filter_by_eligibility(candidates, results)
        assert [c.candidate_id for c in eligible] == ["c1", "c2"]

    def test_order_preserved_in_rejected(self):
        candidates = _passing_candidates()
        thresholds = _default_thresholds(min_ic_ir=1.5)
        results = evaluate_eligibility(candidates, thresholds)
        _, rejected = filter_by_eligibility(candidates, results)
        assert [c.candidate_id for c in rejected] == ["c2", "c3"]


# ---------------------------------------------------------------------------
# Unit Tests: summarize_eligibility()
# ---------------------------------------------------------------------------


class TestSummarizeEligibility:
    """Test eligibility summary aggregation."""

    def test_summary_all_eligible(self):
        candidates = _passing_candidates()
        thresholds = _default_thresholds()
        results = evaluate_eligibility(candidates, thresholds)
        summary = summarize_eligibility(results, thresholds)
        assert summary["total_candidates"] == 3
        assert summary["eligible_candidates"] == 3
        assert summary["rejected_candidates"] == 0
        assert summary["rejection_counts_by_reason"] == {}

    def test_summary_some_rejected(self):
        candidates = _passing_candidates()
        thresholds = _default_thresholds(min_ic_ir=1.0)
        results = evaluate_eligibility(candidates, thresholds)
        summary = summarize_eligibility(results, thresholds)
        assert summary["total_candidates"] == 3
        assert summary["eligible_candidates"] == 2
        assert summary["rejected_candidates"] == 1
        assert summary["rejection_counts_by_reason"][IC_IR_BELOW_THRESHOLD] == 1

    def test_summary_multiple_failures_counted_per_reason(self):
        candidates = [
            _make_candidate("c1", mean_ic=0.01, ic_ir=0.3),
            _make_candidate("c2", mean_ic=0.01, ic_ir=2.0),  # only mean_ic fails
        ]
        thresholds = _default_thresholds(min_mean_ic=0.03, min_ic_ir=1.0)
        results = evaluate_eligibility(candidates, thresholds)
        summary = summarize_eligibility(results, thresholds)
        # c1 fails both mean_ic AND ic_ir; c2 fails only mean_ic
        assert summary["rejection_counts_by_reason"][MEAN_IC_BELOW_THRESHOLD] == 2
        assert summary["rejection_counts_by_reason"][IC_IR_BELOW_THRESHOLD] == 1

    def test_summary_includes_threshold_snapshot(self):
        thresholds = _default_thresholds(min_mean_ic=0.03, min_ic_ir=1.0)
        results = evaluate_eligibility(_passing_candidates(), thresholds)
        summary = summarize_eligibility(results, thresholds)
        assert summary["thresholds"]["min_mean_ic"] == 0.03
        assert summary["thresholds"]["min_ic_ir"] == 1.0

    def test_summary_rejection_counts_sorted(self):
        candidates = _passing_candidates()
        thresholds = _default_thresholds(min_mean_ic=0.03, min_ic_ir=3.0)
        results = evaluate_eligibility(candidates, thresholds)
        summary = summarize_eligibility(results, thresholds)
        keys = list(summary["rejection_counts_by_reason"].keys())
        assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# Integration Tests: Artifact Persistence
# ---------------------------------------------------------------------------


class TestEligibilityArtifactPersistence:
    """Test that eligibility artifacts are written correctly."""

    def _setup(self, tmpdir: str, thresholds_kwargs: dict | None = None):
        """Helper: create candidates, evaluate, and write artifacts."""
        candidates = _passing_candidates()
        thresholds = _default_thresholds(**(thresholds_kwargs or {}))
        results = evaluate_eligibility(candidates, thresholds)
        eligible, rejected = filter_by_eligibility(candidates, results)
        from src.research.candidate_selection import (
            build_candidate_selection_run_id,
            rank_candidates,
            write_candidate_selection_artifacts,
        )
        filters: dict = {}
        run_id = build_candidate_selection_run_id(
            filters=filters,
            candidate_ids=[c.candidate_id for c in candidates],
            primary_metric="ic_ir",
        )
        ranked = rank_candidates(eligible)
        write_candidate_selection_artifacts(
            universe=ranked,
            selected=ranked,
            run_id=run_id,
            filters=filters,
            primary_metric="ic_ir",
            artifacts_root=tmpdir,
        )
        from src.research.candidate_selection.eligibility import summarize_eligibility as _sum
        elig_summary = _sum(results, thresholds)
        elig_csv, sel_csv, rej_csv, sum_json = write_eligibility_artifacts(
            eligibility_results=results,
            eligible_candidates=ranked,
            rejected_candidates=rejected,
            eligibility_summary={**elig_summary, "run_id": run_id},
            run_id=run_id,
            artifacts_root=tmpdir,
        )
        return run_id, results, eligible, rejected, elig_csv, sel_csv, rej_csv, sum_json

    def test_eligibility_csv_exists_and_has_correct_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _, _, _, _, elig_csv, _, _, _ = self._setup(tmpdir)
            assert elig_csv.exists()
            df = pd.read_csv(elig_csv)
            for col in EligibilityResult.csv_columns():
                assert col in df.columns, f"Missing column: {col}"

    def test_eligibility_csv_row_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _, results, _, _, elig_csv, _, _, _ = self._setup(tmpdir)
            df = pd.read_csv(elig_csv)
            assert len(df) == len(results)

    def test_selected_csv_contains_only_eligible(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _, _, eligible, _, _, sel_csv, _, _ = self._setup(
                tmpdir, thresholds_kwargs={"min_ic_ir": 1.0}
            )
            df = pd.read_csv(sel_csv)
            assert len(df) == len(eligible)

    def test_rejected_csv_contains_only_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _, _, _, rejected, _, _, rej_csv, _ = self._setup(
                tmpdir, thresholds_kwargs={"min_ic_ir": 1.0}
            )
            df = pd.read_csv(rej_csv)
            assert len(df) == len(rejected)

    def test_rejected_csv_has_failed_checks_column(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _, _, _, _, _, _, rej_csv, _ = self._setup(
                tmpdir, thresholds_kwargs={"min_ic_ir": 1.0}
            )
            df = pd.read_csv(rej_csv)
            assert "failed_checks" in df.columns

    def test_rejected_csv_failure_labels_correct(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _, _, _, rejected, _, _, rej_csv, _ = self._setup(
                tmpdir, thresholds_kwargs={"min_ic_ir": 1.0}
            )
            if rejected:
                df = pd.read_csv(rej_csv)
                for _, row in df.iterrows():
                    assert IC_IR_BELOW_THRESHOLD in str(row["failed_checks"])

    def test_selection_summary_json_has_eligibility_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _, _, _, _, _, _, _, sum_json = self._setup(
                tmpdir, thresholds_kwargs={"min_ic_ir": 1.0}
            )
            data = json.loads(sum_json.read_text())
            assert "eligible_candidates" in data
            assert "rejected_candidates" in data
            assert "rejection_counts_by_reason" in data
            assert "thresholds" in data

    def test_manifest_updated_with_eligibility_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_id, _, _, _, elig_csv, _, _, _ = self._setup(tmpdir)
            from src.research.candidate_selection import resolve_candidate_selection_artifact_dir
            artifact_dir = resolve_candidate_selection_artifact_dir(run_id, tmpdir)
            manifest = json.loads((artifact_dir / "manifest.json").read_text())
            assert "eligibility_filter_results_csv" in manifest
            assert "rejected_candidates_csv" in manifest
            assert "eligibility_applied" in manifest
            assert manifest["eligibility_applied"] is True

    def test_empty_eligible_writes_empty_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _, _, _, _, _, sel_csv, _, _ = self._setup(
                tmpdir, thresholds_kwargs={"min_ic_ir": 99.0}  # all fail
            )
            df = pd.read_csv(sel_csv)
            assert len(df) == 0

    def test_empty_rejected_writes_empty_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _, _, _, _, _, _, rej_csv, _ = self._setup(tmpdir)
            df = pd.read_csv(rej_csv)
            assert len(df) == 0


# ---------------------------------------------------------------------------
# Integration Tests: End-to-End Pipeline with Gating
# ---------------------------------------------------------------------------


class TestEndToEndWithGating:
    """Test full run_candidate_selection() with eligibility gating."""

    def _build_mock_registry(self, tmpdir: str) -> Path:
        """Write a minimal alpha evaluation registry for integration testing."""
        import json as _json
        alpha_root = Path(tmpdir) / "alpha"
        alpha_root.mkdir()
        registry_path = alpha_root / "registry.jsonl"

        entries = [
            {
                "run_type": "alpha_evaluation",
                "alpha_name": "AlphaA",
                "run_id": "run_a",
                "artifact_path": str(alpha_root / "run_a"),
                "dataset": "features_daily",
                "timeframe": "daily",
                "evaluation_horizon": 5,
                "config": {"signal_mapping": {"policy": "rank_long_short"}},
                "metrics_summary": {
                    "mean_ic": 0.06,
                    "ic_ir": 1.8,
                    "mean_rank_ic": 0.07,
                    "rank_ic_ir": 1.9,
                    "n_periods": 120,
                },
                "promotion_status": "eligible",
                "review_status": "candidate",
            },
            {
                "run_type": "alpha_evaluation",
                "alpha_name": "AlphaB",
                "run_id": "run_b",
                "artifact_path": str(alpha_root / "run_b"),
                "dataset": "features_daily",
                "timeframe": "daily",
                "evaluation_horizon": 5,
                "config": {"signal_mapping": {"policy": "rank_long_short"}},
                "metrics_summary": {
                    "mean_ic": 0.04,
                    "ic_ir": 0.9,
                    "mean_rank_ic": 0.05,
                    "rank_ic_ir": 1.0,
                    "n_periods": 80,
                },
                "promotion_status": "eligible",
                "review_status": "candidate",
            },
        ]
        with registry_path.open("w", encoding="utf-8") as f:
            for entry in entries:
                f.write(_json.dumps(entry) + "\n")
        return alpha_root

    def test_run_returns_required_keys(self):
        from src.research.candidate_selection import run_candidate_selection
        with tempfile.TemporaryDirectory() as tmpdir:
            alpha_root = self._build_mock_registry(tmpdir)
            output_root = Path(tmpdir) / "output"
            result = run_candidate_selection(
                artifacts_root=alpha_root,
                output_artifacts_root=output_root,
            )
            for key in (
                "run_id", "universe_count", "eligible_count", "rejected_count",
                "selected_count", "primary_metric", "filters", "thresholds",
                "eligibility_summary", "redundancy_thresholds", "redundancy_summary",
                "universe_csv", "selected_csv",
                "rejected_csv", "eligibility_csv", "summary_json", "manifest_json",
                "correlation_csv",
            ):
                assert key in result, f"Missing result key: {key}"

    def test_gating_filters_correct_candidates(self):
        from src.research.candidate_selection import run_candidate_selection
        with tempfile.TemporaryDirectory() as tmpdir:
            alpha_root = self._build_mock_registry(tmpdir)
            output_root = Path(tmpdir) / "output"
            # min_ic_ir=1.0 → AlphaA (1.8) passes, AlphaB (0.9) fails
            result = run_candidate_selection(
                artifacts_root=alpha_root,
                output_artifacts_root=output_root,
                min_ic_ir=1.0,
            )
            assert result["eligible_count"] == 1
            assert result["rejected_count"] == 1
            assert result["selected_count"] == 1

    def test_all_pass_when_no_thresholds(self):
        from src.research.candidate_selection import run_candidate_selection
        with tempfile.TemporaryDirectory() as tmpdir:
            alpha_root = self._build_mock_registry(tmpdir)
            output_root = Path(tmpdir) / "output"
            result = run_candidate_selection(
                artifacts_root=alpha_root,
                output_artifacts_root=output_root,
            )
            assert result["eligible_count"] == 2
            assert result["rejected_count"] == 0

    def test_artifacts_created(self):
        from src.research.candidate_selection import run_candidate_selection
        with tempfile.TemporaryDirectory() as tmpdir:
            alpha_root = self._build_mock_registry(tmpdir)
            output_root = Path(tmpdir) / "output"
            result = run_candidate_selection(
                artifacts_root=alpha_root,
                output_artifacts_root=output_root,
                min_ic_ir=1.0,
            )
            for key in ("universe_csv", "selected_csv", "rejected_csv", "eligibility_csv",
                    "correlation_csv", "summary_json", "manifest_json"):
                assert Path(result[key]).exists(), f"Missing artifact: {key}"

    def test_eligibility_csv_structure(self):
        from src.research.candidate_selection import run_candidate_selection
        with tempfile.TemporaryDirectory() as tmpdir:
            alpha_root = self._build_mock_registry(tmpdir)
            output_root = Path(tmpdir) / "output"
            result = run_candidate_selection(
                artifacts_root=alpha_root,
                output_artifacts_root=output_root,
                min_ic_ir=1.0,
            )
            df = pd.read_csv(result["eligibility_csv"])
            assert "is_eligible" in df.columns
            assert "failed_checks" in df.columns
            assert "threshold_min_ic_ir" in df.columns
            assert len(df) == 2  # both candidates appear in eligibility table

    def test_rejected_csv_has_failed_checks_for_gated_candidate(self):
        from src.research.candidate_selection import run_candidate_selection
        with tempfile.TemporaryDirectory() as tmpdir:
            alpha_root = self._build_mock_registry(tmpdir)
            output_root = Path(tmpdir) / "output"
            result = run_candidate_selection(
                artifacts_root=alpha_root,
                output_artifacts_root=output_root,
                min_ic_ir=1.0,
            )
            df = pd.read_csv(result["rejected_csv"])
            assert len(df) == 1
            assert IC_IR_BELOW_THRESHOLD in str(df.iloc[0]["failed_checks"])

    def test_summary_json_has_eligibility_section(self):
        from src.research.candidate_selection import run_candidate_selection
        with tempfile.TemporaryDirectory() as tmpdir:
            alpha_root = self._build_mock_registry(tmpdir)
            output_root = Path(tmpdir) / "output"
            result = run_candidate_selection(
                artifacts_root=alpha_root,
                output_artifacts_root=output_root,
                min_ic_ir=1.0,
            )
            summary = json.loads(Path(result["summary_json"]).read_text())
            assert summary["eligible_candidates"] == 1
            assert summary["rejected_candidates"] == 1
            assert summary["thresholds"]["min_ic_ir"] == 1.0

    def test_max_candidates_applied_after_gating(self):
        from src.research.candidate_selection import run_candidate_selection
        with tempfile.TemporaryDirectory() as tmpdir:
            alpha_root = self._build_mock_registry(tmpdir)
            output_root = Path(tmpdir) / "output"
            # All pass gating, max_candidate_count=1
            result = run_candidate_selection(
                artifacts_root=alpha_root,
                output_artifacts_root=output_root,
                max_candidate_count=1,
            )
            assert result["selected_count"] == 1
            assert result["eligible_count"] == 2


# ---------------------------------------------------------------------------
# Determinism Tests
# ---------------------------------------------------------------------------


class TestDeterminismGating:
    """Eligibility gating must be deterministic across repeated evaluations."""

    def test_repeated_gate_evaluation_identical(self):
        candidates = _passing_candidates()
        thresholds = _default_thresholds(min_ic_ir=1.0, min_mean_ic=0.04)
        results_1 = evaluate_eligibility(candidates, thresholds)
        results_2 = evaluate_eligibility(candidates, thresholds)
        for r1, r2 in zip(results_1, results_2):
            assert r1.candidate_id == r2.candidate_id
            assert r1.is_eligible == r2.is_eligible
            assert r1.failed_checks == r2.failed_checks

    def test_rejection_reason_ordering_stable(self):
        candidate = _make_candidate(mean_ic=0.01, ic_ir=0.3, n_periods=10)
        thresholds = _default_thresholds(min_mean_ic=0.03, min_ic_ir=1.0, min_history_length=50)
        for _ in range(5):
            failed = evaluate_candidate_gates(candidate, thresholds)
            assert failed == [
                MEAN_IC_BELOW_THRESHOLD,
                IC_IR_BELOW_THRESHOLD,
                HISTORY_LENGTH_BELOW_THRESHOLD,
            ]

    def test_filter_split_stable(self):
        candidates = _passing_candidates()
        thresholds = _default_thresholds(min_ic_ir=1.0)
        for _ in range(3):
            results = evaluate_eligibility(candidates, thresholds)
            eligible, rejected = filter_by_eligibility(candidates, results)
            assert [c.candidate_id for c in eligible] == ["c1", "c2"]
            assert [c.candidate_id for c in rejected] == ["c3"]

    def test_failed_checks_string_deterministic(self):
        candidate = _make_candidate(mean_ic=0.01, ic_ir=0.3)
        thresholds = _default_thresholds(min_mean_ic=0.03, min_ic_ir=1.0)
        results_1 = evaluate_eligibility([candidate], thresholds)
        results_2 = evaluate_eligibility([candidate], thresholds)
        assert results_1[0].failed_checks == results_2[0].failed_checks


# ---------------------------------------------------------------------------
# Regression Tests: Issue 1 behavior preserved
# ---------------------------------------------------------------------------


class TestRegressionNoGating:
    """When no thresholds are configured, behavior must be identical to Issue 1."""

    def test_all_candidates_pass_with_default_thresholds(self):
        candidates = _passing_candidates()
        thresholds = EligibilityThresholds()  # all disabled
        results = evaluate_eligibility(candidates, thresholds)
        eligible, rejected = filter_by_eligibility(candidates, results)
        assert len(eligible) == len(candidates)
        assert len(rejected) == 0

    def test_ranking_unaffected_by_gate_pass(self):
        from src.research.candidate_selection import rank_candidates
        candidates = _passing_candidates()
        thresholds = EligibilityThresholds()
        results = evaluate_eligibility(candidates, thresholds)
        eligible, _ = filter_by_eligibility(candidates, results)
        ranked = rank_candidates(eligible, primary_metric="ic_ir")
        assert ranked[0].candidate_id == "c1"
        assert ranked[1].candidate_id == "c2"
        assert ranked[2].candidate_id == "c3"

    def test_universe_count_preserved(self):
        from src.research.candidate_selection import run_candidate_selection
        with tempfile.TemporaryDirectory() as tmpdir:
            # Build a minimal registry
            import json as _json
            alpha_root = Path(tmpdir) / "alpha"
            alpha_root.mkdir()
            registry = alpha_root / "registry.jsonl"
            entries = [
                {
                    "run_type": "alpha_evaluation",
                    "alpha_name": f"Alpha{i}",
                    "run_id": f"run_{i}",
                    "artifact_path": str(alpha_root / f"run_{i}"),
                    "dataset": "features",
                    "timeframe": "daily",
                    "evaluation_horizon": 5,
                    "config": {"signal_mapping": {"policy": "rank_long_short"}},
                    "metrics_summary": {
                        "mean_ic": 0.05 - i * 0.01,
                        "ic_ir": 1.5 - i * 0.3,
                        "mean_rank_ic": None,
                        "rank_ic_ir": None,
                        "n_periods": 100,
                    },
                    "promotion_status": "eligible",
                    "review_status": "candidate",
                }
                for i in range(3)
            ]
            with registry.open("w") as f:
                for e in entries:
                    f.write(_json.dumps(e) + "\n")

            output_root = Path(tmpdir) / "output"
            result = run_candidate_selection(
                artifacts_root=alpha_root,
                output_artifacts_root=output_root,
            )
            assert result["universe_count"] == 3
            assert result["eligible_count"] == 3
            assert result["rejected_count"] == 0

    def test_no_rejected_csv_has_zero_rows(self):
        """With no thresholds, rejected_candidates.csv must be empty."""
        from src.research.candidate_selection import run_candidate_selection
        with tempfile.TemporaryDirectory() as tmpdir:
            import json as _json
            alpha_root = Path(tmpdir) / "alpha"
            alpha_root.mkdir()
            registry = alpha_root / "registry.jsonl"
            entry = {
                "run_type": "alpha_evaluation",
                "alpha_name": "AlphaX",
                "run_id": "run_x",
                "artifact_path": str(alpha_root / "run_x"),
                "dataset": "features",
                "timeframe": "daily",
                "evaluation_horizon": 5,
                "config": {"signal_mapping": {"policy": "rank_long_short"}},
                "metrics_summary": {
                    "mean_ic": 0.05,
                    "ic_ir": 1.5,
                    "mean_rank_ic": None,
                    "rank_ic_ir": None,
                    "n_periods": 100,
                },
                "promotion_status": "eligible",
                "review_status": "candidate",
            }
            with registry.open("w") as f:
                f.write(_json.dumps(entry) + "\n")

            output_root = Path(tmpdir) / "output"
            result = run_candidate_selection(
                artifacts_root=alpha_root,
                output_artifacts_root=output_root,
            )
            df = pd.read_csv(result["rejected_csv"])
            assert len(df) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
