"""Tests for candidate selection pipeline."""

from __future__ import annotations

import tempfile
import json

import pytest

from src.research.candidate_selection import (
    CandidateRecord,
    CandidateValidationError,
    build_candidate_selection_run_id,
    rank_candidates,
    select_top_candidates,
    validate_candidate_universe,
    validate_ranked_universe,
    write_candidate_selection_artifacts,
)
from src.research.candidate_selection.loader import (
    _extract_mapping_name,
    _normalize_registry_entry_to_candidate,
)


def candidates_fixture() -> list[CandidateRecord]:
    """Return sample candidates for testing."""
    return [
        CandidateRecord(
            candidate_id="alpha_1",
            alpha_name="TestAlpha1",
            alpha_run_id="alpha_1",
            sleeve_run_id="sleeve_1",
            mapping_name="rank_long_short",
            dataset="features_daily",
            timeframe="daily",
            evaluation_horizon=5,
            mean_ic=0.05,
            ic_ir=1.5,
            mean_rank_ic=0.06,
            rank_ic_ir=1.6,
            n_periods=100,
            sharpe_ratio=1.2,
            annualized_return=0.15,
            total_return=0.20,
            max_drawdown=-0.10,
            average_turnover=0.05,
            selection_rank=0,
            promotion_status="eligible",
            review_status="candidate",
            artifact_path="artifacts/alpha/alpha_1/",
        ),
        CandidateRecord(
            candidate_id="alpha_2",
            alpha_name="TestAlpha2",
            alpha_run_id="alpha_2",
            sleeve_run_id="sleeve_2",
            mapping_name="rank_long_short",
            dataset="features_daily",
            timeframe="daily",
            evaluation_horizon=5,
            mean_ic=0.04,
            ic_ir=1.2,
            mean_rank_ic=0.05,
            rank_ic_ir=1.3,
            n_periods=100,
            sharpe_ratio=1.0,
            annualized_return=0.12,
            total_return=0.16,
            max_drawdown=-0.12,
            average_turnover=0.06,
            selection_rank=0,
            promotion_status="eligible",
            review_status="candidate",
            artifact_path="artifacts/alpha/alpha_2/",
        ),
        CandidateRecord(
            candidate_id="alpha_3",
            alpha_name="TestAlpha3",
            alpha_run_id="alpha_3",
            sleeve_run_id=None,
            mapping_name=None,
            dataset="features_daily",
            timeframe="daily",
            evaluation_horizon=5,
            mean_ic=0.03,
            ic_ir=None,
            mean_rank_ic=None,
            rank_ic_ir=None,
            n_periods=None,
            sharpe_ratio=None,
            annualized_return=None,
            total_return=None,
            max_drawdown=None,
            average_turnover=None,
            selection_rank=0,
            promotion_status="blocked",
            review_status="needs_review",
            artifact_path="artifacts/alpha/alpha_3/",
        ),
    ]


# Unit Tests


class TestCandidateSchema:
    """Test CandidateRecord schema."""

    def test_candidate_record_creation(self):
        """Test that CandidateRecord can be created and is frozen."""
        c = CandidateRecord(
            candidate_id="test_1",
            alpha_name="Test",
            alpha_run_id="test_1",
            sleeve_run_id="sleeve_1",
            mapping_name="long_short",
            dataset="features",
            timeframe="daily",
            evaluation_horizon=5,
            mean_ic=0.05,
            ic_ir=1.5,
            mean_rank_ic=0.06,
            rank_ic_ir=1.6,
            n_periods=100,
            sharpe_ratio=1.2,
            annualized_return=0.15,
            total_return=0.20,
            max_drawdown=-0.10,
            average_turnover=0.05,
            selection_rank=1,
            promotion_status="eligible",
            review_status="candidate",
            artifact_path="path/to/artifact",
        )
        assert c.candidate_id == "test_1"
        assert c.alpha_name == "Test"
        assert c.selection_rank == 1

    def test_candidate_record_frozen(self):
        """Test that CandidateRecord is frozen."""
        c = CandidateRecord(
            candidate_id="test",
            alpha_name="Test",
            alpha_run_id="test",
            sleeve_run_id=None,
            mapping_name=None,
            dataset="data",
            timeframe="daily",
            evaluation_horizon=5,
            mean_ic=None,
            ic_ir=None,
            mean_rank_ic=None,
            rank_ic_ir=None,
            n_periods=None,
            sharpe_ratio=None,
            annualized_return=None,
            total_return=None,
            max_drawdown=None,
            average_turnover=None,
            selection_rank=1,
            promotion_status="eligible",
            review_status="candidate",
            artifact_path="path",
        )
        with pytest.raises(AttributeError):
            c.selection_rank = 2

    def test_candidate_to_dict(self):
        """Test CandidateRecord.to_dict conversion."""
        c = CandidateRecord(
            candidate_id="test",
            alpha_name="Test",
            alpha_run_id="test",
            sleeve_run_id="sleeve",
            mapping_name="rank",
            dataset="data",
            timeframe="daily",
            evaluation_horizon=5,
            mean_ic=0.05,
            ic_ir=1.5,
            mean_rank_ic=0.06,
            rank_ic_ir=1.6,
            n_periods=100,
            sharpe_ratio=1.2,
            annualized_return=0.15,
            total_return=0.20,
            max_drawdown=-0.10,
            average_turnover=0.05,
            selection_rank=1,
            promotion_status="eligible",
            review_status="candidate",
            artifact_path="path",
        )
        d = c.to_dict()
        assert isinstance(d, dict)
        assert d["candidate_id"] == "test"
        assert d["selection_rank"] == 1
        assert d["mean_ic"] == 0.05


class TestRanking:
    """Test candidate ranking logic."""

    def test_rank_candidates_basic(self):
        """Test basic ranking by ic_ir."""
        candidates = candidates_fixture()
        ranked = rank_candidates(candidates, primary_metric="ic_ir")

        assert len(ranked) == 3
        assert ranked[0].selection_rank == 1
        assert ranked[1].selection_rank == 2
        assert ranked[2].selection_rank == 3
        # alpha_1 has ic_ir=1.5, should be first
        assert ranked[0].alpha_name == "TestAlpha1"
        # alpha_2 has ic_ir=1.2, should be second
        assert ranked[1].alpha_name == "TestAlpha2"
        # alpha_3 has ic_ir=None, should be last
        assert ranked[2].alpha_name == "TestAlpha3"

    def test_rank_candidates_by_mean_ic(self):
        """Test ranking by mean_ic (alternative metric)."""
        candidates = candidates_fixture()
        ranked = rank_candidates(candidates, primary_metric="mean_ic")

        assert ranked[0].selection_rank == 1
        assert ranked[0].alpha_name == "TestAlpha1"  # mean_ic=0.05
        assert ranked[1].alpha_name == "TestAlpha2"  # mean_ic=0.04
        assert ranked[2].alpha_name == "TestAlpha3"  # mean_ic=0.03

    def test_rank_deterministic_tiebreaker(self):
        """Test that ranking is deterministic with proper tiebreakers."""
        c1 = CandidateRecord(
            candidate_id="c1", alpha_name="B", alpha_run_id="c1", sleeve_run_id=None,
            mapping_name=None, dataset="d", timeframe="daily", evaluation_horizon=5,
            mean_ic=0.05, ic_ir=1.5, mean_rank_ic=0.06, rank_ic_ir=1.6, n_periods=100,
            sharpe_ratio=None, annualized_return=None, total_return=None, max_drawdown=None,
            average_turnover=None, selection_rank=0, promotion_status="e",
            review_status="c", artifact_path="p",
        )
        c2 = CandidateRecord(
            candidate_id="c2", alpha_name="A", alpha_run_id="c2", sleeve_run_id=None,
            mapping_name=None, dataset="d", timeframe="daily", evaluation_horizon=5,
            mean_ic=0.05, ic_ir=1.5, mean_rank_ic=0.06, rank_ic_ir=1.6, n_periods=100,
            sharpe_ratio=None, annualized_return=None, total_return=None, max_drawdown=None,
            average_turnover=None, selection_rank=0, promotion_status="e",
            review_status="c", artifact_path="p",
        )
        # Both have same ic_ir, should sort by alpha_name
        ranked = rank_candidates([c1, c2], primary_metric="ic_ir")
        assert ranked[0].alpha_name == "A"  # "A" comes before "B"
        assert ranked[1].alpha_name == "B"

    def test_rank_empty_list(self):
        """Test ranking an empty candidate list."""
        ranked = rank_candidates([])
        assert ranked == []

    def test_rank_unknown_metric_raises(self):
        """Test that unknown metric raises error."""
        candidates = candidates_fixture()
        with pytest.raises(ValueError, match="Unknown primary_metric"):
            rank_candidates(candidates, primary_metric="unknown_metric")


class TestSelection:
    """Test candidate selection logic."""

    def test_select_top_candidates(self):
        """Test selecting top N candidates."""
        candidates = candidates_fixture()
        ranked = rank_candidates(candidates, primary_metric="ic_ir")
        selected = select_top_candidates(ranked, max_count=2)

        assert len(selected) == 2
        assert selected[0].selection_rank == 1
        assert selected[1].selection_rank == 2

    def test_select_all_candidates(self):
        """Test selecting all candidates when max_count is None."""
        candidates = candidates_fixture()
        ranked = rank_candidates(candidates, primary_metric="ic_ir")
        selected = select_top_candidates(ranked, max_count=None)

        assert len(selected) == len(ranked)

    def test_select_more_than_available(self):
        """Test max_count larger than available candidates."""
        candidates = candidates_fixture()
        ranked = rank_candidates(candidates, primary_metric="ic_ir")
        selected = select_top_candidates(ranked, max_count=100)

        assert len(selected) == len(ranked)


class TestValidation:
    """Test validation logic."""

    def test_validate_valid_universe(self):
        """Test validation of valid candidate universe."""
        candidates = candidates_fixture()
        # Should not raise
        validate_candidate_universe(candidates)

    def test_validate_duplicate_ids_raises(self):
        """Test that duplicate IDs are detected."""
        candidates = candidates_fixture()
        candidates = candidates + [candidates[0]]  # Add duplicate
        with pytest.raises(CandidateValidationError, match="Duplicate candidate_id"):
            validate_candidate_universe(candidates)

    def test_validate_wrong_rank_sequence_raises(self):
        """Test that invalid rank sequence is detected."""
        ranked = rank_candidates(candidates_fixture(), primary_metric="ic_ir")
        # Mess up one rank
        bad_ranked = [
            ranked[0],
            CandidateRecord(
                candidate_id="c2", alpha_name="A", alpha_run_id="c2", sleeve_run_id=None,
                mapping_name=None, dataset="d", timeframe="daily", evaluation_horizon=5,
                mean_ic=0.05, ic_ir=None, mean_rank_ic=None, rank_ic_ir=None, n_periods=None,
                sharpe_ratio=None, annualized_return=None, total_return=None, max_drawdown=None,
                average_turnover=None, selection_rank=5,  # Wrong rank
                promotion_status="e", review_status="c", artifact_path="p",
            ),
        ]
        with pytest.raises(CandidateValidationError, match="incorrect rank"):
            validate_ranked_universe(bad_ranked)

    def test_validate_empty_list(self):
        """Test validation of empty list."""
        # Should not raise
        validate_candidate_universe([])
        validate_ranked_universe([])


class TestPersistence:
    """Test artifact persistence."""

    def test_build_run_id_deterministic(self):
        """Test that run ID is deterministic."""
        filters = {"dataset": "data", "timeframe": "daily"}
        candidates = ["c1", "c2"]

        run_id_1 = build_candidate_selection_run_id(
            filters=filters,
            candidate_ids=candidates,
            primary_metric="ic_ir",
        )
        run_id_2 = build_candidate_selection_run_id(
            filters=filters,
            candidate_ids=candidates,
            primary_metric="ic_ir",
        )
        assert run_id_1 == run_id_2
        assert run_id_1.startswith("candidate_selection_")

    def test_build_run_id_differs_with_different_inputs(self):
        """Test that run ID changes with different inputs."""
        filters1 = {"dataset": "data1"}
        filters2 = {"dataset": "data2"}
        candidates = ["c1", "c2"]

        run_id_1 = build_candidate_selection_run_id(
            filters=filters1,
            candidate_ids=candidates,
            primary_metric="ic_ir",
        )
        run_id_2 = build_candidate_selection_run_id(
            filters=filters2,
            candidate_ids=candidates,
            primary_metric="ic_ir",
        )
        assert run_id_1 != run_id_2

    def test_write_artifacts(self):
        """Test writing candidate selection artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            candidates = candidates_fixture()
            ranked = rank_candidates(candidates, primary_metric="ic_ir")
            selected = select_top_candidates(ranked, max_count=2)

            filters = {"dataset": "features_daily"}
            run_id = build_candidate_selection_run_id(
                filters=filters,
                candidate_ids=[c.candidate_id for c in ranked],
                primary_metric="ic_ir",
            )

            universe_csv, selected_csv, summary_json, manifest_json = write_candidate_selection_artifacts(
                universe=ranked,
                selected=selected,
                run_id=run_id,
                filters=filters,
                primary_metric="ic_ir",
                artifacts_root=tmpdir,
            )

            # Check that files exist
            assert universe_csv.exists()
            assert selected_csv.exists()
            assert summary_json.exists()
            assert manifest_json.exists()

            # Check csv contents
            import pandas as pd
            universe_df = pd.read_csv(universe_csv)
            assert len(universe_df) == 3
            assert "selection_rank" in universe_df.columns

            selected_df = pd.read_csv(selected_csv)
            assert len(selected_df) == 2

            # Check JSON manifest
            manifest = json.loads(manifest_json.read_text())
            assert manifest["run_id"] == run_id
            assert manifest["universe_count"] == 3
            assert manifest["selected_count"] == 2


class TestLoaderHelpers:
    """Test loader helper functions."""

    def test_extract_mapping_name_from_metadata(self):
        """Test extracting mapping name from explicit metadata."""
        config = {
            "signal_mapping": {
                "metadata": {"name": "my_mapping"},
                "policy": "rank_long_short",
            }
        }
        name = _extract_mapping_name(config)
        assert name == "my_mapping"

    def test_extract_mapping_name_from_policy(self):
        """Test extracting mapping name from policy."""
        config = {
            "signal_mapping": {
                "policy": "rank_long_short",
            }
        }
        name = _extract_mapping_name(config)
        assert name == "rank_long_short"

    def test_extract_mapping_name_with_quantile(self):
        """Test extracting mapping name with quantile."""
        config = {
            "signal_mapping": {
                "policy": "top_bottom_quantile",
                "quantile": 0.2,
            }
        }
        name = _extract_mapping_name(config)
        assert name == "top_bottom_quantile[q=0.2]"

    def test_extract_mapping_name_none(self):
        """Test that None is returned when mapping is absent."""
        config = {}
        name = _extract_mapping_name(config)
        assert name is None

    def test_normalize_registry_entry_valid(self):
        """Test normalizing a valid registry entry."""
        entry = {
            "alpha_name": "TestAlpha",
            "run_id": "run_1",
            "artifact_path": "path/to/artifact",
            "dataset": "features",
            "timeframe": "daily",
            "evaluation_horizon": 5,
            "config": {
                "signal_mapping": {"policy": "rank_long_short"},
            },
            "metrics_summary": {
                "mean_ic": 0.05,
                "ic_ir": 1.5,
                "mean_rank_ic": 0.06,
                "rank_ic_ir": 1.6,
                "n_periods": 100,
            },
            "promotion_status": "eligible",
            "review_status": "candidate",
        }
        candidate = _normalize_registry_entry_to_candidate(entry)
        assert candidate.alpha_name == "TestAlpha"
        assert candidate.alpha_run_id == "run_1"
        assert candidate.dataset == "features"
        assert candidate.ic_ir == 1.5


# Integration Tests


class TestIntegrationEndToEnd:
    """Integration tests for end-to-end workflow."""

    def test_rank_and_select_workflow(self):
        """Test complete rank and select workflow."""
        candidates = candidates_fixture()
        ranked = rank_candidates(candidates, primary_metric="ic_ir")
        validate_ranked_universe(ranked)

        selected = select_top_candidates(ranked, max_count=2)
        assert len(selected) == 2
        assert selected[0].selection_rank == 1


# Determinism Tests


class TestDeterminism:
    """Test deterministic behavior across runs."""

    def test_ranking_deterministic(self):
        """Test that ranking produces same order across runs."""
        candidates = candidates_fixture()

        ranked1 = rank_candidates(candidates.copy(), primary_metric="ic_ir")
        ranked2 = rank_candidates(candidates.copy(), primary_metric="ic_ir")

        assert [c.candidate_id for c in ranked1] == [c.candidate_id for c in ranked2]
        assert [c.selection_rank for c in ranked1] == [c.selection_rank for c in ranked2]

    def test_run_id_deterministic(self):
        """Test that run ID is deterministic across calls."""
        filters = {"dataset": "d", "timeframe": "daily"}
        candidates = ["c1", "c2", "c3"]

        run_id_1 = build_candidate_selection_run_id(
            filters=filters,
            candidate_ids=candidates,
            primary_metric="ic_ir",
        )
        run_id_2 = build_candidate_selection_run_id(
            filters=filters,
            candidate_ids=candidates,
            primary_metric="ic_ir",
        )
        assert run_id_1 == run_id_2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
