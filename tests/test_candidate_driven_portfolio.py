"""Tests for candidate-driven portfolio construction (Milestone 15 Issue 6)."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.portfolio.candidate_component_loader import (
    CandidateArtifactNotFoundError,
    CandidateArtifactValidationError,
    CandidateComponentLoaderError,
    load_candidate_selection_artifacts,
    validate_candidate_selection_artifacts,
    resolve_candidate_components,
    build_candidate_driven_portfolio_config,
)
from src.portfolio.loaders import load_portfolio_component_runs_returns


class TestCandidateArtifactLoading:
    """Test candidate selection artifact loading."""

    def test_load_artifacts_from_valid_directory(self, tmp_path: Path) -> None:
        """Test loading candidate artifacts from a valid directory."""
        # Setup
        artifact_dir = tmp_path / "candidate_selection_abc123"
        artifact_dir.mkdir(parents=True)

        selected_candidates_data = {
            "candidate_id": ["c1", "c2"],
            "alpha_name": ["alpha_a", "alpha_b"],
            "alpha_run_id": ["alpha_run_1", "alpha_run_2"],
            "sleeve_run_id": ["sleeve_1", "sleeve_2"],
            "mapping_name": ["simple", "simple"],
            "dataset": ["train_data", "train_data"],
            "timeframe": ["1D", "1D"],
            "evaluation_horizon": [252, 252],
            "artifact_path": ["/path/to/sleeve_1", "/path/to/sleeve_2"],
            "selection_rank": [1, 2],
        }

        allocation_data = {
            "candidate_id": ["c1", "c2"],
            "alpha_name": ["alpha_a", "alpha_b"],
            "sleeve_run_id": ["sleeve_1", "sleeve_2"],
            "allocation_weight": [0.5, 0.5],
            "allocation_method": ["equal_weight", "equal_weight"],
            "pre_constraint_weight": [0.5, 0.5],
            "constraint_adjusted_flag": [False, False],
        }

        manifest_data = {
            "run_id": "candidate_selection_abc123",
            "selected_candidates_csv": "selected_candidates.csv",
            "allocation_weights_csv": "allocation_weights.csv",
            "selected_count": 2,
            "rejected_count": 0,
        }

        selected_csv = artifact_dir / "selected_candidates.csv"
        allocation_csv = artifact_dir / "allocation_weights.csv"
        manifest_json = artifact_dir / "manifest.json"

        pd.DataFrame(selected_candidates_data).to_csv(selected_csv, index=False)
        pd.DataFrame(allocation_data).to_csv(allocation_csv, index=False)
        with manifest_json.open("w") as f:
            json.dump(manifest_data, f)

        # Test
        selected, weights, manifest, run_id = load_candidate_selection_artifacts(artifact_dir)

        # Verify
        assert len(selected) == 2
        assert len(weights) == 2
        assert run_id == "candidate_selection_abc123"
        assert all(col in selected.columns for col in ["candidate_id", "sleeve_run_id"])

    def test_load_artifacts_missing_directory_raises_error(self, tmp_path: Path) -> None:
        """Test that missing directory raises appropriate error."""
        nonexistent = tmp_path / "does_not_exist"
        with pytest.raises(CandidateArtifactNotFoundError):
            load_candidate_selection_artifacts(nonexistent)

    def test_load_artifacts_missing_selected_csv_raises_error(self, tmp_path: Path) -> None:
        """Test that missing selected_candidates.csv raises error."""
        artifact_dir = tmp_path / "candidate_selection_abc123"
        artifact_dir.mkdir(parents=True)

        allocation_csv = artifact_dir / "allocation_weights.csv"
        manifest_json = artifact_dir / "manifest.json"

        pd.DataFrame({"candidate_id": ["c1"], "allocation_weight": [1.0]}).to_csv(allocation_csv, index=False)
        with manifest_json.open("w") as f:
            json.dump({}, f)

        with pytest.raises(CandidateArtifactNotFoundError, match="selected_candidates.csv"):
            load_candidate_selection_artifacts(artifact_dir)

    def test_load_artifacts_missing_allocation_csv_raises_error(self, tmp_path: Path) -> None:
        """Test that missing allocation_weights.csv raises error."""
        artifact_dir = tmp_path / "candidate_selection_abc123"
        artifact_dir.mkdir(parents=True)

        selected_csv = artifact_dir / "selected_candidates.csv"
        manifest_json = artifact_dir / "manifest.json"

        pd.DataFrame({"candidate_id": ["c1"], "sleeve_run_id": ["s1"], "artifact_path": ["/p"]}).to_csv(
            selected_csv, index=False
        )
        with manifest_json.open("w") as f:
            json.dump({}, f)

        with pytest.raises(CandidateArtifactNotFoundError, match="allocation_weights.csv"):
            load_candidate_selection_artifacts(artifact_dir)

    def test_load_artifacts_invalid_manifest_json_raises_error(self, tmp_path: Path) -> None:
        """Test that invalid JSON in manifest raises error."""
        artifact_dir = tmp_path / "candidate_selection_abc123"
        artifact_dir.mkdir(parents=True)

        selected_csv = artifact_dir / "selected_candidates.csv"
        allocation_csv = artifact_dir / "allocation_weights.csv"
        manifest_json = artifact_dir / "manifest.json"

        pd.DataFrame({"candidate_id": ["c1"], "sleeve_run_id": ["s1"], "artifact_path": ["/p"]}).to_csv(
            selected_csv, index=False
        )
        pd.DataFrame({"candidate_id": ["c1"], "allocation_weight": [1.0]}).to_csv(allocation_csv, index=False)
        manifest_json.write_text("{ invalid json")

        with pytest.raises(CandidateArtifactValidationError):
            load_candidate_selection_artifacts(artifact_dir)


class TestCandidateArtifactValidation:
    """Test candidate selection artifact validation."""

    def _make_valid_artifacts(self) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """Create valid candidate selection artifacts for testing."""
        selected = pd.DataFrame({
            "candidate_id": ["c1", "c2"],
            "alpha_name": ["alpha_a", "alpha_b"],
            "alpha_run_id": ["ar1", "ar2"],
            "sleeve_run_id": ["s1", "s2"],
            "mapping_name": ["m1", "m2"],
            "dataset": ["d1", "d1"],
            "timeframe": ["1D", "1D"],
            "evaluation_horizon": [252, 252],
            "artifact_path": ["/p1", "/p2"],
        })
        weights = pd.DataFrame({
            "candidate_id": ["c1", "c2"],
            "alpha_name": ["alpha_a", "alpha_b"],
            "sleeve_run_id": ["s1", "s2"],
            "allocation_weight": [0.5, 0.5],
        })
        manifest = {"selected_count": 2}
        return selected, weights, manifest

    def test_validation_passes_for_valid_artifacts(self) -> None:
        """Test that validation passes for valid artifacts."""
        selected, weights, manifest = self._make_valid_artifacts()
        # Should not raise
        validate_candidate_selection_artifacts(selected, weights, manifest)

    def test_validation_fails_for_missing_selected_columns(self) -> None:
        """Test that validation fails when selected_candidates missing required columns."""
        selected, weights, manifest = self._make_valid_artifacts()
        selected = selected.drop(columns=["sleeve_run_id"])  # Remove required column

        with pytest.raises(CandidateArtifactValidationError, match="selected_candidates missing columns"):
            validate_candidate_selection_artifacts(selected, weights, manifest)

    def test_validation_fails_for_duplicate_candidate_ids(self) -> None:
        """Test that validation fails for duplicate candidate IDs."""
        selected, weights, manifest = self._make_valid_artifacts()
        selected = pd.concat([selected.iloc[[0]], selected.iloc[[0]]], ignore_index=True)  # Duplicate first row

        with pytest.raises(CandidateArtifactValidationError, match="duplicate candidate_id"):
            validate_candidate_selection_artifacts(selected, weights, manifest)

    def test_validation_fails_when_selected_and_weights_mismatch(self) -> None:
        """Test that validation fails when selected and weights candidate IDs don't match."""
        selected, weights, manifest = self._make_valid_artifacts()
        weights.loc[0, "candidate_id"] = "c99"  # Change to non-matching ID

        with pytest.raises(CandidateArtifactValidationError, match="candidates in"):
            validate_candidate_selection_artifacts(selected, weights, manifest)

    def test_validation_fails_when_weights_dont_sum_to_one(self) -> None:
        """Test that validation fails when allocation weights don't sum to 1.0."""
        selected, weights, manifest = self._make_valid_artifacts()
        weights["allocation_weight"] = [0.7, 0.2]  # Sum to 0.9, not 1.0

        with pytest.raises(CandidateArtifactValidationError, match="sum to"):
            validate_candidate_selection_artifacts(selected, weights, manifest)

    def test_validation_fails_for_null_sleeve_run_id(self) -> None:
        """Test that validation fails when sleeve_run_id is null."""
        selected, weights, manifest = self._make_valid_artifacts()
        selected.loc[0, "sleeve_run_id"] = None

        with pytest.raises(CandidateArtifactValidationError, match="empty sleeve_run_id"):
            validate_candidate_selection_artifacts(selected, weights, manifest)

    def test_validation_fails_for_negative_weights(self) -> None:
        """Test that validation fails for negative allocation weights."""
        selected, weights, manifest = self._make_valid_artifacts()
        weights.loc[0, "allocation_weight"] = -0.1
        weights.loc[1, "allocation_weight"] = 1.1  # Adjust to sum to 1.0

        with pytest.raises(CandidateArtifactValidationError, match="negative"):
            validate_candidate_selection_artifacts(selected, weights, manifest)


class TestCandidateComponentResolution:
    """Test resolution of candidates to portfolio components."""

    def _make_mock_artifacts(self, tmp_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """Create mock artifacts for testing."""
        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir(parents=True)

        selected = pd.DataFrame({
            "candidate_id": ["c1", "c2"],
            "alpha_name": ["alpha_a", "alpha_b"],
            "alpha_run_id": ["ar1", "ar2"],
            "sleeve_run_id": ["s1", "s2"],
            "mapping_name": ["m1", "m2"],
            "dataset": ["d1", "d1"],
            "timeframe": ["1D", "1D"],
            "evaluation_horizon": [252, 252],
            "artifact_path": [str(artifact_dir / "s1"), str(artifact_dir / "s2")],
            "selection_rank": [1, 2],
        })
        weights = pd.DataFrame({
            "candidate_id": ["c1", "c2"],
            "alpha_name": ["alpha_a", "alpha_b"],
            "sleeve_run_id": ["s1", "s2"],
            "allocation_weight": [0.6, 0.4],
        })
        manifest = {"selected_count": 2}
        return selected, weights, manifest

    def test_component_resolution_creates_components(self, tmp_path: Path) -> None:
        """Test that resolve_candidate_components creates valid components."""
        selected, weights, manifest = self._make_mock_artifacts(tmp_path)

        # Create mock sleeve artifact paths
        for candidate_id in ["c1", "c2"]:
            path = tmp_path / "artifacts" / f"s{candidate_id[-1]}"
            path.mkdir(parents=True, exist_ok=True)

        components = resolve_candidate_components(
            selected,
            weights,
            manifest,
            candidate_selection_artifact_dir=tmp_path / "artifacts",
            candidate_selection_run_id="candidate_selection_test",
        )

        assert len(components) == 2
        assert all("strategy_name" in c for c in components)
        assert all("run_id" in c for c in components)
        assert all("provenance" in c for c in components)

    def test_components_include_provenance(self, tmp_path: Path) -> None:
        """Test that components include complete provenance metadata."""
        selected, weights, manifest = self._make_mock_artifacts(tmp_path)

        # Create mock sleeve artifact paths
        for candidate_id in ["c1", "c2"]:
            path = tmp_path / "artifacts" / f"s{candidate_id[-1]}"
            path.mkdir(parents=True, exist_ok=True)

        components = resolve_candidate_components(
            selected,
            weights,
            manifest,
            candidate_selection_artifact_dir=tmp_path / "artifacts",
            candidate_selection_run_id="candidate_selection_test",
        )

        for component in components:
            provenance = component["provenance"]
            assert "candidate_id" in provenance
            assert "alpha_name" in provenance
            assert "alpha_run_id" in provenance
            assert "sleeve_run_id" in provenance
            assert "candidate_selection_run_id" in provenance
            assert "allocation_weight" in provenance

    def test_components_are_deterministically_sorted(self, tmp_path: Path) -> None:
        """Test that components are sorted deterministically."""
        selected, weights, manifest = self._make_mock_artifacts(tmp_path)

        # Create mock sleeve artifact paths
        for candidate_id in ["c1", "c2"]:
            path = tmp_path / "artifacts" / f"s{candidate_id[-1]}"
            path.mkdir(parents=True, exist_ok=True)

        components1 = resolve_candidate_components(
            selected,
            weights,
            manifest,
            candidate_selection_artifact_dir=tmp_path / "artifacts",
            candidate_selection_run_id="candidate_selection_test",
        )

        # Run again - should get same order
        components2 = resolve_candidate_components(
            selected,
            weights,
            manifest,
            candidate_selection_artifact_dir=tmp_path / "artifacts",
            candidate_selection_run_id="candidate_selection_test",
        )

        assert [c["provenance"]["candidate_id"] for c in components1] == [
            c["provenance"]["candidate_id"] for c in components2
        ]

    def test_component_strategy_names_are_deterministic(self, tmp_path: Path) -> None:
        """Test that strategy names are constructed deterministically."""
        selected, weights, manifest = self._make_mock_artifacts(tmp_path)

        # Create mock sleeve artifact paths
        for candidate_id in ["c1", "c2"]:
            path = tmp_path / "artifacts" / f"s{candidate_id[-1]}"
            path.mkdir(parents=True, exist_ok=True)

        components = resolve_candidate_components(
            selected,
            weights,
            manifest,
            candidate_selection_artifact_dir=tmp_path / "artifacts",
            candidate_selection_run_id="candidate_selection_test",
        )

        # Strategy names should include both alpha name and candidate ID
        for idx, component in enumerate(components):
            strategy_name = component["strategy_name"]
            candidate_id = component["provenance"]["candidate_id"]
            alpha_name = component["provenance"]["alpha_name"]
            assert candidate_id in strategy_name
            assert alpha_name in strategy_name
            assert "__" in strategy_name  # Should have separator


class TestCandidateDrivenPortfolioConfig:
    """Test portfolio config building from candidate inputs."""

    def _make_mock_components(self) -> list[dict]:
        """Create mock components."""
        return [
            {
                "strategy_name": "alpha_a__c1",
                "run_id": "s1",
                "artifact_type": "alpha_sleeve",
                "provenance": {"candidate_id": "c1", "allocation_weight": 0.6},
            },
            {
                "strategy_name": "alpha_b__c2",
                "run_id": "s2",
                "artifact_type": "alpha_sleeve",
                "provenance": {"candidate_id": "c2", "allocation_weight": 0.4},
            }
        ]

    def _make_mock_manifest(self) -> dict:
        """Create mock manifest."""
        return {
            "selected_count": 2,
            "candidate_statistics": {
                "total_candidates": 10,
                "eligible_candidates": 5,
                "rejected_candidates": 3,
            }
        }

    def test_config_includes_portfolio_fields(self) -> None:
        """Test that built config includes required portfolio fields."""
        components = self._make_mock_components()
        manifest = self._make_mock_manifest()

        config = build_candidate_driven_portfolio_config(
            components=components,
            portfolio_name="test_portfolio",
            candidate_selection_run_id="candidate_selection_test",
            manifest=manifest,
            selected_count=2,
        )

        assert config["portfolio_name"] == "test_portfolio"
        assert "allocator" in config
        assert "initial_capital" in config
        assert "alignment_policy" in config

    def test_config_includes_provenance(self) -> None:
        """Test that config includes candidate-selection provenance."""
        components = self._make_mock_components()
        manifest = self._make_mock_manifest()

        config = build_candidate_driven_portfolio_config(
            components=components,
            portfolio_name="test_portfolio",
            candidate_selection_run_id="candidate_selection_test",
            manifest=manifest,
            selected_count=2,
        )

        assert "candidate_selection_provenance" in config
        provenance = config["candidate_selection_provenance"]
        assert provenance["run_id"] == "candidate_selection_test"
        assert provenance["component_count"] == 2
        assert provenance["selected_count"] == 2

    def test_config_preserves_manifest(self) -> None:
        """Test that config preserves manifest snapshot."""
        components = self._make_mock_components()
        manifest = self._make_mock_manifest()

        config = build_candidate_driven_portfolio_config(
            components=components,
            portfolio_name="test_portfolio",
            candidate_selection_run_id="candidate_selection_test",
            manifest=manifest,
            selected_count=2,
        )

        assert "manifest_snapshot" in config["candidate_selection_provenance"]
        assert config["candidate_selection_provenance"]["manifest_snapshot"] == manifest


class TestCandidateDrivenDeterminism:
    """Test deterministic behavior of candidate-driven portfolio construction."""

    def test_repeated_artifact_loads_produce_same_output(self, tmp_path: Path) -> None:
        """Test that repeated artifact loads produce identical output."""
        artifact_dir = tmp_path / "candidate_selection_test123"
        artifact_dir.mkdir(parents=True)

        selected_data = {
            "candidate_id": ["c1", "c2"],
            "alpha_name": ["alpha_a", "alpha_b"],
            "alpha_run_id": ["ar1", "ar2"],
            "sleeve_run_id": ["s1", "s2"],
            "mapping_name": ["m1", "m2"],
            "dataset": ["d1", "d1"],
            "timeframe": ["1D", "1D"],
            "evaluation_horizon": [252, 252],
            "artifact_path": ["/p1", "/p2"],
            "selection_rank": [1, 2],
        }
        weights_data = {
            "candidate_id": ["c1", "c2"],
            "alpha_name": ["alpha_a", "alpha_b"],
            "sleeve_run_id": ["s1", "s2"],
            "allocation_weight": [0.5, 0.5],
            "allocation_method": ["equal_weight", "equal_weight"],
            "pre_constraint_weight": [0.5, 0.5],
            "constraint_adjusted_flag": [False, False],
        }
        manifest_data = {"selected_count": 2}

        pd.DataFrame(selected_data).to_csv(artifact_dir / "selected_candidates.csv", index=False)
        pd.DataFrame(weights_data).to_csv(artifact_dir / "allocation_weights.csv", index=False)
        with (artifact_dir / "manifest.json").open("w") as f:
            json.dump(manifest_data, f)

        # Load multiple times
        selected1, weights1, manifest1, run_id1 = load_candidate_selection_artifacts(artifact_dir)
        selected2, weights2, manifest2, run_id2 = load_candidate_selection_artifacts(artifact_dir)

        # Should be identical
        assert run_id1 == run_id2
        assert selected1.equals(selected2)
        assert weights1.equals(weights2)
        assert manifest1 == manifest2
