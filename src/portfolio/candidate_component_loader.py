"""Candidate-driven portfolio component loading and resolution.

This module implements the bridge between candidate selection outputs and portfolio
construction, enabling portfolio construction directly from selected candidates with
preserved provenance metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


class CandidateComponentLoaderError(ValueError):
    """Raised when candidate-driven component loading fails."""


class CandidateArtifactNotFoundError(CandidateComponentLoaderError):
    """Raised when required candidate selection artifacts are missing."""


class CandidateArtifactValidationError(CandidateComponentLoaderError):
    """Raised when candidate selection artifacts fail validation."""


def load_candidate_selection_artifacts(
    candidate_selection_artifact_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], str]:
    """Load candidate selection artifacts from an artifact directory.

    Args:
        candidate_selection_artifact_dir: Path to candidate selection output directory.

    Returns:
        Tuple of (selected_candidates_df, allocation_weights_df, manifest_dict, run_id).

    Raises:
        CandidateArtifactNotFoundError: If required artifacts are missing.
        CandidateArtifactValidationError: If artifacts cannot be parsed.
    """
    artifact_dir = Path(candidate_selection_artifact_dir)
    if not artifact_dir.exists():
        raise CandidateArtifactNotFoundError(
            f"Candidate selection artifact directory does not exist: {artifact_dir}"
        )

    # Extract run_id from directory name (format: candidate_selection_<hash>)
    run_id = artifact_dir.name
    if not run_id.startswith("candidate_selection_"):
        raise CandidateArtifactNotFoundError(
            f"Artifact directory name does not follow candidate selection naming pattern: {run_id}"
        )

    # Load selected_candidates.csv
    selected_csv = artifact_dir / "selected_candidates.csv"
    if not selected_csv.exists():
        raise CandidateArtifactNotFoundError(
            f"Missing required artifact: selected_candidates.csv in {artifact_dir}"
        )

    try:
        selected_candidates = pd.read_csv(selected_csv)
    except (ValueError, OSError) as exc:
        raise CandidateArtifactValidationError(
            f"Failed to load selected_candidates.csv: {exc}"
        ) from exc

    # Load allocation_weights.csv
    allocation_csv = artifact_dir / "allocation_weights.csv"
    if not allocation_csv.exists():
        raise CandidateArtifactNotFoundError(
            f"Missing required artifact: allocation_weights.csv in {artifact_dir}"
        )

    try:
        allocation_weights = pd.read_csv(allocation_csv)
    except (ValueError, OSError) as exc:
        raise CandidateArtifactValidationError(
            f"Failed to load allocation_weights.csv: {exc}"
        ) from exc

    # Load manifest.json
    manifest_json = artifact_dir / "manifest.json"
    if not manifest_json.exists():
        raise CandidateArtifactNotFoundError(
            f"Missing required artifact: manifest.json in {artifact_dir}"
        )

    try:
        with manifest_json.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if not isinstance(manifest, dict):
            raise ValueError("manifest.json must contain a JSON object at the top level.")
    except (json.JSONDecodeError, ValueError, OSError) as exc:
        raise CandidateArtifactValidationError(f"Failed to load manifest.json: {exc}") from exc

    # Backfill sleeve ids for historical candidate artifacts where sleeve identity
    # is equivalent to alpha run id and was not materialized into CSV outputs.
    if "sleeve_run_id" in selected_candidates.columns and "alpha_run_id" in selected_candidates.columns:
        selected_candidates["sleeve_run_id"] = selected_candidates["sleeve_run_id"].astype("string")
        missing_selected_sleeve = selected_candidates["sleeve_run_id"].isna() | (
            selected_candidates["sleeve_run_id"].astype("string").str.strip() == ""
        )
        if bool(missing_selected_sleeve.any()):
            selected_candidates.loc[missing_selected_sleeve, "sleeve_run_id"] = selected_candidates.loc[
                missing_selected_sleeve, "alpha_run_id"
            ]

    if "sleeve_run_id" in allocation_weights.columns and "candidate_id" in allocation_weights.columns:
        allocation_weights["sleeve_run_id"] = allocation_weights["sleeve_run_id"].astype("string")
        sleeve_by_candidate_id = {
            str(row["candidate_id"]): str(row["sleeve_run_id"])
            for _, row in selected_candidates.iterrows()
            if pd.notna(row.get("candidate_id")) and pd.notna(row.get("sleeve_run_id"))
        }
        missing_weight_sleeve = allocation_weights["sleeve_run_id"].isna() | (
            allocation_weights["sleeve_run_id"].astype("string").str.strip() == ""
        )
        if bool(missing_weight_sleeve.any()):
            allocation_weights.loc[missing_weight_sleeve, "sleeve_run_id"] = allocation_weights.loc[
                missing_weight_sleeve, "candidate_id"
            ].astype("string").map(sleeve_by_candidate_id)

    return selected_candidates, allocation_weights, manifest, run_id


def validate_candidate_selection_artifacts(
    selected_candidates: pd.DataFrame,
    allocation_weights: pd.DataFrame,
    manifest: dict[str, Any],
    *,
    weight_sum_tolerance: float = 1e-10,
) -> None:
    """Validate consistent structure of candidate selection artifacts.

    Args:
        selected_candidates: DataFrame with selected candidate records.
        allocation_weights: DataFrame with allocation decisions.
        manifest: Manifest dictionary.
        weight_sum_tolerance: Tolerance for weight sum validation.

    Raises:
        CandidateArtifactValidationError: If validation fails.
    """
    errors: list[str] = []

    # Validate selected_candidates dataframe
    required_candidate_columns = [
        "candidate_id",
        "alpha_name",
        "alpha_run_id",
        "sleeve_run_id",
        "mapping_name",
        "dataset",
        "timeframe",
        "evaluation_horizon",
        "artifact_path",
    ]
    missing_candidate_columns = [
        col for col in required_candidate_columns if col not in selected_candidates.columns
    ]
    if missing_candidate_columns:
        errors.append(
            f"selected_candidates missing columns: {missing_candidate_columns}"
        )

    # Validate allocation_weights dataframe
    required_weights_columns = [
        "candidate_id",
        "alpha_name",
        "sleeve_run_id",
        "allocation_weight",
    ]
    missing_weights_columns = [
        col for col in required_weights_columns if col not in allocation_weights.columns
    ]
    if missing_weights_columns:
        errors.append(f"allocation_weights missing columns: {missing_weights_columns}")

    if errors:
        raise CandidateArtifactValidationError("; ".join(errors))

    # Check for empty dataframes
    if selected_candidates.empty:
        errors.append("selected_candidates is empty")
    if allocation_weights.empty:
        errors.append("allocation_weights is empty")

    if errors:
        raise CandidateArtifactValidationError("; ".join(errors))

    # Check for duplicate candidate IDs
    selected_ids = selected_candidates["candidate_id"].tolist()
    if len(selected_ids) != len(set(selected_ids)):
        duplicates = [cid for cid in selected_ids if selected_ids.count(cid) > 1]
        errors.append(f"selected_candidates contains duplicate candidate_id values: {sorted(set(duplicates))}")

    weights_ids = allocation_weights["candidate_id"].tolist()
    if len(weights_ids) != len(set(weights_ids)):
        duplicates = [cid for cid in weights_ids if weights_ids.count(cid) > 1]
        errors.append(f"allocation_weights contains duplicate candidate_id values: {sorted(set(duplicates))}")

    # Check that selected and weights match
    selected_set = set(selected_ids)
    weights_set = set(weights_ids)
    if selected_set != weights_set:
        missing_in_weights = sorted(selected_set - weights_set)
        missing_in_selected = sorted(weights_set - selected_set)
        if missing_in_weights:
            errors.append(f"candidates in selected but not in weights: {missing_in_weights}")
        if missing_in_selected:
            errors.append(f"candidates in weights but not in selected: {missing_in_selected}")

    # Check allocation weights sum to 1.0
    if not errors:  # Only check if no structural errors
        weight_sum = float(allocation_weights["allocation_weight"].sum())
        if abs(weight_sum - 1.0) > weight_sum_tolerance:
            errors.append(
                f"allocation_weights sum to {weight_sum}, expected 1.0 "
                f"(tolerance={weight_sum_tolerance})"
            )

        # Check for negative weights
        negative_mask = allocation_weights["allocation_weight"] < -weight_sum_tolerance
        if negative_mask.any():
            negative_ids = allocation_weights.loc[negative_mask, "candidate_id"].tolist()
            errors.append(f"allocation_weights contains negative values for: {negative_ids}")

    # Check that sleeve_run_ids are not empty/null
    if not errors:  # Only check if no prior errors
        null_sleeve_mask = selected_candidates["sleeve_run_id"].isna() | (selected_candidates["sleeve_run_id"] == "")
        if null_sleeve_mask.any():
            null_ids = selected_candidates.loc[null_sleeve_mask, "candidate_id"].tolist()
            errors.append(f"selected_candidates have empty sleeve_run_id for: {null_ids}")

    if errors:
        raise CandidateArtifactValidationError("; ".join(errors))


def resolve_candidate_components(
    selected_candidates: pd.DataFrame,
    allocation_weights: pd.DataFrame,
    manifest: dict[str, Any],
    candidate_selection_artifact_dir: str | Path,
    candidate_selection_run_id: str,
) -> list[dict[str, Any]]:
    """Resolve selected candidates into portfolio components with provenance.

    Args:
        selected_candidates: DataFrame with selected candidate records.
        allocation_weights: DataFrame with allocation decisions.
        manifest: Manifest dictionary from candidate selection.
        candidate_selection_artifact_dir: Path to candidate selection artifact directory.
        candidate_selection_run_id: Candidate selection run ID.

    Returns:
        List of portfolio component descriptors, sorted deterministically.

    Raises:
        CandidateComponentLoaderError: If component resolution fails.
    """
    # Build weight lookup
    weight_by_candidate = {
        row["candidate_id"]: float(row["allocation_weight"])
        for _, row in allocation_weights.iterrows()
    }

    # Resolve components, preserving candidate order with deterministic sorting
    components: list[dict[str, Any]] = []
    errors: list[str] = []

    for idx, (_, row) in enumerate(selected_candidates.iterrows()):
        candidate_id = str(row["candidate_id"])
        alpha_name = str(row["alpha_name"])
        alpha_run_id = str(row["alpha_run_id"])
        sleeve_run_id = str(row["sleeve_run_id"]) if pd.notna(row["sleeve_run_id"]) else None
        mapping_name = str(row["mapping_name"]) if pd.notna(row["mapping_name"]) else None
        artifact_path = str(row["artifact_path"]) if pd.notna(row["artifact_path"]) else None

        if not sleeve_run_id or not artifact_path:
            errors.append(
                f"Candidate {candidate_id} ({alpha_name}) missing "
                f"sleeve_run_id or artifact_path"
            )
            continue

        allocation_weight = weight_by_candidate.get(candidate_id)
        if allocation_weight is None:
            errors.append(f"No allocation weight found for candidate {candidate_id}")
            continue

        # Construct deterministic strategy name from candidate info
        strategy_name = f"{alpha_name}__{candidate_id}"

        # Verify artifact path exists
        artifact_path_obj = Path(artifact_path)
        if not artifact_path_obj.exists():
            errors.append(
                f"Candidate {candidate_id} artifact_path does not exist: {artifact_path}"
            )
            continue

        # Build provenance metadata
        provenance = {
            "candidate_id": candidate_id,
            "alpha_name": alpha_name,
            "alpha_run_id": alpha_run_id,
            "sleeve_run_id": sleeve_run_id,
            "mapping_name": mapping_name,
            "candidate_selection_run_id": candidate_selection_run_id,
            "allocation_weight": allocation_weight,
            "selection_rank": int(row.get("selection_rank", -1)) if pd.notna(row.get("selection_rank")) else None,
        }

        component = {
            "strategy_name": strategy_name,
            "run_id": sleeve_run_id,
            "artifact_type": "alpha_sleeve",
            "source_artifact_path": artifact_path,
            "provenance": provenance,
        }
        components.append(component)

    if errors:
        raise CandidateComponentLoaderError(
            "Failed to resolve candidate components: " + "; ".join(errors)
        )

    # Sort components deterministically by (candidate_id, alpha_name) for stable ordering
    components = sorted(
        components,
        key=lambda c: (
            str(c["provenance"]["candidate_id"]),
            str(c["provenance"]["alpha_name"]),
        ),
    )

    if not components:
        raise CandidateComponentLoaderError("No valid components resolved from candidate selection")

    return components


def build_candidate_driven_portfolio_config(
    components: list[dict[str, Any]],
    portfolio_name: str,
    candidate_selection_run_id: str,
    manifest: dict[str, Any],
    selected_count: int,
    allocator: str = "equal_weight",
    initial_capital: float = 1.0,
    alignment_policy: str = "intersection",
) -> dict[str, Any]:
    """Build a portfolio configuration from candidate-driven inputs.

    Args:
        components: Resolved portfolio components with provenance.
        portfolio_name: Name for the portfolio.
        candidate_selection_run_id: Candidate selection run ID.
        manifest: Candidate selection manifest.
        selected_count: Count of selected candidates.
        allocator: Portfolio allocator method.
        initial_capital: Initial capital for portfolio.
        alignment_policy: Return alignment policy.

    Returns:
        Portfolio configuration dictionary.
    """
    # Extract candidate selection components from manifest for config provenance
    candidate_stats = manifest.get("candidate_statistics", {})

    config = {
        "portfolio_name": portfolio_name,
        "allocator": allocator,
        "initial_capital": initial_capital,
        "alignment_policy": alignment_policy,
        "components": [
            {
                "strategy_name": c["strategy_name"],
                "run_id": c["run_id"],
                "artifact_type": c["artifact_type"],
            }
            for c in components
        ],
        # Store provenance at portfolio level
        "candidate_selection_provenance": {
            "run_id": candidate_selection_run_id,
            "component_count": len(components),
            "selected_count": selected_count,
            "total_universe_count": candidate_stats.get("total_candidates"),
            "eligible_count": candidate_stats.get("eligible_candidates"),
            "rejected_count": candidate_stats.get("rejected_candidates"),
            "manifest_snapshot": manifest,
        },
    }

    return config
