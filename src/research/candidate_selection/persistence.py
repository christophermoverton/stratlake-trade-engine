"""Artifact persistence for candidate selection results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.research.candidate_selection.schema import CandidateRecord
from src.research.registry import canonicalize_value, serialize_canonical_json

if TYPE_CHECKING:
    from src.research.candidate_selection.eligibility import EligibilityResult

DEFAULT_CANDIDATE_ARTIFACTS_ROOT = Path("artifacts") / "candidate_selection"


class CandidatePersistenceError(ValueError):
    """Raised when artifact persistence fails."""


def build_candidate_selection_run_id(
    *,
    filters: dict[str, Any],
    candidate_ids: list[str],
    primary_metric: str = "ic_ir",
) -> str:
    """Build a deterministic run ID for candidate selection.
    
    Uses hash of filters, metric, and candidate IDs to ensure reproducibility.
    """
    payload = {
        "candidate_ids": sorted(candidate_ids),
        "filters": canonicalize_value(filters),
        "primary_metric": primary_metric,
    }
    digest = hashlib.sha256(serialize_canonical_json(payload).encode("utf-8")).hexdigest()[:12]
    return f"candidate_selection_{digest}"


def resolve_candidate_selection_artifact_dir(
    run_id: str,
    artifacts_root: str | Path = DEFAULT_CANDIDATE_ARTIFACTS_ROOT,
) -> Path:
    """Resolve artifact directory for one candidate selection run."""
    return Path(artifacts_root) / run_id


def write_candidate_selection_artifacts(
    *,
    universe: list[CandidateRecord],
    selected: list[CandidateRecord],
    run_id: str,
    filters: dict[str, Any],
    primary_metric: str = "ic_ir",
    artifacts_root: str | Path = DEFAULT_CANDIDATE_ARTIFACTS_ROOT,
) -> tuple[Path, Path, Path, Path]:
    """Write candidate selection artifacts to disk.
    
    Persists:
        - candidate_universe.csv: All loaded candidates (ranked)
        - selected_candidates.csv: Final selected candidates
        - selection_summary.json: Counts and configuration metadata
        - manifest.json: Deterministic run manifest
    
    Args:
        universe: All ranked candidates (full universe).
        selected: Selected candidates (subset of universe).
        run_id: Deterministic run identifier.
        filters: Filter configuration used for loading.
        primary_metric: Primary ranking metric used.
        artifacts_root: Root directory for artifacts.
    
    Returns:
        Tuple of (universe_csv, selected_csv, summary_json, manifest_json) paths.
    
    Raises:
        CandidatePersistenceError: If writing fails.
    """
    artifact_dir = resolve_candidate_selection_artifact_dir(run_id, artifacts_root)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    universe_csv = artifact_dir / "candidate_universe.csv"
    selected_csv = artifact_dir / "selected_candidates.csv"
    summary_json = artifact_dir / "selection_summary.json"
    manifest_json = artifact_dir / "manifest.json"

    try:
        # Write universe CSV
        if universe:
            universe_df = pd.DataFrame([c.to_dict() for c in universe])
            universe_df = universe_df[CandidateRecord.csv_columns()]
        else:
            universe_df = pd.DataFrame(columns=CandidateRecord.csv_columns())
        universe_df.to_csv(universe_csv, index=False, lineterminator="\n", encoding="utf-8")

        # Write selected CSV
        if selected:
            selected_df = pd.DataFrame([c.to_dict() for c in selected])
            selected_df = selected_df[CandidateRecord.csv_columns()]
        else:
            selected_df = pd.DataFrame(columns=CandidateRecord.csv_columns())
        selected_df.to_csv(selected_csv, index=False, lineterminator="\n", encoding="utf-8")

        # Write summary
        summary = {
            "run_id": run_id,
            "universe_count": len(universe),
            "selected_count": len(selected),
            "primary_metric": primary_metric,
            "filters": canonicalize_value(filters),
        }
        summary_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        # Write manifest
        manifest = {
            "run_id": run_id,
            "artifact_dir": str(artifact_dir),
            "created": "candidate_selection",
            "candidate_universe_csv": str(universe_csv),
            "selected_candidates_csv": str(selected_csv),
            "selection_summary_json": str(summary_json),
            "universe_count": len(universe),
            "selected_count": len(selected),
            "primary_metric": primary_metric,
            "filters": canonicalize_value(filters),
        }
        manifest_json.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    except (IOError, OSError) as exc:
        raise CandidatePersistenceError(f"Failed to write artifacts to {artifact_dir}: {exc}") from exc

    return universe_csv, selected_csv, summary_json, manifest_json


def write_eligibility_artifacts(
    *,
    eligibility_results: list["EligibilityResult"],
    eligible_candidates: list[CandidateRecord],
    rejected_candidates: list[CandidateRecord],
    eligibility_summary: dict[str, Any],
    run_id: str,
    artifacts_root: str | Path = DEFAULT_CANDIDATE_ARTIFACTS_ROOT,
) -> tuple[Path, Path, Path, Path]:
    """Write eligibility gate artifacts to the candidate selection artifact directory.

    Persists:
        - eligibility_filter_results.csv: Full per-candidate gate outcome table.
        - selected_candidates.csv: Eligible candidates (overwrites prior version).
        - rejected_candidates.csv: Rejected candidates with failure reasons.
        - selection_summary.json: Updated summary including eligibility counts.
        - manifest.json: Updated manifest referencing all eligibility artifacts.

    Args:
        eligibility_results: Per-candidate gate evaluation outcomes.
        eligible_candidates: Candidates that passed all gates.
        rejected_candidates: Candidates that failed at least one gate.
        eligibility_summary: Aggregate eligibility summary dict.
        run_id: Deterministic run identifier (must already exist as artifact dir).
        artifacts_root: Root artifact directory.

    Returns:
        Tuple of (eligibility_csv, selected_csv, rejected_csv, summary_json) paths.

    Raises:
        CandidatePersistenceError: If writing fails.
    """
    from src.research.candidate_selection.eligibility import EligibilityResult as _EligibilityResult  # noqa: F401

    artifact_dir = resolve_candidate_selection_artifact_dir(run_id, artifacts_root)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    eligibility_csv = artifact_dir / "eligibility_filter_results.csv"
    selected_csv = artifact_dir / "selected_candidates.csv"
    rejected_csv = artifact_dir / "rejected_candidates.csv"
    summary_json = artifact_dir / "selection_summary.json"
    manifest_json = artifact_dir / "manifest.json"

    try:
        # Write eligibility filter results (full gate outcome table)
        from src.research.candidate_selection.eligibility import EligibilityResult
        elig_df = pd.DataFrame([r.to_dict() for r in eligibility_results])
        elig_df = elig_df[EligibilityResult.csv_columns()]
        elig_df.to_csv(eligibility_csv, index=False, lineterminator="\n", encoding="utf-8")

        # Write eligible candidates (selected_candidates.csv)
        if eligible_candidates:
            selected_df = pd.DataFrame([c.to_dict() for c in eligible_candidates])
            selected_df = selected_df[CandidateRecord.csv_columns()]
        else:
            selected_df = pd.DataFrame(columns=CandidateRecord.csv_columns())
        selected_df.to_csv(selected_csv, index=False, lineterminator="\n", encoding="utf-8")

        # Write rejected candidates
        if rejected_candidates:
            # Augment with rejection reasons from eligibility results
            rejection_index = {r.candidate_id: r.failed_checks for r in eligibility_results if not r.is_eligible}
            rejected_rows = []
            for c in rejected_candidates:
                row = c.to_dict()
                row["failed_checks"] = rejection_index.get(c.candidate_id, "")
                rejected_rows.append(row)
            rejected_cols = CandidateRecord.csv_columns() + ["failed_checks"]
            rejected_df = pd.DataFrame(rejected_rows)
            rejected_df = rejected_df[rejected_cols]
        else:
            rejected_df = pd.DataFrame(columns=CandidateRecord.csv_columns() + ["failed_checks"])
        rejected_df.to_csv(rejected_csv, index=False, lineterminator="\n", encoding="utf-8")

        # Write updated summary JSON (merge eligibility summary into existing summary)
        existing_summary: dict[str, Any] = {}
        if summary_json.exists():
            try:
                existing_summary = json.loads(summary_json.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                existing_summary = {}
        updated_summary = {**existing_summary, **eligibility_summary}
        summary_json.write_text(
            json.dumps(updated_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        # Write updated manifest
        existing_manifest: dict[str, Any] = {}
        if manifest_json.exists():
            try:
                existing_manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                existing_manifest = {}
        updated_manifest = {
            **existing_manifest,
            "eligibility_filter_results_csv": str(eligibility_csv),
            "selected_candidates_csv": str(selected_csv),
            "rejected_candidates_csv": str(rejected_csv),
            "eligible_count": len(eligible_candidates),
            "rejected_count": len(rejected_candidates),
            "eligibility_applied": True,
        }
        manifest_json.write_text(
            json.dumps(updated_manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    except (IOError, OSError) as exc:
        raise CandidatePersistenceError(
            f"Failed to write eligibility artifacts to {artifact_dir}: {exc}"
        ) from exc

    return eligibility_csv, selected_csv, rejected_csv, summary_json
