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
    from src.research.candidate_selection.allocation import AllocationDecision
    from src.research.candidate_selection.eligibility import EligibilityResult
    from src.research.candidate_selection.redundancy import RedundancyRejection

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


def write_redundancy_artifacts(
    *,
    eligible_candidates: list[CandidateRecord],
    selected_candidates: list[CandidateRecord],
    rejected_candidates_eligibility: list[CandidateRecord],
    eligibility_results: list["EligibilityResult"],
    redundancy_rejections: list["RedundancyRejection"],
    correlation_matrix: pd.DataFrame,
    redundancy_summary: dict[str, Any],
    run_id: str,
    artifacts_root: str | Path = DEFAULT_CANDIDATE_ARTIFACTS_ROOT,
) -> tuple[Path, Path, Path, Path, Path]:
    """Write redundancy filtering artifacts and update selected/rejected outputs.

    Persists:
        - correlation_matrix.csv
        - selected_candidates.csv (post-redundancy, post-max-candidates)
        - rejected_candidates.csv (eligibility + redundancy provenance)
        - selection_summary.json (merged with redundancy summary)
        - manifest.json (augmented with redundancy metadata)
    """
    artifact_dir = resolve_candidate_selection_artifact_dir(run_id, artifacts_root)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    correlation_csv = artifact_dir / "correlation_matrix.csv"
    selected_csv = artifact_dir / "selected_candidates.csv"
    rejected_csv = artifact_dir / "rejected_candidates.csv"
    summary_json = artifact_dir / "selection_summary.json"
    manifest_json = artifact_dir / "manifest.json"

    try:
        if correlation_matrix.empty:
            corr_df = pd.DataFrame(columns=["candidate_id"])
        else:
            ordered_ids = list(correlation_matrix.index)
            corr_df = correlation_matrix.loc[ordered_ids, ordered_ids].reset_index()
            corr_df.columns = ["candidate_id", *ordered_ids]
        corr_df.to_csv(correlation_csv, index=False, lineterminator="\n", encoding="utf-8")

        if selected_candidates:
            selected_df = pd.DataFrame([c.to_dict() for c in selected_candidates])
            selected_df = selected_df[CandidateRecord.csv_columns()]
        else:
            selected_df = pd.DataFrame(columns=CandidateRecord.csv_columns())
        selected_df.to_csv(selected_csv, index=False, lineterminator="\n", encoding="utf-8")

        eligibility_failed_checks = {
            r.candidate_id: r.failed_checks for r in eligibility_results if not r.is_eligible
        }
        eligible_by_id = {c.candidate_id: c for c in eligible_candidates}

        rejected_rows: list[dict[str, Any]] = []
        for candidate in rejected_candidates_eligibility:
            row = candidate.to_dict()
            row.update(
                {
                    "rejected_stage": "eligibility_gate",
                    "rejection_reason": "eligibility_failed",
                    "failed_checks": eligibility_failed_checks.get(candidate.candidate_id, ""),
                    "rejected_against_candidate_id": None,
                    "observed_correlation": None,
                    "configured_max_correlation": None,
                    "overlap_observations": None,
                }
            )
            rejected_rows.append(row)

        for rejection in redundancy_rejections:
            candidate = eligible_by_id.get(rejection.candidate_id)
            if candidate is None:
                continue
            row = candidate.to_dict()
            row.update(
                {
                    "rejected_stage": rejection.rejected_stage,
                    "rejection_reason": rejection.rejection_reason,
                    "failed_checks": "",
                    "rejected_against_candidate_id": rejection.rejected_against_candidate_id,
                    "observed_correlation": rejection.observed_correlation,
                    "configured_max_correlation": rejection.configured_max_correlation,
                    "overlap_observations": rejection.overlap_observations,
                }
            )
            rejected_rows.append(row)

        rejected_cols = CandidateRecord.csv_columns() + [
            "rejected_stage",
            "rejection_reason",
            "failed_checks",
            "rejected_against_candidate_id",
            "observed_correlation",
            "configured_max_correlation",
            "overlap_observations",
        ]

        if rejected_rows:
            stage_order = {"eligibility_gate": 0, "redundancy_filter": 1}

            def _reject_sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
                return (
                    stage_order.get(str(row.get("rejected_stage")), 9),
                    int(row.get("selection_rank") or 0),
                    str(row.get("candidate_id") or ""),
                )

            rejected_rows = sorted(rejected_rows, key=_reject_sort_key)
            rejected_df = pd.DataFrame(rejected_rows)
            rejected_df = rejected_df[rejected_cols]
        else:
            rejected_df = pd.DataFrame(columns=rejected_cols)
        rejected_df.to_csv(rejected_csv, index=False, lineterminator="\n", encoding="utf-8")

        existing_summary: dict[str, Any] = {}
        if summary_json.exists():
            try:
                existing_summary = json.loads(summary_json.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                existing_summary = {}

        updated_summary = {**existing_summary, **redundancy_summary}
        updated_summary["selected_count"] = len(selected_candidates)
        updated_summary["rejected_count"] = len(rejected_rows)
        summary_json.write_text(
            json.dumps(updated_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        existing_manifest: dict[str, Any] = {}
        if manifest_json.exists():
            try:
                existing_manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                existing_manifest = {}

        updated_manifest = {
            **existing_manifest,
            "correlation_matrix_csv": str(correlation_csv),
            "selected_candidates_csv": str(selected_csv),
            "rejected_candidates_csv": str(rejected_csv),
            "redundancy_applied": bool(redundancy_summary.get("redundancy_enabled", False)),
            "eligible_before_redundancy": redundancy_summary.get("eligible_before_redundancy"),
            "retained_after_redundancy": redundancy_summary.get("retained_after_redundancy"),
            "pruned_by_redundancy": redundancy_summary.get("pruned_by_redundancy"),
            "selected_count": len(selected_candidates),
            "rejected_count": len(rejected_rows),
        }
        manifest_json.write_text(
            json.dumps(updated_manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except (IOError, OSError) as exc:
        raise CandidatePersistenceError(
            f"Failed to write redundancy artifacts to {artifact_dir}: {exc}"
        ) from exc

    return correlation_csv, selected_csv, rejected_csv, summary_json, manifest_json


def write_allocation_artifacts(
    *,
    selected_candidates: list[CandidateRecord],
    allocation_decisions: list["AllocationDecision"],
    allocation_summary: dict[str, Any],
    allocation_constraints: dict[str, Any],
    run_id: str,
    artifacts_root: str | Path = DEFAULT_CANDIDATE_ARTIFACTS_ROOT,
) -> tuple[Path, Path, Path, Path]:
    """Write allocation governance artifacts and update selected outputs.

    Persists:
        - allocation_weights.csv: deterministic final weights per allocated candidate.
        - selected_candidates.csv: selected candidates augmented with allocation columns.
        - selection_summary.json: merged with allocation method/constraints/diagnostics.
        - manifest.json: augmented with allocation artifact references and metadata.
    """

    artifact_dir = resolve_candidate_selection_artifact_dir(run_id, artifacts_root)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    allocation_csv = artifact_dir / "allocation_weights.csv"
    selected_csv = artifact_dir / "selected_candidates.csv"
    summary_json = artifact_dir / "selection_summary.json"
    manifest_json = artifact_dir / "manifest.json"

    try:
        decision_by_id = {decision.candidate_id: decision.to_dict() for decision in allocation_decisions}

        if allocation_decisions:
            from src.research.candidate_selection.allocation import AllocationDecision

            allocation_rows = [
                decision_by_id[candidate.candidate_id]
                for candidate in selected_candidates
                if candidate.candidate_id in decision_by_id
            ]
            allocation_df = pd.DataFrame(allocation_rows)
            allocation_df = allocation_df[AllocationDecision.csv_columns()]
        else:
            from src.research.candidate_selection.allocation import AllocationDecision

            allocation_df = pd.DataFrame(columns=AllocationDecision.csv_columns())
        allocation_df.to_csv(allocation_csv, index=False, lineterminator="\n", encoding="utf-8")

        selected_rows: list[dict[str, Any]] = []
        for candidate in selected_candidates:
            decision = decision_by_id.get(candidate.candidate_id)
            if decision is None:
                continue
            row = candidate.to_dict()
            row.update(
                {
                    "allocation_weight": decision.get("allocation_weight"),
                    "allocation_method": decision.get("allocation_method"),
                    "pre_constraint_weight": decision.get("pre_constraint_weight"),
                    "constraint_adjusted_flag": decision.get("constraint_adjusted_flag"),
                }
            )
            selected_rows.append(row)

        selected_cols = CandidateRecord.csv_columns() + [
            "allocation_weight",
            "allocation_method",
            "pre_constraint_weight",
            "constraint_adjusted_flag",
        ]
        if selected_rows:
            selected_df = pd.DataFrame(selected_rows)
            selected_df = selected_df[selected_cols]
        else:
            selected_df = pd.DataFrame(columns=selected_cols)
        selected_df.to_csv(selected_csv, index=False, lineterminator="\n", encoding="utf-8")

        existing_summary: dict[str, Any] = {}
        if summary_json.exists():
            try:
                existing_summary = json.loads(summary_json.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                existing_summary = {}

        merged_summary = {
            **existing_summary,
            **allocation_summary,
            "allocation_constraints": canonicalize_value(allocation_constraints),
            "selected_count": len(selected_rows),
        }
        summary_json.write_text(
            json.dumps(merged_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        existing_manifest: dict[str, Any] = {}
        if manifest_json.exists():
            try:
                existing_manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                existing_manifest = {}

        merged_manifest = {
            **existing_manifest,
            "allocation_applied": True,
            "allocation_method": allocation_summary.get("allocation_method"),
            "allocation_weights_csv": str(allocation_csv),
            "selected_candidates_csv": str(selected_csv),
            "selected_count": len(selected_rows),
            "allocation_constraints": canonicalize_value(allocation_constraints),
            "allocated_candidates": int(allocation_summary.get("allocated_candidates", len(selected_rows))),
        }
        manifest_json.write_text(
            json.dumps(merged_manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except (IOError, OSError) as exc:
        raise CandidatePersistenceError(
            f"Failed to write allocation artifacts to {artifact_dir}: {exc}"
        ) from exc

    return allocation_csv, selected_csv, summary_json, manifest_json
