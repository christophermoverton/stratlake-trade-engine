"""Load candidate universe from alpha evaluation registry."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from src.research.alpha_eval.registry import load_alpha_evaluation_registry
from src.research.candidate_selection.schema import CandidateRecord


class CandidateSelectionError(ValueError):
    """Raised when candidate loading or validation fails."""


def load_candidate_universe(
    *,
    artifacts_root: str | Path = Path("artifacts") / "alpha",
    alpha_name: str | None = None,
    dataset: str | None = None,
    timeframe: str | None = None,
    evaluation_horizon: int | None = None,
    mapping_name: str | None = None,
) -> list[CandidateRecord]:
    """Load candidate universe from alpha evaluation registry with optional filters.
    
    Args:
        artifacts_root: Root directory for alpha artifacts.
        alpha_name: Optional filter by alpha name.
        dataset: Optional filter by dataset.
        timeframe: Optional filter by timeframe.
        evaluation_horizon: Optional filter by evaluation horizon.
        mapping_name: Optional filter by signal mapping name.
    
    Returns:
        List of CandidateRecord objects in input order (pre-ranking).
    
    Raises:
        CandidateSelectionError: If registry is malformed or required data is missing.
    """
    entries = load_alpha_evaluation_registry(artifacts_root)
    if not entries:
        raise CandidateSelectionError("Alpha evaluation registry is empty.")

    candidates = []
    for entry in entries:
        try:
            candidate = _normalize_registry_entry_to_candidate(entry)
        except (KeyError, ValueError, TypeError) as exc:
            run_id = entry.get("run_id", "<unknown>")
            raise CandidateSelectionError(f"Failed to normalize registry entry '{run_id}': {exc}") from exc

        # Apply filters
        if alpha_name is not None and candidate.alpha_name != alpha_name:
            continue
        if dataset is not None and candidate.dataset != dataset:
            continue
        if timeframe is not None and candidate.timeframe != timeframe:
            continue
        if evaluation_horizon is not None and candidate.evaluation_horizon != evaluation_horizon:
            continue
        if mapping_name is not None and candidate.mapping_name != mapping_name:
            continue

        candidates.append(candidate)

    return candidates


def _normalize_registry_entry_to_candidate(entry: dict[str, Any]) -> CandidateRecord:
    """Convert a registry entry to a CandidateRecord."""

    if not isinstance(entry, dict):
        raise CandidateSelectionError("Registry entry must be a mapping.")

    # Extract basic identifiers
    alpha_name = _extract_string(entry, "alpha_name", required=True)
    alpha_run_id = _extract_string(entry, "run_id", required=True)
    artifact_path = _extract_string(entry, "artifact_path", required=True)

    # Extract top-level fields from entry (registry includes these directly)
    dataset = _extract_string(entry, "dataset", required=True)
    timeframe = _extract_string(entry, "timeframe", required=True)
    evaluation_horizon = _extract_int(entry, "evaluation_horizon", required=True)

    # Extract config-based fields
    config = entry.get("config")
    if not isinstance(config, dict):
        raise CandidateSelectionError(f"Alpha run '{alpha_run_id}' missing 'config' mapping.")

    # Extract metrics from metrics_summary (forecast metrics)
    metrics_summary = entry.get("metrics_summary")
    if not isinstance(metrics_summary, dict):
        raise CandidateSelectionError(f"Alpha run '{alpha_run_id}' missing 'metrics_summary'.")

    mean_ic = _extract_optional_float(metrics_summary, "mean_ic")
    ic_ir = _extract_optional_float(metrics_summary, "ic_ir")
    mean_rank_ic = _extract_optional_float(metrics_summary, "mean_rank_ic")
    rank_ic_ir = _extract_optional_float(metrics_summary, "rank_ic_ir")
    n_periods = _extract_optional_int(metrics_summary, "n_periods")

    # Extract config-based fields
    config = entry.get("config")
    if not isinstance(config, dict):
        raise CandidateSelectionError(f"Alpha run '{alpha_run_id}' missing 'config' mapping.")

    # Extract mapping name from signal_mapping config
    mapping_name = _extract_mapping_name(config)

    # Extract sleeve metrics if available
    sleeve_run_id = None
    sharpe_ratio = None
    annualized_return = None
    total_return = None
    max_drawdown = None
    average_turnover = None

    manifest = entry.get("manifest")
    if isinstance(manifest, dict):
        sleeve = manifest.get("sleeve")
        if isinstance(sleeve, dict):
            sleeve_run_id = _extract_optional_string(sleeve, "run_id")
            metric_summary = sleeve.get("metric_summary")
            if isinstance(metric_summary, dict):
                sharpe_ratio = _extract_optional_float(metric_summary, "sharpe_ratio")
                annualized_return = _extract_optional_float(metric_summary, "annualized_return")
                total_return = _extract_optional_float(metric_summary, "total_return")
                max_drawdown = _extract_optional_float(metric_summary, "max_drawdown")
                average_turnover = _extract_optional_float(metric_summary, "average_turnover")

    # Extract promotion and review status
    promotion_status = entry.get("promotion_status", "unknown")
    review_status = entry.get("review_status", "candidate")

    # Create candidate with rank=0 (will be assigned during ranking)
    return CandidateRecord(
        candidate_id=alpha_run_id,
        alpha_name=alpha_name,
        alpha_run_id=alpha_run_id,
        sleeve_run_id=sleeve_run_id,
        mapping_name=mapping_name,
        dataset=dataset,
        timeframe=timeframe,
        evaluation_horizon=evaluation_horizon,
        mean_ic=mean_ic,
        ic_ir=ic_ir,
        mean_rank_ic=mean_rank_ic,
        rank_ic_ir=rank_ic_ir,
        n_periods=n_periods,
        sharpe_ratio=sharpe_ratio,
        annualized_return=annualized_return,
        total_return=total_return,
        max_drawdown=max_drawdown,
        average_turnover=average_turnover,
        selection_rank=0,  # Assigned during ranking
        promotion_status=promotion_status,
        review_status=review_status,
        artifact_path=artifact_path,
    )


def _extract_string(obj: dict[str, Any], key: str, required: bool = False) -> str:
    """Extract and validate a required string field."""
    value = obj.get(key)
    if value is None:
        if required:
            raise CandidateSelectionError(f"Required field '{key}' is missing.")
        return ""
    if not isinstance(value, str):
        raise CandidateSelectionError(f"Field '{key}' must be a string, got {type(value).__name__}.")
    normalized = value.strip()
    if required and not normalized:
        raise CandidateSelectionError(f"Required field '{key}' is empty.")
    return normalized


def _extract_optional_string(obj: dict[str, Any], key: str) -> str | None:
    """Extract optional string field."""
    value = obj.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise CandidateSelectionError(f"Field '{key}' must be a string or null.")
    normalized = value.strip()
    return normalized if normalized else None


def _extract_int(obj: dict[str, Any], key: str, required: bool = False) -> int:
    """Extract and validate an integer field."""
    value = obj.get(key)
    if value is None:
        if required:
            raise CandidateSelectionError(f"Required field '{key}' is missing.")
        return 0
    if isinstance(value, bool):
        raise CandidateSelectionError(f"Field '{key}' must be an integer, not bool.")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise CandidateSelectionError(f"Field '{key}' must be an integer.") from exc


def _extract_optional_int(obj: dict[str, Any], key: str) -> int | None:
    """Extract optional integer field."""
    value = obj.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        raise CandidateSelectionError(f"Field '{key}' must be an integer or null, not bool.")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise CandidateSelectionError(f"Field '{key}' must be an integer or null.") from exc


def _extract_optional_float(obj: dict[str, Any], key: str) -> float | None:
    """Extract optional float field, handling NaN/Inf."""
    value = obj.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        raise CandidateSelectionError(f"Field '{key}' must be a number or null, not bool.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CandidateSelectionError(f"Field '{key}' must be a number or null.") from exc
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _extract_mapping_name(config: dict[str, Any]) -> str | None:
    """Extract signal mapping name from config."""
    signal_mapping = config.get("signal_mapping")
    if not isinstance(signal_mapping, dict):
        return None

    # Try explicit metadata name first
    metadata = signal_mapping.get("metadata")
    if isinstance(metadata, dict):
        raw_name = metadata.get("name")
        if isinstance(raw_name, str) and raw_name.strip():
            return raw_name.strip()

    # Fall back to policy + quantile
    policy = signal_mapping.get("policy")
    if not isinstance(policy, str) or not policy.strip():
        return None

    quantile = signal_mapping.get("quantile")
    if quantile is None:
        return policy.strip()

    try:
        normalized_quantile = float(quantile)
    except (TypeError, ValueError):
        return policy.strip()

    if math.isnan(normalized_quantile) or math.isinf(normalized_quantile):
        return policy.strip()

    return f"{policy.strip()}[q={normalized_quantile:g}]"
