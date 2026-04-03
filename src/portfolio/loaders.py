from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .contracts import PortfolioContractError, validate_aligned_returns, validate_strategy_returns

_CANONICAL_RETURNS_ARTIFACT = "equity_curve.csv"
_CANONICAL_ALPHA_SLEEVE_RETURNS_ARTIFACT = "sleeve_returns.csv"
_CONFIG_FILENAME = "config.json"
_MANIFEST_FILENAME = "manifest.json"
_REQUIRED_RETURN_COLUMNS: tuple[str, ...] = ("ts_utc", "strategy_return")
_ALIGNMENT_POLICY_INTERSECTION = "intersection"
_DEFAULT_COMPONENT_ARTIFACT_TYPE = "strategy"
_ALPHA_SLEEVE_COMPONENT_ARTIFACT_TYPE = "alpha_sleeve"
SUPPORTED_PORTFOLIO_COMPONENT_ARTIFACT_TYPES: tuple[str, ...] = (
    _DEFAULT_COMPONENT_ARTIFACT_TYPE,
    _ALPHA_SLEEVE_COMPONENT_ARTIFACT_TYPE,
)


def load_strategy_run_returns(run_dir: str | Path) -> pd.DataFrame:
    """
    Load one completed strategy run into normalized portfolio return input.

    The canonical return source is the root-level ``equity_curve.csv`` artifact.
    That file is already the repository's standard inspection/reporting artifact and
    preserves the timestamped strategy return series needed by the portfolio layer.
    """

    root = _normalize_run_dir(run_dir)
    strategy_name = _resolve_strategy_name(root)
    run_id = _resolve_run_id(root)
    returns_artifact = root / _CANONICAL_RETURNS_ARTIFACT
    if not returns_artifact.exists():
        raise FileNotFoundError(
            f"Strategy run directory {root} is missing required artifact {_CANONICAL_RETURNS_ARTIFACT!r}."
        )

    artifact_frame = pd.read_csv(returns_artifact)
    missing_columns = [column for column in _REQUIRED_RETURN_COLUMNS if column not in artifact_frame.columns]
    if missing_columns:
        formatted = ", ".join(repr(column) for column in missing_columns)
        raise ValueError(
            f"Strategy run artifact {returns_artifact} is missing required return columns: {formatted}."
        )

    normalized = _collapse_strategy_return_rows(artifact_frame.loc[:, ["ts_utc", "strategy_return"]].copy())
    normalized["strategy_name"] = strategy_name
    normalized["run_id"] = run_id
    normalized = normalized.loc[:, ["ts_utc", "strategy_name", "strategy_return", "run_id"]]

    try:
        validated = validate_strategy_returns(normalized)
    except PortfolioContractError as exc:
        raise ValueError(f"Strategy run {root} contains invalid portfolio return input: {exc}") from exc

    validated.attrs["portfolio_loader"] = {
        "source_run_dir": str(root),
        "run_id": run_id,
        "strategy_name": strategy_name,
        "returns_artifact": _CANONICAL_RETURNS_ARTIFACT,
        "alignment_policy": _ALIGNMENT_POLICY_INTERSECTION,
    }
    return validated


def load_portfolio_component_returns(
    run_dir: str | Path,
    *,
    artifact_type: str = _DEFAULT_COMPONENT_ARTIFACT_TYPE,
    strategy_name: str | None = None,
) -> pd.DataFrame:
    """Load one supported portfolio component artifact into normalized portfolio return input."""

    normalized_artifact_type = _normalize_component_artifact_type(artifact_type)
    root = _normalize_run_dir(run_dir)
    run_id = _resolve_run_id(root)
    resolved_strategy_name = (
        _resolve_alpha_sleeve_strategy_name(root, explicit_strategy_name=strategy_name)
        if normalized_artifact_type == _ALPHA_SLEEVE_COMPONENT_ARTIFACT_TYPE
        else _resolve_strategy_name(root)
    )
    returns_artifact, return_column = _resolve_component_returns_artifact(
        root,
        artifact_type=normalized_artifact_type,
    )

    artifact_frame = pd.read_csv(returns_artifact)
    required_columns = ("ts_utc", return_column)
    missing_columns = [column for column in required_columns if column not in artifact_frame.columns]
    if missing_columns:
        formatted = ", ".join(repr(column) for column in missing_columns)
        raise ValueError(
            f"Portfolio component artifact {returns_artifact} is missing required return columns: {formatted}."
        )

    normalized = _collapse_strategy_return_rows(
        artifact_frame.loc[:, ["ts_utc", return_column]].rename(columns={return_column: "strategy_return"}).copy()
    )
    normalized["strategy_name"] = resolved_strategy_name
    normalized["run_id"] = run_id
    normalized["artifact_type"] = normalized_artifact_type
    normalized = normalized.loc[:, ["ts_utc", "strategy_name", "strategy_return", "run_id", "artifact_type"]]

    try:
        validated = validate_strategy_returns(normalized)
    except PortfolioContractError as exc:
        raise ValueError(f"Portfolio component {root} contains invalid portfolio return input: {exc}") from exc

    validated.attrs["portfolio_loader"] = {
        "source_run_dir": str(root),
        "run_id": run_id,
        "strategy_name": resolved_strategy_name,
        "artifact_type": normalized_artifact_type,
        "returns_artifact": returns_artifact.name,
        "alignment_policy": _ALIGNMENT_POLICY_INTERSECTION,
    }
    return validated


def _collapse_strategy_return_rows(returns_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize one strategy artifact into one return row per timestamp.

    Strategy artifacts may contain multiple rows for the same timestamp when the
    originating research run spans multiple symbols. The portfolio layer needs a
    single strategy-level return stream, so same-timestamp rows are averaged
    into one effective return for that timestamp.
    """

    collapsed = returns_frame.reset_index(drop=True).copy()
    collapsed["_row_order"] = range(len(collapsed))
    collapsed["_ts_sort"] = pd.to_datetime(collapsed["ts_utc"], utc=True, errors="coerce")
    collapsed = collapsed.sort_values(["_ts_sort", "_row_order"], kind="stable")

    duplicate_mask = collapsed.duplicated(subset=["ts_utc"], keep=False)
    if not duplicate_mask.any():
        return collapsed.loc[:, ["ts_utc", "strategy_return"]].reset_index(drop=True)

    grouped = (
        collapsed.groupby("ts_utc", sort=False, as_index=False)
        .agg(strategy_return=("strategy_return", _average_return_series))
    )
    return grouped.loc[:, ["ts_utc", "strategy_return"]].reset_index(drop=True)


def _average_return_series(series: pd.Series) -> float:
    returns = pd.to_numeric(series, errors="raise").astype("float64")
    return float(returns.mean())


def load_strategy_runs_returns(run_dirs: Sequence[str | Path]) -> pd.DataFrame:
    """Load multiple completed strategy runs into one deterministic long-form return frame."""

    if not isinstance(run_dirs, Sequence) or isinstance(run_dirs, (str, Path)):
        raise ValueError("run_dirs must be provided as a sequence of strategy run directories.")
    if not run_dirs:
        raise ValueError("run_dirs must contain at least one strategy run directory.")

    loaded_frames = [load_strategy_run_returns(run_dir) for run_dir in run_dirs]
    strategy_names = [str(frame["strategy_name"].iloc[0]) for frame in loaded_frames]
    duplicate_strategy_names = sorted({name for name in strategy_names if strategy_names.count(name) > 1})
    if duplicate_strategy_names:
        raise ValueError(
            "Multiple strategy runs resolve to the same strategy identifier. "
            "Pass unique strategy runs or add explicit disambiguation before portfolio loading. "
            f"Duplicate strategy identifiers: {duplicate_strategy_names}."
        )

    combined = pd.concat(loaded_frames, ignore_index=True)
    try:
        validated = validate_strategy_returns(combined)
    except PortfolioContractError as exc:
        raise ValueError(f"Combined strategy returns are invalid for portfolio loading: {exc}") from exc

    validated.attrs["portfolio_loader"] = {
        "run_count": len(loaded_frames),
        "strategy_names": sorted(strategy_names),
        "returns_artifact": _CANONICAL_RETURNS_ARTIFACT,
        "alignment_policy": _ALIGNMENT_POLICY_INTERSECTION,
    }
    return validated


def load_portfolio_component_runs_returns(
    components: Sequence[str | Path | Mapping[str, Any]],
) -> pd.DataFrame:
    """Load multiple supported portfolio component artifacts into one deterministic long-form return frame."""

    if not isinstance(components, Sequence) or isinstance(components, (str, Path)):
        raise ValueError("components must be provided as a sequence of portfolio component descriptors.")
    if not components:
        raise ValueError("components must contain at least one portfolio component descriptor.")

    loaded_frames = [load_portfolio_component_returns(**_component_loader_kwargs(component)) for component in components]
    strategy_names = [str(frame["strategy_name"].iloc[0]) for frame in loaded_frames]
    duplicate_strategy_names = sorted({name for name in strategy_names if strategy_names.count(name) > 1})
    if duplicate_strategy_names:
        raise ValueError(
            "Multiple portfolio components resolve to the same strategy identifier. "
            "Pass unique component identifiers or add explicit disambiguation before portfolio loading. "
            f"Duplicate strategy identifiers: {duplicate_strategy_names}."
        )

    combined = pd.concat(loaded_frames, ignore_index=True)
    try:
        validated = validate_strategy_returns(combined)
    except PortfolioContractError as exc:
        raise ValueError(f"Combined portfolio component returns are invalid for portfolio loading: {exc}") from exc

    artifact_types = sorted(
        {
            str(frame.attrs.get("portfolio_loader", {}).get("artifact_type", _DEFAULT_COMPONENT_ARTIFACT_TYPE))
            for frame in loaded_frames
        }
    )
    validated.attrs["portfolio_loader"] = {
        "run_count": len(loaded_frames),
        "strategy_names": sorted(strategy_names),
        "artifact_types": artifact_types,
        "alignment_policy": _ALIGNMENT_POLICY_INTERSECTION,
    }
    return validated


def build_aligned_return_matrix(strategy_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Build a deterministic wide return matrix from validated long-form strategy returns.

    Milestone 9 v1 supports only ``intersection`` alignment, which retains timestamps
    with non-null returns for every component strategy.
    """

    try:
        normalized = validate_strategy_returns(strategy_returns)
    except PortfolioContractError as exc:
        raise ValueError(f"strategy_returns is not valid normalized portfolio input: {exc}") from exc

    strategy_column = str(
        normalized.attrs.get("portfolio_contract", {}).get("strategy_identifier_column", "strategy_name")
    )
    ordered_strategies = sorted(normalized[strategy_column].astype("string").unique().tolist())

    matrix = normalized.pivot(index="ts_utc", columns=strategy_column, values="strategy_return")
    matrix = matrix.loc[:, ordered_strategies]
    matrix = matrix.sort_index(kind="stable")
    matrix = matrix.dropna(axis="index", how="any")
    matrix.index.name = "ts_utc"
    matrix.columns.name = None

    if matrix.empty:
        raise ValueError(
            "Aligned return matrix is empty under 'intersection' alignment; no shared timestamps exist "
            "across all component strategies."
        )

    try:
        validated = validate_aligned_returns(matrix)
    except PortfolioContractError as exc:
        raise ValueError(f"Aligned return matrix is invalid for portfolio construction: {exc}") from exc

    validated.attrs["portfolio_loader"] = {
        "alignment_policy": _ALIGNMENT_POLICY_INTERSECTION,
        "strategy_count": len(ordered_strategies),
        "timestamp_count": len(validated.index),
        "returns_artifact": _CANONICAL_RETURNS_ARTIFACT,
    }
    return validated


def _normalize_run_dir(run_dir: str | Path) -> Path:
    root = Path(run_dir)
    if not root.exists():
        raise FileNotFoundError(f"Strategy run directory does not exist: {root}.")
    if not root.is_dir():
        raise FileNotFoundError(f"Strategy run path is not a directory: {root}.")
    return root


def _normalize_component_artifact_type(artifact_type: str | None) -> str:
    normalized = str(_DEFAULT_COMPONENT_ARTIFACT_TYPE if artifact_type is None else artifact_type).strip().lower()
    if normalized not in SUPPORTED_PORTFOLIO_COMPONENT_ARTIFACT_TYPES:
        raise ValueError(
            "Unsupported portfolio component artifact_type "
            f"{artifact_type!r}. Supported values: {list(SUPPORTED_PORTFOLIO_COMPONENT_ARTIFACT_TYPES)!r}."
        )
    return normalized


def _resolve_component_returns_artifact(run_dir: Path, *, artifact_type: str) -> tuple[Path, str]:
    if artifact_type == _DEFAULT_COMPONENT_ARTIFACT_TYPE:
        returns_artifact = run_dir / _CANONICAL_RETURNS_ARTIFACT
        return_column = "strategy_return"
    elif artifact_type == _ALPHA_SLEEVE_COMPONENT_ARTIFACT_TYPE:
        returns_artifact = run_dir / _CANONICAL_ALPHA_SLEEVE_RETURNS_ARTIFACT
        return_column = "sleeve_return"
    else:  # pragma: no cover - guarded by _normalize_component_artifact_type
        raise ValueError(f"Unsupported portfolio component artifact_type {artifact_type!r}.")

    if not returns_artifact.exists():
        raise FileNotFoundError(
            f"Portfolio component directory {run_dir} is missing required artifact {returns_artifact.name!r}."
        )
    return returns_artifact, return_column


def _resolve_alpha_sleeve_strategy_name(run_dir: Path, *, explicit_strategy_name: str | None) -> str:
    if explicit_strategy_name is not None:
        normalized = str(explicit_strategy_name).strip()
        if normalized:
            return normalized
        raise ValueError(f"Portfolio component for alpha sleeve {run_dir} contains a blank strategy_name.")

    config_payload = _load_json_if_exists(run_dir / _CONFIG_FILENAME)
    manifest_payload = _load_json_if_exists(run_dir / _MANIFEST_FILENAME)
    for payload_name, payload in (("config.json", config_payload), ("manifest.json", manifest_payload)):
        if isinstance(payload, dict):
            for candidate_key in ("strategy_name", "alpha_name"):
                if payload.get(candidate_key) is not None:
                    normalized = str(payload[candidate_key]).strip()
                    if normalized:
                        return normalized
                    raise ValueError(
                        f"Alpha sleeve metadata file {run_dir / payload_name} contains a blank {candidate_key!r}."
                    )
    raise ValueError(
        f"Alpha sleeve directory {run_dir} is missing a usable component identifier in config.json, manifest.json, "
        "or the portfolio component definition."
    )


def _component_loader_kwargs(component: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(component, (str, Path)):
        return {"run_dir": component}
    if not isinstance(component, Mapping):
        raise ValueError("Portfolio component descriptors must be paths or mappings.")

    run_dir = component.get("source_artifact_path", component.get("run_dir"))
    if run_dir is None:
        raise ValueError("Portfolio component descriptors must include 'source_artifact_path' or 'run_dir'.")
    return {
        "run_dir": run_dir,
        "artifact_type": component.get("artifact_type", _DEFAULT_COMPONENT_ARTIFACT_TYPE),
        "strategy_name": component.get("strategy_name"),
    }


def _resolve_strategy_name(run_dir: Path) -> str:
    config_payload = _load_json_if_exists(run_dir / _CONFIG_FILENAME)
    manifest_payload = _load_json_if_exists(run_dir / _MANIFEST_FILENAME)
    for payload_name, payload in (("config.json", config_payload), ("manifest.json", manifest_payload)):
        if isinstance(payload, dict) and payload.get("strategy_name") is not None:
            strategy_name = str(payload["strategy_name"]).strip()
            if strategy_name:
                return strategy_name
            raise ValueError(
                f"Strategy run metadata file {run_dir / payload_name} contains a blank 'strategy_name'."
            )
    raise ValueError(
        f"Strategy run directory {run_dir} is missing a usable 'strategy_name' in config.json or manifest.json."
    )


def _resolve_run_id(run_dir: Path) -> str:
    manifest_payload = _load_json_if_exists(run_dir / _MANIFEST_FILENAME)
    if isinstance(manifest_payload, dict) and manifest_payload.get("run_id") is not None:
        run_id = str(manifest_payload["run_id"]).strip()
        if run_id:
            return run_id
        raise ValueError(f"Strategy run metadata file {run_dir / _MANIFEST_FILENAME} contains a blank 'run_id'.")
    return run_dir.name


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, found {type(payload).__name__}.")
    return payload


__all__ = [
    "SUPPORTED_PORTFOLIO_COMPONENT_ARTIFACT_TYPES",
    "build_aligned_return_matrix",
    "load_portfolio_component_returns",
    "load_portfolio_component_runs_returns",
    "load_strategy_run_returns",
    "load_strategy_runs_returns",
]
