from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.research.market_simulation.config import (
    MarketSimulationConfig,
    MarketSimulationScenarioConfig,
)
from src.research.market_simulation.ids import generate_path_id
from src.research.market_simulation.validation import MarketSimulationConfigError
from src.research.registry import canonicalize_value

BLOCK_BOOTSTRAP_SCHEMA_VERSION = "1.0"

SOURCE_BLOCK_CATALOG_COLUMNS = (
    "scenario_id",
    "sampled_block_id",
    "source_block_index",
    "source_start_ts_utc",
    "source_end_ts_utc",
    "source_row_count",
    "symbol",
    "primary_regime",
    "regime_values",
    "contains_transition_window",
    "mean_source_return",
    "min_source_return",
    "max_source_return",
)

SAMPLED_BLOCK_INVENTORY_COLUMNS = (
    "scenario_id",
    "path_id",
    "path_index",
    "sample_order",
    "sampled_block_id",
    "source_block_index",
    "source_start_ts_utc",
    "source_end_ts_utc",
    "rows_used",
    "primary_regime",
    "contains_transition_window",
)

BOOTSTRAP_PATH_CATALOG_COLUMNS = (
    "scenario_id",
    "path_id",
    "simulation_type",
    "seed",
    "path_index",
    "path_length_bars",
    "block_length_bars",
    "sampling_mode",
    "target_regimes",
    "source_observation_count",
    "sampled_block_count",
    "transition_block_count",
    "high_vol_block_count",
    "path_start",
    "path_end",
    "notes",
)

SIMULATED_RETURN_PATH_COLUMNS = (
    "scenario_id",
    "path_id",
    "path_step",
    "ts_utc",
    "source_ts_utc",
    "symbol",
    "simulated_return",
    "source_return",
    "regime_label",
    "source_regime_label",
    "is_transition_window",
    "sampled_block_id",
    "gmm_confidence",
    "gmm_entropy",
    "source_symbol",
    "source_row_index",
)

SIMULATED_REGIME_PATH_COLUMNS = (
    "scenario_id",
    "path_id",
    "path_step",
    "ts_utc",
    "regime_label",
    "source_regime_label",
    "is_transition_window",
    "sampled_block_id",
)


@dataclass(frozen=True)
class BlockBootstrapResult:
    scenario_id: str
    scenario_name: str
    output_dir: Path
    bootstrap_config_path: Path
    source_block_catalog_path: Path
    sampled_block_inventory_path: Path
    bootstrap_path_catalog_path: Path
    simulated_return_paths_path: Path
    simulated_regime_paths_path: Path
    bootstrap_sampling_summary_path: Path
    manifest_path: Path
    path_count: int
    simulated_row_count: int
    eligible_block_count: int


@dataclass(frozen=True)
class _BootstrapConfig:
    dataset_path: Path
    display_dataset_path: str
    timestamp_column: str
    return_column: str
    symbol_column: str | None
    regime_column: str | None
    confidence_column: str | None
    entropy_column: str | None
    path_count: int
    path_length_bars: int
    block_length_bars: int
    sampling_mode: str
    target_regimes: tuple[str, ...]
    include_transition_windows: bool
    transition_window_bars: int
    path_start: str

    @classmethod
    def from_scenario(cls, scenario: MarketSimulationScenarioConfig) -> "_BootstrapConfig":
        method_config = scenario.method_config
        dataset_path_value = _required_string(
            method_config.get("dataset_path"), "method_config.dataset_path"
        )
        dataset_path = Path(dataset_path_value)
        resolved_dataset_path = dataset_path if dataset_path.is_absolute() else Path.cwd() / dataset_path
        sampling = _optional_mapping(method_config.get("sampling"), "method_config.sampling")
        sampling_mode = _optional_string(sampling.get("mode"), "method_config.sampling.mode") or "fixed"
        if sampling_mode not in {"fixed", "regime_bucketed", "transition_window", "stress_regime"}:
            raise MarketSimulationConfigError(
                "Regime block bootstrap sampling.mode must be one of: fixed, regime_bucketed, "
                "transition_window, stress_regime."
            )
        target_regimes = _string_tuple(sampling.get("target_regimes"), "method_config.sampling.target_regimes")
        if sampling_mode == "stress_regime" and not target_regimes:
            target_regimes = ("high_vol", "stress")
        path_count = _positive_int(
            method_config.get("path_count", scenario.path_count or 1),
            "method_config.path_count",
        )
        return cls(
            dataset_path=resolved_dataset_path,
            display_dataset_path=_display_path(dataset_path),
            timestamp_column=_required_string(
                method_config.get("timestamp_column"), "method_config.timestamp_column"
            ),
            return_column=_required_string(method_config.get("return_column"), "method_config.return_column"),
            symbol_column=_optional_string(method_config.get("symbol_column"), "method_config.symbol_column"),
            regime_column=_optional_string(method_config.get("regime_column"), "method_config.regime_column"),
            confidence_column=_optional_string(
                method_config.get("confidence_column"), "method_config.confidence_column"
            ),
            entropy_column=_optional_string(method_config.get("entropy_column"), "method_config.entropy_column"),
            path_count=path_count,
            path_length_bars=_positive_int(
                method_config.get("path_length_bars"), "method_config.path_length_bars"
            ),
            block_length_bars=_positive_int(
                method_config.get("block_length_bars"), "method_config.block_length_bars"
            ),
            sampling_mode=sampling_mode,
            target_regimes=target_regimes,
            include_transition_windows=bool(sampling.get("include_transition_windows", False)),
            transition_window_bars=_nonnegative_int(
                sampling.get("transition_window_bars", 0),
                "method_config.sampling.transition_window_bars",
            ),
            path_start=_optional_string(method_config.get("path_start"), "method_config.path_start")
            or "2000-01-01",
        )


@dataclass(frozen=True)
class _SourceBlock:
    sampled_block_id: str
    source_block_index: int
    frame: pd.DataFrame
    metadata: dict[str, Any]


def run_regime_block_bootstrap_scenarios(
    config: MarketSimulationConfig,
    *,
    simulation_run_id: str,
    market_simulations_output_dir: Path,
) -> list[BlockBootstrapResult]:
    results: list[BlockBootstrapResult] = []
    for scenario in config.market_simulations:
        if scenario.simulation_type != "regime_block_bootstrap" or not scenario.enabled:
            continue
        if not _has_bootstrap_method_config(scenario.method_config):
            continue
        results.append(
            run_regime_block_bootstrap(
                scenario,
                simulation_run_id=simulation_run_id,
                market_simulations_output_dir=market_simulations_output_dir,
            )
        )
    return results


def _has_bootstrap_method_config(method_config: Mapping[str, Any]) -> bool:
    return any(
        key in method_config
        for key in ("dataset_path", "path_length_bars", "block_length_bars", "sampling")
    )


def run_regime_block_bootstrap(
    scenario: MarketSimulationScenarioConfig,
    *,
    simulation_run_id: str,
    market_simulations_output_dir: Path,
) -> BlockBootstrapResult:
    bootstrap_config = _BootstrapConfig.from_scenario(scenario)
    source = _normalize_source_frame(_load_dataset(bootstrap_config.dataset_path), bootstrap_config)
    blocks = _build_source_blocks(source, scenario=scenario, bootstrap_config=bootstrap_config)
    eligible_blocks = _eligible_blocks(blocks, bootstrap_config)

    rng = np.random.default_rng(scenario.seed)
    path_catalog_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    simulated_rows: list[dict[str, Any]] = []
    for path_index in range(bootstrap_config.path_count):
        path_id = generate_path_id(
            scenario_id=scenario.scenario_id,
            path_index=path_index,
            seed=scenario.seed,
            metadata={
                "sampling_mode": bootstrap_config.sampling_mode,
                "path_length_bars": bootstrap_config.path_length_bars,
                "block_length_bars": bootstrap_config.block_length_bars,
                "target_regimes": list(bootstrap_config.target_regimes),
            },
        )
        path_rows, path_inventory = _sample_path(
            scenario=scenario,
            bootstrap_config=bootstrap_config,
            eligible_blocks=eligible_blocks,
            rng=rng,
            path_id=path_id,
            path_index=path_index,
        )
        simulated_rows.extend(path_rows)
        inventory_rows.extend(path_inventory)
        path_catalog_rows.append(
            _path_catalog_row(
                scenario=scenario,
                bootstrap_config=bootstrap_config,
                source_observation_count=len(source),
                path_id=path_id,
                path_index=path_index,
                path_inventory=path_inventory,
            )
        )

    source_catalog_rows = [block.metadata for block in blocks]
    scenario_dir = market_simulations_output_dir / scenario.scenario_id
    scenario_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "bootstrap_config_json": scenario_dir / "bootstrap_config.json",
        "source_block_catalog_csv": scenario_dir / "source_block_catalog.csv",
        "sampled_block_inventory_csv": scenario_dir / "sampled_block_inventory.csv",
        "bootstrap_path_catalog_csv": scenario_dir / "bootstrap_path_catalog.csv",
        "simulated_return_paths_parquet": scenario_dir / "simulated_return_paths.parquet",
        "simulated_regime_paths_parquet": scenario_dir / "simulated_regime_paths.parquet",
        "bootstrap_sampling_summary_json": scenario_dir / "bootstrap_sampling_summary.json",
        "bootstrap_manifest_json": scenario_dir / "bootstrap_manifest.json",
    }
    generated_files = {key: path.name for key, path in paths.items()}
    summary = _summary_payload(
        scenario=scenario,
        bootstrap_config=bootstrap_config,
        source_observation_count=len(source),
        eligible_block_count=len(eligible_blocks),
        source_blocks=blocks,
        generated_row_count=len(simulated_rows),
        generated_files=generated_files,
    )
    manifest = _manifest_payload(
        scenario=scenario,
        simulation_run_id=simulation_run_id,
        bootstrap_config=bootstrap_config,
        source_observation_count=len(source),
        source_block_count=len(blocks),
        eligible_block_count=len(eligible_blocks),
        path_row_count=len(path_catalog_rows),
        inventory_row_count=len(inventory_rows),
        simulated_row_count=len(simulated_rows),
        generated_files=generated_files,
    )

    _write_json(paths["bootstrap_config_json"], _normalized_config(scenario, bootstrap_config))
    _write_csv(paths["source_block_catalog_csv"], source_catalog_rows, SOURCE_BLOCK_CATALOG_COLUMNS)
    _write_csv(paths["sampled_block_inventory_csv"], inventory_rows, SAMPLED_BLOCK_INVENTORY_COLUMNS)
    _write_csv(paths["bootstrap_path_catalog_csv"], path_catalog_rows, BOOTSTRAP_PATH_CATALOG_COLUMNS)
    _write_parquet(paths["simulated_return_paths_parquet"], simulated_rows, SIMULATED_RETURN_PATH_COLUMNS)
    _write_parquet(
        paths["simulated_regime_paths_parquet"],
        [{column: row.get(column) for column in SIMULATED_REGIME_PATH_COLUMNS} for row in simulated_rows],
        SIMULATED_REGIME_PATH_COLUMNS,
    )
    _write_json(paths["bootstrap_sampling_summary_json"], summary)
    _write_json(paths["bootstrap_manifest_json"], manifest)

    return BlockBootstrapResult(
        scenario_id=scenario.scenario_id,
        scenario_name=scenario.name,
        output_dir=scenario_dir,
        bootstrap_config_path=paths["bootstrap_config_json"],
        source_block_catalog_path=paths["source_block_catalog_csv"],
        sampled_block_inventory_path=paths["sampled_block_inventory_csv"],
        bootstrap_path_catalog_path=paths["bootstrap_path_catalog_csv"],
        simulated_return_paths_path=paths["simulated_return_paths_parquet"],
        simulated_regime_paths_path=paths["simulated_regime_paths_parquet"],
        bootstrap_sampling_summary_path=paths["bootstrap_sampling_summary_json"],
        manifest_path=paths["bootstrap_manifest_json"],
        path_count=len(path_catalog_rows),
        simulated_row_count=len(simulated_rows),
        eligible_block_count=len(eligible_blocks),
    )


def _load_dataset(dataset_path: Path) -> pd.DataFrame:
    if not dataset_path.exists():
        raise MarketSimulationConfigError(
            f"Regime block bootstrap dataset does not exist: {_display_path(dataset_path)}"
        )
    suffix = dataset_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(dataset_path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(dataset_path)
    raise MarketSimulationConfigError(
        f"Unsupported regime block bootstrap dataset format {dataset_path.suffix!r}. Use CSV or Parquet."
    )


def _normalize_source_frame(frame: pd.DataFrame, bootstrap_config: _BootstrapConfig) -> pd.DataFrame:
    _require_columns(frame, (bootstrap_config.timestamp_column, bootstrap_config.return_column))
    if bootstrap_config.sampling_mode in {"regime_bucketed", "stress_regime", "transition_window"}:
        if bootstrap_config.regime_column is None or bootstrap_config.regime_column not in frame.columns:
            raise MarketSimulationConfigError(
                "Regime block bootstrap regime-conditioned sampling requires "
                "method_config.regime_column to exist in the dataset."
            )
    normalized = frame.copy()
    normalized["_source_ts_utc"] = pd.to_datetime(
        normalized[bootstrap_config.timestamp_column],
        errors="raise",
        utc=True,
    )
    if normalized["_source_ts_utc"].isna().any():
        raise MarketSimulationConfigError(
            f"Regime block bootstrap timestamp column {bootstrap_config.timestamp_column!r} "
            "contains null values."
        )
    normalized["_source_return"] = pd.to_numeric(
        normalized[bootstrap_config.return_column],
        errors="coerce",
    )
    if normalized["_source_return"].isna().any():
        raise MarketSimulationConfigError(
            f"Regime block bootstrap return column {bootstrap_config.return_column!r} "
            "contains null or non-numeric values."
        )
    sort_columns = ["_source_ts_utc"]
    if bootstrap_config.symbol_column is not None and bootstrap_config.symbol_column in normalized.columns:
        sort_columns.append(bootstrap_config.symbol_column)
    normalized = normalized.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    normalized["_source_row_index"] = normalized.index.astype(int)
    normalized["_is_transition_window"] = _transition_window_flags(normalized, bootstrap_config)
    return normalized


def _transition_window_flags(frame: pd.DataFrame, bootstrap_config: _BootstrapConfig) -> pd.Series:
    if bootstrap_config.regime_column is None or bootstrap_config.regime_column not in frame.columns:
        return pd.Series([False] * len(frame), index=frame.index)
    flags = pd.Series([False] * len(frame), index=frame.index)
    group_keys = (
        [bootstrap_config.symbol_column]
        if bootstrap_config.symbol_column is not None and bootstrap_config.symbol_column in frame.columns
        else [None]
    )
    groups = frame.groupby(group_keys[0], sort=True) if group_keys[0] is not None else [(None, frame)]
    window = bootstrap_config.transition_window_bars
    for _, group in groups:
        regimes = group[bootstrap_config.regime_column].astype("string")
        changes = regimes.ne(regimes.shift()).to_numpy()
        if len(changes):
            changes[0] = False
        positions = list(group.index)
        for local_index, changed in enumerate(changes):
            if not changed:
                continue
            start = max(0, local_index - window)
            end = min(len(positions) - 1, local_index + window)
            flags.loc[positions[start : end + 1]] = True
    return flags


def _build_source_blocks(
    frame: pd.DataFrame,
    *,
    scenario: MarketSimulationScenarioConfig,
    bootstrap_config: _BootstrapConfig,
) -> list[_SourceBlock]:
    blocks: list[_SourceBlock] = []
    group_column = (
        bootstrap_config.symbol_column
        if bootstrap_config.symbol_column is not None and bootstrap_config.symbol_column in frame.columns
        else None
    )
    groups = frame.groupby(group_column, sort=True) if group_column is not None else [(None, frame)]
    for symbol, group in groups:
        group = group.sort_values("_source_ts_utc", kind="mergesort").reset_index(drop=True)
        max_start = len(group) - bootstrap_config.block_length_bars
        for start in range(max_start + 1):
            block_frame = group.iloc[start : start + bootstrap_config.block_length_bars].copy()
            block_index = len(blocks)
            sampled_block_id = _generate_sampled_block_id(
                scenario_id=scenario.scenario_id,
                source_block_index=block_index,
                source_row_indices=list(block_frame["_source_row_index"].astype(int)),
            )
            metadata = _source_block_metadata(
                scenario=scenario,
                bootstrap_config=bootstrap_config,
                sampled_block_id=sampled_block_id,
                source_block_index=block_index,
                symbol=None if group_column is None else symbol,
                block_frame=block_frame,
            )
            blocks.append(
                _SourceBlock(
                    sampled_block_id=sampled_block_id,
                    source_block_index=block_index,
                    frame=block_frame,
                    metadata=metadata,
                )
            )
    if not blocks:
        raise MarketSimulationConfigError(
            "Regime block bootstrap produced no source blocks; check block_length_bars and dataset size."
        )
    return blocks


def _eligible_blocks(
    blocks: list[_SourceBlock],
    bootstrap_config: _BootstrapConfig,
) -> list[_SourceBlock]:
    if bootstrap_config.sampling_mode == "fixed":
        eligible = blocks
    elif bootstrap_config.sampling_mode == "transition_window":
        eligible = [block for block in blocks if block.metadata["contains_transition_window"]]
    else:
        if not bootstrap_config.target_regimes:
            raise MarketSimulationConfigError(
                "Regime block bootstrap regime_bucketed sampling requires target_regimes."
            )
        targets = set(bootstrap_config.target_regimes)
        eligible = [block for block in blocks if block.metadata["primary_regime"] in targets]
        if bootstrap_config.include_transition_windows:
            eligible = [
                block
                for block in eligible
                if block.metadata["contains_transition_window"]
                or block.metadata["primary_regime"] in targets
            ]
    if not eligible:
        raise MarketSimulationConfigError(
            "Regime block bootstrap found no eligible source blocks for the configured sampling mode."
        )
    return eligible


def _sample_path(
    *,
    scenario: MarketSimulationScenarioConfig,
    bootstrap_config: _BootstrapConfig,
    eligible_blocks: list[_SourceBlock],
    rng: np.random.Generator,
    path_id: str,
    path_index: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    path_start = _parse_path_start(bootstrap_config.path_start)
    sample_order = 0
    while len(rows) < bootstrap_config.path_length_bars:
        block = eligible_blocks[int(rng.integers(0, len(eligible_blocks)))]
        remaining = bootstrap_config.path_length_bars - len(rows)
        rows_used = min(remaining, len(block.frame))
        inventory_rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "path_id": path_id,
                "path_index": path_index,
                "sample_order": sample_order,
                "sampled_block_id": block.sampled_block_id,
                "source_block_index": block.source_block_index,
                "source_start_ts_utc": block.metadata["source_start_ts_utc"],
                "source_end_ts_utc": block.metadata["source_end_ts_utc"],
                "rows_used": rows_used,
                "primary_regime": block.metadata["primary_regime"],
                "contains_transition_window": block.metadata["contains_transition_window"],
            }
        )
        for _, source_row in block.frame.head(rows_used).iterrows():
            path_step = len(rows)
            rows.append(
                _simulated_row(
                    scenario=scenario,
                    bootstrap_config=bootstrap_config,
                    path_id=path_id,
                    path_step=path_step,
                    ts_utc=path_start + pd.Timedelta(days=path_step),
                    source_row=source_row,
                    sampled_block_id=block.sampled_block_id,
                )
            )
        sample_order += 1
    return rows, inventory_rows


def _source_block_metadata(
    *,
    scenario: MarketSimulationScenarioConfig,
    bootstrap_config: _BootstrapConfig,
    sampled_block_id: str,
    source_block_index: int,
    symbol: Any,
    block_frame: pd.DataFrame,
) -> dict[str, Any]:
    regimes = _regime_values(block_frame, bootstrap_config)
    return {
        "scenario_id": scenario.scenario_id,
        "sampled_block_id": sampled_block_id,
        "source_block_index": source_block_index,
        "source_start_ts_utc": _iso(block_frame["_source_ts_utc"].iloc[0]),
        "source_end_ts_utc": _iso(block_frame["_source_ts_utc"].iloc[-1]),
        "source_row_count": len(block_frame),
        "symbol": None if pd.isna(symbol) else symbol,
        "primary_regime": _primary_regime(regimes),
        "regime_values": "|".join(regimes),
        "contains_transition_window": bool(block_frame["_is_transition_window"].any()),
        "mean_source_return": float(block_frame["_source_return"].mean()),
        "min_source_return": float(block_frame["_source_return"].min()),
        "max_source_return": float(block_frame["_source_return"].max()),
    }


def _simulated_row(
    *,
    scenario: MarketSimulationScenarioConfig,
    bootstrap_config: _BootstrapConfig,
    path_id: str,
    path_step: int,
    ts_utc: pd.Timestamp,
    source_row: pd.Series,
    sampled_block_id: str,
) -> dict[str, Any]:
    regime = _value(source_row, bootstrap_config.regime_column)
    symbol = _value(source_row, bootstrap_config.symbol_column)
    return {
        "scenario_id": scenario.scenario_id,
        "path_id": path_id,
        "path_step": path_step,
        "ts_utc": _iso(ts_utc),
        "source_ts_utc": _iso(source_row["_source_ts_utc"]),
        "symbol": symbol,
        "simulated_return": float(source_row["_source_return"]),
        "source_return": float(source_row["_source_return"]),
        "regime_label": regime,
        "source_regime_label": regime,
        "is_transition_window": bool(source_row["_is_transition_window"]),
        "sampled_block_id": sampled_block_id,
        "gmm_confidence": _value(source_row, bootstrap_config.confidence_column),
        "gmm_entropy": _value(source_row, bootstrap_config.entropy_column),
        "source_symbol": symbol,
        "source_row_index": int(source_row["_source_row_index"]),
    }


def _path_catalog_row(
    *,
    scenario: MarketSimulationScenarioConfig,
    bootstrap_config: _BootstrapConfig,
    source_observation_count: int,
    path_id: str,
    path_index: int,
    path_inventory: list[dict[str, Any]],
) -> dict[str, Any]:
    transition_count = sum(bool(row["contains_transition_window"]) for row in path_inventory)
    high_vol_count = sum(row["primary_regime"] in {"high_vol", "stress"} for row in path_inventory)
    start = _parse_path_start(bootstrap_config.path_start)
    end = start + pd.Timedelta(days=bootstrap_config.path_length_bars - 1)
    return {
        "scenario_id": scenario.scenario_id,
        "path_id": path_id,
        "simulation_type": scenario.simulation_type,
        "seed": scenario.seed,
        "path_index": path_index,
        "path_length_bars": bootstrap_config.path_length_bars,
        "block_length_bars": bootstrap_config.block_length_bars,
        "sampling_mode": bootstrap_config.sampling_mode,
        "target_regimes": "|".join(bootstrap_config.target_regimes),
        "source_observation_count": source_observation_count,
        "sampled_block_count": len(path_inventory),
        "transition_block_count": transition_count,
        "high_vol_block_count": high_vol_count,
        "path_start": _iso(start),
        "path_end": _iso(end),
        "notes": scenario.notes or "",
    }


def _summary_payload(
    *,
    scenario: MarketSimulationScenarioConfig,
    bootstrap_config: _BootstrapConfig,
    source_observation_count: int,
    eligible_block_count: int,
    source_blocks: list[_SourceBlock],
    generated_row_count: int,
    generated_files: Mapping[str, str],
) -> dict[str, Any]:
    return canonicalize_value(
        {
            "scenario_id": scenario.scenario_id,
            "scenario_name": scenario.name,
            "simulation_type": scenario.simulation_type,
            "seed": scenario.seed,
            "path_count": bootstrap_config.path_count,
            "path_length_bars": bootstrap_config.path_length_bars,
            "block_length_bars": bootstrap_config.block_length_bars,
            "sampling_mode": bootstrap_config.sampling_mode,
            "target_regimes": list(bootstrap_config.target_regimes),
            "source_observation_count": source_observation_count,
            "eligible_block_count": eligible_block_count,
            "generated_path_count": bootstrap_config.path_count,
            "generated_row_count": generated_row_count,
            "transition_window_block_count": _transition_block_count(source_blocks),
            "high_vol_or_stress_block_count": _high_vol_block_count(source_blocks),
            "generated_files": dict(generated_files),
            "limitations": _limitations(),
        }
    )


def _manifest_payload(
    *,
    scenario: MarketSimulationScenarioConfig,
    simulation_run_id: str,
    bootstrap_config: _BootstrapConfig,
    source_observation_count: int,
    source_block_count: int,
    eligible_block_count: int,
    path_row_count: int,
    inventory_row_count: int,
    simulated_row_count: int,
    generated_files: Mapping[str, str],
) -> dict[str, Any]:
    return canonicalize_value(
        {
            "artifact_type": "regime_block_bootstrap",
            "schema_version": BLOCK_BOOTSTRAP_SCHEMA_VERSION,
            "simulation_run_id": simulation_run_id,
            "scenario_id": scenario.scenario_id,
            "scenario_name": scenario.name,
            "simulation_type": scenario.simulation_type,
            "generated_files": dict(generated_files),
            "row_counts": {
                "source_block_catalog_csv": source_block_count,
                "sampled_block_inventory_csv": inventory_row_count,
                "bootstrap_path_catalog_csv": path_row_count,
                "simulated_return_paths_parquet": simulated_row_count,
                "simulated_regime_paths_parquet": simulated_row_count,
            },
            "source_dataset_metadata": {
                "dataset_path": bootstrap_config.display_dataset_path,
                "format": bootstrap_config.dataset_path.suffix.lower().lstrip("."),
                "timestamp_column": bootstrap_config.timestamp_column,
                "return_column": bootstrap_config.return_column,
                "symbol_column": bootstrap_config.symbol_column,
                "regime_column": bootstrap_config.regime_column,
                "confidence_column": bootstrap_config.confidence_column,
                "entropy_column": bootstrap_config.entropy_column,
                "source_observation_count": source_observation_count,
            },
            "seed": scenario.seed,
            "path_count": bootstrap_config.path_count,
            "block_length_bars": bootstrap_config.block_length_bars,
            "path_length_bars": bootstrap_config.path_length_bars,
            "eligible_block_count": eligible_block_count,
            "relative_paths": {
                "scenario_dir": scenario.scenario_id,
                **dict(generated_files),
            },
            "limitations": _limitations(),
        }
    )


def _normalized_config(
    scenario: MarketSimulationScenarioConfig,
    bootstrap_config: _BootstrapConfig,
) -> dict[str, Any]:
    return canonicalize_value(
        {
            "scenario": scenario.to_dict(),
            "resolved_bootstrap": {
                "dataset_path": bootstrap_config.display_dataset_path,
                "path_count": bootstrap_config.path_count,
                "path_length_bars": bootstrap_config.path_length_bars,
                "block_length_bars": bootstrap_config.block_length_bars,
                "sampling_mode": bootstrap_config.sampling_mode,
                "target_regimes": list(bootstrap_config.target_regimes),
                "include_transition_windows": bootstrap_config.include_transition_windows,
                "transition_window_bars": bootstrap_config.transition_window_bars,
                "path_start": bootstrap_config.path_start,
                "path_count_precedence": "method_config.path_count > scenario.path_count > 1",
            },
        }
    )


def _limitations() -> list[str]:
    return [
        "Block bootstrap resamples empirical source blocks and does not assume normally distributed returns.",
        "Synthetic path timestamps are deterministic daily bars; source timestamps are preserved separately.",
        "Artifacts are for research stress testing, not live trading or market forecasting.",
        "Shock overlays can consume the stable return/regime columns in a later file-input integration.",
    ]


def _regime_values(frame: pd.DataFrame, bootstrap_config: _BootstrapConfig) -> list[str]:
    if bootstrap_config.regime_column is None or bootstrap_config.regime_column not in frame.columns:
        return []
    values = [
        str(value)
        for value in frame[bootstrap_config.regime_column].dropna().astype(str).tolist()
        if str(value)
    ]
    return sorted(set(values))


def _primary_regime(regimes: list[str]) -> str | None:
    if not regimes:
        return None
    return sorted(regimes)[0]


def _transition_block_count(blocks: list[_SourceBlock]) -> int:
    return sum(bool(block.metadata["contains_transition_window"]) for block in blocks)


def _high_vol_block_count(blocks: list[_SourceBlock]) -> int:
    return sum(block.metadata["primary_regime"] in {"high_vol", "stress"} for block in blocks)


def _generate_sampled_block_id(
    *,
    scenario_id: str,
    source_block_index: int,
    source_row_indices: list[int],
) -> str:
    payload = {
        "scenario_id": scenario_id,
        "source_block_index": source_block_index,
        "source_row_indices": source_row_indices,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return f"{scenario_id}_block_{source_block_index:06d}_{digest}"


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise MarketSimulationConfigError(
            f"Regime block bootstrap dataset is missing required columns: {missing}."
        )


def _required_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MarketSimulationConfigError(
            f"Regime block bootstrap field '{field_name}' must be a non-empty string."
        )
    return value.strip()


def _optional_string(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _required_string(value, field_name)


def _optional_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise MarketSimulationConfigError(
            f"Regime block bootstrap field '{field_name}' must be a mapping."
        )
    return value


def _string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list | tuple):
        raise MarketSimulationConfigError(
            f"Regime block bootstrap field '{field_name}' must be a sequence."
        )
    return tuple(_required_string(item, field_name) for item in value)


def _positive_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise MarketSimulationConfigError(
            f"Regime block bootstrap field '{field_name}' must be a positive integer."
        )
    return int(value)


def _nonnegative_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise MarketSimulationConfigError(
            f"Regime block bootstrap field '{field_name}' must be a non-negative integer."
        )
    return int(value)


def _parse_path_start(value: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except ValueError as exc:
        raise MarketSimulationConfigError(
            f"Regime block bootstrap path_start timestamp {value!r} is invalid."
        ) from exc
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _value(row: pd.Series, column: str | None) -> Any:
    if column is None or column not in row:
        return None
    value = row[column]
    if pd.isna(value):
        return None
    return value


def _iso(value: Any) -> str:
    return pd.Timestamp(value).tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def _write_parquet(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    frame = pd.DataFrame([{column: row.get(column) for column in columns} for row in rows])
    if frame.empty:
        frame = pd.DataFrame(columns=list(columns))
    frame.to_parquet(path, index=False)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(canonicalize_value(payload), indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )


def _display_path(path: Path) -> str:
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        digest = hashlib.sha256(path.as_posix().encode("utf-8")).hexdigest()[:12]
        return f"external/{path.name}_{digest}"
