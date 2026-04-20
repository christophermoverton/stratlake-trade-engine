from __future__ import annotations

import csv
from dataclasses import dataclass
import fnmatch
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from src.cli.run_research_campaign import (
    _SCENARIO_RANKING_METRIC_PRIORITY,
    _scenario_matrix_fieldnames,
    _scenario_matrix_selection_rule,
    _scenario_matrix_sort_key,
    run_research_campaign,
)
from src.config.benchmark_pack import BenchmarkPackConfig
from src.config.research_campaign import (
    ResearchCampaignConfig,
    expand_research_campaign_scenarios,
    load_research_campaign_config,
    resolve_research_campaign_config,
)
from src.research.registry import canonicalize_value

BENCHMARK_CHECKPOINT_FILENAME = "checkpoint.json"
BENCHMARK_MANIFEST_FILENAME = "manifest.json"
BENCHMARK_SUMMARY_FILENAME = "summary.json"
BENCHMARK_CONFIG_FILENAME = "benchmark_pack_config.json"
BENCHMARK_DATASET_SUMMARY_FILENAME = "dataset_summary.json"
BENCHMARK_BATCH_PLAN_FILENAME = "batch_plan.json"
BENCHMARK_BATCH_PLAN_CSV_FILENAME = "batch_plan.csv"
BENCHMARK_MATRIX_CSV_FILENAME = "benchmark_matrix.csv"
BENCHMARK_MATRIX_SUMMARY_FILENAME = "benchmark_matrix.json"
BENCHMARK_INVENTORY_FILENAME = "inventory.json"


@dataclass(frozen=True)
class BenchmarkBatchPlan:
    batch_id: str
    batch_index: int
    input_fingerprint: str
    scenario_ids: tuple[str, ...]
    scenario_fingerprints: tuple[str, ...]
    include_entries: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return canonicalize_value(
            {
                "batch_id": self.batch_id,
                "batch_index": self.batch_index,
                "input_fingerprint": self.input_fingerprint,
                "scenario_ids": list(self.scenario_ids),
                "scenario_fingerprints": list(self.scenario_fingerprints),
                "scenario_count": len(self.scenario_ids),
                "include_entries": [dict(entry) for entry in self.include_entries],
            }
        )


@dataclass(frozen=True)
class BenchmarkPackRunResult:
    pack_id: str
    pack_run_id: str
    status: str
    output_root: Path
    checkpoint_path: Path
    manifest_path: Path
    summary_path: Path
    inventory_path: Path
    batch_plan_path: Path
    batch_plan_csv_path: Path
    benchmark_matrix_csv_path: Path
    benchmark_matrix_summary_path: Path
    comparison_path: Path | None
    checkpoint: dict[str, Any]
    manifest: dict[str, Any]
    summary: dict[str, Any]
    inventory: dict[str, Any]
    comparison: dict[str, Any] | None


def run_benchmark_pack(
    config: BenchmarkPackConfig,
    *,
    output_root: Path | None = None,
    compare_to: Path | None = None,
    stop_after_batches: int | None = None,
    campaign_runner: Callable[[ResearchCampaignConfig], Any] | None = None,
) -> BenchmarkPackRunResult:
    runner = run_research_campaign if campaign_runner is None else campaign_runner
    resolved_output_root = _resolve_output_root(config, output_root=output_root)
    resolved_output_root.mkdir(parents=True, exist_ok=True)
    _write_json(resolved_output_root / BENCHMARK_CONFIG_FILENAME, config.to_dict())

    dataset_summary = _prepare_dataset(config, output_root=resolved_output_root)
    base_campaign_config = _load_campaign_config(config)
    scenario_batches = _build_batch_plan(base_campaign_config, batch_size=config.batching.batch_size)
    batch_plan_payload = {
        "pack_id": config.pack_id,
        "batch_size": config.batching.batch_size,
        "batch_count": len(scenario_batches),
        "scenario_count": sum(len(batch.scenario_ids) for batch in scenario_batches),
        "batches": [batch.to_dict() for batch in scenario_batches],
    }
    batch_plan_path = resolved_output_root / BENCHMARK_BATCH_PLAN_FILENAME
    batch_plan_csv_path = resolved_output_root / BENCHMARK_BATCH_PLAN_CSV_FILENAME
    _write_json(batch_plan_path, batch_plan_payload)
    _write_batch_plan_csv(batch_plan_csv_path, scenario_batches)

    pack_run_id = _build_pack_run_id(config, scenario_batches)
    checkpoint_path = resolved_output_root / BENCHMARK_CHECKPOINT_FILENAME
    manifest_path = resolved_output_root / BENCHMARK_MANIFEST_FILENAME
    summary_path = resolved_output_root / BENCHMARK_SUMMARY_FILENAME
    benchmark_matrix_csv_path = resolved_output_root / BENCHMARK_MATRIX_CSV_FILENAME
    benchmark_matrix_summary_path = resolved_output_root / BENCHMARK_MATRIX_SUMMARY_FILENAME
    inventory_path = resolved_output_root / BENCHMARK_INVENTORY_FILENAME

    existing_checkpoint = _load_existing_checkpoint(checkpoint_path, pack_run_id=pack_run_id)
    batch_records = _seed_batch_records(scenario_batches, existing_checkpoint=existing_checkpoint)
    processed_new_batches = 0

    with _features_root_environment():
        os.environ["FEATURES_ROOT"] = dataset_summary["features_root"]
        for batch in scenario_batches:
            existing_record = batch_records[batch.batch_id]
            if _can_reuse_batch(existing_record, batch=batch):
                loaded_summary = _read_json(Path(str(existing_record["orchestration_summary_path"])))
                batch_records[batch.batch_id] = canonicalize_value(
                    {
                        **dict(existing_record),
                        "status": "reused",
                        "scenario_status_counts": dict(loaded_summary.get("scenario_status_counts", {})),
                    }
                )
                _persist_benchmark_state(
                    config=config,
                    pack_run_id=pack_run_id,
                    output_root=resolved_output_root,
                    checkpoint_path=checkpoint_path,
                    manifest_path=manifest_path,
                    summary_path=summary_path,
                    batch_plan_path=batch_plan_path,
                    batch_plan_csv_path=batch_plan_csv_path,
                    benchmark_matrix_csv_path=benchmark_matrix_csv_path,
                    benchmark_matrix_summary_path=benchmark_matrix_summary_path,
                    inventory_path=inventory_path,
                    dataset_summary=dataset_summary,
                    batch_records=batch_records,
                    status="partial",
                    comparison=None,
                    inventory_payload=None,
                )
                continue

            batch_output_root = resolved_output_root / "batches" / batch.batch_id
            batch_output_root.mkdir(parents=True, exist_ok=True)
            batch_config = _build_batch_config(
                base_campaign_config,
                batch=batch,
                batch_output_root=batch_output_root,
            )
            _write_json(batch_output_root / "batch_config.json", batch_config.to_dict())
            with _working_directory(batch_output_root):
                result = runner(batch_config)
            summary_payload, _manifest_payload, orchestration_run_id = _extract_orchestration_payloads(result)
            batch_records[batch.batch_id] = canonicalize_value(
                {
                    "batch_id": batch.batch_id,
                    "batch_index": batch.batch_index,
                    "status": "completed",
                    "input_fingerprint": batch.input_fingerprint,
                    "scenario_ids": list(batch.scenario_ids),
                    "scenario_fingerprints": list(batch.scenario_fingerprints),
                    "scenario_count": len(batch.scenario_ids),
                    "orchestration_run_id": orchestration_run_id,
                    "orchestration_artifact_dir": str(
                        getattr(result, "orchestration_artifact_dir", batch_output_root).as_posix()
                    ),
                    "orchestration_summary_path": str(getattr(result, "orchestration_summary_path").as_posix()),
                    "orchestration_manifest_path": str(getattr(result, "orchestration_manifest_path").as_posix()),
                    "scenario_matrix_summary_path": str(getattr(result, "scenario_matrix_summary_path").as_posix()),
                    "scenario_matrix_csv_path": str(getattr(result, "scenario_matrix_csv_path").as_posix()),
                    "scenario_status_counts": dict(summary_payload.get("scenario_status_counts", {})),
                    "scenario_count_completed": int(summary_payload.get("scenario_count", 0)),
                }
            )
            processed_new_batches += 1
            _persist_benchmark_state(
                config=config,
                pack_run_id=pack_run_id,
                output_root=resolved_output_root,
                checkpoint_path=checkpoint_path,
                manifest_path=manifest_path,
                summary_path=summary_path,
                batch_plan_path=batch_plan_path,
                batch_plan_csv_path=batch_plan_csv_path,
                benchmark_matrix_csv_path=benchmark_matrix_csv_path,
                benchmark_matrix_summary_path=benchmark_matrix_summary_path,
                inventory_path=inventory_path,
                dataset_summary=dataset_summary,
                batch_records=batch_records,
                status="partial",
                comparison=None,
                inventory_payload=None,
            )
            if stop_after_batches is not None and processed_new_batches >= stop_after_batches:
                return _finalize_partial_result(
                    config=config,
                    pack_run_id=pack_run_id,
                    output_root=resolved_output_root,
                    checkpoint_path=checkpoint_path,
                    manifest_path=manifest_path,
                    summary_path=summary_path,
                    inventory_path=inventory_path,
                    batch_plan_path=batch_plan_path,
                    batch_plan_csv_path=batch_plan_csv_path,
                    benchmark_matrix_csv_path=benchmark_matrix_csv_path,
                    benchmark_matrix_summary_path=benchmark_matrix_summary_path,
                )

    inventory_payload = _build_inventory(
        resolved_output_root,
        exclude_globs=config.outputs.inventory_excludes,
    )
    _write_json(inventory_path, inventory_payload)
    comparison_payload = None
    comparison_path = None
    if compare_to is not None:
        reference_inventory = _read_json(compare_to)
        comparison_payload = _compare_inventories(
            reference_inventory=reference_inventory,
            current_inventory=inventory_payload,
            reference_path=compare_to,
            current_path=inventory_path,
        )
        comparison_path = resolved_output_root / "comparisons" / "inventory_comparison.json"
        _write_json(comparison_path, comparison_payload)

    _persist_benchmark_state(
        config=config,
        pack_run_id=pack_run_id,
        output_root=resolved_output_root,
        checkpoint_path=checkpoint_path,
        manifest_path=manifest_path,
        summary_path=summary_path,
        batch_plan_path=batch_plan_path,
        batch_plan_csv_path=batch_plan_csv_path,
        benchmark_matrix_csv_path=benchmark_matrix_csv_path,
        benchmark_matrix_summary_path=benchmark_matrix_summary_path,
        inventory_path=inventory_path,
        dataset_summary=dataset_summary,
        batch_records=batch_records,
        status="completed",
        comparison=comparison_payload,
        inventory_payload=inventory_payload,
    )

    final_checkpoint = _read_json(checkpoint_path)
    final_manifest = _read_json(manifest_path)
    final_summary = _read_json(summary_path)
    return BenchmarkPackRunResult(
        pack_id=config.pack_id,
        pack_run_id=pack_run_id,
        status=str(final_summary.get("status", "unknown")),
        output_root=resolved_output_root,
        checkpoint_path=checkpoint_path,
        manifest_path=manifest_path,
        summary_path=summary_path,
        inventory_path=inventory_path,
        batch_plan_path=batch_plan_path,
        batch_plan_csv_path=batch_plan_csv_path,
        benchmark_matrix_csv_path=benchmark_matrix_csv_path,
        benchmark_matrix_summary_path=benchmark_matrix_summary_path,
        comparison_path=comparison_path,
        checkpoint=final_checkpoint,
        manifest=final_manifest,
        summary=final_summary,
        inventory=inventory_payload,
        comparison=comparison_payload,
    )


def _finalize_partial_result(
    *,
    config: BenchmarkPackConfig,
    pack_run_id: str,
    output_root: Path,
    checkpoint_path: Path,
    manifest_path: Path,
    summary_path: Path,
    inventory_path: Path,
    batch_plan_path: Path,
    batch_plan_csv_path: Path,
    benchmark_matrix_csv_path: Path,
    benchmark_matrix_summary_path: Path,
) -> BenchmarkPackRunResult:
    inventory_payload = _build_inventory(
        output_root,
        exclude_globs=config.outputs.inventory_excludes,
    )
    _write_json(inventory_path, inventory_payload)
    summary_payload = _read_json(summary_path)
    summary_payload["inventory"] = inventory_payload
    summary_payload["output_paths"]["inventory"] = inventory_path.as_posix()
    _write_json(summary_path, summary_payload)
    return BenchmarkPackRunResult(
        pack_id=config.pack_id,
        pack_run_id=pack_run_id,
        status="partial",
        output_root=output_root,
        checkpoint_path=checkpoint_path,
        manifest_path=manifest_path,
        summary_path=summary_path,
        inventory_path=inventory_path,
        batch_plan_path=batch_plan_path,
        batch_plan_csv_path=batch_plan_csv_path,
        benchmark_matrix_csv_path=benchmark_matrix_csv_path,
        benchmark_matrix_summary_path=benchmark_matrix_summary_path,
        comparison_path=None,
        checkpoint=_read_json(checkpoint_path),
        manifest=_read_json(manifest_path),
        summary=summary_payload,
        inventory=inventory_payload,
        comparison=None,
    )


def _load_campaign_config(config: BenchmarkPackConfig) -> ResearchCampaignConfig:
    loaded = load_research_campaign_config(Path(config.campaign.config_path))
    payload = loaded.to_dict()
    config_dir = Path(config.campaign.config_path).parent
    targets = dict(payload.get("targets", {}))
    for key in ("alpha_catalog_path", "strategy_config_path", "portfolio_config_path"):
        value = targets.get(key)
        if isinstance(value, str) and value.strip():
            targets[key] = _resolve_existing_path_string(value, config_dir=config_dir)
    payload["targets"] = targets
    dataset_selection = dict(payload.get("dataset_selection", {}))
    tickers_path = dataset_selection.get("tickers_path")
    if isinstance(tickers_path, str) and tickers_path.strip():
        dataset_selection["tickers_path"] = _resolve_existing_path_string(
            tickers_path,
            config_dir=config_dir,
        )
    payload["dataset_selection"] = dataset_selection
    return resolve_research_campaign_config(payload, config.campaign.overrides)


def _build_pack_run_id(config: BenchmarkPackConfig, batches: Sequence[BenchmarkBatchPlan]) -> str:
    payload = canonicalize_value(
        {
            "pack_id": config.pack_id,
            "dataset": config.dataset.to_dict(),
            "campaign": config.campaign.to_dict(),
            "batching": config.batching.to_dict(),
            "batches": [batch.to_dict() for batch in batches],
        }
    )
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    return f"{config.pack_id}_{digest}"


def _build_batch_plan(
    base_campaign_config: ResearchCampaignConfig,
    *,
    batch_size: int,
) -> tuple[BenchmarkBatchPlan, ...]:
    scenarios = expand_research_campaign_scenarios(base_campaign_config)
    batches: list[BenchmarkBatchPlan] = []
    for index in range(0, len(scenarios), batch_size):
        batch_index = index // batch_size
        chunk = scenarios[index : index + batch_size]
        fingerprint_payload = [
            {
                "scenario_id": scenario.scenario_id,
                "fingerprint": scenario.fingerprint,
                "source": scenario.source,
            }
            for scenario in chunk
        ]
        input_fingerprint = hashlib.sha256(
            json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]
        include_entries = tuple(
            canonicalize_value(
                {
                    "scenario_id": scenario.scenario_id,
                    "description": scenario.description,
                    "overrides": getattr(scenario, "overrides", {}),
                }
            )
            for scenario in chunk
        )
        batches.append(
            BenchmarkBatchPlan(
                batch_id=f"batch_{batch_index:04d}",
                batch_index=batch_index,
                input_fingerprint=input_fingerprint,
                scenario_ids=tuple(scenario.scenario_id for scenario in chunk),
                scenario_fingerprints=tuple(scenario.fingerprint for scenario in chunk),
                include_entries=include_entries,
            )
        )
    return tuple(batches)


def _build_batch_config(
    base_campaign_config: ResearchCampaignConfig,
    *,
    batch: BenchmarkBatchPlan,
    batch_output_root: Path,
) -> ResearchCampaignConfig:
    payload = dict(base_campaign_config.to_dict())
    payload["outputs"] = {
        "alpha_artifacts_root": (batch_output_root / "artifacts" / "alpha").as_posix(),
        "candidate_selection_output_path": (
            batch_output_root / "artifacts" / "candidate_selection"
        ).as_posix(),
        "candidate_selection_registry_path": None,
        "campaign_artifacts_root": (batch_output_root / "artifacts" / "research_campaigns").as_posix(),
        "portfolio_artifacts_root": (batch_output_root / "artifacts" / "portfolios").as_posix(),
        "comparison_output_path": (batch_output_root / "artifacts" / "campaign_comparisons").as_posix(),
        "review_output_path": (batch_output_root / "artifacts" / "reviews" / "research_review").as_posix(),
    }
    payload["candidate_selection"]["artifacts_root"] = (batch_output_root / "artifacts" / "alpha").as_posix()
    payload["candidate_selection"]["output"] = {
        "path": (batch_output_root / "artifacts" / "candidate_selection").as_posix(),
        "registry_path": None,
        "review_output_path": (
            batch_output_root / "artifacts" / "reviews" / "candidate_review"
        ).as_posix(),
    }
    payload["review"]["output"]["path"] = (
        batch_output_root / "artifacts" / "reviews" / "research_review"
    ).as_posix()
    payload["scenarios"] = {
        "enabled": True,
        "matrix": [],
        "include": [dict(entry) for entry in batch.include_entries],
        "max_scenarios": None,
        "max_values_per_axis": None,
    }
    return resolve_research_campaign_config(payload)


def _prepare_dataset(config: BenchmarkPackConfig, *, output_root: Path) -> dict[str, Any]:
    features_root = output_root / Path(config.dataset.features_root)
    dataset_root = features_root / "curated" / config.dataset.dataset_name
    dataset_summary_path = output_root / BENCHMARK_DATASET_SUMMARY_FILENAME
    if config.dataset.generator == "synthetic_features_daily":
        _write_synthetic_features_dataset(
            dataset_root=dataset_root,
            start_date=config.dataset.start_date,
            periods=config.dataset.periods,
            frequency=config.dataset.frequency,
            symbols=config.dataset.symbols,
        )
    summary = {
        "generator": config.dataset.generator,
        "features_root": features_root.as_posix(),
        "dataset_root": dataset_root.as_posix(),
        "dataset_name": config.dataset.dataset_name,
        "start_date": config.dataset.start_date,
        "periods": config.dataset.periods,
        "frequency": config.dataset.frequency,
        "symbols": list(config.dataset.symbols),
        "parquet_file_count": sum(1 for _ in dataset_root.glob("**/*.parquet")),
    }
    _write_json(dataset_summary_path, summary)
    return summary


def _write_synthetic_features_dataset(
    *,
    dataset_root: Path,
    start_date: str,
    periods: int,
    frequency: str,
    symbols: Sequence[str],
) -> None:
    expected_files = [dataset_root / f"symbol={symbol}" / "year=2025" / "part-0.parquet" for symbol in symbols]
    if all(path.exists() for path in expected_files):
        return

    timestamps = pd.date_range(start_date, periods=periods, freq=frequency, tz="UTC")
    rows: list[dict[str, Any]] = []
    for symbol_index, symbol in enumerate(symbols):
        base = float(symbol_index + 1)
        for ts_index, ts_utc in enumerate(timestamps):
            close = 100.0 + base * 9.5 + ts_index * (0.45 + base * 0.11)
            feature_ret_1d = base * 0.009 + ts_index * 0.00035
            feature_ret_5d = base * 0.014 + ts_index * 0.00042
            feature_ret_20d = base * 0.018 + ts_index * 0.00051
            feature_vol_20d = None if ts_index < 19 else 0.011 + base * 0.002 + ts_index * 0.00005
            feature_close_to_sma20 = None if ts_index < 19 else 0.95 + base * 0.02 + ts_index * 0.001
            rows.append(
                {
                    "symbol": symbol,
                    "ts_utc": ts_utc,
                    "timeframe": "1D",
                    "date": ts_utc.strftime("%Y-%m-%d"),
                    "close": close,
                    "feature_ret_1d": feature_ret_1d,
                    "feature_ret_5d": feature_ret_5d,
                    "feature_ret_20d": feature_ret_20d,
                    "feature_vol_20d": feature_vol_20d,
                    "feature_close_to_sma20": feature_close_to_sma20,
                    "target_ret_1d": feature_ret_1d * 0.82,
                    "target_ret_5d": feature_ret_5d * 0.79,
                }
            )
    frame = pd.DataFrame(rows).sort_values(["symbol", "ts_utc"], kind="stable").reset_index(drop=True)
    for column in ("symbol", "timeframe", "date"):
        frame[column] = frame[column].astype("string")
    for symbol in symbols:
        symbol_frame = frame.loc[frame["symbol"].eq(symbol)].copy(deep=True)
        output_dir = dataset_root / f"symbol={symbol}" / f"year={timestamps[0].year}"
        output_dir.mkdir(parents=True, exist_ok=True)
        symbol_frame.to_parquet(output_dir / "part-0.parquet", index=False)


def _seed_batch_records(
    batches: Sequence[BenchmarkBatchPlan],
    *,
    existing_checkpoint: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    existing_by_id: dict[str, dict[str, Any]] = {}
    if isinstance(existing_checkpoint, Mapping):
        for batch in existing_checkpoint.get("batches", []):
            if isinstance(batch, Mapping) and isinstance(batch.get("batch_id"), str):
                existing_by_id[str(batch["batch_id"])] = dict(batch)
    for batch in batches:
        existing = existing_by_id.get(batch.batch_id)
        if existing is None:
            records[batch.batch_id] = canonicalize_value(
                {
                    "batch_id": batch.batch_id,
                    "batch_index": batch.batch_index,
                    "status": "pending",
                    "input_fingerprint": batch.input_fingerprint,
                    "scenario_ids": list(batch.scenario_ids),
                    "scenario_fingerprints": list(batch.scenario_fingerprints),
                    "scenario_count": len(batch.scenario_ids),
                }
            )
            continue
        records[batch.batch_id] = canonicalize_value(dict(existing))
    return records


def _can_reuse_batch(existing_record: Mapping[str, Any], *, batch: BenchmarkBatchPlan) -> bool:
    if str(existing_record.get("status") or "") not in {"completed", "reused"}:
        return False
    if str(existing_record.get("input_fingerprint") or "") != batch.input_fingerprint:
        return False
    summary_path = existing_record.get("orchestration_summary_path")
    manifest_path = existing_record.get("orchestration_manifest_path")
    if not isinstance(summary_path, str) or not Path(summary_path).exists():
        return False
    if not isinstance(manifest_path, str) or not Path(manifest_path).exists():
        return False
    return True


def _persist_benchmark_state(
    *,
    config: BenchmarkPackConfig,
    pack_run_id: str,
    output_root: Path,
    checkpoint_path: Path,
    manifest_path: Path,
    summary_path: Path,
    batch_plan_path: Path,
    batch_plan_csv_path: Path,
    benchmark_matrix_csv_path: Path,
    benchmark_matrix_summary_path: Path,
    inventory_path: Path,
    dataset_summary: Mapping[str, Any],
    batch_records: Mapping[str, Mapping[str, Any]],
    status: str,
    comparison: Mapping[str, Any] | None,
    inventory_payload: Mapping[str, Any] | None,
) -> None:
    batch_rows = [canonicalize_value(dict(batch_records[batch_id])) for batch_id in sorted(batch_records)]
    benchmark_matrix = _write_benchmark_matrix(
        batch_rows,
        csv_path=benchmark_matrix_csv_path,
        summary_path=benchmark_matrix_summary_path,
        pack_run_id=pack_run_id,
    )
    status_counts: dict[str, int] = {}
    for batch in batch_rows:
        state = str(batch.get("status") or "unknown")
        status_counts[state] = status_counts.get(state, 0) + 1
    checkpoint_payload = canonicalize_value(
        {
            "schema_version": 1,
            "run_type": "benchmark_pack_checkpoint",
            "pack_id": config.pack_id,
            "pack_run_id": pack_run_id,
            "status": status,
            "output_root": output_root.as_posix(),
            "batch_count": len(batch_rows),
            "batch_status_counts": dict(sorted(status_counts.items())),
            "completed_batch_ids": [
                batch["batch_id"] for batch in batch_rows if batch.get("status") == "completed"
            ],
            "reused_batch_ids": [
                batch["batch_id"] for batch in batch_rows if batch.get("status") == "reused"
            ],
            "pending_batch_ids": [
                batch["batch_id"] for batch in batch_rows if batch.get("status") == "pending"
            ],
            "batches": batch_rows,
        }
    )
    _write_json(checkpoint_path, checkpoint_payload)
    output_paths = {
        "output_root": output_root.as_posix(),
        "config": (output_root / BENCHMARK_CONFIG_FILENAME).as_posix(),
        "dataset_summary": (output_root / BENCHMARK_DATASET_SUMMARY_FILENAME).as_posix(),
        "batch_plan": batch_plan_path.as_posix(),
        "batch_plan_csv": batch_plan_csv_path.as_posix(),
        "benchmark_matrix_csv": benchmark_matrix_csv_path.as_posix(),
        "benchmark_matrix_summary": benchmark_matrix_summary_path.as_posix(),
        "checkpoint": checkpoint_path.as_posix(),
        "manifest": manifest_path.as_posix(),
        "summary": summary_path.as_posix(),
        "inventory": inventory_path.as_posix(),
    }
    if comparison is not None:
        output_paths["comparison"] = (output_root / "comparisons" / "inventory_comparison.json").as_posix()
    summary_payload = canonicalize_value(
        {
            "run_type": "benchmark_pack",
            "pack_id": config.pack_id,
            "pack_run_id": pack_run_id,
            "status": status,
            "description": config.description,
            "batch_size": config.batching.batch_size,
            "batch_count": len(batch_rows),
            "scenario_count": int(benchmark_matrix.get("row_count", 0)),
            "batch_status_counts": dict(sorted(status_counts.items())),
            "resume": {
                "completed_batch_ids": checkpoint_payload["completed_batch_ids"],
                "reused_batch_ids": checkpoint_payload["reused_batch_ids"],
                "pending_batch_ids": checkpoint_payload["pending_batch_ids"],
            },
            "dataset": canonicalize_value(dict(dataset_summary)),
            "benchmark_matrix": benchmark_matrix,
            "batches": batch_rows,
            "comparison": None if comparison is None else canonicalize_value(dict(comparison)),
            "inventory": None if inventory_payload is None else canonicalize_value(dict(inventory_payload)),
            "output_paths": output_paths,
        }
    )
    _write_json(summary_path, summary_payload)
    manifest_payload = canonicalize_value(
        {
            "run_type": "benchmark_pack",
            "pack_id": config.pack_id,
            "pack_run_id": pack_run_id,
            "status": status,
            "artifact_files": sorted(
                [
                    BENCHMARK_CONFIG_FILENAME,
                    BENCHMARK_DATASET_SUMMARY_FILENAME,
                    BENCHMARK_BATCH_PLAN_FILENAME,
                    BENCHMARK_BATCH_PLAN_CSV_FILENAME,
                    BENCHMARK_CHECKPOINT_FILENAME,
                    BENCHMARK_MANIFEST_FILENAME,
                    BENCHMARK_SUMMARY_FILENAME,
                    BENCHMARK_MATRIX_CSV_FILENAME,
                    BENCHMARK_MATRIX_SUMMARY_FILENAME,
                    BENCHMARK_INVENTORY_FILENAME,
                ]
            ),
            "artifact_groups": canonicalize_value(
                {
                    "benchmark_pack": [
                        BENCHMARK_CONFIG_FILENAME,
                        BENCHMARK_DATASET_SUMMARY_FILENAME,
                        BENCHMARK_BATCH_PLAN_FILENAME,
                        BENCHMARK_BATCH_PLAN_CSV_FILENAME,
                        BENCHMARK_CHECKPOINT_FILENAME,
                        BENCHMARK_MANIFEST_FILENAME,
                        BENCHMARK_SUMMARY_FILENAME,
                        BENCHMARK_MATRIX_CSV_FILENAME,
                        BENCHMARK_MATRIX_SUMMARY_FILENAME,
                        BENCHMARK_INVENTORY_FILENAME,
                    ],
                }
            ),
            "summary_path": BENCHMARK_SUMMARY_FILENAME,
            "batch_status_counts": dict(sorted(status_counts.items())),
            "batches": [
                {
                    "batch_id": batch["batch_id"],
                    "status": batch["status"],
                    "orchestration_run_id": batch.get("orchestration_run_id"),
                    "orchestration_summary_path": batch.get("orchestration_summary_path"),
                    "orchestration_manifest_path": batch.get("orchestration_manifest_path"),
                    "scenario_ids": batch.get("scenario_ids"),
                }
                for batch in batch_rows
            ],
        }
    )
    _write_json(manifest_path, manifest_payload)


def _write_benchmark_matrix(
    batch_rows: Sequence[Mapping[str, Any]],
    *,
    csv_path: Path,
    summary_path: Path,
    pack_run_id: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    sweep_keys: list[str] = []
    for batch in batch_rows:
        summary_path_text = batch.get("scenario_matrix_summary_path")
        if not isinstance(summary_path_text, str) or not Path(summary_path_text).exists():
            continue
        matrix_summary = _read_json(Path(summary_path_text))
        for row in matrix_summary.get("leaderboard", []):
            row_payload = dict(row)
            row_payload["batch_id"] = batch.get("batch_id")
            row_payload.pop("rank", None)
            rows.append(canonicalize_value(row_payload))
            for key in row_payload:
                if key.startswith("sweep_"):
                    sweep_key = key.removeprefix("sweep_")
                    if sweep_key not in sweep_keys:
                        sweep_keys.append(sweep_key)
    ordered_rows = sorted(rows, key=_scenario_matrix_sort_key)
    ranked_rows = [canonicalize_value({"rank": index, **row}) for index, row in enumerate(ordered_rows, start=1)]
    fieldnames = _scenario_matrix_fieldnames(sweep_keys) + ["batch_id"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in ranked_rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    summary = canonicalize_value(
        {
            "benchmark_matrix_id": pack_run_id,
            "pack_run_id": pack_run_id,
            "row_count": len(ranked_rows),
            "metric_priority": list(_SCENARIO_RANKING_METRIC_PRIORITY),
            "selection_rule": _scenario_matrix_selection_rule(),
            "sweep_keys": sweep_keys,
            "leaderboard": ranked_rows,
            "csv_path": csv_path.as_posix(),
            "summary_path": summary_path.as_posix(),
        }
    )
    _write_json(summary_path, summary)
    return summary


def _build_inventory(output_root: Path, *, exclude_globs: Sequence[str]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for path in sorted(output_root.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(output_root).as_posix()
        if any(fnmatch.fnmatch(relative_path, pattern) for pattern in exclude_globs):
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append(
            {
                "path": relative_path,
                "sha256": digest,
                "size_bytes": path.stat().st_size,
            }
        )
    aggregate_payload = "|".join(f"{entry['path']}:{entry['sha256']}" for entry in entries)
    aggregate_digest = hashlib.sha256(aggregate_payload.encode("utf-8")).hexdigest()
    return canonicalize_value(
        {
            "file_count": len(entries),
            "aggregate_digest": aggregate_digest,
            "entries": entries,
        }
    )


def _compare_inventories(
    *,
    reference_inventory: Mapping[str, Any],
    current_inventory: Mapping[str, Any],
    reference_path: Path,
    current_path: Path,
) -> dict[str, Any]:
    reference_entries = {
        str(entry["path"]): str(entry["sha256"])
        for entry in reference_inventory.get("entries", [])
        if isinstance(entry, Mapping) and "path" in entry and "sha256" in entry
    }
    current_entries = {
        str(entry["path"]): str(entry["sha256"])
        for entry in current_inventory.get("entries", [])
        if isinstance(entry, Mapping) and "path" in entry and "sha256" in entry
    }
    added = sorted(path for path in current_entries if path not in reference_entries)
    removed = sorted(path for path in reference_entries if path not in current_entries)
    changed = sorted(
        path
        for path in current_entries
        if path in reference_entries and current_entries[path] != reference_entries[path]
    )
    return canonicalize_value(
        {
            "reference_inventory_path": reference_path.as_posix(),
            "current_inventory_path": current_path.as_posix(),
            "reference_aggregate_digest": reference_inventory.get("aggregate_digest"),
            "current_aggregate_digest": current_inventory.get("aggregate_digest"),
            "matches": not added and not removed and not changed,
            "added_files": added,
            "removed_files": removed,
            "changed_files": changed,
        }
    )


def _extract_orchestration_payloads(result: Any) -> tuple[dict[str, Any], dict[str, Any], str]:
    summary_payload = getattr(result, "orchestration_summary", None)
    manifest_payload = getattr(result, "orchestration_manifest", None)
    if not isinstance(summary_payload, dict):
        raise ValueError("Benchmark-pack batches must return research-campaign orchestration results.")
    if not isinstance(manifest_payload, dict):
        raise ValueError("Benchmark-pack batches must return orchestration manifests.")
    run_id = getattr(result, "orchestration_run_id", None)
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("Benchmark-pack batches must expose orchestration_run_id.")
    return summary_payload, manifest_payload, run_id


def _load_existing_checkpoint(path: Path, *, pack_run_id: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = _read_json(path)
    if str(payload.get("pack_run_id") or "") != pack_run_id:
        return None
    return payload


def _resolve_output_root(config: BenchmarkPackConfig, *, output_root: Path | None) -> Path:
    if output_root is not None:
        return output_root.resolve()
    if config.outputs.root is not None:
        return Path(config.outputs.root).resolve()
    return (Path("artifacts") / "benchmark_packs" / config.pack_id).resolve()


def _resolve_existing_path_string(value: str, *, config_dir: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return path.as_posix()
    config_relative = (config_dir / path).resolve()
    if config_relative.exists():
        return config_relative.as_posix()
    return path.resolve().as_posix()


def _write_batch_plan_csv(path: Path, batches: Sequence[BenchmarkBatchPlan]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "batch_id",
        "batch_index",
        "input_fingerprint",
        "scenario_count",
        "scenario_ids",
        "scenario_fingerprints",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for batch in batches:
            writer.writerow(
                {
                    "batch_id": batch.batch_id,
                    "batch_index": batch.batch_index,
                    "input_fingerprint": batch.input_fingerprint,
                    "scenario_count": len(batch.scenario_ids),
                    "scenario_ids": "|".join(batch.scenario_ids),
                    "scenario_fingerprints": "|".join(batch.scenario_fingerprints),
                }
            )


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8", newline="\n")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class _features_root_environment:
    def __init__(self) -> None:
        self.previous = os.environ.get("FEATURES_ROOT")

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.previous is None:
            os.environ.pop("FEATURES_ROOT", None)
        else:
            os.environ["FEATURES_ROOT"] = self.previous


class _working_directory:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.previous = Path.cwd()

    def __enter__(self) -> None:
        os.chdir(self.path)
        return None

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        os.chdir(self.previous)


__all__ = ["BenchmarkPackRunResult", "run_benchmark_pack"]
