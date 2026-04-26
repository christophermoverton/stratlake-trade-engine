from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from src.config.regime_benchmark_pack import (
    RegimeBenchmarkPackConfig,
    RegimeBenchmarkVariantConfig,
)
from src.data.load_features import FeaturePaths, load_features
from src.research.metrics import max_drawdown, total_return
from src.research.regimes import (
    REGIME_AUDIT_COLUMNS,
    RegimeTransitionConfig,
    analyze_portfolio_regime_transitions,
    apply_regime_calibration,
    apply_regime_policy,
    classify_market_regimes,
    classify_regime_shifts_with_gmm,
    detect_regime_transitions,
    resolve_regime_policy_config,
    write_regime_artifacts,
    write_regime_calibration_artifacts,
    write_regime_gmm_artifacts,
    write_regime_policy_artifacts,
    write_regime_transition_artifacts,
)
from src.research.regimes.gmm_classifier import resolve_regime_gmm_classifier_config
from src.research.registry import canonicalize_value


BENCHMARK_MATRIX_COLUMNS = (
    "benchmark_run_id",
    "variant_name",
    "variant_type",
    "regime_source",
    "calibration_profile",
    "classifier_model",
    "policy_name",
    "is_static_baseline",
    "observation_count",
    "regime_count",
    "avg_regime_duration",
    "median_regime_duration",
    "min_regime_duration",
    "max_regime_duration",
    "dominant_regime",
    "dominant_regime_share",
    "transition_count",
    "transition_rate",
    "mean_confidence",
    "median_confidence",
    "min_confidence",
    "mean_entropy",
    "max_entropy",
    "high_entropy_observation_count",
    "low_confidence_observation_count",
    "taxonomy_ml_disagreement_rate",
    "policy_turnover",
    "policy_state_change_count",
    "adaptive_vs_static_return_delta",
    "adaptive_vs_static_sharpe_delta",
    "adaptive_vs_static_max_drawdown_delta",
    "source_artifact_path",
    "warnings",
)
MODEL_COMPARISON_COLUMNS = (
    "variant_name",
    "classifier_model",
    "observation_count",
    "mean_confidence",
    "median_confidence",
    "min_confidence",
    "mean_entropy",
    "max_entropy",
    "high_entropy_observation_count",
    "low_confidence_observation_count",
    "taxonomy_ml_disagreement_rate",
)
CALIBRATION_COMPARISON_COLUMNS = (
    "variant_name",
    "calibration_profile",
    "regime_count",
    "avg_regime_duration",
    "median_regime_duration",
    "transition_count",
    "transition_rate",
    "low_confidence_observation_count",
    "warnings",
)
POLICY_COMPARISON_COLUMNS = (
    "variant_name",
    "policy_name",
    "policy_turnover",
    "policy_state_change_count",
    "adaptive_vs_static_return_delta",
    "adaptive_vs_static_sharpe_delta",
    "adaptive_vs_static_max_drawdown_delta",
    "adaptive_allocation_change_count",
    "high_volatility_regime_drawdown",
    "transition_window_performance_delta",
    "warnings",
)
STATIC_VARIANT_SUMMARY_FILENAME = "static_variant_summary.json"


@dataclass(frozen=True)
class RegimeBenchmarkPackRunResult:
    benchmark_name: str
    benchmark_run_id: str
    variant_count: int
    output_root: Path
    config_path: Path
    benchmark_matrix_csv_path: Path
    benchmark_matrix_json_path: Path
    model_comparison_csv_path: Path
    calibration_comparison_csv_path: Path
    policy_comparison_csv_path: Path
    stability_summary_path: Path
    transition_summary_path: Path
    summary_path: Path
    manifest_path: Path
    input_inventory_path: Path
    provenance_path: Path
    summary: dict[str, Any]
    manifest: dict[str, Any]


@dataclass(frozen=True)
class _SharedInputs:
    market_data: pd.DataFrame
    baseline_frame: pd.DataFrame
    classification_labels: pd.DataFrame
    classification_metadata: dict[str, Any]
    labels_with_confidence: pd.DataFrame
    gmm_result: Any | None
    input_inventory: dict[str, Any]


@dataclass(frozen=True)
class _VariantArtifacts:
    row: dict[str, Any]
    stability_summary: dict[str, Any] | None
    transition_summary: dict[str, Any] | None
    policy_summary: dict[str, Any] | None
    source_artifact_path: str
    generated_files: tuple[str, ...]
    warnings: tuple[str, ...]


def run_regime_benchmark_pack(
    config: RegimeBenchmarkPackConfig,
    *,
    output_root: Path | None = None,
) -> RegimeBenchmarkPackRunResult:
    shared = _build_shared_inputs(config)
    benchmark_run_id = _build_benchmark_run_id(config, shared.input_inventory)
    resolved_output_root = _resolve_output_root(
        config,
        benchmark_run_id=benchmark_run_id,
        output_root=output_root,
    )
    resolved_output_root.mkdir(parents=True, exist_ok=True)

    config_path = resolved_output_root / "config.json"
    benchmark_matrix_csv_path = resolved_output_root / "benchmark_matrix.csv"
    benchmark_matrix_json_path = resolved_output_root / "benchmark_matrix.json"
    model_comparison_csv_path = resolved_output_root / "model_comparison.csv"
    calibration_comparison_csv_path = resolved_output_root / "calibration_comparison.csv"
    policy_comparison_csv_path = resolved_output_root / "policy_comparison.csv"
    stability_summary_path = resolved_output_root / "stability_summary.json"
    transition_summary_path = resolved_output_root / "transition_summary.json"
    summary_path = resolved_output_root / "benchmark_summary.json"
    manifest_path = resolved_output_root / "manifest.json"
    input_inventory_path = resolved_output_root / "input_inventory.json"
    provenance_path = resolved_output_root / "artifact_provenance.json"

    _write_json(config_path, config.to_dict())
    _write_json(input_inventory_path, shared.input_inventory)

    variant_artifacts: list[_VariantArtifacts] = []
    for variant_index, variant in enumerate(config.variants):
        variant_artifacts.append(
            _run_variant(
                config=config,
                shared=shared,
                benchmark_run_id=benchmark_run_id,
                variant=variant,
                variant_index=variant_index,
                output_root=resolved_output_root,
            )
        )

    matrix_rows = [artifact.row for artifact in variant_artifacts]
    _write_csv(benchmark_matrix_csv_path, matrix_rows, fieldnames=BENCHMARK_MATRIX_COLUMNS)
    _write_json(
        benchmark_matrix_json_path,
        {
            "benchmark_name": config.benchmark_name,
            "benchmark_run_id": benchmark_run_id,
            "row_count": len(matrix_rows),
            "columns": list(BENCHMARK_MATRIX_COLUMNS),
            "rows": matrix_rows,
        },
    )

    model_rows = [
        {column: row.get(column) for column in MODEL_COMPARISON_COLUMNS}
        for row in matrix_rows
        if row.get("classifier_model") is not None
    ]
    calibration_rows = [
        {column: row.get(column) for column in CALIBRATION_COMPARISON_COLUMNS}
        for row in matrix_rows
        if row.get("calibration_profile") is not None
    ]
    policy_rows = []
    for artifact in variant_artifacts:
        row = artifact.row
        if row.get("policy_name") is None:
            continue
        policy_row = {column: row.get(column) for column in POLICY_COMPARISON_COLUMNS}
        if artifact.policy_summary is not None:
            policy_row["adaptive_allocation_change_count"] = artifact.policy_summary.get(
                "adaptive_allocation_change_count"
            )
            policy_row["high_volatility_regime_drawdown"] = artifact.policy_summary.get(
                "high_volatility_regime_drawdown"
            )
            policy_row["transition_window_performance_delta"] = artifact.policy_summary.get(
                "transition_window_performance_delta"
            )
        policy_rows.append(canonicalize_value(policy_row))

    _write_csv(model_comparison_csv_path, model_rows, fieldnames=MODEL_COMPARISON_COLUMNS)
    _write_csv(
        calibration_comparison_csv_path,
        calibration_rows,
        fieldnames=CALIBRATION_COMPARISON_COLUMNS,
    )
    _write_csv(policy_comparison_csv_path, policy_rows, fieldnames=POLICY_COMPARISON_COLUMNS)

    stability_summary = {
        "benchmark_name": config.benchmark_name,
        "benchmark_run_id": benchmark_run_id,
        "variants": [
            artifact.stability_summary
            for artifact in variant_artifacts
            if artifact.stability_summary is not None
        ],
    }
    transition_summary = {
        "benchmark_name": config.benchmark_name,
        "benchmark_run_id": benchmark_run_id,
        "variants": [
            artifact.transition_summary
            for artifact in variant_artifacts
            if artifact.transition_summary is not None
        ],
    }
    _write_json(stability_summary_path, stability_summary)
    _write_json(transition_summary_path, transition_summary)

    provenance = {
        "benchmark_name": config.benchmark_name,
        "benchmark_run_id": benchmark_run_id,
        "variant_provenance": [
            {
                "variant_name": artifact.row["variant_name"],
                "source_artifact_path": artifact.source_artifact_path,
                "generated_files": list(artifact.generated_files),
                "warnings": list(artifact.warnings),
            }
            for artifact in variant_artifacts
        ],
    }
    _write_json(provenance_path, provenance)

    summary = canonicalize_value(
        {
            "benchmark_name": config.benchmark_name,
            "benchmark_run_id": benchmark_run_id,
            "variant_count": len(variant_artifacts),
            "variant_names": [artifact.row["variant_name"] for artifact in variant_artifacts],
            "output_root": ".",
            "matrix_row_count": len(matrix_rows),
            "source_artifact_paths": [artifact.source_artifact_path for artifact in variant_artifacts],
            "warning_count": int(sum(len(artifact.warnings) for artifact in variant_artifacts)),
            "paths": {
                "benchmark_matrix_csv": "benchmark_matrix.csv",
                "benchmark_matrix_json": "benchmark_matrix.json",
                "model_comparison_csv": "model_comparison.csv",
                "calibration_comparison_csv": "calibration_comparison.csv",
                "policy_comparison_csv": "policy_comparison.csv",
                "stability_summary_json": "stability_summary.json",
                "transition_summary_json": "transition_summary.json",
                "config_json": "config.json",
                "manifest_json": "manifest.json",
                "input_inventory_json": "input_inventory.json",
                "artifact_provenance_json": "artifact_provenance.json",
            },
        }
    )
    _write_json(summary_path, summary)

    manifest = _build_manifest(
        output_root=resolved_output_root,
        benchmark_name=config.benchmark_name,
        benchmark_run_id=benchmark_run_id,
        variant_artifacts=variant_artifacts,
    )
    _write_json(manifest_path, manifest)

    return RegimeBenchmarkPackRunResult(
        benchmark_name=config.benchmark_name,
        benchmark_run_id=benchmark_run_id,
        variant_count=len(variant_artifacts),
        output_root=resolved_output_root,
        config_path=config_path,
        benchmark_matrix_csv_path=benchmark_matrix_csv_path,
        benchmark_matrix_json_path=benchmark_matrix_json_path,
        model_comparison_csv_path=model_comparison_csv_path,
        calibration_comparison_csv_path=calibration_comparison_csv_path,
        policy_comparison_csv_path=policy_comparison_csv_path,
        stability_summary_path=stability_summary_path,
        transition_summary_path=transition_summary_path,
        summary_path=summary_path,
        manifest_path=manifest_path,
        input_inventory_path=input_inventory_path,
        provenance_path=provenance_path,
        summary=summary,
        manifest=manifest,
    )


def _build_shared_inputs(config: RegimeBenchmarkPackConfig) -> _SharedInputs:
    end_exclusive = _end_exclusive(config.end, timeframe=config.timeframe)
    market_data = load_features(
        config.dataset,
        start=config.start,
        end=end_exclusive,
        symbols=list(config.symbols) if config.symbols else None,
        paths=FeaturePaths(root=Path(config.features_root)),
        validate_integrity=False,
    )
    if market_data.empty:
        raise ValueError(
            "Regime benchmark-pack market input is empty for the requested dataset window."
        )
    required_columns = {"ts_utc", "symbol", "close"}
    missing = sorted(required_columns - set(market_data.columns))
    if missing:
        raise ValueError(
            f"Regime benchmark-pack market input is missing required columns: {missing}."
        )

    market = market_data.loc[:, ["ts_utc", "symbol", "close"]].copy(deep=True)
    market["ts_utc"] = pd.to_datetime(market["ts_utc"], utc=True, errors="coerce")
    market["symbol"] = market["symbol"].astype("string")
    market["close"] = pd.to_numeric(market["close"], errors="coerce").astype("float64")
    market = market.dropna(subset=["ts_utc", "symbol", "close"])
    market = market.sort_values(["ts_utc", "symbol"], kind="stable").reset_index(drop=True)
    if market.empty:
        raise ValueError("Regime benchmark-pack market input became empty after normalization.")

    baseline_frame = _build_baseline_frame(market)
    classification = classify_market_regimes(market, config=dict(config.classification))

    gmm_result = None
    labels_with_confidence = classification.labels.copy(deep=True)
    if _requires_gmm(config):
        gmm_input = classification.labels.dropna(subset=list(REGIME_AUDIT_COLUMNS)).reset_index(drop=True)
        resolved_gmm_config = resolve_regime_gmm_classifier_config(dict(config.gmm))
        if len(gmm_input) < resolved_gmm_config.min_observations:
            raise ValueError(
                "Regime benchmark-pack GMM inputs do not satisfy the configured minimum observation count. "
                f"Need at least {resolved_gmm_config.min_observations} observations after taxonomy warmup "
                f"(got {len(gmm_input)})."
            )
        gmm_result = classify_regime_shifts_with_gmm(
            gmm_input.loc[:, ["ts_utc", *list(REGIME_AUDIT_COLUMNS)]].copy(deep=True),
            config=resolved_gmm_config,
        )
        labels_with_confidence = classification.labels.merge(
            gmm_result.labels.loc[:, ["ts_utc", "gmm_confidence_score"]],
            on="ts_utc",
            how="left",
            sort=False,
        )

    input_inventory = canonicalize_value(
        {
            "dataset": {
                "dataset_name": config.dataset,
                "features_root": Path(config.features_root).as_posix(),
                "start": config.start,
                "end": config.end,
                "end_exclusive": end_exclusive,
                "row_count": int(len(market)),
                "timestamp_count": int(market["ts_utc"].nunique()),
                "symbol_count": int(market["symbol"].nunique()),
                "symbols": sorted(str(symbol) for symbol in market["symbol"].dropna().unique().tolist()),
                "columns": ["ts_utc", "symbol", "close"],
            },
            "classification": {
                "defined_row_count": int(classification.labels["is_defined"].astype("bool").sum()),
                "row_count": int(len(classification.labels)),
                "taxonomy_version": classification.taxonomy_version,
            },
            "gmm_enabled": bool(gmm_result is not None),
        }
    )
    return _SharedInputs(
        market_data=market,
        baseline_frame=baseline_frame,
        classification_labels=classification.labels,
        classification_metadata=classification.metadata,
        labels_with_confidence=labels_with_confidence,
        gmm_result=gmm_result,
        input_inventory=input_inventory,
    )


def _run_variant(
    *,
    config: RegimeBenchmarkPackConfig,
    shared: _SharedInputs,
    benchmark_run_id: str,
    variant: RegimeBenchmarkVariantConfig,
    variant_index: int,
    output_root: Path,
) -> _VariantArtifacts:
    variant_dir = output_root / "variants" / f"{variant_index:02d}_{variant.name}"
    variant_dir.mkdir(parents=True, exist_ok=True)

    variant_type = _variant_type(variant)
    warnings: list[str] = []
    stability_summary = None
    transition_summary = None
    policy_summary = None
    generated_files: list[str] = []

    row: dict[str, Any] = {
        "benchmark_run_id": benchmark_run_id,
        "variant_name": variant.name,
        "variant_type": variant_type,
        "regime_source": variant.regime_source,
        "calibration_profile": variant.calibration_profile,
        "classifier_model": None,
        "policy_name": None,
        "is_static_baseline": bool(variant.regime_source == "static"),
        "observation_count": int(len(shared.baseline_frame)),
        "regime_count": None,
        "avg_regime_duration": None,
        "median_regime_duration": None,
        "min_regime_duration": None,
        "max_regime_duration": None,
        "dominant_regime": None,
        "dominant_regime_share": None,
        "transition_count": None,
        "transition_rate": None,
        "mean_confidence": None,
        "median_confidence": None,
        "min_confidence": None,
        "mean_entropy": None,
        "max_entropy": None,
        "high_entropy_observation_count": None,
        "low_confidence_observation_count": None,
        "taxonomy_ml_disagreement_rate": None,
        "policy_turnover": None,
        "policy_state_change_count": None,
        "adaptive_vs_static_return_delta": None,
        "adaptive_vs_static_sharpe_delta": None,
        "adaptive_vs_static_max_drawdown_delta": None,
        "source_artifact_path": None,
        "warnings": "",
    }

    if variant.regime_source == "static":
        static_summary_path = variant_dir / STATIC_VARIANT_SUMMARY_FILENAME
        _write_json(static_summary_path, _static_variant_summary(shared.baseline_frame))
        source_artifact_path = _relative_path(static_summary_path, base_dir=output_root)
        generated_files.append(source_artifact_path)
    else:
        labels = shared.classification_labels
        if variant.regime_source in {"gmm", "policy"} and shared.gmm_result is not None:
            row["classifier_model"] = "gaussian_mixture_model"
            row.update(
                _gmm_confidence_summary(
                    shared.gmm_result.labels,
                    taxonomy_labels=shared.classification_labels,
                    high_entropy_threshold=config.metrics.high_entropy_threshold,
                )
            )
            if variant.regime_source == "gmm":
                warnings.append(
                    "Classifier variants preserve taxonomy labels for regime metrics and use GMM outputs as a confidence overlay."
                )

        if variant.regime_source == "policy":
            if shared.gmm_result is None:
                warnings.append(
                    "Policy variant is running without GMM confidence inputs because no GMM variant requested them."
                )
            labels = _labels_for_policy_variant(shared, variant)
            labels_dir = variant_dir / "labels"
            write_regime_artifacts(labels_dir, labels, metadata={"benchmark_variant": variant.name})
            generated_files.extend(
                [
                    _relative_path(labels_dir / "manifest.json", base_dir=output_root),
                    _relative_path(labels_dir / "regime_labels.csv", base_dir=output_root),
                    _relative_path(labels_dir / "regime_summary.json", base_dir=output_root),
                ]
            )
            policy_result = _run_policy_variant(
                shared=shared,
                variant=variant,
                labels=labels,
                config=config,
            )
            policy_dir = variant_dir / "policy"
            write_regime_policy_artifacts(
                policy_dir,
                policy_result,
                run_id=f"{benchmark_run_id}_{variant.name}",
                source_regime_artifact_references={"labels": "../labels/manifest.json"},
                calibration_profile=variant.calibration_profile,
                confidence_artifact_references=(
                    {"gmm_manifest": "../gmm/regime_gmm_manifest.json"}
                    if shared.gmm_result is not None
                    else {}
                ),
                extra_metadata={"benchmark_variant": variant.name},
            )
            source_artifact_path = _relative_path(
                policy_dir / "adaptive_policy_manifest.json",
                base_dir=output_root,
            )
            generated_files.extend(
                [
                    _relative_path(policy_dir / "adaptive_policy_manifest.json", base_dir=output_root),
                    _relative_path(policy_dir / "regime_policy_summary.json", base_dir=output_root),
                    _relative_path(policy_dir / "adaptive_vs_static_comparison.csv", base_dir=output_root),
                ]
            )
            row["policy_name"] = Path(str(variant.policy_config)).stem if variant.policy_config else "policy"
            row.update(_policy_matrix_summary(policy_result))
            policy_summary = _policy_variant_summary(policy_result, labels=labels, shared=shared, config=config)
        elif variant.calibration_profile is not None:
            calibration_result = _apply_calibration_for_variant(shared, variant)
            labels = calibration_result.labels
            labels_dir = variant_dir / "labels"
            calibration_dir = variant_dir / "calibration"
            write_regime_artifacts(labels_dir, labels, metadata={"benchmark_variant": variant.name})
            write_regime_calibration_artifacts(
                calibration_dir,
                calibration_result,
                run_id=f"{benchmark_run_id}_{variant.name}",
                source_regime_artifact_references={"classification_manifest": "../labels/manifest.json"},
                taxonomy_metadata={"taxonomy_version": calibration_result.taxonomy_version},
                extra_metadata={"benchmark_variant": variant.name},
            )
            source_artifact_path = _relative_path(
                calibration_dir / "regime_calibration.json",
                base_dir=output_root,
            )
            generated_files.extend(
                [
                    _relative_path(labels_dir / "manifest.json", base_dir=output_root),
                    _relative_path(calibration_dir / "regime_calibration.json", base_dir=output_root),
                    _relative_path(calibration_dir / "regime_calibration_summary.json", base_dir=output_root),
                    _relative_path(calibration_dir / "regime_stability_metrics.json", base_dir=output_root),
                ]
            )
            if row["low_confidence_observation_count"] is None:
                row["low_confidence_observation_count"] = int(
                    calibration_result.audit["is_low_confidence"].astype("bool").sum()
                )
        elif variant.regime_source == "taxonomy":
            write_regime_artifacts(
                variant_dir,
                labels,
                metadata={"benchmark_variant": variant.name, **shared.classification_metadata},
            )
            source_artifact_path = _relative_path(variant_dir / "manifest.json", base_dir=output_root)
            generated_files.extend(
                [
                    _relative_path(variant_dir / "manifest.json", base_dir=output_root),
                    _relative_path(variant_dir / "regime_labels.csv", base_dir=output_root),
                    _relative_path(variant_dir / "regime_summary.json", base_dir=output_root),
                ]
            )
        elif variant.regime_source == "gmm":
            if shared.gmm_result is None:
                raise ValueError(f"Variant {variant.name!r} requires GMM outputs but none were generated.")
            gmm_dir = variant_dir / "gmm"
            write_regime_gmm_artifacts(
                gmm_dir,
                shared.gmm_result,
                run_id=f"{benchmark_run_id}_{variant.name}",
                source_regime_artifact_references={"taxonomy_summary": "../regime_summary.json"},
                extra_metadata={"benchmark_variant": variant.name},
            )
            source_artifact_path = _relative_path(
                gmm_dir / "regime_gmm_manifest.json",
                base_dir=output_root,
            )
            generated_files.extend(
                [
                    _relative_path(gmm_dir / "regime_gmm_manifest.json", base_dir=output_root),
                    _relative_path(gmm_dir / "regime_gmm_summary.json", base_dir=output_root),
                    _relative_path(gmm_dir / "regime_gmm_labels.csv", base_dir=output_root),
                ]
            )
        else:
            raise ValueError(f"Unsupported variant regime_source: {variant.regime_source!r}.")

        labels_for_summary = labels
        if config.metrics.include_stability_metrics:
            stability_summary = _stability_summary(labels_for_summary, variant_name=variant.name)
            row.update(
                {
                    "regime_count": stability_summary["regime_count"],
                    "avg_regime_duration": stability_summary["avg_regime_duration"],
                    "median_regime_duration": stability_summary["median_regime_duration"],
                    "min_regime_duration": stability_summary["min_regime_duration"],
                    "max_regime_duration": stability_summary["max_regime_duration"],
                    "dominant_regime": stability_summary["dominant_regime"],
                    "dominant_regime_share": stability_summary["dominant_regime_share"],
                }
            )
        if config.metrics.include_transition_metrics:
            transition_result = analyze_portfolio_regime_transitions(
                shared.baseline_frame.loc[:, ["ts_utc", "portfolio_return"]],
                labels_for_summary,
                return_column="portfolio_return",
                config=RegimeTransitionConfig(**dict(config.transition)),
            )
            transition_dir = variant_dir / "transition"
            write_regime_transition_artifacts(
                transition_dir,
                transition_result,
                run_id=f"{benchmark_run_id}_{variant.name}",
                extra_metadata={"benchmark_variant": variant.name},
            )
            transition_summary = _transition_summary(
                labels_for_summary,
                transition_result=transition_result,
                variant_name=variant.name,
            )
            row.update(
                {
                    "transition_count": transition_summary["transition_count"],
                    "transition_rate": transition_summary["transition_rate"],
                }
            )
            generated_files.extend(
                [
                    _relative_path(transition_dir / "regime_transition_summary.json", base_dir=output_root),
                    _relative_path(transition_dir / "regime_transition_events.csv", base_dir=output_root),
                    _relative_path(transition_dir / "regime_transition_manifest.json", base_dir=output_root),
                ]
            )

    row["source_artifact_path"] = source_artifact_path
    row["warnings"] = " | ".join(sorted(dict.fromkeys(warnings)))
    return _VariantArtifacts(
        row=canonicalize_value(_normalize_row(row)),
        stability_summary=stability_summary,
        transition_summary=transition_summary,
        policy_summary=policy_summary,
        source_artifact_path=source_artifact_path,
        generated_files=tuple(sorted(dict.fromkeys(generated_files))),
        warnings=tuple(sorted(dict.fromkeys(warnings))),
    )


def _requires_gmm(config: RegimeBenchmarkPackConfig) -> bool:
    return any(variant.regime_source in {"gmm", "policy"} for variant in config.variants)


def _apply_calibration_for_variant(shared: _SharedInputs, variant: RegimeBenchmarkVariantConfig):
    if variant.regime_source == "gmm":
        if shared.gmm_result is None:
            raise ValueError(
                f"Variant {variant.name!r} requires GMM confidence for calibration but no GMM outputs exist."
            )
        return apply_regime_calibration(
            shared.labels_with_confidence,
            profile=variant.calibration_profile,
            confidence_column="gmm_confidence_score",
            metadata={"benchmark_variant": variant.name},
        )
    return apply_regime_calibration(
        shared.classification_labels,
        profile=variant.calibration_profile,
        metadata={"benchmark_variant": variant.name},
    )


def _labels_for_policy_variant(
    shared: _SharedInputs,
    variant: RegimeBenchmarkVariantConfig,
) -> pd.DataFrame:
    if variant.calibration_profile is None:
        return shared.classification_labels.copy(deep=True)
    return _apply_calibration_for_variant(shared, variant).labels


def _run_policy_variant(
    *,
    shared: _SharedInputs,
    variant: RegimeBenchmarkVariantConfig,
    labels: pd.DataFrame,
    config: RegimeBenchmarkPackConfig,
):
    if variant.policy_config is None:
        raise ValueError(f"Variant {variant.name!r} requires a policy_config path.")
    payload = yaml.safe_load(Path(variant.policy_config).read_text(encoding="utf-8")) or {}
    policy_config = resolve_regime_policy_config(payload)
    confidence_frame = None
    if shared.gmm_result is not None:
        confidence_frame = shared.gmm_result.labels.loc[
            :,
            [
                column
                for column in (
                    "ts_utc",
                    "confidence_score",
                    "confidence_bucket",
                    "fallback_flag",
                    "fallback_reason",
                    "predicted_label",
                )
                if column in shared.gmm_result.labels.columns
            ],
        ].copy(deep=True)
    return apply_regime_policy(
        labels,
        config=policy_config,
        confidence_frame=confidence_frame,
        baseline_frame=shared.baseline_frame.copy(deep=True),
        surface=config.policy_surface,
        return_column=config.policy_return_column,
        weight_column=config.policy_weight_column,
        calibration_profile=variant.calibration_profile,
        metadata={"benchmark_variant": variant.name},
    )


def _variant_type(variant: RegimeBenchmarkVariantConfig) -> str:
    if variant.regime_source == "static":
        return "static_baseline"
    if variant.regime_source == "taxonomy" and variant.calibration_profile is None:
        return "taxonomy_only"
    if variant.regime_source == "taxonomy" and variant.calibration_profile is not None:
        return "calibrated_taxonomy"
    if variant.regime_source == "gmm" and variant.calibration_profile is None:
        return "gmm_classifier"
    if variant.regime_source == "gmm" and variant.calibration_profile is not None:
        return "gmm_calibrated_overlay"
    if variant.regime_source == "policy":
        return "policy_optimized"
    return variant.regime_source


def _build_baseline_frame(market_data: pd.DataFrame) -> pd.DataFrame:
    ordered = market_data.sort_values(["symbol", "ts_utc"], kind="stable").reset_index(drop=True)
    ordered["market_return_symbol"] = ordered.groupby("symbol", sort=False)["close"].pct_change()
    baseline = (
        ordered.groupby("ts_utc", sort=True)["market_return_symbol"]
        .mean()
        .reset_index(name="portfolio_return")
    )
    baseline["portfolio_return"] = pd.to_numeric(
        baseline["portfolio_return"], errors="coerce"
    ).fillna(0.0)
    baseline["weight"] = 1.0
    baseline["market_return"] = baseline["portfolio_return"]
    return baseline.loc[:, ["ts_utc", "portfolio_return", "market_return", "weight"]].copy(deep=True)


def _static_variant_summary(baseline_frame: pd.DataFrame) -> dict[str, Any]:
    returns = pd.to_numeric(baseline_frame["portfolio_return"], errors="coerce").dropna()
    return canonicalize_value(
        {
            "artifact_type": "regime_benchmark_static_variant",
            "observation_count": int(len(baseline_frame)),
            "baseline_total_return": _finite_float(total_return(returns)),
            "baseline_max_drawdown": _finite_float(max_drawdown(returns)),
        }
    )


def _stability_summary(labels: pd.DataFrame, *, variant_name: str) -> dict[str, Any]:
    runs = _label_runs(labels)
    durations = [run["duration"] for run in runs]
    counts = labels["regime_label"].astype("string").value_counts(sort=False).sort_index()
    dominant_regime = None if counts.empty else str(counts.sort_values(ascending=False, kind="stable").index[0])
    dominant_count = 0 if dominant_regime is None else int(counts.loc[dominant_regime])
    return canonicalize_value(
        {
            "variant_name": variant_name,
            "observation_count": int(len(labels)),
            "regime_count": int(len(runs)),
            "distinct_regime_label_count": int(labels["regime_label"].astype("string").nunique()),
            "avg_regime_duration": _finite_float(pd.Series(durations, dtype="float64").mean()) if durations else None,
            "median_regime_duration": _finite_float(pd.Series(durations, dtype="float64").median()) if durations else None,
            "min_regime_duration": int(min(durations)) if durations else None,
            "max_regime_duration": int(max(durations)) if durations else None,
            "dominant_regime": dominant_regime,
            "dominant_regime_share": (
                _finite_float(dominant_count / max(len(labels), 1)) if dominant_regime is not None else None
            ),
        }
    )


def _transition_summary(
    labels: pd.DataFrame,
    *,
    transition_result: Any,
    variant_name: str,
) -> dict[str, Any]:
    events = detect_regime_transitions(labels, dimensions="composite")
    transition_pairs = (
        events["transition_label"].astype("string").value_counts(sort=False).sort_index()
        if not events.empty
        else pd.Series(dtype="int64")
    )
    most_common_transition = None
    most_common_transition_count = 0
    if not transition_pairs.empty:
        sorted_pairs = transition_pairs.sort_values(ascending=False, kind="stable")
        most_common_transition = str(sorted_pairs.index[0])
        most_common_transition_count = int(sorted_pairs.iloc[0])
    transition_count = int(len(events))
    observation_count = int(len(labels))
    event_summaries = transition_result.event_summaries
    return canonicalize_value(
        {
            "variant_name": variant_name,
            "transition_count": transition_count,
            "transition_rate": _finite_float(transition_count / max(observation_count - 1, 1)),
            "most_common_transition": most_common_transition,
            "most_common_transition_count": most_common_transition_count,
            "rare_transition_count": int((transition_pairs == 1).sum()) if not transition_pairs.empty else 0,
            "transition_concentration": (
                _finite_float(most_common_transition_count / transition_count) if transition_count else None
            ),
            "transition_instability_score": (
                _finite_float(transition_count / max(observation_count, 1)) if transition_count else 0.0
            ),
            "transition_matrix_summary": [
                {"transition_label": str(index), "count": int(value)}
                for index, value in transition_pairs.items()
            ],
            "coverage_counts": (
                {}
                if event_summaries.empty
                else {
                    str(key): int(value)
                    for key, value in event_summaries["coverage_status"].value_counts(sort=False).sort_index().items()
                }
            ),
        }
    )


def _gmm_confidence_summary(
    gmm_labels: pd.DataFrame,
    *,
    taxonomy_labels: pd.DataFrame,
    high_entropy_threshold: float,
) -> dict[str, Any]:
    merged = taxonomy_labels.loc[:, ["ts_utc", "regime_label"]].merge(
        gmm_labels,
        on="ts_utc",
        how="inner",
        sort=False,
    )
    confidence = pd.to_numeric(merged["gmm_confidence_score"], errors="coerce").dropna()
    entropy = pd.to_numeric(merged["gmm_posterior_entropy"], errors="coerce").dropna()
    taxonomy_changed = merged["regime_label"].astype("string").ne(
        merged["regime_label"].astype("string").shift(1)
    )
    cluster_changed = merged["gmm_cluster_label"].astype("string").ne(
        merged["gmm_previous_cluster_label"].astype("string")
    )
    valid_change_mask = merged["gmm_previous_cluster_label"].notna()
    disagreement_rate = None
    if bool(valid_change_mask.any()):
        disagreement_rate = _finite_float(
            (taxonomy_changed.loc[valid_change_mask] != cluster_changed.loc[valid_change_mask])
            .astype("float64")
            .mean()
        )
    return {
        "mean_confidence": _finite_float(confidence.mean()) if not confidence.empty else None,
        "median_confidence": _finite_float(confidence.median()) if not confidence.empty else None,
        "min_confidence": _finite_float(confidence.min()) if not confidence.empty else None,
        "mean_entropy": _finite_float(entropy.mean()) if not entropy.empty else None,
        "max_entropy": _finite_float(entropy.max()) if not entropy.empty else None,
        "high_entropy_observation_count": int(entropy.gt(high_entropy_threshold).sum())
        if not entropy.empty
        else None,
        "low_confidence_observation_count": int(merged["gmm_is_low_confidence"].astype("bool").sum()),
        "taxonomy_ml_disagreement_rate": disagreement_rate,
    }


def _policy_matrix_summary(result: Any) -> dict[str, Any]:
    comparison = result.comparison.iloc[0].to_dict() if not result.comparison.empty else {}
    decisions = result.decisions.copy(deep=True)
    turnover = _policy_turnover(decisions)
    return {
        "policy_turnover": 0.0 if turnover is None else turnover,
        "policy_state_change_count": _policy_state_change_count(decisions),
        "adaptive_vs_static_return_delta": _delta(
            comparison.get("adaptive_total_return"),
            comparison.get("baseline_total_return"),
        ),
        "adaptive_vs_static_sharpe_delta": _delta(
            comparison.get("adaptive_sharpe"),
            comparison.get("baseline_sharpe"),
        ),
        "adaptive_vs_static_max_drawdown_delta": _delta(
            comparison.get("adaptive_max_drawdown"),
            comparison.get("baseline_max_drawdown"),
        ),
    }


def _policy_variant_summary(
    result: Any,
    *,
    labels: pd.DataFrame,
    shared: _SharedInputs,
    config: RegimeBenchmarkPackConfig,
) -> dict[str, Any]:
    decisions = result.decisions.copy(deep=True)
    decisions["ts_utc"] = pd.to_datetime(decisions["ts_utc"], utc=True, errors="coerce")
    adaptive_weight = pd.to_numeric(decisions.get("adaptive_weight"), errors="coerce")
    allocation_scale = pd.to_numeric(decisions.get("allocation_scale"), errors="coerce")
    transition_result = analyze_portfolio_regime_transitions(
        shared.baseline_frame.loc[:, ["ts_utc", "portfolio_return"]],
        labels,
        return_column="portfolio_return",
        config=RegimeTransitionConfig(**dict(config.transition)),
    )
    windows = transition_result.windows.loc[
        :,
        [column for column in ("transition_event_id", "ts_utc") if column in transition_result.windows.columns],
    ].copy(deep=True)
    merged_windows = windows.merge(
        decisions.loc[:, ["ts_utc", "adaptive_return"]],
        on="ts_utc",
        how="left",
        sort=False,
    ).merge(
        shared.baseline_frame.loc[:, ["ts_utc", "portfolio_return"]],
        on="ts_utc",
        how="left",
        sort=False,
    )
    deltas: list[float] = []
    for _, group in merged_windows.groupby("transition_event_id", sort=False):
        adaptive = pd.to_numeric(group["adaptive_return"], errors="coerce").dropna().astype("float64")
        baseline = pd.to_numeric(group["portfolio_return"], errors="coerce").dropna().astype("float64")
        if adaptive.empty or baseline.empty:
            continue
        deltas.append(float(total_return(adaptive) - total_return(baseline)))

    high_volatility_returns = decisions.merge(
        labels.loc[:, ["ts_utc", "volatility_state"]],
        on="ts_utc",
        how="left",
        sort=False,
    )
    high_volatility_adaptive = pd.to_numeric(
        high_volatility_returns.loc[
            high_volatility_returns["volatility_state"].eq("high_volatility"),
            "adaptive_return",
        ],
        errors="coerce",
    ).dropna()
    adaptive_change_source = adaptive_weight if adaptive_weight.notna().any() else allocation_scale
    return canonicalize_value(
        {
            "adaptive_allocation_change_count": int(adaptive_change_source.diff().fillna(0.0).ne(0.0).sum())
            if adaptive_change_source.notna().any()
            else None,
            "high_volatility_regime_drawdown": (
                _finite_float(max_drawdown(high_volatility_adaptive))
                if not high_volatility_adaptive.empty
                else None
            ),
            "transition_window_performance_delta": (
                _finite_float(pd.Series(deltas, dtype="float64").mean()) if deltas else None
            ),
        }
    )


def _policy_state_change_count(decisions: pd.DataFrame) -> int | None:
    if decisions.empty:
        return 0
    key = decisions["matched_policy_key"].astype("string").fillna("default")
    fallback = decisions["fallback_policy_applied"].astype("string").fillna("none")
    state = key + "|" + fallback
    return int(state.ne(state.shift(1)).iloc[1:].sum()) if len(state) > 1 else 0


def _policy_turnover(decisions: pd.DataFrame) -> float | None:
    adaptive_weight = pd.to_numeric(decisions.get("adaptive_weight"), errors="coerce")
    if adaptive_weight.notna().any():
        return _finite_float(adaptive_weight.diff().abs().fillna(0.0).sum())
    allocation_scale = pd.to_numeric(decisions.get("allocation_scale"), errors="coerce")
    if allocation_scale.notna().any():
        return _finite_float(allocation_scale.diff().abs().fillna(0.0).sum())
    return None


def _label_runs(labels: pd.DataFrame) -> list[dict[str, Any]]:
    series = labels["regime_label"].astype("string").tolist()
    if not series:
        return []
    runs: list[dict[str, Any]] = []
    start = 0
    for position in range(1, len(series) + 1):
        if position < len(series) and series[position] == series[start]:
            continue
        runs.append({"label": series[start], "duration": position - start})
        start = position
    return runs


def _build_benchmark_run_id(
    config: RegimeBenchmarkPackConfig,
    input_inventory: Mapping[str, Any],
) -> str:
    payload = canonicalize_value({"config": config.to_dict(), "input_inventory": dict(input_inventory)})
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    return f"{config.benchmark_name}_{digest}"


def _resolve_output_root(
    config: RegimeBenchmarkPackConfig,
    *,
    benchmark_run_id: str,
    output_root: Path | None,
) -> Path:
    root = output_root if output_root is not None else Path(config.output_root)
    return root.resolve() / benchmark_run_id


def _build_manifest(
    *,
    output_root: Path,
    benchmark_name: str,
    benchmark_run_id: str,
    variant_artifacts: list[_VariantArtifacts],
) -> dict[str, Any]:
    file_inventory: dict[str, Any] = {}
    generated_files: list[str] = []
    for path in sorted(output_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(output_root).as_posix()
        generated_files.append(relative)
        if relative == "manifest.json":
            file_inventory[relative] = {"path": relative}
            continue
        file_inventory[relative] = {
            "path": relative,
            "sha256": _sha256_file(path),
            "rows": _row_count(path),
        }
    return canonicalize_value(
        {
            "artifact_type": "regime_benchmark_pack",
            "schema_version": 1,
            "benchmark_name": benchmark_name,
            "benchmark_run_id": benchmark_run_id,
            "variant_count": len(variant_artifacts),
            "generated_files": generated_files,
            "row_counts": {
                "benchmark_matrix": len(variant_artifacts),
                "model_comparison": int(
                    sum(1 for artifact in variant_artifacts if artifact.row.get("classifier_model") is not None)
                ),
                "calibration_comparison": int(
                    sum(1 for artifact in variant_artifacts if artifact.row.get("calibration_profile") is not None)
                ),
                "policy_comparison": int(
                    sum(1 for artifact in variant_artifacts if artifact.row.get("policy_name") is not None)
                ),
            },
            "source_artifact_paths": [artifact.source_artifact_path for artifact in variant_artifacts],
            "variant_warnings": {
                artifact.row["variant_name"]: list(artifact.warnings)
                for artifact in variant_artifacts
            },
            "deterministic_run_metadata": {
                "path_style": "relative",
                "stable_variant_order": [artifact.row["variant_name"] for artifact in variant_artifacts],
            },
            "file_inventory": file_inventory,
        }
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(canonicalize_value(payload), indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )


def _end_exclusive(end: str, *, timeframe: str) -> str:
    timestamp = pd.Timestamp(end)
    if timeframe == "1D":
        return (timestamp + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return timestamp.isoformat()


def _relative_path(path: Path, *, base_dir: Path) -> str:
    try:
        return path.relative_to(base_dir).as_posix()
    except ValueError:
        return path.as_posix()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_count(path: Path) -> int | None:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            return 0
        count = text.count("\n")
        if text.endswith("\n"):
            count -= 1
        return max(count, 0)
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            rows = payload.get("rows")
            if isinstance(rows, list):
                return int(len(rows))
        return None
    return None


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            normalized[key] = None
        else:
            normalized[key] = value
    return normalized


def _finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def _delta(left: Any, right: Any) -> float | None:
    left_value = _finite_float(left)
    right_value = _finite_float(right)
    if left_value is None or right_value is None:
        return None
    return _finite_float(left_value - right_value)


__all__ = ["RegimeBenchmarkPackRunResult", "run_regime_benchmark_pack"]
