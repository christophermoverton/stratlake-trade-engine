from __future__ import annotations

from dataclasses import dataclass, field
import csv
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import pandas as pd

from src.research.registry import canonicalize_value

from .models import SCHEMA_VERSION, RobustnessFinding, sanitize_portable_value

PURGED_SPLIT_PLAN_FILENAME = "purged_split_plan.json"
PURGED_SPLIT_SUMMARY_FILENAME = "purged_split_summary.csv"
LEAKAGE_VALIDATION_FILENAME = "leakage_validation.json"

PURGED_SPLIT_SUMMARY_COLUMNS: list[str] = [
    "split_id",
    "train_start",
    "train_end",
    "validation_start",
    "validation_end",
    "purge_window",
    "embargo_window",
    "n_train_observations",
    "n_validation_observations",
    "n_purged_observations",
    "n_embargoed_observations",
    "leakage_status",
]

TEMPORAL_VALIDATION_STATUSES: tuple[str, ...] = ("pass", "needs_review", "blocked", "missing")
SUPPORTED_WINDOW_UNITS = {"D", "W", "M", "Y", "H", "MIN", "T"}


@dataclass(frozen=True)
class TemporalObservation:
    observation_id: str
    timestamp: pd.Timestamp
    label_start: pd.Timestamp
    label_end: pd.Timestamp
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return canonicalize_value(
            {
                "label_end": _format_timestamp(self.label_end),
                "label_start": _format_timestamp(self.label_start),
                "metadata": dict(self.metadata),
                "observation_id": self.observation_id,
                "timestamp": _format_timestamp(self.timestamp),
            }
        )


@dataclass(frozen=True)
class PurgedSplitConfig:
    n_splits: int = 3
    validation_window_size: int | None = None
    label_horizon: str | pd.Timedelta | None = None
    embargo_window: str | pd.Timedelta | None = "0D"
    time_unit: str = "timestamp"
    min_train_observations: int = 1
    min_validation_observations: int = 1

    def to_dict(self) -> dict[str, Any]:
        return canonicalize_value(
            {
                "embargo_window": _format_window_config(self.embargo_window),
                "label_horizon": _format_window_config(self.label_horizon),
                "min_train_observations": self.min_train_observations,
                "min_validation_observations": self.min_validation_observations,
                "n_splits": self.n_splits,
                "time_unit": self.time_unit,
                "validation_window_size": self.validation_window_size,
            }
        )


@dataclass(frozen=True)
class PurgedSplit:
    split_id: str
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    purge_window: str
    embargo_window: str
    train_observation_ids: tuple[str, ...] = ()
    validation_observation_ids: tuple[str, ...] = ()
    purged_observation_ids: tuple[str, ...] = ()
    embargoed_observation_ids: tuple[str, ...] = ()
    n_train_observations: int = 0
    n_validation_observations: int = 0
    leakage_status: str = "missing"
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return canonicalize_value(
            {
                "details": dict(self.details),
                "embargo_window": self.embargo_window,
                "embargoed_observation_ids": list(self.embargoed_observation_ids),
                "leakage_status": self.leakage_status,
                "n_embargoed_observations": len(self.embargoed_observation_ids),
                "n_purged_observations": len(self.purged_observation_ids),
                "n_train_observations": self.n_train_observations,
                "n_validation_observations": self.n_validation_observations,
                "purge_window": self.purge_window,
                "purged_observation_ids": list(self.purged_observation_ids),
                "split_id": self.split_id,
                "train_end": self.train_end,
                "train_observation_ids": list(self.train_observation_ids),
                "train_start": self.train_start,
                "validation_end": self.validation_end,
                "validation_observation_ids": list(self.validation_observation_ids),
                "validation_start": self.validation_start,
            }
        )

    def to_summary_row(self) -> dict[str, Any]:
        return {
            "embargo_window": self.embargo_window,
            "leakage_status": self.leakage_status,
            "n_embargoed_observations": len(self.embargoed_observation_ids),
            "n_purged_observations": len(self.purged_observation_ids),
            "n_train_observations": self.n_train_observations,
            "n_validation_observations": self.n_validation_observations,
            "purge_window": self.purge_window,
            "split_id": self.split_id,
            "train_end": self.train_end,
            "train_start": self.train_start,
            "validation_end": self.validation_end,
            "validation_start": self.validation_start,
        }


@dataclass(frozen=True)
class LeakageValidationResult:
    check_id: str
    status: str
    split_id: str = ""
    message: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return canonicalize_value(
            {
                "check_id": self.check_id,
                "details": dict(self.details),
                "message": self.message,
                "split_id": self.split_id,
                "status": self.status,
            }
        )


@dataclass(frozen=True)
class PurgedSplitPlan:
    workflow_type: str
    run_id: str
    config: PurgedSplitConfig
    splits: tuple[PurgedSplit, ...] = ()
    validation_results: tuple[LeakageValidationResult, ...] = ()
    observation_count: int = 0
    valid_observation_count: int = 0
    generated_artifacts: tuple[str, ...] = (
        LEAKAGE_VALIDATION_FILENAME,
        PURGED_SPLIT_PLAN_FILENAME,
        PURGED_SPLIT_SUMMARY_FILENAME,
    )

    @property
    def overall_status(self) -> str:
        return _overall_status([result.status for result in self.validation_results])

    def to_dict(self) -> dict[str, Any]:
        return canonicalize_value(
            {
                "config": self.config.to_dict(),
                "generated_artifacts": sorted(self.generated_artifacts),
                "observation_count": self.observation_count,
                "overall_status": self.overall_status,
                "run_id": self.run_id,
                "schema_version": SCHEMA_VERSION,
                "split_count": len(self.splits),
                "splits": [split.to_dict() for split in sorted(self.splits, key=lambda item: item.split_id)],
                "valid_observation_count": self.valid_observation_count,
                "workflow_type": self.workflow_type,
            }
        )


@dataclass(frozen=True)
class PurgedSplitWriteResult:
    output_dir: Path
    purged_split_plan_path: Path
    purged_split_summary_path: Path
    leakage_validation_path: Path


def intervals_overlap(
    left_start: Any,
    left_end: Any,
    right_start: Any,
    right_end: Any,
) -> bool:
    left_start_ts = _coerce_timestamp(left_start, field_name="left_start")
    left_end_ts = _coerce_timestamp(left_end, field_name="left_end")
    right_start_ts = _coerce_timestamp(right_start, field_name="right_start")
    right_end_ts = _coerce_timestamp(right_end, field_name="right_end")
    return left_start_ts < right_end_ts and left_end_ts > right_start_ts


def apply_purge_and_embargo(
    training_observations: Sequence[TemporalObservation],
    validation_observations: Sequence[TemporalObservation],
    *,
    embargo_window: str | pd.Timedelta | None = "0D",
) -> tuple[list[TemporalObservation], list[TemporalObservation], list[TemporalObservation]]:
    embargo_delta = _coerce_window(embargo_window, field_name="embargo_window", allow_zero=True)
    validation_intervals = [(item.label_start, item.label_end) for item in validation_observations]
    validation_end = max(item.timestamp for item in validation_observations)
    embargo_end = validation_end + embargo_delta
    kept: list[TemporalObservation] = []
    purged: list[TemporalObservation] = []
    embargoed: list[TemporalObservation] = []

    for observation in sorted(training_observations, key=_observation_sort_key):
        if any(
            intervals_overlap(
                observation.label_start,
                observation.label_end,
                validation_start,
                validation_end_value,
            )
            for validation_start, validation_end_value in validation_intervals
        ):
            purged.append(observation)
        elif embargo_delta > pd.Timedelta(0) and validation_end <= observation.timestamp < embargo_end:
            embargoed.append(observation)
        else:
            kept.append(observation)
    return kept, purged, embargoed


def build_purged_split_plan(
    records: Sequence[TemporalObservation | Mapping[str, Any]],
    *,
    config: PurgedSplitConfig | Mapping[str, Any] | None = None,
    workflow_type: str = "temporal_validation",
    run_id: str = "temporal_validation",
) -> PurgedSplitPlan:
    resolved_config = _coerce_config(config)
    observations, input_checks = _coerce_observations(records, config=resolved_config)
    config_checks = _validate_config(resolved_config, len(observations))
    if config_checks:
        return PurgedSplitPlan(
            workflow_type=workflow_type,
            run_id=run_id,
            config=resolved_config,
            validation_results=tuple(sorted((*input_checks, *config_checks), key=_validation_sort_key)),
            observation_count=len(records),
            valid_observation_count=len(observations),
        )

    ordered = sorted(observations, key=_observation_sort_key)
    splits: list[PurgedSplit] = []
    split_checks: list[LeakageValidationResult] = list(input_checks)
    validation_window_size = resolved_config.validation_window_size or math.ceil(len(ordered) / resolved_config.n_splits)
    embargo_delta = _coerce_window(resolved_config.embargo_window, field_name="embargo_window", allow_zero=True)

    for split_index in range(resolved_config.n_splits):
        start_index = split_index * validation_window_size
        end_index = min(len(ordered), start_index + validation_window_size)
        validation_observations = ordered[start_index:end_index]
        split_id = f"purged_{split_index:04d}"
        if not validation_observations:
            split_checks.append(
                LeakageValidationResult(
                    check_id="temporal_validation.insufficient_validation_observations",
                    status="needs_review",
                    split_id=split_id,
                    message="Validation block contains no observations.",
                    details={"required": resolved_config.min_validation_observations, "observed": 0},
                )
            )
            continue

        validation_ids = {item.observation_id for item in validation_observations}
        base_training = [item for item in ordered if item.observation_id not in validation_ids]
        train_observations, purged, embargoed = apply_purge_and_embargo(
            base_training,
            validation_observations,
            embargo_window=resolved_config.embargo_window,
        )
        validation_start = min(item.timestamp for item in validation_observations)
        validation_end = max(item.timestamp for item in validation_observations)
        embargo_end = validation_end + embargo_delta
        split = PurgedSplit(
            split_id=split_id,
            train_start=_format_timestamp(min((item.timestamp for item in train_observations), default=None)),
            train_end=_format_timestamp(max((item.timestamp for item in train_observations), default=None)),
            validation_start=_format_timestamp(validation_start),
            validation_end=_format_timestamp(validation_end),
            purge_window=_format_validation_label_window(validation_observations),
            embargo_window=f"[{_format_timestamp(validation_end)}, {_format_timestamp(embargo_end)})",
            train_observation_ids=tuple(item.observation_id for item in train_observations),
            validation_observation_ids=tuple(item.observation_id for item in validation_observations),
            purged_observation_ids=tuple(item.observation_id for item in purged),
            embargoed_observation_ids=tuple(item.observation_id for item in embargoed),
            n_train_observations=len(train_observations),
            n_validation_observations=len(validation_observations),
            leakage_status="pass",
            details={
                "embargo_end": _format_timestamp(embargo_end),
                "embargo_window": _format_window_config(resolved_config.embargo_window),
                "train_label_intervals": _label_interval_details(train_observations),
                "train_timestamps": {item.observation_id: _format_timestamp(item.timestamp) for item in train_observations},
                "validation_label_intervals": _label_interval_details(validation_observations),
                "window_semantics": "[start, end)",
            },
        )
        splits.append(split)

    plan = PurgedSplitPlan(
        workflow_type=workflow_type,
        run_id=run_id,
        config=resolved_config,
        splits=tuple(sorted(splits, key=lambda item: item.split_id)),
        validation_results=tuple(sorted(split_checks, key=_validation_sort_key)),
        observation_count=len(records),
        valid_observation_count=len(observations),
    )
    validation_results = validate_purged_split_plan(plan)
    return PurgedSplitPlan(
        workflow_type=plan.workflow_type,
        run_id=plan.run_id,
        config=plan.config,
        splits=plan.splits,
        validation_results=tuple(sorted((*plan.validation_results, *validation_results), key=_validation_sort_key)),
        observation_count=plan.observation_count,
        valid_observation_count=plan.valid_observation_count,
    )


def validate_purged_split_plan(plan: PurgedSplitPlan) -> list[LeakageValidationResult]:
    checks: list[LeakageValidationResult] = []
    for split in sorted(plan.splits, key=lambda item: item.split_id):
        checks.extend(_validate_split(plan, split))
    if plan.splits and not checks:
        checks.append(
            LeakageValidationResult(
                check_id="temporal_validation.pass",
                status="pass",
                message="Purged temporal split plan passed leakage checks.",
                details={"split_count": len(plan.splits)},
            )
        )
    return sorted(checks, key=_validation_sort_key)


def build_purged_split_findings(plan: PurgedSplitPlan, *, include_pass: bool = False) -> list[RobustnessFinding]:
    findings: list[RobustnessFinding] = []
    for result in sorted(plan.validation_results, key=_validation_sort_key):
        if result.status == "pass" and not include_pass:
            continue
        findings.append(
            RobustnessFinding(
                check_id=result.check_id,
                severity=_severity_for_status(result.status),
                workflow_type=plan.workflow_type,
                run_id=plan.run_id,
                message=result.message or _finding_message(result),
                details={"split_id": result.split_id, **dict(result.details)},
            )
        )
    return sorted(findings, key=lambda item: (item.workflow_type, item.run_id, item.check_id, str(item.details.get("split_id", ""))))


def build_purged_split_evidence(
    records: Sequence[TemporalObservation | Mapping[str, Any]],
    *,
    config: PurgedSplitConfig | Mapping[str, Any] | None = None,
    workflow_type: str = "temporal_validation",
    run_id: str = "temporal_validation",
    include_pass_findings: bool = False,
) -> tuple[PurgedSplitPlan, list[RobustnessFinding]]:
    plan = build_purged_split_plan(records, config=config, workflow_type=workflow_type, run_id=run_id)
    return plan, build_purged_split_findings(plan, include_pass=include_pass_findings)


def write_purged_split_artifacts(
    plan: PurgedSplitPlan,
    *,
    output_root: str | Path,
) -> PurgedSplitWriteResult:
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    roots = (Path.cwd(), output_dir)
    plan_path = output_dir / PURGED_SPLIT_PLAN_FILENAME
    summary_path = output_dir / PURGED_SPLIT_SUMMARY_FILENAME
    validation_path = output_dir / LEAKAGE_VALIDATION_FILENAME

    _write_json(plan_path, sanitize_portable_value(plan.to_dict(), roots=roots))
    _write_summary_csv(summary_path, [split.to_summary_row() for split in plan.splits])
    _write_json(validation_path, sanitize_portable_value(_leakage_validation_payload(plan), roots=roots))
    return PurgedSplitWriteResult(
        output_dir=output_dir,
        purged_split_plan_path=plan_path,
        purged_split_summary_path=summary_path,
        leakage_validation_path=validation_path,
    )


def _validate_split(plan: PurgedSplitPlan, split: PurgedSplit) -> list[LeakageValidationResult]:
    checks: list[LeakageValidationResult] = []
    train_ids = set(split.train_observation_ids)
    validation_ids = set(split.validation_observation_ids)
    if len(split.train_observation_ids) != split.n_train_observations:
        checks.append(_split_check("temporal_validation.invalid_split_config", "blocked", split, "Train count does not match serialized train IDs."))
    if len(split.validation_observation_ids) != split.n_validation_observations:
        checks.append(_split_check("temporal_validation.invalid_split_config", "blocked", split, "Validation count does not match serialized validation IDs."))
    if train_ids & validation_ids:
        checks.append(
            _split_check(
                "temporal_validation.train_validation_overlap",
                "blocked",
                split,
                "Train and validation observation IDs overlap.",
                {"overlap_ids": sorted(train_ids & validation_ids)},
            )
        )
    if split.n_train_observations < plan.config.min_train_observations:
        checks.append(
            _split_check(
                "temporal_validation.insufficient_train_observations",
                "needs_review",
                split,
                "Split has fewer training observations than configured.",
                {"observed": split.n_train_observations, "required": plan.config.min_train_observations},
            )
        )
    if split.n_validation_observations < plan.config.min_validation_observations:
        checks.append(
            _split_check(
                "temporal_validation.insufficient_validation_observations",
                "needs_review",
                split,
                "Split has fewer validation observations than configured.",
                {"observed": split.n_validation_observations, "required": plan.config.min_validation_observations},
            )
        )
    details = dict(split.details)
    train_intervals = _interval_details(details.get("train_label_intervals"))
    validation_intervals = _interval_details(details.get("validation_label_intervals"))
    overlapping_ids = sorted(
        train_id
        for train_id, train_start, train_end in train_intervals
        if any(intervals_overlap(train_start, train_end, validation_start, validation_end) for _, validation_start, validation_end in validation_intervals)
    )
    if overlapping_ids:
        checks.append(
            _split_check(
                "temporal_validation.purged_interval_overlap",
                "blocked",
                split,
                "Training label intervals overlap validation label intervals.",
                {"overlap_ids": overlapping_ids},
            )
        )
    train_timestamps = _timestamp_details(details.get("train_timestamps"))
    embargo_end_raw = details.get("embargo_end")
    if split.validation_end and embargo_end_raw:
        validation_end = _coerce_timestamp(split.validation_end, field_name="validation_end")
        embargo_end = _coerce_timestamp(embargo_end_raw, field_name="embargo_end")
        embargo_violations = sorted(
            observation_id
            for observation_id, timestamp in train_timestamps.items()
            if validation_end <= timestamp < embargo_end
        )
        if embargo_violations:
            checks.append(
                _split_check(
                    "temporal_validation.embargo_violation",
                    "blocked",
                    split,
                    "Training observations fall inside the configured embargo window.",
                    {"embargo_violation_ids": embargo_violations},
                )
            )
    return checks


def _coerce_observations(
    records: Sequence[TemporalObservation | Mapping[str, Any]],
    *,
    config: PurgedSplitConfig,
) -> tuple[list[TemporalObservation], list[LeakageValidationResult]]:
    observations: list[TemporalObservation] = []
    checks: list[LeakageValidationResult] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(records):
        try:
            observation = _coerce_observation(raw, index=index, config=config)
        except ValueError as exc:
            check_id = str(exc).split(":", maxsplit=1)[0]
            checks.append(
                LeakageValidationResult(
                    check_id=check_id,
                    status="missing" if "missing" in check_id else "blocked",
                    message=str(exc).split(":", maxsplit=1)[-1].strip(),
                    details={"row_index": index},
                )
            )
            continue
        if observation.observation_id in seen_ids:
            checks.append(
                LeakageValidationResult(
                    check_id="temporal_validation.invalid_split_config",
                    status="blocked",
                    message="Duplicate observation_id encountered.",
                    details={"observation_id": observation.observation_id, "row_index": index},
                )
            )
            continue
        seen_ids.add(observation.observation_id)
        observations.append(observation)
    return sorted(observations, key=_observation_sort_key), checks


def _coerce_observation(
    record: TemporalObservation | Mapping[str, Any],
    *,
    index: int,
    config: PurgedSplitConfig,
) -> TemporalObservation:
    if isinstance(record, TemporalObservation):
        return record
    observation_id = str(record.get("observation_id") or record.get("id") or f"obs_{index:04d}").strip()
    if not observation_id:
        raise ValueError("temporal_validation.invalid_split_config: observation_id must be non-empty.")
    timestamp = _required_timestamp(record.get("timestamp"), "timestamp")
    label_start_raw = record.get("label_start")
    label_end_raw = record.get("label_end")
    label_start = _coerce_timestamp(label_start_raw, field_name="label_start") if not _is_missing(label_start_raw) else None
    label_end = _coerce_timestamp(label_end_raw, field_name="label_end") if not _is_missing(label_end_raw) else None
    label_horizon = _coerce_optional_window(config.label_horizon, field_name="label_horizon")
    if label_start is None and label_end is None and label_horizon is not None:
        label_start = timestamp
        label_end = timestamp + label_horizon
    elif label_start is None:
        raise ValueError("temporal_validation.missing_timestamp: label_start is required unless label_horizon is configured.")
    elif label_end is None and label_horizon is not None:
        label_end = label_start + label_horizon
    elif label_end is None:
        raise ValueError("temporal_validation.missing_timestamp: label_end is required unless label_horizon is configured.")
    if label_end < label_start:
        raise ValueError("temporal_validation.invalid_split_config: label_end must be greater than or equal to label_start.")
    metadata = dict(record.get("metadata", {})) if isinstance(record.get("metadata", {}), Mapping) else {}
    return TemporalObservation(
        observation_id=observation_id,
        timestamp=timestamp,
        label_start=label_start,
        label_end=label_end,
        metadata=metadata,
    )


def _validate_config(config: PurgedSplitConfig, observation_count: int) -> list[LeakageValidationResult]:
    checks: list[LeakageValidationResult] = []
    if config.n_splits <= 0:
        checks.append(_config_check("n_splits must be greater than zero."))
    if config.validation_window_size is not None and config.validation_window_size <= 0:
        checks.append(_config_check("validation_window_size must be greater than zero when provided."))
    if config.min_train_observations < 0 or config.min_validation_observations < 0:
        checks.append(_config_check("minimum observation thresholds must be non-negative."))
    if observation_count == 0:
        checks.append(
            LeakageValidationResult(
                check_id="temporal_validation.missing_timestamp",
                status="missing",
                message="No valid temporal observations are available.",
                details={"observation_count": 0},
            )
        )
    if observation_count and config.n_splits > observation_count:
        checks.append(
            LeakageValidationResult(
                check_id="temporal_validation.insufficient_validation_observations",
                status="needs_review",
                message="Requested split count exceeds valid observation count.",
                details={"n_splits": config.n_splits, "valid_observation_count": observation_count},
            )
        )
    return checks


def _config_check(message: str) -> LeakageValidationResult:
    return LeakageValidationResult(
        check_id="temporal_validation.invalid_split_config",
        status="blocked",
        message=message,
    )


def _required_timestamp(value: Any, field_name: str) -> pd.Timestamp:
    if _is_missing(value):
        raise ValueError(f"temporal_validation.missing_timestamp: {field_name} is required.")
    return _coerce_timestamp(value, field_name=field_name)


def _coerce_timestamp(value: Any, *, field_name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"temporal_validation.missing_timestamp: {field_name} must be a valid timestamp.") from exc
    if pd.isna(timestamp):
        raise ValueError(f"temporal_validation.missing_timestamp: {field_name} must be a valid timestamp.")
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _coerce_config(config: PurgedSplitConfig | Mapping[str, Any] | None) -> PurgedSplitConfig:
    if config is None:
        return PurgedSplitConfig()
    if isinstance(config, PurgedSplitConfig):
        return config
    return PurgedSplitConfig(
        n_splits=int(config.get("n_splits", 3)),
        validation_window_size=None if config.get("validation_window_size") is None else int(config.get("validation_window_size")),
        label_horizon=config.get("label_horizon"),
        embargo_window=config.get("embargo_window", "0D"),
        time_unit=str(config.get("time_unit", "timestamp")),
        min_train_observations=int(config.get("min_train_observations", 1)),
        min_validation_observations=int(config.get("min_validation_observations", 1)),
    )


def _coerce_optional_window(value: str | pd.Timedelta | None, *, field_name: str) -> pd.Timedelta | None:
    if value is None or value == "":
        return None
    return _coerce_window(value, field_name=field_name, allow_zero=True)


def _coerce_window(value: str | pd.Timedelta | None, *, field_name: str, allow_zero: bool) -> pd.Timedelta:
    if value is None or value == "":
        return pd.Timedelta(0)
    if isinstance(value, pd.Timedelta):
        if value < pd.Timedelta(0) or (not allow_zero and value == pd.Timedelta(0)):
            raise ValueError(f"{field_name} must be non-negative.")
        return value
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a duration string.")
    compact = value.replace(" ", "")
    match = re.fullmatch(r"(?P<count>\d+)(?P<unit>MIN|[DWMYHT])", compact, re.IGNORECASE)
    if match is None:
        raise ValueError(f"{field_name} must use a duration like '5D', '12H', or '30MIN'.")
    count = int(match.group("count"))
    unit = match.group("unit").upper()
    if count < 0 or (count == 0 and not allow_zero):
        raise ValueError(f"{field_name} must be greater than zero.")
    if unit not in SUPPORTED_WINDOW_UNITS:
        raise ValueError(f"{field_name} must use one of: {', '.join(sorted(SUPPORTED_WINDOW_UNITS))}.")
    return pd.Timedelta(**{_timedelta_unit(unit): count})


def _timedelta_unit(unit: str) -> str:
    return {
        "D": "days",
        "W": "weeks",
        "H": "hours",
        "MIN": "minutes",
        "T": "minutes",
        "M": "days",
        "Y": "days",
    }[unit]


def _label_interval_details(observations: Sequence[TemporalObservation]) -> dict[str, dict[str, str]]:
    return {
        item.observation_id: {
            "label_end": _format_timestamp(item.label_end),
            "label_start": _format_timestamp(item.label_start),
        }
        for item in sorted(observations, key=_observation_sort_key)
    }


def _interval_details(value: Any) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    if not isinstance(value, Mapping):
        return []
    intervals: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    for observation_id, payload in sorted(value.items()):
        if not isinstance(payload, Mapping):
            continue
        start = payload.get("label_start")
        end = payload.get("label_end")
        if _is_missing(start) or _is_missing(end):
            continue
        intervals.append(
            (
                str(observation_id),
                _coerce_timestamp(start, field_name="label_start"),
                _coerce_timestamp(end, field_name="label_end"),
            )
        )
    return intervals


def _timestamp_details(value: Any) -> dict[str, pd.Timestamp]:
    if not isinstance(value, Mapping):
        return {}
    timestamps: dict[str, pd.Timestamp] = {}
    for observation_id, timestamp in sorted(value.items()):
        if _is_missing(timestamp):
            continue
        timestamps[str(observation_id)] = _coerce_timestamp(timestamp, field_name="timestamp")
    return timestamps


def _format_window_config(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, pd.Timedelta):
        return str(value)
    return str(value)


def _format_timestamp(value: pd.Timestamp | None) -> str:
    if value is None:
        return ""
    return value.tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _format_validation_label_window(observations: Sequence[TemporalObservation]) -> str:
    start = min(item.label_start for item in observations)
    end = max(item.label_end for item in observations)
    return f"[{_format_timestamp(start)}, {_format_timestamp(end)})"


def _observation_sort_key(observation: TemporalObservation) -> tuple[pd.Timestamp, str]:
    return (observation.timestamp, observation.observation_id)


def _validation_sort_key(result: LeakageValidationResult) -> tuple[str, str, str, str]:
    return (result.split_id, result.status, result.check_id, result.message)


def _split_check(
    check_id: str,
    status: str,
    split: PurgedSplit,
    message: str,
    details: Mapping[str, Any] | None = None,
) -> LeakageValidationResult:
    return LeakageValidationResult(
        check_id=check_id,
        status=status,
        split_id=split.split_id,
        message=message,
        details=details or {},
    )


def _overall_status(statuses: Sequence[str]) -> str:
    if not statuses:
        return "missing"
    order = {"pass": 0, "needs_review": 1, "missing": 2, "blocked": 3}
    return max(statuses, key=lambda item: order.get(item, 3))


def _severity_for_status(status: str) -> str:
    if status == "pass":
        return "info"
    if status == "blocked":
        return "blocked"
    return "needs_review"


def _finding_message(result: LeakageValidationResult) -> str:
    if result.status == "pass":
        return "Temporal validation passed leakage checks."
    return f"Temporal validation check '{result.check_id}' requires review."


def _leakage_validation_payload(plan: PurgedSplitPlan) -> dict[str, Any]:
    checks = [result.to_dict() for result in sorted(plan.validation_results, key=_validation_sort_key)]
    return canonicalize_value(
        {
            "checks": checks,
            "finding_count": len([check for check in checks if check["status"] != "pass"]),
            "overall_status": plan.overall_status,
            "schema_version": SCHEMA_VERSION,
            "split_count": len(plan.splits),
        }
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(canonicalize_value(dict(payload)), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=PURGED_SPLIT_SUMMARY_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in sorted(rows, key=lambda item: item["split_id"]):
            writer.writerow({field: row.get(field, "") for field in PURGED_SPLIT_SUMMARY_COLUMNS})


def _is_missing(value: Any) -> bool:
    return value is None or value == ""


__all__ = [
    "LEAKAGE_VALIDATION_FILENAME",
    "PURGED_SPLIT_PLAN_FILENAME",
    "PURGED_SPLIT_SUMMARY_COLUMNS",
    "PURGED_SPLIT_SUMMARY_FILENAME",
    "LeakageValidationResult",
    "PurgedSplit",
    "PurgedSplitConfig",
    "PurgedSplitPlan",
    "PurgedSplitWriteResult",
    "TEMPORAL_VALIDATION_STATUSES",
    "TemporalObservation",
    "apply_purge_and_embargo",
    "build_purged_split_evidence",
    "build_purged_split_findings",
    "build_purged_split_plan",
    "intervals_overlap",
    "validate_purged_split_plan",
    "write_purged_split_artifacts",
]
