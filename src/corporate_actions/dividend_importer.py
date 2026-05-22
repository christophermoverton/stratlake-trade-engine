from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from src.artifacts.safety import atomic_write_json, portable_path
from src.corporate_actions.dividend_contract import (
    DIVIDEND_PRIMARY_KEY_FIELDS,
    DIVIDEND_REQUIRED_FIELDS,
    DIVIDEND_REQUIRED_NULLABLE_FIELDS,
    DIVIDEND_SCHEMA_NAME,
    DIVIDEND_SCHEMA_VERSION,
    DIVIDEND_SUPPORTED_EVENT_TYPES,
    DIVIDEND_UPSTREAM_FIELD_MAPPING,
    DividendContractError,
    build_dividend_schema_contract,
    validate_dividend_event_schema,
)


DIVIDEND_SORT_COLUMNS: tuple[str, ...] = (
    "symbol",
    "ex_date",
    "event_type",
    "process_date",
    "source_event_id",
    "source",
)


@dataclass(frozen=True)
class DividendImportResult:
    run_id: str
    source_data_path: str
    source_metadata_path: str
    output_root: str
    artifact_root: str
    artifact_path: str
    start: str
    end: str
    input_row_count: int
    normalized_row_count: int
    written_row_count: int
    filtered_row_count: int
    duplicate_event_count: int
    invalid_event_count: int
    symbols: tuple[str, ...]
    partitions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "source_data_path": self.source_data_path,
            "source_metadata_path": self.source_metadata_path,
            "output_root": self.output_root,
            "artifact_root": self.artifact_root,
            "artifact_path": self.artifact_path,
            "start": self.start,
            "end": self.end,
            "input_row_count": self.input_row_count,
            "normalized_row_count": self.normalized_row_count,
            "written_row_count": self.written_row_count,
            "filtered_row_count": self.filtered_row_count,
            "duplicate_event_count": self.duplicate_event_count,
            "invalid_event_count": self.invalid_event_count,
            "symbols": list(self.symbols),
            "partitions": list(self.partitions),
        }


class DividendImportError(ValueError):
    """Raised when local dividend event import cannot complete deterministically."""


def read_upstream_dividend_artifacts(
    source_data_path: str | Path,
    source_metadata_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read local upstream dividend Parquet and metadata JSON artifacts."""

    data_path = _resolve_local_file(source_data_path, label="source dividend data")
    metadata_path = _resolve_local_file(source_metadata_path, label="source dividend metadata")

    data = pd.read_parquet(data_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise DividendImportError("source dividend metadata JSON must contain an object.")
    return data, metadata


def normalize_upstream_dividend_events(upstream: pd.DataFrame) -> pd.DataFrame:
    """Map upstream dividend artifacts into the StratLake dividend event contract."""

    if not isinstance(upstream, pd.DataFrame):
        raise DividendImportError("upstream dividend events must be provided as a pandas DataFrame.")

    normalized = pd.DataFrame(index=upstream.index)
    for stratlake_field in DIVIDEND_REQUIRED_FIELDS + DIVIDEND_REQUIRED_NULLABLE_FIELDS:
        upstream_field = DIVIDEND_UPSTREAM_FIELD_MAPPING[stratlake_field]
        if stratlake_field == "schema_version":
            normalized[stratlake_field] = DIVIDEND_SCHEMA_VERSION
        elif stratlake_field == "as_of_date" and stratlake_field in upstream.columns:
            normalized[stratlake_field] = upstream[stratlake_field]
        elif upstream_field is not None and upstream_field in upstream.columns:
            normalized[stratlake_field] = upstream[upstream_field]
        else:
            normalized[stratlake_field] = pd.NA

    normalized["symbol"] = _normalize_string_series(normalized["symbol"], uppercase=True)
    for column in ("event_type", "source", "source_event_id", "source_payload_fingerprint", "currency"):
        normalized[column] = _normalize_string_series(normalized[column])

    for column in ("ex_date", "process_date", "as_of_date"):
        normalized[column] = _normalize_date_series(normalized[column], column_name=column, nullable=False)
    for column in ("declaration_date", "record_date", "payable_date"):
        normalized[column] = _normalize_date_series(normalized[column], column_name=column, nullable=True)

    normalized["raw_payload"] = normalized["raw_payload"].map(_normalize_raw_payload)
    normalized["year"] = pd.to_datetime(
        normalized["ex_date"],
        errors="coerce",
        format="mixed",
        utc=True,
    ).dt.strftime("%Y")
    return normalized


def filter_dividend_events_by_ex_date(
    events: pd.DataFrame,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    """Apply half-open StratLake import windows: start <= ex_date < end."""

    start_date = _normalize_boundary_date(start, label="start")
    end_date = _normalize_boundary_date(end, label="end")
    if start_date >= end_date:
        raise DividendImportError("dividend import start must be earlier than end.")

    ex_dates = pd.to_datetime(events["ex_date"], errors="coerce", format="mixed", utc=True).dt.tz_localize(None)
    mask = (ex_dates >= pd.Timestamp(start_date)) & (ex_dates < pd.Timestamp(end_date))
    return events.loc[mask].copy()


def write_dividend_event_dataset(events: pd.DataFrame, output_root: str | Path) -> tuple[str, ...]:
    """Write deterministic symbol/year partitioned dividend event Parquet output."""

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    if events.empty:
        return ()

    required_for_write = set(DIVIDEND_REQUIRED_FIELDS + DIVIDEND_REQUIRED_NULLABLE_FIELDS + ("year",))
    missing = sorted(required_for_write - set(events.columns))
    if missing:
        formatted = ", ".join(repr(column) for column in missing)
        raise DividendImportError(f"dividend events are missing required writer columns: {formatted}.")

    ordered = _sort_dividend_events(events)
    partitions: list[str] = []
    for (symbol, year), partition in ordered.groupby(["symbol", "year"], sort=True):
        partition_path = root / f"symbol={symbol}" / f"year={year}"
        partition_path.mkdir(parents=True, exist_ok=True)
        for existing in sorted(partition_path.glob("part-*.parquet")):
            existing.unlink()

        output_columns = [
            column
            for column in DIVIDEND_REQUIRED_FIELDS + DIVIDEND_REQUIRED_NULLABLE_FIELDS
            if column != "symbol"
        ]
        partition.loc[:, output_columns].reset_index(drop=True).to_parquet(
            partition_path / "part-0.parquet",
            index=False,
        )
        partitions.append(portable_path(partition_path, roots=(Path.cwd(), root)))

    return tuple(sorted(partitions))


def load_dividend_events(dataset_root: str | Path) -> pd.DataFrame:
    """
    Load a curated dividend event dataset from its partitioned dataset root.

    Dividend event datasets use Hive-style partition directories
    ``symbol=<SYMBOL>/year=<YYYY>/``. Downstream consumers should read from the
    dataset root so pandas/pyarrow reconstructs partition columns.
    """

    root = Path(dataset_root)
    if not root.exists():
        raise DividendImportError(f"dividend event dataset root does not exist: {root}.")
    if not root.is_dir():
        raise DividendImportError(f"dividend event dataset root must be a directory: {root}.")

    frame = pd.read_parquet(root)
    if frame.empty:
        return frame
    return _sort_loaded_dividend_events(frame)


def import_dividend_events(
    *,
    source_data_path: str | Path,
    source_metadata_path: str | Path,
    output_root: str | Path,
    artifact_root: str | Path = "artifacts/corporate_actions",
    run_id: str | None = None,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    strict: bool = True,
) -> DividendImportResult:
    """Import local upstream dividend artifacts into StratLake-curated event evidence."""

    source_data = _resolve_local_file(source_data_path, label="source dividend data")
    source_metadata = _resolve_local_file(source_metadata_path, label="source dividend metadata")
    output = Path(output_root)
    artifacts = Path(artifact_root)
    start_date = _normalize_boundary_date(start, label="start")
    end_date = _normalize_boundary_date(end, label="end")
    if start_date >= end_date:
        raise DividendImportError("dividend import start must be earlier than end.")

    upstream, metadata = read_upstream_dividend_artifacts(source_data, source_metadata)
    source_dataset_fingerprint = _fingerprint_file(source_data)
    import_config = _build_import_config(
        source_data_path=source_data,
        source_metadata_path=source_metadata,
        output_root=output,
        artifact_root=artifacts,
        start=start_date,
        end=end_date,
        strict=strict,
    )
    import_config_fingerprint = _stable_fingerprint(import_config)
    resolved_run_id = run_id or _build_run_id(
        source_dataset_fingerprint=source_dataset_fingerprint,
        import_config_fingerprint=import_config_fingerprint,
    )
    artifact_path = artifacts / resolved_run_id

    missing_required_columns = _missing_upstream_columns(upstream, DIVIDEND_REQUIRED_FIELDS)
    missing_required_nullable_columns = _missing_upstream_columns(
        upstream,
        DIVIDEND_REQUIRED_NULLABLE_FIELDS,
    )
    normalized = normalize_upstream_dividend_events(upstream)
    filtered = filter_dividend_events_by_ex_date(normalized, start=start_date, end=end_date)
    rows_outside_import_window_count = int(len(normalized) - len(filtered))
    qa_counts = _build_qa_counts(
        normalized,
        filtered,
        missing_required_columns=missing_required_columns,
        missing_required_nullable_columns=missing_required_nullable_columns,
    )

    preliminary_duplicate_event_count = _duplicate_event_count(filtered)
    duplicate_events = _duplicate_events(filtered)
    invalid_events = _invalid_events(filtered)
    if preliminary_duplicate_event_count and strict:
        valid_events = filtered.iloc[0:0].copy()
        result = _build_import_result(
            run_id=resolved_run_id,
            source_data=source_data,
            source_metadata=source_metadata,
            output=output,
            artifacts=artifacts,
            artifact_path=artifact_path,
            start_date=start_date,
            end_date=end_date,
            upstream=upstream,
            normalized=normalized,
            valid_events=valid_events,
            rows_outside_import_window_count=rows_outside_import_window_count,
            duplicate_event_count=preliminary_duplicate_event_count,
            invalid_event_count=len(invalid_events),
            partitions=(),
        )
        _write_dividend_import_artifacts(
            artifact_path=artifact_path,
            result=result,
            import_config=import_config,
            import_config_fingerprint=import_config_fingerprint,
            metadata=metadata,
            source_data=source_data,
            source_metadata=source_metadata,
            source_dataset_fingerprint=source_dataset_fingerprint,
            qa_counts=qa_counts,
            duplicate_events=duplicate_events,
            invalid_events=invalid_events,
            strict=strict,
        )
        duplicate_key = _first_duplicate_key(filtered)
        raise DividendImportError(
            "dividend import contains duplicate primary-key rows. "
            f"First duplicate key: {duplicate_key}."
        )

    valid_events, invalid_event_count = _validate_or_filter_invalid_events(filtered, strict=strict)
    invalid_events = _invalid_events(filtered)
    if invalid_event_count and strict:
        result = _build_import_result(
            run_id=resolved_run_id,
            source_data=source_data,
            source_metadata=source_metadata,
            output=output,
            artifacts=artifacts,
            artifact_path=artifact_path,
            start_date=start_date,
            end_date=end_date,
            upstream=upstream,
            normalized=normalized,
            valid_events=filtered.iloc[0:0].copy(),
            rows_outside_import_window_count=rows_outside_import_window_count,
            duplicate_event_count=preliminary_duplicate_event_count,
            invalid_event_count=invalid_event_count,
            partitions=(),
        )
        _write_dividend_import_artifacts(
            artifact_path=artifact_path,
            result=result,
            import_config=import_config,
            import_config_fingerprint=import_config_fingerprint,
            metadata=metadata,
            source_data=source_data,
            source_metadata=source_metadata,
            source_dataset_fingerprint=source_dataset_fingerprint,
            qa_counts=qa_counts,
            duplicate_events=duplicate_events,
            invalid_events=invalid_events,
            strict=strict,
        )
        try:
            validate_dividend_event_schema(filtered.drop(columns=["year"], errors="ignore"))
        except DividendContractError as exc:
            raise DividendImportError(
                "dividend import contains invalid rows that violate the dividend event schema."
            ) from exc

    duplicate_event_count = _duplicate_event_count(valid_events)
    if duplicate_event_count:
        valid_events = valid_events.drop_duplicates(subset=list(DIVIDEND_PRIMARY_KEY_FIELDS), keep="first")

    valid_events = _sort_dividend_events(valid_events).reset_index(drop=True)
    validate_dividend_event_schema(valid_events.drop(columns=["year"], errors="ignore"))
    partitions = write_dividend_event_dataset(valid_events, output)

    result = _build_import_result(
        run_id=resolved_run_id,
        source_data=source_data,
        source_metadata=source_metadata,
        output=output,
        artifacts=artifacts,
        artifact_path=artifact_path,
        start_date=start_date,
        end_date=end_date,
        upstream=upstream,
        normalized=normalized,
        valid_events=valid_events,
        rows_outside_import_window_count=rows_outside_import_window_count,
        duplicate_event_count=duplicate_event_count,
        invalid_event_count=invalid_event_count,
        partitions=partitions,
    )
    _write_dividend_import_artifacts(
        artifact_path=artifact_path,
        result=result,
        import_config=import_config,
        import_config_fingerprint=import_config_fingerprint,
        metadata=metadata,
        source_data=source_data,
        source_metadata=source_metadata,
        source_dataset_fingerprint=source_dataset_fingerprint,
        qa_counts=qa_counts,
        duplicate_events=duplicate_events,
        invalid_events=invalid_events,
        strict=strict,
    )
    return result


def _build_import_config(
    *,
    source_data_path: Path,
    source_metadata_path: Path,
    output_root: Path,
    artifact_root: Path,
    start: str,
    end: str,
    strict: bool,
) -> dict[str, Any]:
    return {
        "artifact_root": _portable_import_path(artifact_root, artifact_root=artifact_root),
        "event_evidence_policy": "dividend events are explicit event evidence, not adjusted price data",
        "fallback_key_behavior": "deferred_inactive",
        "output_root": _portable_import_path(output_root, artifact_root=artifact_root),
        "schema_name": DIVIDEND_SCHEMA_NAME,
        "schema_version": DIVIDEND_SCHEMA_VERSION,
        "source_data_path": _portable_import_path(source_data_path, artifact_root=artifact_root),
        "source_metadata_path": _portable_import_path(source_metadata_path, artifact_root=artifact_root),
        "strict": strict,
        "window": {
            "end": end,
            "semantics": "start <= ex_date < end",
            "start": start,
        },
    }


def _build_run_id(*, source_dataset_fingerprint: str, import_config_fingerprint: str) -> str:
    digest = _stable_fingerprint(
        {
            "import_config_fingerprint": import_config_fingerprint,
            "source_dataset_fingerprint": source_dataset_fingerprint,
        }
    )[:12]
    return f"dividend_import_{digest}"


def _build_import_result(
    *,
    run_id: str,
    source_data: Path,
    source_metadata: Path,
    output: Path,
    artifacts: Path,
    artifact_path: Path,
    start_date: str,
    end_date: str,
    upstream: pd.DataFrame,
    normalized: pd.DataFrame,
    valid_events: pd.DataFrame,
    rows_outside_import_window_count: int,
    duplicate_event_count: int,
    invalid_event_count: int,
    partitions: tuple[str, ...],
) -> DividendImportResult:
    symbols = tuple(sorted(valid_events["symbol"].dropna().astype("string").unique().tolist()))
    return DividendImportResult(
        run_id=run_id,
        source_data_path=_portable_import_path(source_data, artifact_root=artifacts),
        source_metadata_path=_portable_import_path(source_metadata, artifact_root=artifacts),
        output_root=_portable_import_path(output, artifact_root=artifacts),
        artifact_root=_portable_import_path(artifacts, artifact_root=artifacts),
        artifact_path=_portable_import_path(artifact_path, artifact_root=artifacts),
        start=start_date,
        end=end_date,
        input_row_count=int(len(upstream)),
        normalized_row_count=int(len(normalized)),
        written_row_count=int(len(valid_events)),
        filtered_row_count=rows_outside_import_window_count,
        duplicate_event_count=duplicate_event_count,
        invalid_event_count=invalid_event_count,
        symbols=symbols,
        partitions=partitions,
    )


def _write_dividend_import_artifacts(
    *,
    artifact_path: Path,
    result: DividendImportResult,
    import_config: dict[str, Any],
    import_config_fingerprint: str,
    metadata: dict[str, Any],
    source_data: Path,
    source_metadata: Path,
    source_dataset_fingerprint: str,
    qa_counts: dict[str, int],
    duplicate_events: pd.DataFrame,
    invalid_events: pd.DataFrame,
    strict: bool,
) -> None:
    artifact_path.mkdir(parents=True, exist_ok=True)

    qa_summary = _build_qa_summary(
        result=result,
        qa_counts=qa_counts,
        strict=strict,
    )
    summary = {
        "artifact_type": "corporate_action_event_import_summary",
        "event_evidence_policy": "dividend events are explicit event evidence, not adjusted price data",
        "fallback_key_behavior": "deferred_inactive",
        "import_result": result.to_dict(),
        "qa_status": qa_summary["qa_status"],
        "schema_name": DIVIDEND_SCHEMA_NAME,
        "schema_version": DIVIDEND_SCHEMA_VERSION,
    }
    provenance = _build_source_provenance(
        metadata=metadata,
        source_data=source_data,
        source_metadata=source_metadata,
        artifact_root=artifact_path.parent,
        source_dataset_fingerprint=source_dataset_fingerprint,
        import_config_fingerprint=import_config_fingerprint,
    )
    manifest = _build_manifest(
        result=result,
        qa_status=qa_summary["qa_status"],
    )

    atomic_write_json(artifact_path / "import_config.json", import_config, sort_keys=True)
    atomic_write_json(artifact_path / "schema_contract.json", build_dividend_schema_contract(), sort_keys=True)
    atomic_write_json(artifact_path / "qa_summary.json", qa_summary, sort_keys=True)
    atomic_write_json(artifact_path / "summary.json", summary, sort_keys=True)
    atomic_write_json(artifact_path / "source_provenance.json", provenance, sort_keys=True)
    _write_csv(artifact_path / "duplicate_events.csv", duplicate_events)
    _write_csv(artifact_path / "invalid_events.csv", invalid_events)
    atomic_write_json(artifact_path / "manifest.json", manifest, sort_keys=True)


def _build_manifest(*, result: DividendImportResult, qa_status: str) -> dict[str, Any]:
    artifact_files = (
        "duplicate_events.csv",
        "import_config.json",
        "invalid_events.csv",
        "manifest.json",
        "qa_summary.json",
        "schema_contract.json",
        "source_provenance.json",
        "summary.json",
    )
    return {
        "artifact_files": list(artifact_files),
        "artifact_root": result.artifact_path,
        "artifact_type": "corporate_action_event_import",
        "canonical_dataset_root": result.output_root,
        "event_evidence_policy": "dividend events are explicit event evidence, not adjusted price data",
        "evidence_type": "dividend_events",
        "fallback_key_behavior": "deferred_inactive",
        "paths": {filename: f"{result.artifact_path}/{filename}" for filename in artifact_files},
        "qa_status": qa_status,
        "run_id": result.run_id,
        "schema_name": DIVIDEND_SCHEMA_NAME,
        "schema_version": DIVIDEND_SCHEMA_VERSION,
    }


def _build_qa_summary(
    *,
    result: DividendImportResult,
    qa_counts: dict[str, int],
    strict: bool,
) -> dict[str, Any]:
    blocking_count = (
        result.invalid_event_count
        + result.duplicate_event_count
        + qa_counts["missing_required_column_count"]
        + qa_counts["invalid_event_type_count"]
        + qa_counts["invalid_date_count"]
        + qa_counts["null_required_value_count"]
        + qa_counts["negative_cash_amount_count"]
        + qa_counts["negative_stock_amount_count"]
        + qa_counts["currency_missing_for_cash_dividend_count"]
    )
    qa_status = "pass" if blocking_count == 0 else "fail" if strict else "warn"
    return {
        "advisory_duplicate_policy": (
            "strict mode rejects duplicate primary-key rows"
            if strict
            else "advisory mode reports duplicate primary-key row counts and keeps the first deterministic occurrence"
        ),
        "currency_missing_for_cash_dividend_count": qa_counts["currency_missing_for_cash_dividend_count"],
        "duplicate_event_count": result.duplicate_event_count,
        "event_evidence_policy": "dividend events are explicit event evidence, not adjusted price data",
        "fallback_key_behavior": "deferred_inactive",
        "filtered_row_count": result.filtered_row_count,
        "input_row_count": result.input_row_count,
        "invalid_date_count": qa_counts["invalid_date_count"],
        "invalid_event_count": result.invalid_event_count,
        "invalid_event_type_count": qa_counts["invalid_event_type_count"],
        "missing_required_column_count": qa_counts["missing_required_column_count"],
        "missing_required_nullable_column_count": qa_counts["missing_required_nullable_column_count"],
        "negative_cash_amount_count": qa_counts["negative_cash_amount_count"],
        "negative_stock_amount_count": qa_counts["negative_stock_amount_count"],
        "normalized_row_count": result.normalized_row_count,
        "null_required_value_count": qa_counts["null_required_value_count"],
        "qa_status": qa_status,
        "rows_outside_import_window_count": qa_counts["rows_outside_import_window_count"],
        "strict_mode": strict,
        "written_row_count": result.written_row_count,
    }


def _build_source_provenance(
    *,
    metadata: dict[str, Any],
    source_data: Path,
    source_metadata: Path,
    artifact_root: Path,
    source_dataset_fingerprint: str,
    import_config_fingerprint: str,
) -> dict[str, Any]:
    source_vendor = _metadata_string(metadata, "source_vendor") or _metadata_string(metadata, "source")
    return {
        "credentials_used": False,
        "fallback_key_behavior": "deferred_inactive",
        "import_config_fingerprint": import_config_fingerprint,
        "live_network_used": False,
        "path_policy": "repository-relative POSIX paths only; no absolute local paths",
        "schema_name": DIVIDEND_SCHEMA_NAME,
        "schema_version": DIVIDEND_SCHEMA_VERSION,
        "source_data_path": _portable_import_path(source_data, artifact_root=artifact_root),
        "source_dataset_fingerprint": source_dataset_fingerprint,
        "source_metadata_path": _portable_import_path(source_metadata, artifact_root=artifact_root),
        "source_vendor": source_vendor,
        "upstream_metadata": _safe_metadata(metadata),
        "upstream_package_name": _metadata_string(metadata, "upstream_package_name")
        or "fintech-market-ingestion",
        "upstream_package_version": _metadata_string(metadata, "upstream_package_version"),
        "upstream_project": _metadata_string(metadata, "upstream_project") or "fintech-market-ingestion",
        "upstream_source_repository": _metadata_string(metadata, "upstream_source_repository"),
    }


def _build_qa_counts(
    normalized: pd.DataFrame,
    filtered: pd.DataFrame,
    *,
    missing_required_columns: tuple[str, ...],
    missing_required_nullable_columns: tuple[str, ...],
) -> dict[str, int]:
    return {
        "currency_missing_for_cash_dividend_count": _currency_missing_for_cash_dividend_count(filtered),
        "invalid_date_count": _invalid_date_count(normalized),
        "invalid_event_type_count": _invalid_event_type_count(filtered),
        "missing_required_column_count": len(missing_required_columns),
        "missing_required_nullable_column_count": len(missing_required_nullable_columns),
        "negative_cash_amount_count": _negative_numeric_count(filtered, "cash_amount"),
        "negative_stock_amount_count": _negative_numeric_count(filtered, "stock_amount"),
        "null_required_value_count": _null_required_value_count(filtered),
        "rows_outside_import_window_count": int(len(normalized) - len(filtered)),
    }


def _missing_upstream_columns(upstream: pd.DataFrame, fields: tuple[str, ...]) -> tuple[str, ...]:
    missing: list[str] = []
    for stratlake_field in fields:
        upstream_field = DIVIDEND_UPSTREAM_FIELD_MAPPING[stratlake_field]
        if stratlake_field == "schema_version":
            continue
        if stratlake_field == "as_of_date" and stratlake_field in upstream.columns:
            continue
        if upstream_field is not None and upstream_field in upstream.columns:
            continue
        missing.append(stratlake_field)
    return tuple(missing)


def _invalid_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return _csv_frame([])

    contract_events = events.drop(columns=["year"], errors="ignore")
    issues_by_index: dict[Any, list[str]] = {}

    def append_issue(row_index: Any, issue: str) -> None:
        if not issue:
            return
        row_issues = issues_by_index.setdefault(row_index, [])
        if issue not in row_issues:
            row_issues.append(issue)

    missing_required_columns = tuple(
        field for field in DIVIDEND_REQUIRED_FIELDS if field not in contract_events.columns
    )
    if missing_required_columns:
        missing_columns_issue = "Missing required columns: " + ", ".join(missing_required_columns)
        for row_index in contract_events.index:
            append_issue(row_index, missing_columns_issue)

    if "event_type" in contract_events.columns:
        event_type = contract_events["event_type"]
        invalid_event_type_mask = event_type.notna() & ~event_type.isin(DIVIDEND_SUPPORTED_EVENT_TYPES)
        for row_index, value in event_type.loc[invalid_event_type_mask].items():
            append_issue(row_index, f"Unsupported event_type: {value}")

    date_columns = [column for column in contract_events.columns if column.endswith("_date")]
    for column in date_columns:
        values = contract_events[column]
        non_empty_mask = values.notna() & values.astype("string").str.strip().ne("")
        invalid_date_mask = non_empty_mask & pd.to_datetime(values, errors="coerce").isna()
        for row_index, value in values.loc[invalid_date_mask].items():
            append_issue(row_index, f"Unparseable {column}: {value}")

    present_required_non_nullable = [
        field
        for field in DIVIDEND_REQUIRED_FIELDS
        if field in contract_events.columns and field not in DIVIDEND_REQUIRED_NULLABLE_FIELDS
    ]
    if present_required_non_nullable:
        null_required = contract_events[present_required_non_nullable].isna()
        null_required_mask = null_required.any(axis=1)
        for row_index in contract_events.index[null_required_mask]:
            missing_fields = [field for field in present_required_non_nullable if bool(null_required.at[row_index, field])]
            append_issue(row_index, "Null required values: " + ", ".join(missing_fields))

    if not issues_by_index:
        return _csv_frame([])

    rows: list[dict[str, Any]] = []
    invalid_rows = contract_events.loc[list(issues_by_index.keys())]
    for index, row in invalid_rows.iterrows():
        row_frame = pd.DataFrame([row])
        issues = list(issues_by_index.get(index, []))
        for schema_issue in _row_issues(row_frame):
            if schema_issue not in issues:
                issues.append(schema_issue)
        if not issues:
            continue
        payload = _jsonable_row(events.loc[index])
        payload["issue"] = "; ".join(issues)
        payload["source_row_index"] = int(index) if isinstance(index, int) else str(index)
        rows.append(payload)
    return _csv_frame(rows)


def _duplicate_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return _csv_frame([])
    duplicate_mask = events.duplicated(subset=list(DIVIDEND_PRIMARY_KEY_FIELDS), keep=False)
    duplicate_rows = events.loc[duplicate_mask].copy()
    if duplicate_rows.empty:
        return _csv_frame([])
    duplicate_rows = _sort_dividend_events(duplicate_rows)
    rows = []
    for _, row in duplicate_rows.iterrows():
        payload = _jsonable_row(row)
        payload["duplicate_key"] = _stable_json(
            {column: payload.get(column) for column in DIVIDEND_PRIMARY_KEY_FIELDS}
        )
        rows.append(payload)
    return _csv_frame(rows)


def _row_issues(row_frame: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    try:
        validate_dividend_event_schema(row_frame)
    except DividendContractError as exc:
        issues.append(str(exc))
    return issues


def _invalid_event_type_count(events: pd.DataFrame) -> int:
    if "event_type" not in events.columns:
        return 0
    observed = events["event_type"].astype("string")
    invalid = observed.notna() & ~observed.isin(DIVIDEND_SUPPORTED_EVENT_TYPES)
    return int(invalid.sum())


def _invalid_date_count(events: pd.DataFrame) -> int:
    count = 0
    for column in ("ex_date", "process_date", "as_of_date", "declaration_date", "record_date", "payable_date"):
        if column not in events.columns:
            continue
        values = events[column].dropna()
        if values.empty:
            continue
        parsed = pd.to_datetime(values, errors="coerce", format="mixed", utc=True)
        count += int(parsed.isna().sum())
    return count


def _null_required_value_count(events: pd.DataFrame) -> int:
    count = 0
    for column in DIVIDEND_REQUIRED_FIELDS:
        if column in events.columns:
            count += int(events[column].isna().sum())
    return count


def _negative_numeric_count(events: pd.DataFrame, column: str) -> int:
    if column not in events.columns:
        return 0
    values = pd.to_numeric(events[column], errors="coerce")
    return int((values < 0).sum())


def _currency_missing_for_cash_dividend_count(events: pd.DataFrame) -> int:
    if "event_type" not in events.columns or "currency" not in events.columns:
        return 0
    mask = (events["event_type"].astype("string") == "cash_dividend") & events["currency"].isna()
    return int(mask.sum())


def _safe_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key in sorted(metadata):
        lowered = key.lower()
        if any(token in lowered for token in ("secret", "token", "password", "credential", "api_key")):
            safe[key] = "[redacted]"
        else:
            safe[key] = _jsonable_value(metadata[key])
    return safe


def _metadata_string(metadata: dict[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = frame.copy()
    if not frame.empty:
        frame = frame.sort_values(list(frame.columns), kind="mergesort").reset_index(drop=True)
    text = frame.to_csv(index=False, lineterminator="\n")
    path.write_text(text, encoding="utf-8", newline="")


def _csv_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if rows:
        columns = sorted({column for row in rows for column in row})
    else:
        columns = [
            "symbol",
            "event_type",
            "source_event_id",
            "ex_date",
            "process_date",
            "source",
            "issue",
        ]
    return pd.DataFrame(rows, columns=columns)


def _jsonable_row(row: pd.Series) -> dict[str, Any]:
    return {str(key): _jsonable_value(value) for key, value in row.items()}


def _jsonable_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, Mapping):
        return {str(key): _jsonable_value(val) for key, val in sorted(value.items())}
    if isinstance(value, list | tuple):
        return [_jsonable_value(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _fingerprint_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_fingerprint(payload: Any) -> str:
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _stable_json(payload: Any) -> str:
    return json.dumps(_jsonable_value(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _resolve_local_file(path: str | Path, *, label: str) -> Path:
    raw = str(path)
    if "://" in raw:
        parsed = urlparse(raw)
        if parsed.scheme != "file":
            raise DividendImportError(f"{label} path must be a local file path, not a URI: {raw!r}.")
        resolved = Path(parsed.path)
    else:
        resolved = Path(path)
    if not resolved.exists():
        raise DividendImportError(f"{label} path does not exist: {raw!r}.")
    if not resolved.is_file():
        raise DividendImportError(f"{label} path must be a file: {raw!r}.")
    return resolved


def _portable_import_path(path: str | Path, *, artifact_root: Path) -> str:
    return portable_path(path, roots=(Path.cwd(), *_import_portable_roots(artifact_root)))


def _import_portable_roots(artifact_root: Path) -> tuple[Path, ...]:
    resolved = artifact_root.resolve()
    roots: list[Path] = []
    if resolved.name == "corporate_actions" and resolved.parent.name == "artifacts":
        roots.append(resolved.parent.parent)
    roots.append(resolved.parent)
    return tuple(roots)


def _normalize_string_series(series: pd.Series, *, uppercase: bool = False) -> pd.Series:
    normalized = series.astype("string").str.strip()
    if uppercase:
        normalized = normalized.str.upper()
    return normalized.mask(normalized == "", pd.NA)


def _normalize_date_series(series: pd.Series, *, column_name: str, nullable: bool) -> pd.Series:
    values = series.copy()
    if nullable:
        non_null = values.dropna()
        if non_null.empty:
            return values.where(values.isna(), pd.NA)
        parsed = pd.to_datetime(non_null, errors="coerce", format="mixed", utc=True)
        normalized = pd.Series(pd.NA, index=values.index, dtype="string")
        normalized.loc[non_null.index] = parsed.dt.strftime("%Y-%m-%d")
        invalid = parsed.isna()
        if invalid.any():
            normalized.loc[non_null.index[invalid]] = non_null.loc[non_null.index[invalid]].astype("string").str.strip()
        return normalized

    parsed = pd.to_datetime(values, errors="coerce", format="mixed", utc=True)
    normalized = parsed.dt.strftime("%Y-%m-%d")
    invalid = parsed.isna() & values.notna()
    if invalid.any():
        normalized.loc[invalid] = values.loc[invalid].astype("string").str.strip()
    if values.isna().any():
        null_count = int(values.isna().sum())
        raise DividendImportError(f"date column {column_name!r} contains {null_count} null value(s).")
    return normalized


def _normalize_boundary_date(value: str | pd.Timestamp, *, label: str) -> str:
    try:
        parsed = pd.to_datetime(value, errors="raise", format="mixed", utc=True)
    except Exception as exc:
        raise DividendImportError(f"dividend import {label} date is not parseable: {value!r}.") from exc
    if pd.isna(parsed):
        raise DividendImportError(f"dividend import {label} date must not be null.")
    return pd.Timestamp(parsed).strftime("%Y-%m-%d")


def _normalize_raw_payload(value: Any) -> Any:
    if value is None:
        return pd.NA
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return pd.NA
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return stripped
        return json.dumps(parsed, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if isinstance(value, Mapping) or isinstance(value, list | tuple):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if pd.isna(value):
        return pd.NA
    return str(value)


def _validate_or_filter_invalid_events(events: pd.DataFrame, *, strict: bool) -> tuple[pd.DataFrame, int]:
    try:
        validate_dividend_event_schema(events.drop(columns=["year"], errors="ignore"))
    except DividendContractError:
        if strict:
            raise
    else:
        return events.copy(), 0

    valid_indices: list[Any] = []
    invalid_count = 0
    for index, row in events.iterrows():
        row_frame = pd.DataFrame([row.drop(labels=["year"], errors="ignore")])
        try:
            validate_dividend_event_schema(row_frame)
        except DividendContractError:
            invalid_count += 1
        else:
            valid_indices.append(index)

    return events.loc[valid_indices].copy(), invalid_count


def _duplicate_event_count(events: pd.DataFrame) -> int:
    if events.empty:
        return 0
    duplicate_mask = events.duplicated(subset=list(DIVIDEND_PRIMARY_KEY_FIELDS), keep=False)
    return int(duplicate_mask.sum())


def _first_duplicate_key(events: pd.DataFrame) -> dict[str, Any]:
    duplicate_mask = events.duplicated(subset=list(DIVIDEND_PRIMARY_KEY_FIELDS), keep=False)
    if not duplicate_mask.any():
        return {}
    return events.loc[duplicate_mask, list(DIVIDEND_PRIMARY_KEY_FIELDS)].iloc[0].to_dict()


def _sort_dividend_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    return events.sort_values(list(DIVIDEND_SORT_COLUMNS), kind="mergesort").reset_index(drop=True)


def _sort_loaded_dividend_events(events: pd.DataFrame) -> pd.DataFrame:
    sort_columns = [column for column in DIVIDEND_SORT_COLUMNS if column in events.columns]
    if "year" in events.columns:
        sort_columns.append("year")
    return events.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
