"""Deterministic read-only query helpers for M29 catalog records."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.catalog.lineage import build_lineage_edges
from src.catalog.models import CatalogRecord


@dataclass(frozen=True)
class CatalogQuery:
    """Filter object for querying in-memory catalog records."""

    run_types: tuple[str, ...] | None = None
    statuses: tuple[str, ...] | None = None
    strategy_name: str | None = None
    portfolio_name: str | None = None
    allocator_name: str | None = None
    alpha_model_name: str | None = None
    regime_method: str | None = None
    campaign_id: str | None = None
    scenario_id: str | None = None
    record_family: str | None = None
    robustness_status: str | None = None
    wfe_status: str | None = None
    sample_size_status: str | None = None
    trade_count_status: str | None = None
    sensitivity_status: str | None = None
    fragility_status: str | None = None
    multiple_testing_status: str | None = None
    temporal_validation_status: str | None = None
    governance_status: str | None = None
    promotion_review_status: str | None = None
    review_status: str | None = None
    promotion_status: str | None = None
    validation_readiness_present: bool | None = None
    release_validation_present: bool | None = None
    validation_bundle_present: bool | None = None
    release_readiness_present: bool | None = None
    artifact_type: str | None = None
    evidence_type: str | None = None
    source_domain: str | None = None
    event_domain: str | None = None
    schema_version: str | None = None
    min_metric: tuple[str, float] | None = None
    max_metric: tuple[str, float] | None = None
    metric_equals: tuple[str, float] | None = None
    start_ts: str | None = None
    end_ts: str | None = None
    include_templates: bool = False
    include_unknown: bool = True


def query_catalog(
    records: Iterable[CatalogRecord],
    query: CatalogQuery | None = None,
) -> list[CatalogRecord]:
    """Return catalog records matching *query*, sorted deterministically."""
    q = query or CatalogQuery()
    return filter_catalog_records(
        records,
        run_types=q.run_types,
        statuses=q.statuses,
        strategy_name=q.strategy_name,
        portfolio_name=q.portfolio_name,
        allocator_name=q.allocator_name,
        alpha_model_name=q.alpha_model_name,
        regime_method=q.regime_method,
        campaign_id=q.campaign_id,
        scenario_id=q.scenario_id,
        record_family=q.record_family,
        robustness_status=q.robustness_status,
        wfe_status=q.wfe_status,
        sample_size_status=q.sample_size_status,
        trade_count_status=q.trade_count_status,
        sensitivity_status=q.sensitivity_status,
        fragility_status=q.fragility_status,
        multiple_testing_status=q.multiple_testing_status,
        temporal_validation_status=q.temporal_validation_status,
        governance_status=q.governance_status,
        promotion_review_status=q.promotion_review_status,
        review_status=q.review_status,
        promotion_status=q.promotion_status,
        validation_readiness_present=q.validation_readiness_present,
        release_validation_present=q.release_validation_present,
        validation_bundle_present=q.validation_bundle_present,
        release_readiness_present=q.release_readiness_present,
        artifact_type=q.artifact_type,
        evidence_type=q.evidence_type,
        source_domain=q.source_domain,
        event_domain=q.event_domain,
        schema_version=q.schema_version,
        min_metric=q.min_metric,
        max_metric=q.max_metric,
        metric_equals=q.metric_equals,
        start_ts=q.start_ts,
        end_ts=q.end_ts,
        include_templates=q.include_templates,
        include_unknown=q.include_unknown,
    )


def filter_catalog_records(
    records: Iterable[CatalogRecord],
    *,
    run_types: Iterable[str] | None = None,
    statuses: Iterable[str] | None = None,
    strategy_name: str | None = None,
    portfolio_name: str | None = None,
    allocator_name: str | None = None,
    alpha_model_name: str | None = None,
    regime_method: str | None = None,
    campaign_id: str | None = None,
    scenario_id: str | None = None,
    record_family: str | None = None,
    robustness_status: str | None = None,
    wfe_status: str | None = None,
    sample_size_status: str | None = None,
    trade_count_status: str | None = None,
    sensitivity_status: str | None = None,
    fragility_status: str | None = None,
    multiple_testing_status: str | None = None,
    temporal_validation_status: str | None = None,
    governance_status: str | None = None,
    promotion_review_status: str | None = None,
    review_status: str | None = None,
    promotion_status: str | None = None,
    validation_readiness_present: bool | None = None,
    release_validation_present: bool | None = None,
    validation_bundle_present: bool | None = None,
    release_readiness_present: bool | None = None,
    artifact_type: str | None = None,
    evidence_type: str | None = None,
    source_domain: str | None = None,
    event_domain: str | None = None,
    schema_version: str | None = None,
    min_metric: tuple[str, float] | None = None,
    max_metric: tuple[str, float] | None = None,
    metric_equals: tuple[str, float] | None = None,
    start_ts: str | None = None,
    end_ts: str | None = None,
    include_templates: bool = False,
    include_unknown: bool = True,
) -> list[CatalogRecord]:
    """Filter records without mutating them.

    Name and ID filters are exact string matches. Time filters use lexicographic
    comparison of ISO-like strings, matching the M29 query limitation.
    """
    run_type_set = set(run_types) if run_types is not None else None
    status_set = set(statuses) if statuses is not None else None
    resolved_validation_readiness_present = _coalesce_bool_filter(
        validation_readiness_present,
        validation_bundle_present,
        alias="validation_bundle_present",
        canonical="validation_readiness_present",
    )
    resolved_release_validation_present = _coalesce_bool_filter(
        release_validation_present,
        release_readiness_present,
        alias="release_readiness_present",
        canonical="release_validation_present",
    )
    results: list[CatalogRecord] = []

    for record in records:
        if not include_templates and record.run_type == "portfolio_template":
            continue
        if not include_unknown and (record.run_type == "unknown" or record.status == "unknown"):
            continue
        if run_type_set is not None and record.run_type not in run_type_set:
            continue
        if status_set is not None and record.status not in status_set:
            continue
        if strategy_name is not None and record.strategy_name != strategy_name:
            continue
        if portfolio_name is not None and record.portfolio_name != portfolio_name:
            continue
        if allocator_name is not None and record.allocator_name != allocator_name:
            continue
        if alpha_model_name is not None and record.alpha_model_name != alpha_model_name:
            continue
        if regime_method is not None and record.regime_method != regime_method:
            continue
        if campaign_id is not None and record.campaign_id != campaign_id:
            continue
        if scenario_id is not None and record.scenario_id != scenario_id:
            continue
        if record_family is not None and record.record_family != record_family:
            continue
        if robustness_status is not None and record.robustness_status != robustness_status:
            continue
        if wfe_status is not None and record.wfe_status != wfe_status:
            continue
        if sample_size_status is not None and record.sample_size_status != sample_size_status:
            continue
        if trade_count_status is not None and record.trade_count_status != trade_count_status:
            continue
        if sensitivity_status is not None and record.sensitivity_status != sensitivity_status:
            continue
        if fragility_status is not None and record.fragility_status != fragility_status:
            continue
        if multiple_testing_status is not None and record.multiple_testing_status != multiple_testing_status:
            continue
        if temporal_validation_status is not None and record.temporal_validation_status != temporal_validation_status:
            continue
        if governance_status is not None and record.governance_status != governance_status:
            continue
        if promotion_review_status is not None and record.promotion_review_status != promotion_review_status:
            continue
        if review_status is not None and record.review_status != review_status:
            continue
        if promotion_status is not None and record.promotion_status != promotion_status:
            continue
        if (
            resolved_validation_readiness_present is not None
            and record.validation_readiness_present is not resolved_validation_readiness_present
        ):
            continue
        if (
            resolved_release_validation_present is not None
            and record.release_validation_present is not resolved_release_validation_present
        ):
            continue
        if artifact_type is not None and _evidence_value(record, "artifact_type") != artifact_type:
            continue
        if evidence_type is not None and _evidence_value(record, "evidence_type") != evidence_type:
            continue
        if source_domain is not None and _evidence_value(record, "source_domain") != source_domain:
            continue
        if event_domain is not None and _evidence_value(record, "event_domain") != event_domain:
            continue
        if schema_version is not None and _evidence_value(record, "schema_version") != schema_version:
            continue
        if start_ts is not None and (record.start_ts is None or record.start_ts < start_ts):
            continue
        if end_ts is not None and (record.end_ts is None or record.end_ts > end_ts):
            continue
        if min_metric is not None and not _metric_at_least(record, *min_metric):
            continue
        if max_metric is not None and not _metric_at_most(record, *max_metric):
            continue
        if metric_equals is not None and not _metric_equals(record, *metric_equals):
            continue
        results.append(record)

    return _sort_records(results)


def summarize_catalog(records: Iterable[CatalogRecord]) -> dict[str, object]:
    """Return deterministic summary counts for catalog records."""
    record_list = list(records)
    by_run_type = Counter(record.run_type for record in record_list)
    by_status = Counter(record.status for record in record_list)
    by_run_type_status = Counter((record.run_type, record.status) for record in record_list)
    return {
        "total_count": len(record_list),
        "by_run_type": {key: by_run_type[key] for key in sorted(by_run_type)},
        "by_status": {key: by_status[key] for key in sorted(by_status)},
        "by_run_type_status": {
            f"{run_type}:{status}": by_run_type_status[(run_type, status)]
            for run_type, status in sorted(by_run_type_status)
        },
    }


def related_records(
    record: CatalogRecord,
    records: Iterable[CatalogRecord],
    *,
    direction: str = "both",
    edge_types: Iterable[str] | None = None,
    repo_root: str | Path | None = None,
) -> list[CatalogRecord]:
    """Return records related to *record* through lineage edges."""
    if direction not in {"upstream", "downstream", "both"}:
        raise ValueError("direction must be 'upstream', 'downstream', or 'both'")

    record_list = list(records)
    edge_type_set = set(edge_types) if edge_types is not None else None
    by_catalog_id = {item.catalog_id: item for item in record_list}
    related: dict[str, CatalogRecord] = {}

    for edge in build_lineage_edges(record_list, repo_root=repo_root):
        if edge_type_set is not None and edge.edge_type not in edge_type_set:
            continue
        if direction in {"upstream", "both"} and edge.target_catalog_id == record.catalog_id:
            source = by_catalog_id.get(edge.source_catalog_id or "")
            if source is not None:
                related[source.catalog_id] = source
        if direction in {"downstream", "both"} and edge.source_catalog_id == record.catalog_id:
            target = by_catalog_id.get(edge.target_catalog_id or "")
            if target is not None:
                related[target.catalog_id] = target

    return _sort_records(related.values())


def get_upstream_records(
    record: CatalogRecord,
    records: Iterable[CatalogRecord],
    *,
    edge_types: Iterable[str] | None = None,
    repo_root: str | Path | None = None,
) -> list[CatalogRecord]:
    """Return records feeding into *record*."""
    return related_records(record, records, direction="upstream", edge_types=edge_types, repo_root=repo_root)


def get_downstream_records(
    record: CatalogRecord,
    records: Iterable[CatalogRecord],
    *,
    edge_types: Iterable[str] | None = None,
    repo_root: str | Path | None = None,
) -> list[CatalogRecord]:
    """Return records that depend on *record*."""
    return related_records(record, records, direction="downstream", edge_types=edge_types, repo_root=repo_root)


def records_to_dicts(records: Iterable[CatalogRecord]) -> list[dict[str, object]]:
    """Serialize records to deterministic dictionaries."""
    return [_json_like(record.to_dict()) for record in _sort_records(records)]


def records_to_rows(records: Iterable[CatalogRecord]) -> list[dict[str, object]]:
    """Serialize records to stable table-friendly rows."""
    rows: list[dict[str, object]] = []
    for record in _sort_records(records):
        rows.append(
            {
                "catalog_id": record.catalog_id,
                "run_id": record.run_id,
                "run_type": record.run_type,
                "status": record.status,
                "artifact_root": record.artifact_root,
                "strategy_name": record.strategy_name,
                "portfolio_name": record.portfolio_name,
                "allocator_name": record.allocator_name,
                "alpha_model_name": record.alpha_model_name,
                "timeframe": record.timeframe,
                "start_ts": record.start_ts,
                "end_ts": record.end_ts,
                "review_status": record.review_status,
                "promotion_status": record.promotion_status,
                "record_family": record.record_family,
                "robustness_status": record.robustness_status,
                "wfe_status": record.wfe_status,
                "sample_size_status": record.sample_size_status,
                "trade_count_status": record.trade_count_status,
                "sensitivity_status": record.sensitivity_status,
                "fragility_status": record.fragility_status,
                "multiple_testing_status": record.multiple_testing_status,
                "temporal_validation_status": record.temporal_validation_status,
                "governance_status": record.governance_status,
                "promotion_review_status": record.promotion_review_status,
                "validation_readiness_present": record.validation_readiness_present,
                "release_validation_present": record.release_validation_present,
                "artifact_type": _evidence_value(record, "artifact_type"),
                "evidence_type": _evidence_value(record, "evidence_type"),
                "source_domain": _evidence_value(record, "source_domain"),
                "event_domain": _evidence_value(record, "event_domain"),
                "schema_version": _evidence_value(record, "schema_version"),
            }
        )
    return rows


def _coalesce_bool_filter(
    canonical_value: bool | None,
    alias_value: bool | None,
    *,
    alias: str,
    canonical: str,
) -> bool | None:
    if canonical_value is not None and alias_value is not None and canonical_value != alias_value:
        raise ValueError(f"{alias} conflicts with {canonical}")
    return canonical_value if canonical_value is not None else alias_value


def _metric_at_least(record: CatalogRecord, name: str, value: float) -> bool:
    metric = _metric_value(record, name)
    return metric is not None and metric >= value


def _metric_at_most(record: CatalogRecord, name: str, value: float) -> bool:
    metric = _metric_value(record, name)
    return metric is not None and metric <= value


def _metric_equals(record: CatalogRecord, name: str, value: float) -> bool:
    metric = _metric_value(record, name)
    return metric is not None and metric == value


def _metric_value(record: CatalogRecord, name: str) -> float | None:
    metrics = record.metrics_summary or {}
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _evidence_value(record: CatalogRecord, name: str) -> str | None:
    evidence = record.metadata.get("evidence") if isinstance(record.metadata, dict) else None
    if not isinstance(evidence, dict):
        return None
    value = evidence.get(name)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _sort_records(records: Iterable[CatalogRecord]) -> list[CatalogRecord]:
    return sorted(
        records,
        key=lambda record: (
            record.run_type,
            record.run_id or "",
            record.catalog_id,
            record.artifact_root,
        ),
    )


def _json_like(value: dict[str, Any]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key in sorted(value):
        item = value[key]
        if isinstance(item, dict):
            result[key] = _json_like(item)
        elif isinstance(item, list):
            result[key] = [_json_like(v) if isinstance(v, dict) else v for v in item]
        else:
            result[key] = item
    return result
