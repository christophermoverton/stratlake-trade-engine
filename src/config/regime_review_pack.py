from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.research.registry import canonicalize_value


class RegimeReviewPackConfigError(ValueError):
    """Raised when a regime review-pack configuration is malformed."""


_ROOT_KEYS = frozenset({"review_name", "benchmark_path", "promotion_gates_path", "output_root", "ranking", "report"})
_RANKING_KEYS = frozenset({"decision_order", "metric_priority", "tie_breakers"})
_REPORT_KEYS = frozenset(
    {"write_markdown", "include_rejected_candidates", "include_warning_summary", "include_evidence_index"}
)
_VALID_DECISIONS = frozenset({"accepted", "accepted_with_warnings", "needs_review", "rejected"})
_DEFAULT_DECISION_ORDER = ("accepted", "accepted_with_warnings", "needs_review", "rejected")
_DEFAULT_METRIC_PRIORITY = (
    "adaptive_vs_static_sharpe_delta",
    "adaptive_vs_static_return_delta",
    "adaptive_vs_static_max_drawdown_delta",
    "mean_confidence",
    "transition_rate",
    "policy_turnover",
)
_DEFAULT_TIE_BREAKERS = ("variant_name",)
_SUPPORTED_METRICS = frozenset(
    {
        "adaptive_vs_static_sharpe_delta",
        "adaptive_vs_static_return_delta",
        "adaptive_vs_static_max_drawdown_delta",
        "mean_confidence",
        "mean_entropy",
        "transition_rate",
        "avg_regime_duration",
        "dominant_regime_share",
        "policy_turnover",
        "variant_name",
    }
)


@dataclass(frozen=True)
class RegimeReviewRankingConfig:
    decision_order: tuple[str, ...] = _DEFAULT_DECISION_ORDER
    metric_priority: tuple[str, ...] = _DEFAULT_METRIC_PRIORITY
    tie_breakers: tuple[str, ...] = _DEFAULT_TIE_BREAKERS

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_order": list(self.decision_order),
            "metric_priority": list(self.metric_priority),
            "tie_breakers": list(self.tie_breakers),
        }


@dataclass(frozen=True)
class RegimeReviewReportConfig:
    write_markdown: bool = True
    include_rejected_candidates: bool = True
    include_warning_summary: bool = True
    include_evidence_index: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "write_markdown": self.write_markdown,
            "include_rejected_candidates": self.include_rejected_candidates,
            "include_warning_summary": self.include_warning_summary,
            "include_evidence_index": self.include_evidence_index,
        }


@dataclass(frozen=True)
class RegimeReviewPackConfig:
    review_name: str
    benchmark_path: str
    promotion_gates_path: str
    output_root: str = "artifacts/regime_reviews"
    ranking: RegimeReviewRankingConfig = field(default_factory=RegimeReviewRankingConfig)
    report: RegimeReviewReportConfig = field(default_factory=RegimeReviewReportConfig)

    def to_dict(self) -> dict[str, Any]:
        return canonicalize_value(
            {
                "review_name": self.review_name,
                "benchmark_path": Path(self.benchmark_path).as_posix(),
                "promotion_gates_path": Path(self.promotion_gates_path).as_posix(),
                "output_root": Path(self.output_root).as_posix(),
                "ranking": self.ranking.to_dict(),
                "report": self.report.to_dict(),
            }
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RegimeReviewPackConfig":
        unknown_keys = sorted(set(payload) - _ROOT_KEYS)
        if unknown_keys:
            raise RegimeReviewPackConfigError(
                f"Regime review-pack config contains unsupported keys: {unknown_keys}."
            )
        return cls(
            review_name=_normalize_required_string(payload.get("review_name"), field_name="review_name"),
            benchmark_path=_normalize_path_string(payload.get("benchmark_path"), field_name="benchmark_path"),
            promotion_gates_path=_normalize_path_string(
                payload.get("promotion_gates_path"), field_name="promotion_gates_path"
            ),
            output_root=_normalize_path_string(payload.get("output_root", "artifacts/regime_reviews"), field_name="output_root"),
            ranking=_resolve_ranking(payload.get("ranking")),
            report=_resolve_report(payload.get("report")),
        )


def load_regime_review_pack_config(path: Path) -> RegimeReviewPackConfig:
    if not path.exists():
        raise RegimeReviewPackConfigError(f"Regime review-pack config does not exist: {path.as_posix()}")
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() == ".json":
            payload = json.load(handle)
        elif path.suffix.lower() in {".yml", ".yaml"}:
            payload = yaml.safe_load(handle) or {}
        else:
            raise RegimeReviewPackConfigError(
                f"Unsupported regime review-pack config format {path.suffix!r}. Use JSON, YAML, or YML."
            )
    if not isinstance(payload, Mapping):
        raise RegimeReviewPackConfigError("Regime review-pack config must contain a top-level mapping.")
    return RegimeReviewPackConfig.from_mapping(payload)


def _resolve_ranking(value: Any) -> RegimeReviewRankingConfig:
    if value is None:
        return RegimeReviewRankingConfig()
    if not isinstance(value, Mapping):
        raise RegimeReviewPackConfigError("Regime review-pack field 'ranking' must be a mapping.")
    unknown_keys = sorted(set(value) - _RANKING_KEYS)
    if unknown_keys:
        raise RegimeReviewPackConfigError(f"Regime review-pack field 'ranking' contains unsupported keys: {unknown_keys}.")
    decision_order = _normalize_string_sequence(value.get("decision_order", _DEFAULT_DECISION_ORDER), field_name="ranking.decision_order")
    if set(decision_order) != _VALID_DECISIONS or len(decision_order) != len(_VALID_DECISIONS):
        expected = ", ".join(_DEFAULT_DECISION_ORDER)
        raise RegimeReviewPackConfigError(f"Regime review-pack field 'ranking.decision_order' must contain exactly: {expected}.")
    metric_priority = _normalize_string_sequence(value.get("metric_priority", _DEFAULT_METRIC_PRIORITY), field_name="ranking.metric_priority")
    _validate_supported(metric_priority, field_name="ranking.metric_priority")
    tie_breakers = _normalize_string_sequence(value.get("tie_breakers", _DEFAULT_TIE_BREAKERS), field_name="ranking.tie_breakers")
    _validate_supported(tie_breakers, field_name="ranking.tie_breakers")
    return RegimeReviewRankingConfig(decision_order=decision_order, metric_priority=metric_priority, tie_breakers=tie_breakers)


def _resolve_report(value: Any) -> RegimeReviewReportConfig:
    if value is None:
        return RegimeReviewReportConfig()
    if not isinstance(value, Mapping):
        raise RegimeReviewPackConfigError("Regime review-pack field 'report' must be a mapping.")
    unknown_keys = sorted(set(value) - _REPORT_KEYS)
    if unknown_keys:
        raise RegimeReviewPackConfigError(f"Regime review-pack field 'report' contains unsupported keys: {unknown_keys}.")
    return RegimeReviewReportConfig(
        write_markdown=_coerce_bool(value.get("write_markdown", True), field_name="report.write_markdown"),
        include_rejected_candidates=_coerce_bool(
            value.get("include_rejected_candidates", True), field_name="report.include_rejected_candidates"
        ),
        include_warning_summary=_coerce_bool(value.get("include_warning_summary", True), field_name="report.include_warning_summary"),
        include_evidence_index=_coerce_bool(value.get("include_evidence_index", True), field_name="report.include_evidence_index"),
    )


def _validate_supported(values: tuple[str, ...], *, field_name: str) -> None:
    unknown = sorted(set(values) - _SUPPORTED_METRICS)
    if unknown:
        expected = ", ".join(sorted(_SUPPORTED_METRICS))
        raise RegimeReviewPackConfigError(f"Regime review-pack field '{field_name}' contains unsupported metrics {unknown}. Expected one of: {expected}.")


def _normalize_required_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise RegimeReviewPackConfigError(f"Regime review-pack field '{field_name}' must be a non-empty string.")
    normalized = value.strip()
    if not normalized:
        raise RegimeReviewPackConfigError(f"Regime review-pack field '{field_name}' must be a non-empty string.")
    return normalized


def _normalize_path_string(value: Any, *, field_name: str) -> str:
    return Path(_normalize_required_string(value, field_name=field_name)).as_posix()


def _normalize_string_sequence(value: Any, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise RegimeReviewPackConfigError(f"Regime review-pack field '{field_name}' must be a sequence.")
    return tuple(_normalize_required_string(item, field_name=field_name) for item in value)


def _coerce_bool(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise RegimeReviewPackConfigError(f"Regime review-pack field '{field_name}' must be boolean.")
    return value


__all__ = [
    "RegimeReviewPackConfig",
    "RegimeReviewPackConfigError",
    "RegimeReviewRankingConfig",
    "RegimeReviewReportConfig",
    "load_regime_review_pack_config",
]
