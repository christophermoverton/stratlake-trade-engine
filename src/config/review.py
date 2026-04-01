from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
REVIEW_CONFIG = REPO_ROOT / "configs" / "review.yml"
SUPPORTED_REVIEW_RUN_TYPES = frozenset({"alpha_evaluation", "strategy", "portfolio"})
_REVIEW_SECTION_KEYS = frozenset({"filters", "ranking", "output", "promotion_gates"})
_FILTER_KEYS = frozenset(
    {"run_types", "timeframe", "dataset", "alpha_name", "strategy_name", "portfolio_name", "top_k_per_type"}
)
_RANKING_KEYS = frozenset(
    {
        "alpha_evaluation_primary_metric",
        "alpha_evaluation_secondary_metric",
        "strategy_primary_metric",
        "strategy_secondary_metric",
        "portfolio_primary_metric",
        "portfolio_secondary_metric",
    }
)
_OUTPUT_KEYS = frozenset({"path", "emit_plots"})


class ReviewConfigError(ValueError):
    """Raised when review configuration is malformed."""


@dataclass(frozen=True)
class ReviewFiltersConfig:
    run_types: list[str]
    timeframe: str | None
    dataset: str | None
    alpha_name: str | None
    strategy_name: str | None
    portfolio_name: str | None
    top_k_per_type: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_types": list(self.run_types),
            "timeframe": self.timeframe,
            "dataset": self.dataset,
            "alpha_name": self.alpha_name,
            "strategy_name": self.strategy_name,
            "portfolio_name": self.portfolio_name,
            "top_k_per_type": self.top_k_per_type,
        }


@dataclass(frozen=True)
class ReviewRankingConfig:
    alpha_evaluation_primary_metric: str
    alpha_evaluation_secondary_metric: str
    strategy_primary_metric: str
    strategy_secondary_metric: str
    portfolio_primary_metric: str
    portfolio_secondary_metric: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha_evaluation_primary_metric": self.alpha_evaluation_primary_metric,
            "alpha_evaluation_secondary_metric": self.alpha_evaluation_secondary_metric,
            "strategy_primary_metric": self.strategy_primary_metric,
            "strategy_secondary_metric": self.strategy_secondary_metric,
            "portfolio_primary_metric": self.portfolio_primary_metric,
            "portfolio_secondary_metric": self.portfolio_secondary_metric,
        }


@dataclass(frozen=True)
class ReviewOutputConfig:
    path: str | None
    emit_plots: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "emit_plots": self.emit_plots,
        }


@dataclass(frozen=True)
class ReviewConfig:
    filters: ReviewFiltersConfig
    ranking: ReviewRankingConfig
    output: ReviewOutputConfig
    promotion_gates: dict[str, Any] | None

    @classmethod
    def default(cls) -> "ReviewConfig":
        return cls(
            filters=ReviewFiltersConfig(
                run_types=["alpha_evaluation", "strategy", "portfolio"],
                timeframe=None,
                dataset=None,
                alpha_name=None,
                strategy_name=None,
                portfolio_name=None,
                top_k_per_type=None,
            ),
            ranking=ReviewRankingConfig(
                alpha_evaluation_primary_metric="ic_ir",
                alpha_evaluation_secondary_metric="mean_ic",
                strategy_primary_metric="sharpe_ratio",
                strategy_secondary_metric="total_return",
                portfolio_primary_metric="sharpe_ratio",
                portfolio_secondary_metric="total_return",
            ),
            output=ReviewOutputConfig(path=None, emit_plots=True),
            promotion_gates=None,
        )

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any] | None,
        *,
        base: "ReviewConfig" | None = None,
    ) -> "ReviewConfig":
        if payload is None:
            return base or cls.default()
        if not isinstance(payload, Mapping):
            raise ReviewConfigError("Review configuration must be a mapping.")

        seed = base or cls.default()
        container, promotion_payload = _extract_review_container(payload)
        filters = _resolve_filters(container.get("filters"), base=seed.filters)
        ranking = _resolve_ranking(container.get("ranking"), base=seed.ranking)
        output = _resolve_output(container.get("output"), base=seed.output)
        if promotion_payload is None:
            promotion_gates = seed.promotion_gates
        else:
            promotion_gates = _resolve_promotion_gates(promotion_payload)
        return cls(filters=filters, ranking=ranking, output=output, promotion_gates=promotion_gates)

    def to_dict(self) -> dict[str, Any]:
        return _canonicalize_value(
            {
                "filters": self.filters.to_dict(),
                "ranking": self.ranking.to_dict(),
                "output": self.output.to_dict(),
                "promotion_gates": self.promotion_gates,
            }
        )

    def identity_payload(self) -> dict[str, Any]:
        payload = self.to_dict()
        output = dict(payload["output"])
        output.pop("path", None)
        payload["output"] = output
        return _canonicalize_value(payload)


def load_review_config(path: Path = REVIEW_CONFIG) -> ReviewConfig:
    """Load repository review defaults from YAML/JSON, or return code defaults when absent."""

    if not path.exists():
        return ReviewConfig.default()

    with path.open("r", encoding="utf-8") as handle:
        suffix = path.suffix.lower()
        if suffix == ".json":
            payload = json.load(handle)
        elif suffix in {".yaml", ".yml"}:
            payload = yaml.safe_load(handle) or {}
        else:
            raise ReviewConfigError(
                f"Unsupported review config format {path.suffix!r}. Use JSON, YAML, or YML."
            )
    if not isinstance(payload, Mapping):
        raise ReviewConfigError("Review config file must contain a top-level mapping.")
    return ReviewConfig.from_mapping(payload)


def resolve_review_config(
    *sources: Mapping[str, Any] | None,
    cli_overrides: Mapping[str, Any] | None = None,
    defaults_path: Path | None = None,
) -> ReviewConfig:
    """
    Resolve repository defaults, config values, and CLI overrides into one review contract.

    Precedence is deterministic: repository defaults < config < CLI overrides.
    """

    resolved = load_review_config(REVIEW_CONFIG if defaults_path is None else defaults_path)
    for source in sources:
        if source is None:
            continue
        resolved = ReviewConfig.from_mapping(source, base=resolved)
    if cli_overrides is not None:
        resolved = ReviewConfig.from_mapping(cli_overrides, base=resolved)
    return resolved


def _extract_review_container(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], Any]:
    if "review" in payload:
        container = payload.get("review")
        if not isinstance(container, Mapping):
            raise ReviewConfigError("Review configuration field 'review' must be a mapping.")
        unknown_keys = sorted(set(container) - _REVIEW_SECTION_KEYS)
        if unknown_keys:
            raise ReviewConfigError(f"Review configuration field 'review' contains unsupported keys: {unknown_keys}.")
        promotion_payload = container.get("promotion_gates", payload.get("promotion_gates"))
        return container, promotion_payload

    unknown_keys = sorted(set(payload) - _REVIEW_SECTION_KEYS)
    if unknown_keys:
        raise ReviewConfigError(f"Review configuration contains unsupported keys: {unknown_keys}.")
    return payload, payload.get("promotion_gates")


def _resolve_filters(payload: Any, *, base: ReviewFiltersConfig) -> ReviewFiltersConfig:
    if payload is None:
        return base
    if not isinstance(payload, Mapping):
        raise ReviewConfigError("Review configuration field 'filters' must be a mapping.")
    unknown_keys = sorted(set(payload) - _FILTER_KEYS)
    if unknown_keys:
        raise ReviewConfigError(f"Review configuration field 'filters' contains unsupported keys: {unknown_keys}.")

    return ReviewFiltersConfig(
        run_types=_resolve_run_types(payload.get("run_types"), base.run_types),
        timeframe=_normalize_optional_string(payload.get("timeframe"), field_name="filters.timeframe", default=base.timeframe),
        dataset=_normalize_optional_string(payload.get("dataset"), field_name="filters.dataset", default=base.dataset),
        alpha_name=_normalize_optional_string(payload.get("alpha_name"), field_name="filters.alpha_name", default=base.alpha_name),
        strategy_name=_normalize_optional_string(
            payload.get("strategy_name"),
            field_name="filters.strategy_name",
            default=base.strategy_name,
        ),
        portfolio_name=_normalize_optional_string(
            payload.get("portfolio_name"),
            field_name="filters.portfolio_name",
            default=base.portfolio_name,
        ),
        top_k_per_type=_resolve_positive_int(
            payload.get("top_k_per_type"),
            field_name="filters.top_k_per_type",
            default=base.top_k_per_type,
        ),
    )


def _resolve_ranking(payload: Any, *, base: ReviewRankingConfig) -> ReviewRankingConfig:
    if payload is None:
        return base
    if not isinstance(payload, Mapping):
        raise ReviewConfigError("Review configuration field 'ranking' must be a mapping.")
    unknown_keys = sorted(set(payload) - _RANKING_KEYS)
    if unknown_keys:
        raise ReviewConfigError(f"Review configuration field 'ranking' contains unsupported keys: {unknown_keys}.")

    return ReviewRankingConfig(
        alpha_evaluation_primary_metric=_normalize_required_string(
            payload.get("alpha_evaluation_primary_metric", base.alpha_evaluation_primary_metric),
            field_name="ranking.alpha_evaluation_primary_metric",
        ),
        alpha_evaluation_secondary_metric=_normalize_required_string(
            payload.get("alpha_evaluation_secondary_metric", base.alpha_evaluation_secondary_metric),
            field_name="ranking.alpha_evaluation_secondary_metric",
        ),
        strategy_primary_metric=_normalize_required_string(
            payload.get("strategy_primary_metric", base.strategy_primary_metric),
            field_name="ranking.strategy_primary_metric",
        ),
        strategy_secondary_metric=_normalize_required_string(
            payload.get("strategy_secondary_metric", base.strategy_secondary_metric),
            field_name="ranking.strategy_secondary_metric",
        ),
        portfolio_primary_metric=_normalize_required_string(
            payload.get("portfolio_primary_metric", base.portfolio_primary_metric),
            field_name="ranking.portfolio_primary_metric",
        ),
        portfolio_secondary_metric=_normalize_required_string(
            payload.get("portfolio_secondary_metric", base.portfolio_secondary_metric),
            field_name="ranking.portfolio_secondary_metric",
        ),
    )


def _resolve_output(payload: Any, *, base: ReviewOutputConfig) -> ReviewOutputConfig:
    if payload is None:
        return base
    if not isinstance(payload, Mapping):
        raise ReviewConfigError("Review configuration field 'output' must be a mapping.")
    unknown_keys = sorted(set(payload) - _OUTPUT_KEYS)
    if unknown_keys:
        raise ReviewConfigError(f"Review configuration field 'output' contains unsupported keys: {unknown_keys}.")

    emit_plots = payload.get("emit_plots", base.emit_plots)
    if not isinstance(emit_plots, bool):
        raise ReviewConfigError("Review configuration field 'output.emit_plots' must be a boolean.")
    return ReviewOutputConfig(
        path=_normalize_optional_string(payload.get("path"), field_name="output.path", default=base.path),
        emit_plots=emit_plots,
    )


def _resolve_promotion_gates(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ReviewConfigError("Review configuration field 'promotion_gates' must be a mapping.")
    promotion_payload = _canonicalize_value(dict(payload))
    if "gates" not in promotion_payload:
        raise ReviewConfigError("Review configuration field 'promotion_gates' must define a 'gates' list.")
    return promotion_payload


def _resolve_run_types(value: Any, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    if not isinstance(value, list):
        raise ReviewConfigError("Review configuration field 'filters.run_types' must be a list of strings.")

    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ReviewConfigError("Review configuration field 'filters.run_types' must contain only strings.")
        text = item.strip().lower()
        if text not in SUPPORTED_REVIEW_RUN_TYPES:
            formatted = ", ".join(sorted(SUPPORTED_REVIEW_RUN_TYPES))
            raise ReviewConfigError(
                f"Review configuration field 'filters.run_types' must contain only supported values: {formatted}."
            )
        if text not in normalized:
            normalized.append(text)
    if not normalized:
        raise ReviewConfigError(
            "Review configuration field 'filters.run_types' must contain at least one supported run type."
        )
    return normalized


def _resolve_positive_int(value: Any, *, field_name: str, default: int | None) -> int | None:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ReviewConfigError(f"Review configuration field '{field_name}' must be a positive integer.")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ReviewConfigError(f"Review configuration field '{field_name}' must be a positive integer.") from exc
    if integer <= 0:
        raise ReviewConfigError(f"Review configuration field '{field_name}' must be a positive integer.")
    return integer


def _normalize_required_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ReviewConfigError(f"Review configuration field '{field_name}' must be a non-empty string.")
    normalized = value.strip()
    if not normalized:
        raise ReviewConfigError(f"Review configuration field '{field_name}' must be a non-empty string.")
    return normalized


def _normalize_optional_string(value: Any, *, field_name: str, default: str | None) -> str | None:
    if value is None:
        return default
    if not isinstance(value, str):
        raise ReviewConfigError(f"Review configuration field '{field_name}' must be a string when provided.")
    normalized = value.strip()
    return normalized or None


def _canonicalize_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _canonicalize_value(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonicalize_value(item) for item in value]
    return value
