from __future__ import annotations

from dataclasses import dataclass
from typing import Any

TAXONOMY_VERSION = "regime_taxonomy_v1"
UNDEFINED_REGIME_LABEL = "undefined"

VOLATILITY_DIMENSION = "volatility"
TREND_DIMENSION = "trend"
DRAWDOWN_RECOVERY_DIMENSION = "drawdown_recovery"
STRESS_DIMENSION = "stress"

REGIME_DIMENSIONS: tuple[str, ...] = (
    VOLATILITY_DIMENSION,
    TREND_DIMENSION,
    DRAWDOWN_RECOVERY_DIMENSION,
    STRESS_DIMENSION,
)

REGIME_STATE_COLUMNS: dict[str, str] = {
    VOLATILITY_DIMENSION: "volatility_state",
    TREND_DIMENSION: "trend_state",
    DRAWDOWN_RECOVERY_DIMENSION: "drawdown_recovery_state",
    STRESS_DIMENSION: "stress_state",
}

REGIME_METRIC_COLUMNS: dict[str, tuple[str, ...]] = {
    VOLATILITY_DIMENSION: ("volatility_metric",),
    TREND_DIMENSION: ("trend_metric",),
    DRAWDOWN_RECOVERY_DIMENSION: ("drawdown_metric",),
    STRESS_DIMENSION: ("stress_correlation_metric", "stress_dispersion_metric"),
}

REGIME_LABEL_COLUMNS: tuple[str, ...] = (
    "ts_utc",
    "volatility_state",
    "trend_state",
    "drawdown_recovery_state",
    "stress_state",
    "regime_label",
    "is_defined",
)

REGIME_AUDIT_COLUMNS: tuple[str, ...] = (
    "volatility_metric",
    "trend_metric",
    "drawdown_metric",
    "stress_correlation_metric",
    "stress_dispersion_metric",
)

REGIME_OUTPUT_COLUMNS: tuple[str, ...] = (*REGIME_LABEL_COLUMNS, *REGIME_AUDIT_COLUMNS)


@dataclass(frozen=True)
class RegimeStateDefinition:
    """Auditable definition for one deterministic regime state label."""

    label: str
    description: str

    def to_dict(self) -> dict[str, str]:
        return {"description": self.description, "label": self.label}


@dataclass(frozen=True)
class RegimeDimensionDefinition:
    """Canonical taxonomy entry for one regime dimension."""

    dimension: str
    state_column: str
    metric_columns: tuple[str, ...]
    states: tuple[RegimeStateDefinition, ...]
    purpose: str

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(state.label for state in self.states)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "metric_columns": list(self.metric_columns),
            "purpose": self.purpose,
            "state_column": self.state_column,
            "states": [state.to_dict() for state in self.states],
        }


REGIME_TAXONOMY: dict[str, RegimeDimensionDefinition] = {
    VOLATILITY_DIMENSION: RegimeDimensionDefinition(
        dimension=VOLATILITY_DIMENSION,
        state_column=REGIME_STATE_COLUMNS[VOLATILITY_DIMENSION],
        metric_columns=REGIME_METRIC_COLUMNS[VOLATILITY_DIMENSION],
        purpose="Bucket rolling realized market volatility into deterministic low, normal, and high states.",
        states=(
            RegimeStateDefinition(UNDEFINED_REGIME_LABEL, "Insufficient history or unusable returns."),
            RegimeStateDefinition("low_volatility", "Rolling realized volatility is at or below the low threshold."),
            RegimeStateDefinition("normal_volatility", "Rolling realized volatility is between the configured thresholds."),
            RegimeStateDefinition("high_volatility", "Rolling realized volatility is at or above the high threshold."),
        ),
    ),
    TREND_DIMENSION: RegimeDimensionDefinition(
        dimension=TREND_DIMENSION,
        state_column=REGIME_STATE_COLUMNS[TREND_DIMENSION],
        metric_columns=REGIME_METRIC_COLUMNS[TREND_DIMENSION],
        purpose="Classify rolling compounded market return into up, down, or sideways trend.",
        states=(
            RegimeStateDefinition(UNDEFINED_REGIME_LABEL, "Insufficient history or unusable returns."),
            RegimeStateDefinition("downtrend", "Rolling compounded return is at or below the negative trend threshold."),
            RegimeStateDefinition("sideways", "Rolling compounded return is inside the trend threshold band."),
            RegimeStateDefinition("uptrend", "Rolling compounded return is at or above the positive trend threshold."),
        ),
    ),
    DRAWDOWN_RECOVERY_DIMENSION: RegimeDimensionDefinition(
        dimension=DRAWDOWN_RECOVERY_DIMENSION,
        state_column=REGIME_STATE_COLUMNS[DRAWDOWN_RECOVERY_DIMENSION],
        metric_columns=REGIME_METRIC_COLUMNS[DRAWDOWN_RECOVERY_DIMENSION],
        purpose="Describe whether the market equity curve is near a high, underwater, in drawdown, or recovering.",
        states=(
            RegimeStateDefinition(UNDEFINED_REGIME_LABEL, "Current return cannot produce an auditable drawdown state."),
            RegimeStateDefinition("near_peak", "Drawdown is at or below the near-peak threshold."),
            RegimeStateDefinition("underwater", "Drawdown is positive but below the drawdown threshold and not improving."),
            RegimeStateDefinition("drawdown", "Drawdown is at or above the drawdown threshold and not improving."),
            RegimeStateDefinition("recovery", "Drawdown is positive and improving relative to the prior timestamp."),
        ),
    ),
    STRESS_DIMENSION: RegimeDimensionDefinition(
        dimension=STRESS_DIMENSION,
        state_column=REGIME_STATE_COLUMNS[STRESS_DIMENSION],
        metric_columns=REGIME_METRIC_COLUMNS[STRESS_DIMENSION],
        purpose="Flag cross-sectional correlation or dispersion stress when enough symbols are available.",
        states=(
            RegimeStateDefinition(UNDEFINED_REGIME_LABEL, "Insufficient cross-section, history, or usable returns."),
            RegimeStateDefinition("normal_stress", "Correlation and dispersion metrics are below stress thresholds."),
            RegimeStateDefinition("correlation_stress", "Average pairwise rolling correlation is at or above threshold."),
            RegimeStateDefinition("dispersion_stress", "Cross-sectional dispersion is at or above threshold."),
        ),
    ),
}


def taxonomy_payload() -> dict[str, Any]:
    """Return the stable, JSON-friendly taxonomy payload used in metadata."""

    return {
        "dimensions": {
            dimension: REGIME_TAXONOMY[dimension].to_dict()
            for dimension in REGIME_DIMENSIONS
        },
        "label_columns": list(REGIME_LABEL_COLUMNS),
        "metric_columns": list(REGIME_AUDIT_COLUMNS),
        "version": TAXONOMY_VERSION,
    }


__all__ = [
    "DRAWDOWN_RECOVERY_DIMENSION",
    "REGIME_AUDIT_COLUMNS",
    "REGIME_DIMENSIONS",
    "REGIME_LABEL_COLUMNS",
    "REGIME_METRIC_COLUMNS",
    "REGIME_OUTPUT_COLUMNS",
    "REGIME_STATE_COLUMNS",
    "REGIME_TAXONOMY",
    "RegimeDimensionDefinition",
    "RegimeStateDefinition",
    "STRESS_DIMENSION",
    "TAXONOMY_VERSION",
    "TREND_DIMENSION",
    "UNDEFINED_REGIME_LABEL",
    "VOLATILITY_DIMENSION",
    "taxonomy_payload",
]
