"""
Strategy Archetype Contract and Formal Definitions

This module formalizes the Strategy Archetype Library, establishing:
- Formal contracts for all strategies
- Mathematical definitions
- Input/output schema
- Signal semantics integration
- Execution semantics
- Failure modes and constraints

Each strategy is a first-class research primitive with explicit typed signals
and deterministic outputs suitable for reproducible research and production.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

import pandas as pd


class StrategyArchetype(str, Enum):
    """Canonical strategy archetypes in the StratLake library."""

    TIME_SERIES_MOMENTUM = "time_series_momentum"
    CROSS_SECTION_MOMENTUM = "cross_section_momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    PAIRS_TRADING = "pairs_trading"
    RESIDUAL_MOMENTUM = "residual_momentum"
    VOLATILITY_REGIME_MOMENTUM = "volatility_regime_momentum"
    WEIGHTED_CROSS_SECTION_ENSEMBLE = "weighted_cross_section_ensemble"


class RebalanceFrequency(str, Enum):
    """Supported rebalance frequencies."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class MissingDataHandling(str, Enum):
    """Strategy behavior for missing data."""

    SKIP_EPOCH = "skip_epoch"
    FORWARD_FILL = "forward_fill"
    NEUTRAL = "neutral"


@dataclass(frozen=True)
class InputSchemaField:
    """Definition of a single required input field."""

    name: str
    dtype: str
    required: bool = True
    description: str = ""
    examples: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class InputSchema:
    """Formal specification of strategy inputs."""

    required_columns: list[str]
    required_fields: list[InputSchemaField]
    cross_sectional_required: bool = False
    time_series_required: bool = True
    min_lookback_periods: int | None = None
    min_cross_section_size: int | None = None
    supported_timeframes: tuple[str, ...] = ("daily",)

    def validate(self, df: pd.DataFrame) -> None:
        """Validate a DataFrame against this schema."""
        missing = set(self.required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        if self.cross_sectional_required and "symbol" not in df.columns:
            raise ValueError("Cross-sectional data required but 'symbol' column not found.")

        if self.cross_sectional_required and self.min_cross_section_size:
            if "symbol" in df.columns:
                sizes = df.groupby("ts_utc")["symbol"].nunique()
                if (sizes < self.min_cross_section_size).any():
                    raise ValueError(
                        f"Insufficient cross-sectional size: minimum {self.min_cross_section_size} required, "
                        f"got minimum {sizes.min()}"
                    )


@dataclass(frozen=True)
class OutputSignalSchema:
    """Formal specification of signal output."""

    signal_type: str
    allowed_values: list[float] | None = None
    value_range: tuple[float, float] | None = None
    description: str = ""

    def validate(self, signal: pd.Series) -> None:
        """Validate a signal against this schema."""
        if self.allowed_values is not None:
            invalid = signal[~signal.isin(self.allowed_values)]
            if not invalid.empty:
                raise ValueError(
                    f"Signal contains invalid values for {self.signal_type}: {set(invalid.unique())}"
                )

        if self.value_range is not None:
            min_val, max_val = self.value_range
            out_of_range = signal[(signal < min_val) | (signal > max_val)]
            if not out_of_range.empty:
                raise ValueError(
                    f"Signal values outside range [{min_val}, {max_val}]: "
                    f"min={signal.min()}, max={signal.max()}"
                )


@dataclass(frozen=True)
class ExecutionSemantics:
    """Formal definition of execution behavior."""

    rebalance_frequency: RebalanceFrequency
    execution_lag_days: int = 1
    missing_data_handling: MissingDataHandling = MissingDataHandling.SKIP_EPOCH
    requires_forward_fill: bool = False
    ordering_stable: bool = True
    deterministic: bool = True
    description: str = ""


@dataclass(frozen=True)
class FailureMode:
    """Known failure mode and constraint."""

    name: str
    description: str
    regime: str | None = None
    sensitivity: str = "moderate"
    mitigation: str | None = None


@dataclass(frozen=True)
class StrategyDefinition:
    """
    Formal contract for a strategy archetype.

    This captures the complete definition of a strategy including:
    - Mathematical specification
    - Input/output contracts
    - Execution semantics
    - Known failure modes
    """

    strategy_id: str
    version: str
    archetype: StrategyArchetype
    name: str
    description: str
    mathematical_definition: str
    research_hypothesis: str
    input_schema: InputSchema
    output_signal_schema: OutputSignalSchema
    execution_semantics: ExecutionSemantics
    failure_modes: tuple[FailureMode, ...] = ()
    assumptions: tuple[str, ...] = ()
    regime_dependencies: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    paper_references: tuple[str, ...] = ()

    def to_registry_entry(self) -> dict[str, Any]:
        """Convert to registry JSON entry."""
        return {
            "strategy_id": self.strategy_id,
            "version": self.version,
            "status": "active",
            "archetype": self.archetype.value,
            "name": self.name,
            "description": self.description,
            "mathematical_definition": self.mathematical_definition,
            "research_hypothesis": self.research_hypothesis,
            "input_schema": {
                "required_columns": self.input_schema.required_columns,
                "cross_sectional_required": self.input_schema.cross_sectional_required,
                "time_series_required": self.input_schema.time_series_required,
                "min_lookback_periods": self.input_schema.min_lookback_periods,
                "min_cross_section_size": self.input_schema.min_cross_section_size,
                "supported_timeframes": list(self.input_schema.supported_timeframes),
            },
            "output_signal": {
                "signal_type": self.output_signal_schema.signal_type,
                "allowed_values": self.output_signal_schema.allowed_values,
                "value_range": self.output_signal_schema.value_range,
            },
            "execution_semantics": {
                "rebalance_frequency": self.execution_semantics.rebalance_frequency.value,
                "execution_lag_days": self.execution_semantics.execution_lag_days,
                "missing_data_handling": self.execution_semantics.missing_data_handling.value,
                "deterministic": self.execution_semantics.deterministic,
            },
            "failure_modes": [
                {
                    "name": fm.name,
                    "description": fm.description,
                    "regime": fm.regime,
                    "sensitivity": fm.sensitivity,
                    "mitigation": fm.mitigation,
                }
                for fm in self.failure_modes
            ],
            "assumptions": list(self.assumptions),
            "regime_dependencies": list(self.regime_dependencies),
            "tags": list(self.tags),
            "paper_references": list(self.paper_references),
        }


class ArchetypeStrategy(ABC):
    """
    Enhanced abstract base strategy enforcing the archetype contract.

    This extends BaseStrategy with:
    - Formal definition metadata
    - Input/output schema validation
    - Execution semantics specification
    - Explicit failure mode documentation
    """

    # Subclasses must provide this metadata
    strategy_definition: StrategyDefinition

    @property
    def strategy_id(self) -> str:
        """Return stable strategy identifier."""
        return self.strategy_definition.strategy_id

    @property
    def version(self) -> str:
        """Return strategy version."""
        return self.strategy_definition.version

    @property
    def archetype(self) -> StrategyArchetype:
        """Return the archetype classification."""
        return self.strategy_definition.archetype

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate deterministic trading signals from feature data.

        Args:
            df: Feature dataset containing required columns per input_schema.

        Returns:
            A pandas Series aligned to df.index with signal values.

        Raises:
            ValueError: If input data violates input_schema or output violates output_signal_schema.
        """
        ...

    def validate_inputs(self, df: pd.DataFrame) -> None:
        """Validate that input DataFrame satisfies input_schema."""
        self.strategy_definition.input_schema.validate(df)

    def validate_outputs(self, signal: pd.Series) -> None:
        """Validate that output signal satisfies output_signal_schema."""
        self.strategy_definition.output_signal_schema.validate(signal)

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        """
        Call the strategy with automatic input/output validation.

        This ensures deterministic behavior and contract enforcement.
        """
        self.validate_inputs(df)
        signal = self.generate_signals(df)
        self.validate_outputs(signal)
        return signal
