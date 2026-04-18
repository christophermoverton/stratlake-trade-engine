from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.config.settings import load_yaml_config

REPO_ROOT = Path(__file__).resolve().parents[2]
EXECUTION_CONFIG = REPO_ROOT / "configs" / "execution.yml"
SUPPORTED_FIXED_FEE_MODELS = frozenset({"per_rebalance"})
SUPPORTED_SLIPPAGE_MODELS = frozenset({"constant", "turnover_scaled", "volatility_scaled"})


@dataclass(frozen=True)
class ExecutionConfig:
    """Deterministic execution assumptions used by strategy and portfolio runs."""

    enabled: bool
    execution_delay: int
    transaction_cost_bps: float
    slippage_bps: float
    fixed_fee: float = 0.0
    fixed_fee_model: str = "per_rebalance"
    slippage_model: str = "constant"
    slippage_turnover_scale: float = 1.0
    slippage_volatility_scale: float = 1.0
    # Directional cost controls (optional - enable asymmetric long/short costs)
    long_transaction_cost_bps: float | None = None
    short_transaction_cost_bps: float | None = None
    short_slippage_multiplier: float = 1.0
    short_borrow_cost_bps: float = 0.0

    @property
    def friction_bps(self) -> float:
        return float(self.transaction_cost_bps + self.slippage_bps)

    @property
    def friction_rate(self) -> float:
        return float(self.friction_bps / 10_000.0)

    @property
    def has_execution_friction(self) -> bool:
        return bool(
            self.enabled
            and (
                self.transaction_cost_bps > 0.0
                or self.slippage_bps > 0.0
                or self.fixed_fee > 0.0
                or (self.long_transaction_cost_bps is not None and self.long_transaction_cost_bps > 0.0)
                or (self.short_transaction_cost_bps is not None and self.short_transaction_cost_bps > 0.0)
                or self.short_borrow_cost_bps > 0.0
            )
        )

    @property
    def has_directional_asymmetry(self) -> bool:
        """True if directional cost asymmetry is configured."""
        return bool(
            self.long_transaction_cost_bps is not None
            or self.short_transaction_cost_bps is not None
            or self.short_slippage_multiplier != 1.0
            or self.short_borrow_cost_bps > 0.0
        )

    def get_long_transaction_cost_bps(self) -> float:
        """Get effective long transaction cost (uses directional if set, falls back to symmetric)."""
        if self.long_transaction_cost_bps is not None:
            return self.long_transaction_cost_bps
        return self.transaction_cost_bps

    def get_short_transaction_cost_bps(self) -> float:
        """Get effective short transaction cost (uses directional if set, falls back to symmetric)."""
        if self.short_transaction_cost_bps is not None:
            return self.short_transaction_cost_bps
        return self.transaction_cost_bps

    def get_short_slippage_bps(self) -> float:
        """Get effective short slippage (applies multiplier to base slippage)."""
        return self.slippage_bps * self.short_slippage_multiplier

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "enabled": self.enabled,
            "execution_delay": self.execution_delay,
            "transaction_cost_bps": self.transaction_cost_bps,
            "slippage_bps": self.slippage_bps,
            "fixed_fee": self.fixed_fee,
            "fixed_fee_model": self.fixed_fee_model,
            "slippage_model": self.slippage_model,
            "slippage_turnover_scale": self.slippage_turnover_scale,
            "slippage_volatility_scale": self.slippage_volatility_scale,
        }
        if self.long_transaction_cost_bps is not None:
            payload["long_transaction_cost_bps"] = self.long_transaction_cost_bps
        if self.short_transaction_cost_bps is not None:
            payload["short_transaction_cost_bps"] = self.short_transaction_cost_bps
        if self.short_slippage_multiplier != 1.0:
            payload["short_slippage_multiplier"] = self.short_slippage_multiplier
        if self.short_borrow_cost_bps != 0.0:
            payload["short_borrow_cost_bps"] = self.short_borrow_cost_bps
        return payload

    @classmethod
    def default(cls) -> "ExecutionConfig":
        return cls(
            enabled=False,
            execution_delay=1,
            transaction_cost_bps=0.0,
            slippage_bps=0.0,
            fixed_fee=0.0,
            fixed_fee_model="per_rebalance",
            slippage_model="constant",
            slippage_turnover_scale=1.0,
            slippage_volatility_scale=1.0,
            long_transaction_cost_bps=None,
            short_transaction_cost_bps=None,
            short_slippage_multiplier=1.0,
            short_borrow_cost_bps=0.0,
        )

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any] | None,
        *,
        base: "ExecutionConfig" | None = None,
    ) -> "ExecutionConfig":
        if payload is None:
            return base or cls.default()
        if not isinstance(payload, Mapping):
            raise ValueError("Execution configuration must be a mapping.")

        seed = base or cls.default()
        transaction_cost_bps = _coerce_non_negative_float(
            payload.get(
                "transaction_cost_bps",
                payload.get("proportional_transaction_cost_bps", seed.transaction_cost_bps),
            ),
            field_name="transaction_cost_bps",
        )
        slippage_bps = _coerce_non_negative_float(
            payload.get(
                "slippage_bps",
                payload.get("slippage_rate_bps", seed.slippage_bps),
            ),
            field_name="slippage_bps",
        )
        fixed_fee = _coerce_non_negative_float(
            payload.get("fixed_fee", seed.fixed_fee),
            field_name="fixed_fee",
        )
        execution_delay = _coerce_execution_delay(
            payload.get("execution_delay", seed.execution_delay),
        )
        fixed_fee_model = _coerce_choice(
            payload.get("fixed_fee_model", seed.fixed_fee_model),
            field_name="fixed_fee_model",
            supported_values=SUPPORTED_FIXED_FEE_MODELS,
        )
        slippage_model = _coerce_choice(
            payload.get("slippage_model", seed.slippage_model),
            field_name="slippage_model",
            supported_values=SUPPORTED_SLIPPAGE_MODELS,
        )
        slippage_turnover_scale = _coerce_non_negative_float(
            payload.get("slippage_turnover_scale", seed.slippage_turnover_scale),
            field_name="slippage_turnover_scale",
        )
        slippage_volatility_scale = _coerce_non_negative_float(
            payload.get("slippage_volatility_scale", seed.slippage_volatility_scale),
            field_name="slippage_volatility_scale",
        )

        # Directional cost controls (optional)
        long_transaction_cost_bps = _coerce_optional_non_negative_float(
            payload.get("long_transaction_cost_bps", seed.long_transaction_cost_bps),
            field_name="long_transaction_cost_bps",
        )
        short_transaction_cost_bps = _coerce_optional_non_negative_float(
            payload.get("short_transaction_cost_bps", seed.short_transaction_cost_bps),
            field_name="short_transaction_cost_bps",
        )
        short_slippage_multiplier = _coerce_non_negative_float(
            payload.get("short_slippage_multiplier", seed.short_slippage_multiplier),
            field_name="short_slippage_multiplier",
        )
        short_borrow_cost_bps = _coerce_non_negative_float(
            payload.get("short_borrow_cost_bps", seed.short_borrow_cost_bps),
            field_name="short_borrow_cost_bps",
        )

        enabled_value = payload.get("enabled")
        if enabled_value is None:
            enabled = bool(
                seed.enabled
                or transaction_cost_bps > 0.0
                or slippage_bps > 0.0
                or fixed_fee > 0.0
                or (long_transaction_cost_bps is not None and long_transaction_cost_bps > 0.0)
                or (short_transaction_cost_bps is not None and short_transaction_cost_bps > 0.0)
                or short_borrow_cost_bps > 0.0
            )
        elif isinstance(enabled_value, bool):
            enabled = enabled_value
        else:
            raise ValueError("Execution configuration field 'enabled' must be a boolean when provided.")

        return cls(
            enabled=enabled,
            execution_delay=execution_delay,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
            fixed_fee=fixed_fee,
            fixed_fee_model=fixed_fee_model,
            slippage_model=slippage_model,
            slippage_turnover_scale=slippage_turnover_scale,
            slippage_volatility_scale=slippage_volatility_scale,
            long_transaction_cost_bps=long_transaction_cost_bps,
            short_transaction_cost_bps=short_transaction_cost_bps,
            short_slippage_multiplier=short_slippage_multiplier,
            short_borrow_cost_bps=short_borrow_cost_bps,
        )


def load_execution_config(path: Path = EXECUTION_CONFIG) -> ExecutionConfig:
    """Load execution defaults from YAML, or return deterministic code defaults when absent."""

    if not path.exists():
        return ExecutionConfig.default()

    payload = load_yaml_config(path)
    if not isinstance(payload, dict):
        raise ValueError("Execution configuration file must contain a top-level mapping.")

    execution_payload = payload.get("execution", payload)
    if not isinstance(execution_payload, Mapping):
        raise ValueError("Execution configuration must define an 'execution' mapping.")
    return ExecutionConfig.from_mapping(execution_payload)


def resolve_execution_config(
    *sources: Mapping[str, Any] | None,
    base: ExecutionConfig | None = None,
) -> ExecutionConfig:
    """Merge execution settings in order, validating each layer deterministically."""

    resolved = base or load_execution_config()
    for source in sources:
        if source is None:
            continue
        resolved = ExecutionConfig.from_mapping(source, base=resolved)
    return resolved


def _coerce_execution_delay(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("Execution configuration field 'execution_delay' must be an integer >= 1.")
    try:
        execution_delay = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Execution configuration field 'execution_delay' must be an integer >= 1.") from exc
    if execution_delay < 1:
        raise ValueError("Execution configuration field 'execution_delay' must be >= 1.")
    return execution_delay


def _coerce_non_negative_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Execution configuration field '{field_name}' must be a non-negative number.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Execution configuration field '{field_name}' must be a non-negative number.") from exc
    if numeric < 0.0:
        raise ValueError(f"Execution configuration field '{field_name}' must be non-negative.")
    return numeric


def _coerce_optional_non_negative_float(value: Any, *, field_name: str) -> float | None:
    """Coerce optional float field (None or non-negative number)."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"Execution configuration field '{field_name}' must be None or a non-negative number.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Execution configuration field '{field_name}' must be None or a non-negative number.") from exc
    if numeric < 0.0:
        raise ValueError(f"Execution configuration field '{field_name}' must be non-negative.")
    return numeric


def _coerce_choice(
    value: Any,
    *,
    field_name: str,
    supported_values: frozenset[str],
) -> str:
    if not isinstance(value, str):
        raise ValueError(
            f"Execution configuration field '{field_name}' must be one of {sorted(supported_values)}."
        )
    normalized = value.strip().lower()
    if normalized not in supported_values:
        raise ValueError(
            f"Execution configuration field '{field_name}' must be one of {sorted(supported_values)}."
        )
    return normalized
