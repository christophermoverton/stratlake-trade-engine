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
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
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

        enabled_value = payload.get("enabled")
        if enabled_value is None:
            enabled = bool(
                seed.enabled
                or transaction_cost_bps > 0.0
                or slippage_bps > 0.0
                or fixed_fee > 0.0
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
