from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from src.config.execution import ExecutionConfig, load_execution_config
from src.config.sanity import SanityCheckConfig, load_sanity_config, resolve_sanity_check_config
from src.research.strict_mode import (
    StrictModePolicy,
    apply_strict_mode_to_sanity_config,
    apply_strict_mode_to_validation_config,
    resolve_strict_mode_policy,
)

if TYPE_CHECKING:
    from src.portfolio.contracts import PortfolioValidationConfig
    from src.portfolio.risk import PortfolioRiskConfig


_RUNTIME_SECTION_KEYS = frozenset(("execution", "sanity", "validation", "portfolio_validation", "risk", "strict_mode"))
_STRICT_MODE_KEYS = frozenset(("enabled", "source"))
_EXECUTION_OVERRIDE_KEYS = frozenset(
    (
        "enabled",
        "execution_delay",
        "transaction_cost_bps",
        "proportional_transaction_cost_bps",
        "slippage_bps",
        "slippage_rate_bps",
        "fixed_fee",
        "fixed_fee_model",
        "slippage_model",
        "slippage_turnover_scale",
        "slippage_volatility_scale",
        "long_transaction_cost_bps",
        "short_transaction_cost_bps",
        "short_slippage_multiplier",
        "short_borrow_cost_bps",
    )
)
_SANITY_KEYS = frozenset(
    (
        "max_abs_period_return",
        "max_annualized_return",
        "max_sharpe_ratio",
        "max_equity_multiple",
        "strict_sanity_checks",
        "min_annualized_volatility_floor",
        "min_volatility_trigger_sharpe",
        "min_volatility_trigger_annualized_return",
        "smoothness_min_sharpe",
        "smoothness_min_annualized_return",
        "smoothness_max_drawdown",
        "smoothness_min_positive_return_fraction",
    )
)
_PORTFOLIO_VALIDATION_KEYS = frozenset(
    (
        "long_only",
        "target_weight_sum",
        "weight_sum_tolerance",
        "target_net_exposure",
        "net_exposure_tolerance",
        "max_gross_exposure",
        "max_leverage",
        "max_single_sleeve_weight",
        "min_single_sleeve_weight",
        "max_abs_period_return",
        "max_equity_multiple",
        "strict_sanity_checks",
    )
)
_PORTFOLIO_RISK_KEYS = frozenset(
    (
        "volatility_window",
        "target_volatility",
        "min_volatility_scale",
        "max_volatility_scale",
        "allow_scale_up",
        "var_confidence_level",
        "cvar_confidence_level",
        "volatility_epsilon",
        "periods_per_year_override",
    )
)


@dataclass(frozen=True)
class RuntimeConfig:
    """Unified normalized runtime controls used by strategy and portfolio workflows."""

    execution: ExecutionConfig
    sanity: SanityCheckConfig
    portfolio_validation: "PortfolioValidationConfig"
    risk: "PortfolioRiskConfig"
    strict_mode: StrictModePolicy

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution": self.execution.to_dict(),
            "sanity": self.sanity.to_dict(),
            "portfolio_validation": self.portfolio_validation.to_dict(),
            "risk": self.risk.to_dict(),
            "strict_mode": self.strict_mode.to_dict(),
        }

    def apply_to_payload(
        self,
        payload: Mapping[str, Any] | None = None,
        *,
        include_runtime_section: bool = True,
        include_validation_section: bool = True,
        validation_key: str = "validation",
    ) -> dict[str, Any]:
        merged = dict(payload or {})
        merged["execution"] = self.execution.to_dict()
        merged["sanity"] = self.sanity.to_dict()
        if include_validation_section:
            merged[validation_key] = self.portfolio_validation.to_dict()
        merged["risk"] = self.risk.to_dict()
        merged["strict_mode"] = self.strict_mode.to_dict()
        if include_runtime_section:
            merged["runtime"] = self.to_dict()
        return merged


def resolve_runtime_config(
    *runtime_sources: Mapping[str, Any] | None,
    cli_overrides: Mapping[str, Any] | None = None,
    cli_strict: bool = False,
    execution_defaults_path: Path | None = None,
    sanity_defaults_path: Path | None = None,
) -> RuntimeConfig:
    """
    Resolve repository defaults, config values, and CLI overrides into one runtime contract.

    Precedence is deterministic: repository defaults < runtime_sources < cli_overrides.
    """

    execution = load_execution_config() if execution_defaults_path is None else load_execution_config(execution_defaults_path)
    sanity = load_sanity_config() if sanity_defaults_path is None else load_sanity_config(sanity_defaults_path)
    portfolio_validation = _default_portfolio_validation_config()
    risk = _default_portfolio_risk_config()
    strict_mode_payload: Mapping[str, Any] | None = None

    for source in runtime_sources:
        if source is None:
            continue
        extracted = _extract_runtime_sections(source)
        if extracted["execution"] is not None:
            execution = ExecutionConfig.from_mapping(extracted["execution"], base=execution)
        if extracted["sanity"] is not None:
            sanity = resolve_sanity_check_config(extracted["sanity"], base=sanity)
        if extracted["portfolio_validation"] is not None:
            portfolio_validation = _resolve_portfolio_validation_with_base(
                extracted["portfolio_validation"],
                base=portfolio_validation,
            )
        if extracted["risk"] is not None:
            risk = _resolve_portfolio_risk_with_base(
                extracted["risk"],
                base=risk,
            )
        if extracted["strict_mode"] is not None:
            strict_mode_payload = extracted["strict_mode"]

    if cli_overrides is not None:
        extracted_overrides = _extract_runtime_sections(cli_overrides, owner="cli overrides")
        if extracted_overrides["execution"] is not None:
            execution = ExecutionConfig.from_mapping(extracted_overrides["execution"], base=execution)
        if extracted_overrides["sanity"] is not None:
            sanity = resolve_sanity_check_config(extracted_overrides["sanity"], base=sanity)
        if extracted_overrides["portfolio_validation"] is not None:
            portfolio_validation = _resolve_portfolio_validation_with_base(
                extracted_overrides["portfolio_validation"],
                base=portfolio_validation,
            )
        if extracted_overrides["risk"] is not None:
            risk = _resolve_portfolio_risk_with_base(
                extracted_overrides["risk"],
                base=risk,
            )
        if extracted_overrides["strict_mode"] is not None:
            strict_mode_payload = extracted_overrides["strict_mode"]

    strict_policy = resolve_strict_mode_policy(
        cli_strict=cli_strict,
        strict_mode_config=strict_mode_payload,
        sanity_config=sanity,
        validation_config=portfolio_validation,
    )
    return RuntimeConfig(
        execution=execution,
        sanity=apply_strict_mode_to_sanity_config(sanity, strict_policy),
        portfolio_validation=apply_strict_mode_to_validation_config(portfolio_validation, strict_policy),
        risk=risk,
        strict_mode=strict_policy,
    )


def _extract_runtime_sections(
    payload: Mapping[str, Any],
    *,
    owner: str = "runtime config",
) -> dict[str, Mapping[str, Any] | None]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{owner} must be a mapping.")

    runtime_payload = payload.get("runtime")
    if runtime_payload is not None:
        if not isinstance(runtime_payload, Mapping):
            raise ValueError(f"{owner} field 'runtime' must be a mapping.")
        unknown_runtime_keys = sorted(set(runtime_payload) - _RUNTIME_SECTION_KEYS)
        if unknown_runtime_keys:
            raise ValueError(
                f"{owner} field 'runtime' contains unsupported keys: {unknown_runtime_keys}."
            )
        container = runtime_payload
    else:
        container = payload

    execution_payload = _resolve_section_mapping(
        container,
        primary_key="execution",
        allowed_keys=_EXECUTION_OVERRIDE_KEYS,
        owner=owner,
    )
    sanity_payload = _resolve_section_mapping(
        container,
        primary_key="sanity",
        allowed_keys=_SANITY_KEYS,
        owner=owner,
    )
    validation_payload = _resolve_portfolio_validation_section(container, owner=owner)
    risk_payload = _resolve_section_mapping(
        container,
        primary_key="risk",
        allowed_keys=_PORTFOLIO_RISK_KEYS,
        owner=owner,
    )
    strict_mode_payload = _resolve_section_mapping(
        container,
        primary_key="strict_mode",
        allowed_keys=_STRICT_MODE_KEYS,
        owner=owner,
    )
    return {
        "execution": execution_payload,
        "sanity": sanity_payload,
        "portfolio_validation": validation_payload,
        "risk": risk_payload,
        "strict_mode": strict_mode_payload,
    }


def _resolve_section_mapping(
    payload: Mapping[str, Any],
    *,
    primary_key: str,
    allowed_keys: frozenset[str],
    owner: str,
) -> Mapping[str, Any] | None:
    section = payload.get(primary_key)
    if section is None:
        return None
    if not isinstance(section, Mapping):
        raise ValueError(f"{owner} section '{primary_key}' must be a mapping.")
    unknown_keys = sorted(set(section) - allowed_keys)
    if unknown_keys:
        raise ValueError(f"{owner} section '{primary_key}' contains unsupported keys: {unknown_keys}.")
    return dict(section)


def _resolve_portfolio_validation_section(
    payload: Mapping[str, Any],
    *,
    owner: str,
) -> Mapping[str, Any] | None:
    validation_payload = payload.get("validation")
    portfolio_validation_payload = payload.get("portfolio_validation")
    if validation_payload is not None and portfolio_validation_payload is not None:
        if dict(validation_payload) != dict(portfolio_validation_payload):
            raise ValueError(
                f"{owner} cannot define both 'validation' and 'portfolio_validation' with different values."
            )
        portfolio_validation_payload = validation_payload
    elif portfolio_validation_payload is None:
        portfolio_validation_payload = validation_payload

    if portfolio_validation_payload is None:
        return None
    if not isinstance(portfolio_validation_payload, Mapping):
        raise ValueError(f"{owner} section 'portfolio_validation' must be a mapping.")
    unknown_keys = sorted(set(portfolio_validation_payload) - _PORTFOLIO_VALIDATION_KEYS)
    if unknown_keys:
        raise ValueError(
            f"{owner} section 'portfolio_validation' contains unsupported keys: {unknown_keys}."
        )
    return dict(portfolio_validation_payload)


def _resolve_portfolio_validation_with_base(
    payload: Mapping[str, Any],
    *,
    base: "PortfolioValidationConfig",
) -> "PortfolioValidationConfig":
    from src.portfolio.contracts import PortfolioContractError, resolve_portfolio_validation_config

    if not isinstance(payload, Mapping):
        raise PortfolioContractError("portfolio validation config must be a dictionary when provided.")

    merged = {
        **base.to_dict(),
        **dict(payload),
    }
    return resolve_portfolio_validation_config(merged)


def _default_portfolio_validation_config() -> "PortfolioValidationConfig":
    from src.portfolio.contracts import PortfolioValidationConfig

    return PortfolioValidationConfig()


def _resolve_portfolio_risk_with_base(
    payload: Mapping[str, Any],
    *,
    base: "PortfolioRiskConfig",
) -> "PortfolioRiskConfig":
    from src.portfolio.risk import PortfolioRiskError, resolve_portfolio_risk_config

    if not isinstance(payload, Mapping):
        raise PortfolioRiskError("portfolio risk config must be a dictionary when provided.")

    merged = {
        **base.to_dict(),
        **dict(payload),
    }
    return resolve_portfolio_risk_config(merged)


def _default_portfolio_risk_config() -> "PortfolioRiskConfig":
    from src.portfolio.risk import PortfolioRiskConfig

    return PortfolioRiskConfig()
