from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Mapping

from src.config.sanity import SanityCheckConfig, resolve_sanity_check_config

if TYPE_CHECKING:
    from src.portfolio.contracts import PortfolioValidationConfig


class ResearchStrictModeError(ValueError):
    """Raised when a run fails under the unified strict-mode contract."""


@dataclass(frozen=True)
class StrictModePolicy:
    enabled: bool
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "source": self.source,
        }


def resolve_strict_mode_policy(
    *,
    cli_strict: bool = False,
    sanity_config: SanityCheckConfig | Mapping[str, Any] | None = None,
    validation_config: "PortfolioValidationConfig | Mapping[str, Any] | None" = None,
) -> StrictModePolicy:
    from src.portfolio.contracts import resolve_portfolio_validation_config

    config_enabled = False
    if sanity_config is not None:
        config_enabled = config_enabled or resolve_sanity_check_config(sanity_config).strict_sanity_checks
    if validation_config is not None:
        config_enabled = config_enabled or resolve_portfolio_validation_config(validation_config).strict_sanity_checks

    if cli_strict:
        return StrictModePolicy(enabled=True, source="cli")
    if config_enabled:
        return StrictModePolicy(enabled=True, source="config")
    return StrictModePolicy(enabled=False, source="default")


def apply_strict_mode_to_sanity_config(
    sanity_config: SanityCheckConfig | Mapping[str, Any] | None,
    policy: StrictModePolicy,
) -> SanityCheckConfig:
    resolved = resolve_sanity_check_config(sanity_config)
    return replace(resolved, strict_sanity_checks=policy.enabled)


def apply_strict_mode_to_validation_config(
    validation_config: "PortfolioValidationConfig | Mapping[str, Any] | None",
    policy: StrictModePolicy,
) -> "PortfolioValidationConfig":
    from src.portfolio.contracts import resolve_portfolio_validation_config

    resolved = resolve_portfolio_validation_config(validation_config)
    return replace(resolved, strict_sanity_checks=policy.enabled)


def raise_research_validation_error(
    *,
    validator: str,
    scope: str,
    exc: Exception,
    strict_mode: bool,
) -> None:
    message = str(exc).strip()
    is_threshold_failure = _looks_like_threshold_failure(message)
    if strict_mode and is_threshold_failure:
        prefix = "Strict mode failure"
        suffix = "Artifacts and registry writes were prevented."
    else:
        prefix = "Research validation failure"
        suffix = "Artifacts and registry writes were prevented."
    raise ResearchStrictModeError(f"{prefix} [{validator}] ({scope}): {message} {suffix}") from exc


def _looks_like_threshold_failure(message: str) -> bool:
    normalized = message.lower()
    return "flagged" in normalized or "sanity check failed:" in normalized
