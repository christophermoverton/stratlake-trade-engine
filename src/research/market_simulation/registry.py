from __future__ import annotations

from dataclasses import dataclass

from src.research.market_simulation.validation import MarketSimulationConfigError


@dataclass(frozen=True)
class SimulationTypeMetadata:
    simulation_type: str
    display_name: str
    uses_historical_data: bool
    uses_synthetic_generation: bool
    uses_shock_overlay: bool
    status: str = "reserved"

    def to_dict(self) -> dict[str, object]:
        return {
            "simulation_type": self.simulation_type,
            "display_name": self.display_name,
            "uses_historical_data": self.uses_historical_data,
            "uses_synthetic_generation": self.uses_synthetic_generation,
            "uses_shock_overlay": self.uses_shock_overlay,
            "status": self.status,
        }


_REGISTRY: dict[str, SimulationTypeMetadata] = {
    "historical_episode_replay": SimulationTypeMetadata(
        simulation_type="historical_episode_replay",
        display_name="Historical Episode Replay",
        uses_historical_data=True,
        uses_synthetic_generation=False,
        uses_shock_overlay=False,
    ),
    "shock_overlay": SimulationTypeMetadata(
        simulation_type="shock_overlay",
        display_name="Shock Overlay",
        uses_historical_data=True,
        uses_synthetic_generation=False,
        uses_shock_overlay=True,
    ),
    "regime_block_bootstrap": SimulationTypeMetadata(
        simulation_type="regime_block_bootstrap",
        display_name="Regime Block Bootstrap",
        uses_historical_data=True,
        uses_synthetic_generation=True,
        uses_shock_overlay=False,
    ),
    "transition_block_bootstrap": SimulationTypeMetadata(
        simulation_type="transition_block_bootstrap",
        display_name="Transition Block Bootstrap",
        uses_historical_data=True,
        uses_synthetic_generation=True,
        uses_shock_overlay=False,
    ),
    "regime_transition_monte_carlo": SimulationTypeMetadata(
        simulation_type="regime_transition_monte_carlo",
        display_name="Regime Transition Monte Carlo",
        uses_historical_data=False,
        uses_synthetic_generation=True,
        uses_shock_overlay=False,
    ),
    "hybrid_simulation": SimulationTypeMetadata(
        simulation_type="hybrid_simulation",
        display_name="Hybrid Simulation",
        uses_historical_data=True,
        uses_synthetic_generation=True,
        uses_shock_overlay=True,
    ),
}

RESERVED_SIMULATION_TYPES = tuple(sorted(_REGISTRY))


def is_supported_simulation_type(simulation_type: str) -> bool:
    return simulation_type in _REGISTRY


def get_simulation_type_metadata(simulation_type: str) -> SimulationTypeMetadata:
    try:
        return _REGISTRY[simulation_type]
    except KeyError as exc:
        expected = ", ".join(RESERVED_SIMULATION_TYPES)
        raise MarketSimulationConfigError(
            f"Market simulation type {simulation_type!r} is unsupported. Expected one of: {expected}."
        ) from exc
