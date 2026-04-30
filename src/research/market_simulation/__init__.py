from __future__ import annotations

from src.research.market_simulation.artifacts import (
    MarketSimulationFrameworkResult,
    run_market_simulation_framework,
)
from src.research.market_simulation.block_bootstrap import (
    BlockBootstrapResult,
    run_regime_block_bootstrap,
    run_regime_block_bootstrap_scenarios,
)
from src.research.market_simulation.config import (
    MarketSimulationConfig,
    MarketSimulationConfigError,
    MarketSimulationScenarioConfig,
    load_market_simulation_config,
)
from src.research.market_simulation.ids import generate_path_id, generate_scenario_id
from src.research.market_simulation.historical_replay import (
    HistoricalEpisodeReplayResult,
    generate_episode_id,
    run_historical_episode_replay,
    run_historical_episode_replays,
)
from src.research.market_simulation.registry import (
    RESERVED_SIMULATION_TYPES,
    get_simulation_type_metadata,
    is_supported_simulation_type,
)

__all__ = [
    "MarketSimulationConfig",
    "MarketSimulationConfigError",
    "MarketSimulationFrameworkResult",
    "MarketSimulationScenarioConfig",
    "BlockBootstrapResult",
    "HistoricalEpisodeReplayResult",
    "RESERVED_SIMULATION_TYPES",
    "generate_episode_id",
    "generate_path_id",
    "generate_scenario_id",
    "get_simulation_type_metadata",
    "is_supported_simulation_type",
    "load_market_simulation_config",
    "run_historical_episode_replay",
    "run_historical_episode_replays",
    "run_market_simulation_framework",
    "run_regime_block_bootstrap",
    "run_regime_block_bootstrap_scenarios",
]
