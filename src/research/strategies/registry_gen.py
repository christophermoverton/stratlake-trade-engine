"""
Registry generation utilities for strategy archetypes.

Converts strategy definitions to JSONL registry format.
"""

import json
from pathlib import Path

from src.research.strategies.archetypes import (
    TimeSeriesMomentumStrategy,
    CrossSectionMomentumStrategy,
    MeanReversionStrategy,
    BreakoutStrategy,
    PairsTradingStrategy,
    ResidualMomentumStrategy,
)


def generate_strategies_registry() -> list[dict]:
    """Generate full registry of canonical strategies."""
    strategies = [
        TimeSeriesMomentumStrategy.strategy_definition,
        CrossSectionMomentumStrategy.strategy_definition,
        MeanReversionStrategy.strategy_definition,
        BreakoutStrategy.strategy_definition,
        PairsTradingStrategy.strategy_definition,
        ResidualMomentumStrategy.strategy_definition,
    ]

    return [s.to_registry_entry() for s in strategies]


def write_strategies_registry(output_path: Path) -> None:
    """Write registry as JSONL to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    entries = generate_strategies_registry()
    with open(output_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    print(f"Wrote {len(entries)} strategy entries to {output_path}")


if __name__ == "__main__":
    # Generate registry
    repo_root = Path(__file__).resolve().parents[3]
    registry_path = repo_root / "artifacts" / "registry" / "strategies.jsonl"
    write_strategies_registry(registry_path)
