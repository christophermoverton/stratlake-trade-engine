from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.research.reporting import generate_strategy_plots


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for strategy-run plot generation."""

    parser = argparse.ArgumentParser(
        description="Generate deterministic visualization artifacts from an existing strategy run directory."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Existing strategy run artifact directory under artifacts/strategies/.",
    )
    return parser.parse_args(argv)


def run_cli(argv: Sequence[str] | None = None) -> dict[str, Path]:
    """Execute the plot-generation CLI flow from parsed command-line arguments."""

    args = parse_args(argv)
    plot_paths = generate_strategy_plots(Path(args.run_dir))
    print(f"run_dir: {Path(args.run_dir)}")
    print(f"plot_count: {len(plot_paths)}")
    for name in sorted(plot_paths):
        print(f"{name}: {plot_paths[name]}")
    return plot_paths


def main() -> None:
    """CLI entrypoint used by direct module execution."""

    run_cli()


if __name__ == "__main__":
    main()
