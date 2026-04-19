from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from src.pipeline.builder import PipelineBuilder, PipelineBuilderError, load_builder_config


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse builder CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Generate an M20-compatible pipeline from a declarative builder config."
    )
    parser.add_argument("--config", required=True, help="Builder-friendly YAML/JSON config path.")
    parser.add_argument("--output", help="Optional pipeline YAML output path.")
    parser.add_argument("--run", action="store_true", help="Execute the generated pipeline after rendering it.")
    return parser.parse_args(argv)


def run_cli(argv: Sequence[str] | None = None):
    """Render or execute one declarative builder config."""

    args = parse_args(argv)
    payload = load_builder_config(args.config)
    builder = PipelineBuilder.from_mapping(payload)
    if args.run:
        return builder.run(path=None if args.output is None else Path(args.output))

    yaml_text = builder.to_yaml(path=None if args.output is None else Path(args.output))
    if args.output is None:
        print(yaml_text)
    else:
        print(f"wrote: {Path(args.output).as_posix()}")
    return yaml_text


def main() -> None:
    """CLI entrypoint used by direct module execution."""

    try:
        run_cli()
    except PipelineBuilderError as exc:
        print(f"Pipeline build failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
