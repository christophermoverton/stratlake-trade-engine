from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

DEFAULT_CONSOLE_ROWS = 10


def add_dual_flag_argument(
    parser: argparse.ArgumentParser,
    preferred_flag: str,
    legacy_flag: str,
    /,
    **kwargs: Any,
) -> None:
    """Register one preferred flag with a backward-compatible alias."""

    parser.add_argument(preferred_flag, legacy_flag, **kwargs)


def parse_csv_or_space_separated(values: Sequence[str] | None) -> list[str] | None:
    """Return normalized values from comma-separated and/or repeated CLI inputs."""

    if values is None:
        return None
    normalized: list[str] = []
    for raw_value in values:
        for value in raw_value.split(","):
            text = value.strip()
            if text:
                normalized.append(text)
    return normalized


def require_registry_mode(enabled: bool, *, surface_name: str) -> None:
    """Raise a consistent error when a compare surface is registry-backed only."""

    if not enabled:
        raise ValueError(
            f"{surface_name} currently supports registry-backed inputs only. Pass --from-registry."
        )


def optional_output_path(raw_output_path: str | None) -> Path | None:
    """Convert an optional CLI output path string into a Path."""

    return None if raw_output_path is None else Path(raw_output_path)


def render_console_table(table: str, *, max_rows: int = DEFAULT_CONSOLE_ROWS) -> str:
    """Limit console tables to a consistent number of body rows while preserving headers."""

    lines = table.splitlines()
    header_lines = 2
    if len(lines) <= header_lines + max_rows:
        return table
    hidden_rows = len(lines) - header_lines - max_rows
    return "\n".join([*lines[: header_lines + max_rows], f"... ({hidden_rows} more rows)"])


def print_comparison_summary(
    *,
    identifier_label: str,
    identifier: str,
    row_count: int,
    table: str,
    csv_path: Path,
    json_path: Path,
    extra_fields: Sequence[tuple[str, Any]] = (),
) -> None:
    """Print a consistent comparison/review CLI summary."""

    print(f"{identifier_label}: {identifier}")
    for key, value in extra_fields:
        print(f"{key}: {value}")
    print(f"rows: {row_count}")
    print(render_console_table(table))
    print(f"leaderboard_csv: {csv_path}")
    print(f"leaderboard_json: {json_path}")
