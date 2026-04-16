from __future__ import annotations

from typing import Any, Sequence


def run_cli(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Tiny helper module used by pipeline tests and example specs."""

    return {
        "argv": [] if argv is None else [str(item) for item in argv],
        "argc": 0 if argv is None else len(argv),
    }
