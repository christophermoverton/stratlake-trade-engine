from __future__ import annotations

from typing import Sequence

from src.cli.session_copy_common import main_for_operation, run_copy_cli


def run_cli(argv: Sequence[str] | None = None) -> dict[str, object]:
    return run_copy_cli("export", argv)


def main(argv: Sequence[str] | None = None) -> int:
    return main_for_operation("export", argv)


if __name__ == "__main__":
    raise SystemExit(main())
