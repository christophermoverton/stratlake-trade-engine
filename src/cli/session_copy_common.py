from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from src.session.drive_adapter import (
    COPY_CATEGORIES,
    SessionCopyResult,
    export_session_to_drive,
    import_session_from_drive,
)

CATEGORY_FLAGS = {
    "configs": "--include-configs",
    "contracts": "--include-contracts",
    "docs": "--include-docs",
    "artifacts": "--include-artifacts",
    "derived_artifacts": "--include-derived-artifacts",
    "features": "--include-features",
    "market_data": "--include-market-data",
    "session_metadata": "--include-session-metadata",
}


def build_parser(operation: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            f"{operation.title()} selected StratLake notebook-session files to or from "
            "a mounted filesystem Drive root."
        )
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Local StratLake session root or a path inside it. Defaults to current directory.",
    )
    parser.add_argument(
        "--drive-root",
        default=None,
        help="Mounted Drive/project root. If omitted, uses drive_root from session metadata.",
    )
    for category, flag in CATEGORY_FLAGS.items():
        parser.add_argument(
            flag,
            action="store_true",
            help=f"Include {category.replace('_', ' ')} in this explicit one-shot copy.",
        )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan the copy without copying files. No manifest is written unless --write-manifest is set.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting existing destination files.",
    )
    parser.add_argument(
        "--operation-id",
        default="latest",
        help="Deterministic manifest operation id. Defaults to 'latest'.",
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Write a manifest even for dry-runs.",
    )
    return parser


def run_copy_cli(operation: str, argv: Sequence[str] | None = None) -> dict[str, object]:
    parser = build_parser(operation)
    args = parser.parse_args(argv)
    categories = _selected_categories(args)
    copy_fn = export_session_to_drive if operation == "export" else import_session_from_drive
    result = copy_fn(
        root=Path(args.root),
        drive_root=args.drive_root,
        include_categories=categories,
        force=args.force,
        dry_run=args.dry_run,
        operation_id=args.operation_id,
        write_manifest=True if args.write_manifest else None,
    )
    print_copy_summary(result)
    return result.to_dict()


def main_for_operation(operation: str, argv: Sequence[str] | None = None) -> int:
    try:
        run_copy_cli(operation, argv)
    except (FileExistsError, FileNotFoundError, NotADirectoryError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def print_copy_summary(result: SessionCopyResult) -> None:
    plan = result.plan
    print(f"StratLake session {plan.operation}:")
    print(f"  local_root: {plan.local_root.as_posix()}")
    print(f"  drive_root: {plan.drive_root.as_posix()}")
    print(f"  categories: {', '.join(plan.include_categories)}")
    print(f"  dry_run: {str(plan.dry_run).lower()}")
    print(f"  copied: {result.copied_count}")
    print(f"  skipped: {result.skipped_count}")
    print(f"  overwritten: {result.overwritten_count}")
    if result.manifest_path is not None:
        print(f"  manifest: {result.manifest_path.as_posix()}")


def _selected_categories(args: argparse.Namespace) -> tuple[str, ...]:
    selected = []
    for category in COPY_CATEGORIES:
        attr = f"include_{category}"
        if getattr(args, attr):
            selected.append(category)
    return tuple(selected)
