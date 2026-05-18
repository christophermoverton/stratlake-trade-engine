from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from src.catalog import (
    DEFAULT_DERIVED_INDEX_PATH,
    DerivedIndexError,
    build_derived_index,
    validate_derived_index,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or validate the optional derived catalog index.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build a disposable derived catalog index.")
    build.add_argument("--artifacts-root", default="artifacts")
    build.add_argument("--repo-root", default=".")
    build.add_argument("--output", default=DEFAULT_DERIVED_INDEX_PATH)

    validate = subparsers.add_parser("validate", help="Validate a derived catalog index.")
    validate.add_argument("--index", required=True)
    validate.add_argument("--artifacts-root", default="artifacts")
    validate.add_argument("--repo-root", default=".")
    return parser.parse_args(argv)


def run_cli(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    repo_root = Path(args.repo_root)
    artifacts_root = Path(args.artifacts_root)
    if not artifacts_root.is_absolute():
        artifacts_root = repo_root / artifacts_root
    if args.command == "build":
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = repo_root / output_path
        payload = build_derived_index(artifacts_root, output_path, repo_root=repo_root)
    else:
        validation = validate_derived_index(args.index, artifacts_root=artifacts_root, repo_root=repo_root)
        payload = {"valid": True, **validation.metadata}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run_cli(argv)
    except (DerivedIndexError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
