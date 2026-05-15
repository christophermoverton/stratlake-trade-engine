from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from src.catalog import build_lineage_export_for_workflow


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export deterministic catalog lineage JSON.")
    parser.add_argument("--artifacts-root", default="artifacts")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--format", choices=("openlineage", "prov"), default="openlineage")
    parser.add_argument("--selected-run-id")
    parser.add_argument("--output")
    parser.add_argument("--index")
    parser.add_argument("--index-mode", choices=("direct", "index", "auto"), default="direct")
    return parser.parse_args(argv)


def run_cli(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    payload = build_lineage_export_for_workflow(
        args.artifacts_root,
        repo_root=args.repo_root,
        index_path=args.index,
        index_mode=args.index_mode,
        export_format=args.format,
        selected_run_id=args.selected_run_id,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8", newline="\n")
    else:
        print(text)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    run_cli(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
