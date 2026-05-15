from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from src.catalog import build_lineage_edges, export_lineage, load_catalog_records


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
    repo_root = Path(args.repo_root)
    artifacts_root = Path(args.artifacts_root)
    if not artifacts_root.is_absolute():
        artifacts_root = repo_root / artifacts_root

    records = load_catalog_records(
        artifacts_root,
        repo_root=repo_root,
        index_path=args.index,
        mode=args.index_mode,
    )
    edges = build_lineage_edges(records, repo_root=repo_root)
    payload = export_lineage(
        records,
        edges,
        format=args.format,
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
