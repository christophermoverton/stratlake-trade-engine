from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from src.catalog import (
    EvidenceReviewError,
    build_evidence_review_for_workflow,
    review_pack_root,
    validate_evidence_review_pack,
    write_evidence_review_pack,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or validate static derived evidence review packs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build one static evidence review pack.")
    build.add_argument("--artifacts-root", default="artifacts")
    build.add_argument("--repo-root", default=".")
    build.add_argument("--index-path")
    build.add_argument("--index-mode", choices=("direct", "index", "auto"), default="direct")
    build.add_argument("--selected-run-id")
    build.add_argument("--selected-catalog-id")
    build.add_argument("--review-id")
    build.add_argument("--resolve-related", action="store_true")
    build.add_argument("--lineage-format", choices=("openlineage", "prov", "both"), default="both")
    build.add_argument("--include-html", action="store_true")
    build.add_argument("--overwrite", action="store_true")

    validate = subparsers.add_parser("validate", help="Validate one existing static evidence review pack.")
    validate.add_argument("--repo-root", default=".")
    root_group = validate.add_mutually_exclusive_group(required=True)
    root_group.add_argument("--pack-root")
    root_group.add_argument("--review-id")
    validate.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def run_cli(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    repo_root = Path(args.repo_root)
    if args.command == "build":
        model = build_evidence_review_for_workflow(
            args.artifacts_root,
            repo_root=repo_root,
            index_path=args.index_path,
            index_mode=args.index_mode,
            selected_run_id=args.selected_run_id,
            selected_catalog_id=args.selected_catalog_id,
            review_id=args.review_id,
            resolve_related=args.resolve_related,
            lineage_format=args.lineage_format,
        )
        result = write_evidence_review_pack(
            model,
            repo_root=repo_root,
            include_html=args.include_html,
            overwrite=args.overwrite,
        )
        payload = {
            "review_id": result["review_id"],
            "output_root": result["output_root"],
            "generated_file_count": len(result["generated_files"]),
            "diagnostics_overall_status": model["catalog_health_diagnostics"]["summary"]["overall_status"],
            "validation_status": result["validation"]["status"],
            "report_path": f"{result['output_root']}/report.md",
        }
        _print_summary("build", payload)
        return payload

    pack_root = args.pack_root or review_pack_root(args.review_id)
    payload = validate_evidence_review_pack(pack_root, repo_root=repo_root, strict=args.strict)
    _print_summary("validate", payload)
    if payload["status"] == "fail":
        raise EvidenceReviewError("Evidence review pack validation failed.")
    return payload


def _print_summary(command: str, payload: dict[str, object]) -> None:
    if command == "build":
        keys = (
            "review_id",
            "output_root",
            "generated_file_count",
            "diagnostics_overall_status",
            "validation_status",
            "report_path",
        )
    else:
        keys = (
            "review_id",
            "pack_root",
            "status",
            "validation_status",
            "diagnostics_overall_status",
        )
    for key in keys:
        print(f"{key}: {payload.get(key)}")
    if command == "validate":
        print(f"missing_file_count: {len(payload['missing_files'])}")
        print(f"invalid_file_count: {len(payload['invalid_files'])}")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run_cli(argv)
    except (EvidenceReviewError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
