from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.research.governance import run_promotion_governance_report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic promotion governance observability artifacts from existing promotion outputs."
    )
    parser.add_argument("--registry-path", "--registry_path", dest="registry_path", help="Optional registry.jsonl path.")
    parser.add_argument(
        "--artifact-root",
        "--artifact_root",
        dest="artifact_root",
        default="artifacts",
        help="Artifact root used to resolve manifests and optional review contexts.",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        help="Output root or concrete report directory. Defaults to artifacts/promotion_governance/<report_id>.",
    )
    parser.add_argument("--report-id", "--report_id", dest="report_id", help="Optional deterministic report id override.")
    parser.add_argument(
        "--strict-validation",
        "--strict_validation",
        dest="strict_validation",
        action="store_true",
        help="Raise a non-zero CLI error after writing artifacts when consistency validation fails.",
    )
    return parser.parse_args(argv)


def run_cli(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    result = run_promotion_governance_report(
        registry_path=None if args.registry_path is None else Path(args.registry_path),
        artifact_root=Path(args.artifact_root),
        output_dir=None if args.output_dir is None else Path(args.output_dir),
        report_id=args.report_id,
        strict_validation=args.strict_validation,
    )
    print(f"report_id: {result.report_id}")
    print(f"output_dir: {result.output_dir.as_posix()}")
    print(f"summary: {result.summary_path.as_posix()}")
    print(f"outcome_matrix: {result.outcome_matrix_path.as_posix()}")
    print(f"validation_status: {result.validation['status']}")
    return result


def main() -> None:
    run_cli()


if __name__ == "__main__":
    main()
