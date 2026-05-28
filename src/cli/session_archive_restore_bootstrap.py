from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from src.artifacts.safety import atomic_write_text
from src.session_archive import (
    SessionArchiveError,
    SessionArchiveRestoreRequest,
    build_session_archive_restore_plan,
    inspect_session_archive,
    restore_session_archive_pack,
    validate_session_archive,
)

BOUNDARY_TEXT = (
    "Session archive packs are derived, disposable, transport-only snapshots; "
    "they are not canonical storage, not canonical evidence, and not a registry."
)
SUPPORTED_BOOTSTRAP_OVERWRITE_POLICIES = frozenset(
    {"fail_if_exists", "skip_existing", "overwrite_allowed"}
)
_RESTORE_POLICY_BY_BOOTSTRAP_POLICY = {
    "fail_if_exists": "fail_if_exists",
    "skip_existing": "skip_existing",
    "overwrite_allowed": "replace_existing",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Restore a portable M43 session archive pack into an explicit local target workspace."
        ),
        epilog=BOUNDARY_TEXT,
    )
    parser.add_argument("--archive-root", required=True, help="Archive pack root to restore.")
    parser.add_argument("--target-root", required=True, help="Explicit local restore target root.")
    parser.add_argument(
        "--overwrite-policy",
        choices=sorted(SUPPORTED_BOOTSTRAP_OVERWRITE_POLICIES),
        default="fail_if_exists",
        help="Restore collision policy for files under --target-root.",
    )
    parser.add_argument("--verify-checksums", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validate-before-restore", action="store_true")
    parser.add_argument("--inspect-before-restore", action="store_true")
    parser.add_argument(
        "--report-root",
        default=None,
        help=(
            "Optional report directory under --target-root. Defaults to "
            "artifacts/_derived/session_archives/<archive_id>/."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true", help="Emit deterministic JSON output.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args: argparse.Namespace | None = None
    try:
        args = parse_args(argv)
        summary = _run_args(args)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 1
    except SessionArchiveError as exc:
        if args is not None and args.json:
            _print_json(_runtime_failure_summary(args, str(exc)))
            return 2
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (FileNotFoundError, FileExistsError, NotADirectoryError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return int(summary.get("exit_code", 1))


def run_cli(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    return _run_args(args)


def _run_args(args: argparse.Namespace) -> dict[str, Any]:
    validation_status, inspection_status, warnings, errors = _preflight(args)
    if errors:
        summary = _preflight_failure_summary(
            args,
            validation_status=validation_status,
            inspection_status=inspection_status,
            warnings=warnings,
            errors=errors,
        )
        _emit(args, summary)
        return summary
    if args.dry_run:
        return _run_dry(args, validation_status, inspection_status, warnings)
    return _run_restore(args, validation_status, inspection_status, warnings)


def _preflight(
    args: argparse.Namespace,
) -> tuple[str, str, list[str], list[str]]:
    warnings: list[str] = []
    errors: list[str] = []
    validation_status = "not_requested"
    inspection_status = "not_requested"

    if args.validate_before_restore:
        validation = validate_session_archive(
            args.archive_root, verify_checksums=args.verify_checksums
        )
        validation_status = validation.status
        warnings.extend(_issue_lines(validation.issues, "warning"))
        errors.extend(_issue_lines(validation.issues, "error"))
        if not validation.passed:
            return validation_status, inspection_status, warnings, errors

    if args.inspect_before_restore:
        inspection = inspect_session_archive(
            args.archive_root, verify_checksums=args.verify_checksums
        )
        inspection_status = inspection.status
        warnings.extend(_issue_lines(inspection.issues, "warning"))
        errors.extend(_issue_lines(inspection.issues, "error"))

    return validation_status, inspection_status, warnings, errors


def _run_dry(
    args: argparse.Namespace,
    validation_status: str,
    inspection_status: str,
    warnings: list[str],
) -> dict[str, Any]:
    request = _restore_request(args, write_report=False)
    plan = build_session_archive_restore_plan(request)
    summary: dict[str, Any] = {
        "archive_id": plan.archive_id,
        "archive_root": Path(args.archive_root).resolve().as_posix(),
        "bootstrap_report_path": None,
        "boundaries": _boundaries(),
        "checksum_status": plan.checksum_status,
        "dry_run": True,
        "errors": [],
        "exit_code": 0,
        "inspection_status": inspection_status,
        "overwrite_policy": args.overwrite_policy,
        "planned_file_count": len(plan.restore_entries),
        "report_path": None,
        "restored_file_count": 0,
        "restore_overwrite_policy": plan.overwrite_policy,
        "skipped_file_count": len(plan.skipped_entries),
        "status": "planned",
        "target_root": Path(args.target_root).resolve().as_posix(),
        "validation_status": validation_status,
        "verify_checksums": args.verify_checksums,
        "warnings": sorted(set(warnings)),
    }
    _emit(args, summary)
    return summary


def _run_restore(
    args: argparse.Namespace,
    validation_status: str,
    inspection_status: str,
    warnings: list[str],
) -> dict[str, Any]:
    request = _restore_request(args, write_report=True)
    result = restore_session_archive_pack(request)
    summary: dict[str, Any] = {
        "archive_id": result.plan.archive_id,
        "archive_root": Path(args.archive_root).resolve().as_posix(),
        "bootstrap_report_path": None,
        "boundaries": _boundaries(),
        "checksum_status": result.plan.checksum_status,
        "dry_run": False,
        "errors": [],
        "exit_code": 0,
        "inspection_status": inspection_status,
        "overwrite_policy": args.overwrite_policy,
        "planned_file_count": len(result.plan.restore_entries),
        "report_path": None if result.report_path is None else result.report_path.as_posix(),
        "restored_file_count": len(result.restored_paths),
        "restore_overwrite_policy": result.plan.overwrite_policy,
        "skipped_file_count": len(result.skipped_paths),
        "status": "restored",
        "target_root": Path(args.target_root).resolve().as_posix(),
        "validation_status": validation_status,
        "verify_checksums": args.verify_checksums,
        "warnings": sorted(set(warnings)),
    }
    report_path = _write_restore_bootstrap_report(result.plan.target_root, summary)
    summary["bootstrap_report_path"] = report_path.as_posix()
    _emit(args, summary)
    return summary


def _preflight_failure_summary(
    args: argparse.Namespace,
    *,
    validation_status: str,
    inspection_status: str,
    warnings: list[str],
    errors: list[str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "archive_id": _safe_archive_id(args.archive_root),
        "archive_root": Path(args.archive_root).resolve().as_posix(),
        "bootstrap_report_path": None,
        "boundaries": _boundaries(),
        "checksum_status": "failed" if args.verify_checksums else "not_requested",
        "dry_run": args.dry_run,
        "errors": sorted(set(errors)),
        "exit_code": 1,
        "inspection_status": inspection_status,
        "overwrite_policy": args.overwrite_policy,
        "planned_file_count": 0,
        "report_path": None,
        "restored_file_count": 0,
        "restore_overwrite_policy": _restore_policy(args.overwrite_policy),
        "skipped_file_count": 0,
        "status": "failed",
        "target_root": Path(args.target_root).resolve().as_posix(),
        "validation_status": validation_status,
        "verify_checksums": args.verify_checksums,
        "warnings": sorted(set(warnings)),
    }
    return summary


def _runtime_failure_summary(args: argparse.Namespace, message: str) -> dict[str, Any]:
    return {
        "archive_id": _safe_archive_id(args.archive_root),
        "archive_root": Path(args.archive_root).resolve().as_posix(),
        "bootstrap_report_path": None,
        "boundaries": _boundaries(),
        "checksum_status": "failed" if args.verify_checksums else "not_requested",
        "dry_run": args.dry_run,
        "errors": [message],
        "exit_code": 2,
        "inspection_status": "not_requested",
        "overwrite_policy": args.overwrite_policy,
        "planned_file_count": 0,
        "report_path": None,
        "restored_file_count": 0,
        "restore_overwrite_policy": _restore_policy(args.overwrite_policy),
        "skipped_file_count": 0,
        "status": "failed",
        "target_root": Path(args.target_root).resolve().as_posix(),
        "validation_status": "not_requested",
        "verify_checksums": args.verify_checksums,
        "warnings": [],
    }


def _safe_archive_id(archive_root: str | Path) -> str | None:
    try:
        payload = json.loads((Path(archive_root) / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(payload, Mapping):
        return None
    archive_id = payload.get("archive_id")
    return archive_id if isinstance(archive_id, str) else None


def _restore_request(
    args: argparse.Namespace, *, write_report: bool
) -> SessionArchiveRestoreRequest:
    return SessionArchiveRestoreRequest(
        archive_root=args.archive_root,
        target_root=args.target_root,
        overwrite_policy=_restore_policy(args.overwrite_policy),
        verify_checksums=args.verify_checksums,
        write_report=write_report,
        report_root=args.report_root,
    )


def _restore_policy(value: str) -> str:
    try:
        return _RESTORE_POLICY_BY_BOOTSTRAP_POLICY[value]
    except KeyError as exc:
        raise SessionArchiveError(
            "Session archive restore bootstrap overwrite_policy must be one of "
            f"{sorted(SUPPORTED_BOOTSTRAP_OVERWRITE_POLICIES)}."
        ) from exc


def _write_restore_bootstrap_report(
    target_root: Path,
    summary: Mapping[str, Any],
) -> Path:
    archive_id = str(summary["archive_id"])
    report_root = (
        Path(summary["report_path"]).parent
        if summary.get("report_path")
        else (target_root / "artifacts" / "_derived" / "session_archives" / archive_id)
    )
    report_path = report_root / "restore_bootstrap_report.json"
    payload = {key: summary[key] for key in sorted(summary)}
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    return atomic_write_text(report_path, text)


def _issue_lines(issues: Sequence[Any], severity: str) -> list[str]:
    return [f"{issue.code}: {issue.message}" for issue in issues if issue.severity == severity]


def _boundaries() -> dict[str, bool]:
    return {
        "authoritative": False,
        "canonical_storage": False,
        "derived": True,
        "disposable": True,
        "requires_credentials": False,
        "requires_live_market_data": False,
        "requires_network": False,
        "transport_only": True,
    }


def _emit(args: argparse.Namespace, payload: Mapping[str, Any]) -> None:
    if args.json:
        _print_json(payload)
    else:
        _print_human(payload)


def _print_json(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


def _print_human(payload: Mapping[str, Any]) -> None:
    title = (
        "Session archive restore bootstrap dry-run complete"
        if payload["dry_run"]
        else "Session archive restore bootstrap complete"
    )
    if payload["status"] == "failed":
        title = "Session archive restore bootstrap failed"
    print(title)
    print(f"Archive ID: {payload['archive_id']}")
    print(f"Archive root: {payload['archive_root']}")
    print(f"Target root: {payload['target_root']}")
    print(f"Dry run: {payload['dry_run']}")
    print(f"Overwrite policy: {payload['overwrite_policy']}")
    print(f"Checksum status: {payload['checksum_status']}")
    print(f"Validation status: {payload['validation_status']}")
    print(f"Inspection status: {payload['inspection_status']}")
    print(f"Planned files: {payload['planned_file_count']}")
    print(f"Restored files: {payload['restored_file_count']}")
    print(f"Skipped files: {payload['skipped_file_count']}")
    if payload.get("report_path"):
        print(f"Restore report: {payload['report_path']}")
    if payload.get("bootstrap_report_path"):
        print(f"Restore bootstrap report: {payload['bootstrap_report_path']}")
    for warning in payload.get("warnings", []):
        print(f"Warning: {warning}")
    for error in payload.get("errors", []):
        print(f"Error: {error}")
    print(f"Status: {payload['status']}")


if __name__ == "__main__":
    raise SystemExit(main())
