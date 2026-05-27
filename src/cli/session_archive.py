from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Mapping, Sequence

from src.session_archive import (
    DEFAULT_EXCLUDE_PATTERNS,
    DEFAULT_MAX_ENTRIES_PER_SHARD,
    DEFAULT_MAX_SHARD_SIZE_BYTES,
    SUPPORTED_RESTORE_OVERWRITE_POLICIES,
    SessionArchiveError,
    SessionArchiveIncludePolicy,
    SessionArchiveLogicalGroup,
    SessionArchiveRestoreRequest,
    SessionArchiveWriteRequest,
    build_session_archive_plan,
    build_session_archive_restore_plan,
    inspect_session_archive,
    restore_session_archive_pack,
    validate_session_archive,
    write_session_archive_inspection_report,
    write_session_archive_pack,
    write_session_archive_validation_report,
)

BOUNDARY_TEXT = (
    "Session archive packs are derived, disposable, transport-only snapshots; "
    "they are not canonical storage, canonical evidence, or a registry."
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create, restore, validate, and inspect portable StratLake session archives.",
        epilog=BOUNDARY_TEXT,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_pack_parser(subparsers)
    _add_restore_parser(subparsers)
    _add_validate_parser(subparsers)
    _add_inspect_parser(subparsers)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        return _run(args)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 1
    except SessionArchiveError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _run(args: argparse.Namespace) -> int:
    if args.command == "pack":
        return _run_pack(args)
    if args.command == "restore":
        return _run_restore(args)
    if args.command == "validate":
        return _run_validate(args)
    if args.command == "inspect":
        return _run_inspect(args)
    raise SessionArchiveError(f"Unsupported session archive command: {args.command}.")


def _run_pack(args: argparse.Namespace) -> int:
    request = _write_request(args)
    if args.dry_run:
        plan = build_session_archive_plan(request)
        summary = {
            "archive_id": plan.manifest.archive_id,
            "archive_root": plan.archive_root.as_posix(),
            "dry_run": True,
            "file_count": len(plan.entries),
            "groups": sorted({entry.logical_group.value for entry in plan.entries}),
            "shard_count": len(plan.shards),
            "size_bytes": sum(entry.size_bytes for entry in plan.entries),
            "status": "planned",
        }
        _emit(args, summary, _pack_lines("Session archive pack dry-run complete", summary))
        return 0

    result = write_session_archive_pack(request)
    summary = {
        "archive_id": result.plan.manifest.archive_id,
        "archive_root": result.archive_root.as_posix(),
        "dry_run": False,
        "file_count": len(result.plan.entries),
        "groups": sorted({entry.logical_group.value for entry in result.plan.entries}),
        "shard_count": len(result.plan.shards),
        "size_bytes": sum(entry.size_bytes for entry in result.plan.entries),
        "status": "created",
    }
    _emit(args, summary, _pack_lines("Session archive pack created", summary))
    return 0


def _run_restore(args: argparse.Namespace) -> int:
    request = SessionArchiveRestoreRequest(
        archive_root=args.archive_root,
        target_root=args.target_root,
        overwrite_policy=args.overwrite_policy,
        verify_checksums=args.verify_checksums,
        write_report=not args.dry_run,
        report_root=args.report_root,
    )
    if args.dry_run:
        plan = build_session_archive_restore_plan(request)
        summary = {
            "archive_id": plan.archive_id,
            "checksum_status": plan.checksum_status,
            "dry_run": True,
            "overwrite_policy": plan.overwrite_policy,
            "planned_files": len(plan.restore_entries),
            "skipped_files": len(plan.skipped_entries),
            "status": "planned",
            "target_root": plan.target_root.name,
        }
        _emit(args, summary, _restore_lines("Session archive restore dry-run complete", summary))
        return 0

    result = restore_session_archive_pack(request)
    summary = {
        "archive_id": result.plan.archive_id,
        "checksum_status": result.plan.checksum_status,
        "dry_run": False,
        "overwrite_policy": result.plan.overwrite_policy,
        "report_path": None if result.report_path is None else result.report_path.as_posix(),
        "restored_files": len(result.restored_paths),
        "skipped_files": len(result.skipped_paths),
        "status": "restored",
        "target_root": result.plan.target_root.name,
    }
    _emit(args, summary, _restore_lines("Session archive restore complete", summary))
    return 0


def _run_validate(args: argparse.Namespace) -> int:
    result = validate_session_archive(args.archive_root, verify_checksums=args.verify_checksums)
    report_path = None
    if args.output_path is not None or args.output_root is not None:
        report_path = write_session_archive_validation_report(
            args.archive_root,
            output_path=args.output_path,
            output_root=args.output_root,
            verify_checksums=args.verify_checksums,
        )
    summary = {
        "archive_id": result.archive_id,
        "checksum_status": result.checksum_status,
        "error_count": _issue_count(result.issues, "error"),
        "report_path": None if report_path is None else report_path.as_posix(),
        "status": result.status,
        "warning_count": _issue_count(result.issues, "warning"),
    }
    if args.json:
        _print_json(result.report)
    elif not args.quiet:
        _print_lines(_validation_lines(result, summary))
    return 0 if result.passed else 1


def _run_inspect(args: argparse.Namespace) -> int:
    result = inspect_session_archive(args.archive_root, verify_checksums=args.verify_checksums)
    report_path = None
    if args.output_path is not None or args.output_root is not None:
        report_path = write_session_archive_inspection_report(
            args.archive_root,
            output_path=args.output_path,
            output_root=args.output_root,
            verify_checksums=args.verify_checksums,
        )
    summary = {
        "archive_id": result.summary.archive_id,
        "duckdb_snapshot_status": result.summary.duckdb_snapshot_status,
        "error_count": _issue_count(result.issues, "error"),
        "estimated_restored_file_count": result.summary.estimated_restored_file_count,
        "estimated_restored_total_size": result.summary.estimated_restored_total_size,
        "groups": list(result.summary.logical_groups_included),
        "portability_status": result.summary.portability_status,
        "report_path": None if report_path is None else report_path.as_posix(),
        "schema_version": result.summary.schema_version,
        "shard_count": result.summary.shard_count,
        "status": result.status,
        "warning_count": _issue_count(result.issues, "warning"),
    }
    if args.json:
        _print_json(result.report)
    elif not args.quiet:
        _print_lines(_inspection_lines(summary))
    return 0 if not any(issue.severity == "error" for issue in result.issues) else 1


def _add_pack_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "pack",
        help="Create a derived portable session archive pack.",
        description=f"Create a portable session archive pack. {BOUNDARY_TEXT}",
        epilog=BOUNDARY_TEXT,
    )
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--archive-id", required=True)
    parser.add_argument("--output-root", default="artifacts/_derived/session_archives")
    parser.add_argument("--profile-name", dest="source_runtime_profile")
    parser.add_argument("--profile-path", dest="source_profile_path")
    parser.add_argument("--session-id")
    parser.add_argument("--include-group", action="append", choices=_logical_group_choices())
    parser.add_argument(
        "--include-path",
        action="append",
        default=None,
        metavar="GROUP=PATH",
        help="Repository-relative include path for a logical group; repeatable.",
    )
    parser.add_argument("--exclude-pattern", action="append", default=None)
    parser.add_argument("--max-shard-size-bytes", type=int, default=DEFAULT_MAX_SHARD_SIZE_BYTES)
    parser.add_argument("--max-entries-per-shard", type=int, default=DEFAULT_MAX_ENTRIES_PER_SHARD)
    parser.add_argument("--duckdb-snapshot-source-path")
    parser.add_argument("--duckdb-snapshot-description")
    parser.add_argument(
        "--collision-policy",
        choices=("fail_if_exists", "overwrite_allowed"),
        default="fail_if_exists",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON summary.")


def _add_restore_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "restore",
        help="Restore a portable session archive pack to a local target root.",
        description=f"Restore a portable session archive pack. {BOUNDARY_TEXT}",
        epilog=BOUNDARY_TEXT,
    )
    parser.add_argument("--archive-root", required=True)
    parser.add_argument("--target-root", required=True)
    parser.add_argument(
        "--overwrite-policy",
        choices=sorted(SUPPORTED_RESTORE_OVERWRITE_POLICIES),
        default="fail_if_exists",
    )
    parser.add_argument("--verify-checksums", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-root")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON summary.")


def _add_validate_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "validate",
        help="Validate a portable session archive pack before restore.",
        description=f"Validate a portable session archive pack. {BOUNDARY_TEXT}",
        epilog=BOUNDARY_TEXT,
    )
    _add_report_args(parser)


def _add_inspect_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "inspect",
        help="Inspect a portable session archive pack without extracting it.",
        description=f"Inspect a portable session archive pack. {BOUNDARY_TEXT}",
        epilog=BOUNDARY_TEXT,
    )
    _add_report_args(parser)


def _add_report_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--archive-root", required=True)
    parser.add_argument("--verify-checksums", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-path")
    parser.add_argument(
        "--output-root",
        help="Write report under OUTPUT_ROOT/_derived/session_archives/<archive_id>/.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the deterministic report JSON.")
    parser.add_argument("--quiet", action="store_true", help="Suppress human-readable output.")


def _write_request(args: argparse.Namespace) -> SessionArchiveWriteRequest:
    include_groups = (
        tuple(SessionArchiveLogicalGroup(group) for group in args.include_group)
        if args.include_group
        else SessionArchiveIncludePolicy().include_groups
    )
    include_policy = SessionArchiveIncludePolicy(
        include_groups=include_groups,
        include_paths=_include_paths(args.include_path or ()),
        exclude_patterns=(
            tuple(args.exclude_pattern) if args.exclude_pattern else DEFAULT_EXCLUDE_PATTERNS
        ),
    )
    return SessionArchiveWriteRequest(
        archive_id=args.archive_id,
        repository_root=args.repository_root,
        output_root=args.output_root,
        include_policy=include_policy,
        max_shard_size_bytes=args.max_shard_size_bytes,
        max_entries_per_shard=args.max_entries_per_shard,
        session_id=args.session_id,
        source_runtime_profile=args.source_runtime_profile,
        source_profile_path=args.source_profile_path,
        duckdb_snapshot_source_path=args.duckdb_snapshot_source_path,
        duckdb_snapshot_description=args.duckdb_snapshot_description,
        collision_policy=args.collision_policy,
    )


def _include_paths(values: Sequence[str]) -> dict[SessionArchiveLogicalGroup, tuple[str, ...]]:
    resolved: dict[SessionArchiveLogicalGroup, list[str]] = {}
    for value in values:
        if "=" not in value:
            raise SessionArchiveError(
                "Session archive --include-path values must use GROUP=PATH syntax."
            )
        group_text, path = value.split("=", 1)
        try:
            group = SessionArchiveLogicalGroup(group_text.strip())
        except ValueError as exc:
            raise SessionArchiveError(
                f"Unsupported session archive include path logical group: {group_text!r}."
            ) from exc
        resolved.setdefault(group, []).append(path.strip())
    return {group: tuple(paths) for group, paths in resolved.items()}


def _logical_group_choices() -> tuple[str, ...]:
    return tuple(group.value for group in SessionArchiveLogicalGroup)


def _emit(args: argparse.Namespace, payload: Mapping[str, Any], lines: Sequence[str]) -> None:
    if args.json:
        _print_json(payload)
    else:
        _print_lines(lines)


def _print_json(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


def _print_lines(lines: Sequence[str]) -> None:
    for line in lines:
        print(line)


def _pack_lines(title: str, summary: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        title,
        f"Archive ID: {summary['archive_id']}",
        f"Archive root: {summary['archive_root']}",
        f"Groups: {', '.join(summary['groups'])}",
        f"Shards: {summary['shard_count']}",
        f"Files: {summary['file_count']}",
        f"Size bytes: {summary['size_bytes']}",
    )


def _restore_lines(title: str, summary: Mapping[str, Any]) -> tuple[str, ...]:
    lines = [
        title,
        f"Archive ID: {summary['archive_id']}",
        f"Target root: {summary['target_root']}",
        f"Overwrite policy: {summary['overwrite_policy']}",
        f"Checksum status: {summary['checksum_status']}",
    ]
    if summary["dry_run"]:
        lines.append(f"Planned files: {summary['planned_files']}")
    else:
        lines.append(f"Restored files: {summary['restored_files']}")
    lines.append(f"Skipped files: {summary['skipped_files']}")
    if summary.get("report_path"):
        lines.append(f"Report: {summary['report_path']}")
    return tuple(lines)


def _validation_lines(
    result: Any,
    summary: Mapping[str, Any],
) -> tuple[str, ...]:
    lines = [
        f"Session archive validation: {summary['status']}",
        f"Archive ID: {summary['archive_id']}",
        f"Checksum status: {summary['checksum_status']}",
        f"Issues: {summary['error_count']} errors, {summary['warning_count']} warnings",
    ]
    for issue in result.issues:
        lines.append(f"{issue.severity}: {issue.code}: {issue.message}")
    if summary.get("report_path"):
        lines.append(f"Report: {summary['report_path']}")
    return tuple(lines)


def _inspection_lines(summary: Mapping[str, Any]) -> tuple[str, ...]:
    lines = [
        f"Session archive inspection: {summary['status']}",
        f"Archive ID: {summary['archive_id']}",
        f"Schema version: {summary['schema_version']}",
        f"Groups: {', '.join(summary['groups'])}",
        f"Shards: {summary['shard_count']}",
        f"Estimated restored files: {summary['estimated_restored_file_count']}",
        f"Estimated restored size bytes: {summary['estimated_restored_total_size']}",
        f"Portability: {summary['portability_status']}",
        f"DuckDB snapshot: {summary['duckdb_snapshot_status']}",
        f"Issues: {summary['error_count']} errors, {summary['warning_count']} warnings",
    ]
    if summary.get("report_path"):
        lines.append(f"Report: {summary['report_path']}")
    return tuple(lines)


def _issue_count(issues: Sequence[Any], severity: str) -> int:
    return sum(1 for issue in issues if issue.severity == severity)


if __name__ == "__main__":
    raise SystemExit(main())
