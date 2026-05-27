from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

from src.artifacts.safety import atomic_write_text
from src.session_archive import (
    DEFAULT_EXCLUDE_PATTERNS,
    DEFAULT_MAX_ENTRIES_PER_SHARD,
    DEFAULT_MAX_SHARD_SIZE_BYTES,
    SessionArchiveError,
    SessionArchiveIncludePolicy,
    SessionArchiveLogicalGroup,
    SessionArchiveWriteRequest,
    build_session_archive_plan,
    inspect_session_archive,
    validate_session_archive,
    write_session_archive_pack,
)

BOUNDARY_TEXT = (
    "Session archive packs are derived, disposable, transport-only snapshots; "
    "they are not canonical storage, canonical evidence, or a registry."
)
SUPPORTED_COPY_POLICIES = frozenset({"fail_if_exists", "skip_existing", "overwrite_allowed"})
SUPPORTED_ARCHIVE_COLLISION_POLICIES = frozenset({"fail_if_exists", "overwrite_allowed"})


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a portable M43 session archive pack and optionally copy it to a mounted "
            "filesystem persistence root."
        ),
        epilog=BOUNDARY_TEXT,
    )
    parser.add_argument("--root", required=True, help="Repository root used to build the archive.")
    parser.add_argument("--archive-id", required=True, help="Deterministic archive identifier.")
    parser.add_argument(
        "--output-root",
        default="artifacts/_derived/session_archives",
        help="Repository-relative or absolute output root for the local archive pack.",
    )
    parser.add_argument(
        "--drive-root",
        default=None,
        help="Optional mounted filesystem root for copied archive output.",
    )
    parser.add_argument("--include-features", action="store_true")
    parser.add_argument("--include-artifacts", action="store_true")
    parser.add_argument("--include-configs", action="store_true")
    parser.add_argument("--include-duckdb-snapshot", action="store_true")
    parser.add_argument("--duckdb-snapshot-source-path", default=None)
    parser.add_argument("--duckdb-snapshot-description", default=None)
    parser.add_argument("--max-shard-size-bytes", type=int, default=DEFAULT_MAX_SHARD_SIZE_BYTES)
    parser.add_argument("--max-entries-per-shard", type=int, default=DEFAULT_MAX_ENTRIES_PER_SHARD)
    parser.add_argument("--exclude-pattern", action="append", default=None)
    parser.add_argument(
        "--archive-collision-policy",
        choices=sorted(SUPPORTED_ARCHIVE_COLLISION_POLICIES),
        default="fail_if_exists",
        help="Collision policy for local derived archive creation.",
    )
    parser.add_argument(
        "--copy-policy",
        choices=sorted(SUPPORTED_COPY_POLICIES),
        default="fail_if_exists",
        help="Copy policy when --drive-root is supplied.",
    )
    parser.add_argument("--validate-after-copy", action="store_true")
    parser.add_argument("--inspect-after-copy", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true", help="Emit deterministic JSON output.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        summary = run_cli(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 1
    except SessionArchiveError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (FileNotFoundError, FileExistsError, NotADirectoryError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return int(summary.get("exit_code", 1))


def run_cli(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    request = _build_write_request(args)
    if args.dry_run:
        return _run_dry(args, request)
    return _run_write(args, request)


def _run_dry(
    args: argparse.Namespace,
    request: SessionArchiveWriteRequest,
) -> dict[str, Any]:
    plan = build_session_archive_plan(request)
    groups = sorted({entry.logical_group.value for entry in plan.entries})
    summary: dict[str, Any] = {
        "archive_id": plan.manifest.archive_id,
        "archive_collision_policy": args.archive_collision_policy,
        "bootstrap_report_path": None,
        "boundaries": _boundaries(),
        "copied_file_count": 0,
        "copy_policy": args.copy_policy,
        "copy_status": "not_started_dry_run",
        "destination_archive_root": _destination_archive_root(args),
        "destination_root": None if args.drive_root is None else Path(args.drive_root).resolve().as_posix(),
        "dry_run": True,
        "errors": [],
        "file_count": len(plan.entries),
        "included_logical_groups": groups,
        "inspection_status": "not_requested",
        "local_archive_root": plan.archive_root.as_posix(),
        "shard_count": len(plan.shards),
        "size_bytes": sum(entry.size_bytes for entry in plan.entries),
        "skipped_file_count": 0,
        "source_root": Path(args.root).resolve().as_posix(),
        "status": "planned",
        "validation_status": "not_requested",
        "warnings": [],
        "exit_code": 0,
    }
    _emit(args, summary)
    return summary


def _run_write(
    args: argparse.Namespace,
    request: SessionArchiveWriteRequest,
) -> dict[str, Any]:
    result = write_session_archive_pack(request)
    destination_root = None if args.drive_root is None else Path(args.drive_root).resolve()
    destination_archive_root = None
    copy_status = "not_requested"
    copied_file_count = 0
    skipped_file_count = 0
    if destination_root is not None:
        destination_archive_root = (destination_root / result.plan.manifest.archive_id).resolve()
        copy_status, copied_file_count, skipped_file_count = _copy_archive_pack(
            source_archive_root=result.archive_root,
            destination_archive_root=destination_archive_root,
            copy_policy=args.copy_policy,
        )

    inspect_target = destination_archive_root if destination_archive_root is not None else result.archive_root
    warnings: list[str] = []
    errors: list[str] = []
    validation_status = "not_requested"
    inspection_status = "not_requested"
    exit_code = 0

    if args.validate_after_copy:
        validation = validate_session_archive(inspect_target)
        validation_status = validation.status
        warnings.extend(_issue_lines(validation.issues, "warning"))
        errors.extend(_issue_lines(validation.issues, "error"))
        if not validation.passed:
            exit_code = 1

    if args.inspect_after_copy:
        inspection = inspect_session_archive(inspect_target)
        inspection_status = inspection.status
        warnings.extend(_issue_lines(inspection.issues, "warning"))
        errors.extend(_issue_lines(inspection.issues, "error"))
        if any(issue.severity == "error" for issue in inspection.issues):
            exit_code = 1

    groups = sorted({entry.logical_group.value for entry in result.plan.entries})
    summary: dict[str, Any] = {
        "archive_id": result.plan.manifest.archive_id,
        "archive_collision_policy": args.archive_collision_policy,
        "boundaries": _boundaries(),
        "copied_file_count": copied_file_count,
        "copy_policy": args.copy_policy,
        "copy_status": copy_status,
        "destination_archive_root": None
        if destination_archive_root is None
        else destination_archive_root.as_posix(),
        "destination_root": None if destination_root is None else destination_root.as_posix(),
        "dry_run": False,
        "errors": sorted(set(errors)),
        "file_count": len(result.plan.entries),
        "included_logical_groups": groups,
        "inspection_status": inspection_status,
        "local_archive_root": result.archive_root.as_posix(),
        "shard_count": len(result.plan.shards),
        "size_bytes": sum(entry.size_bytes for entry in result.plan.entries),
        "skipped_file_count": skipped_file_count,
        "source_root": Path(args.root).resolve().as_posix(),
        "status": "failed" if exit_code else "bootstrapped",
        "validation_status": validation_status,
        "warnings": sorted(set(warnings)),
    }
    report_path = _write_bootstrap_report(Path(args.root).resolve(), args.archive_id, summary)
    summary["bootstrap_report_path"] = report_path.as_posix()
    summary["exit_code"] = exit_code
    _emit(args, summary)
    return summary


def _build_write_request(args: argparse.Namespace) -> SessionArchiveWriteRequest:
    include_policy = SessionArchiveIncludePolicy(
        include_groups=_include_groups(args),
        exclude_patterns=(
            tuple(args.exclude_pattern) if args.exclude_pattern is not None else DEFAULT_EXCLUDE_PATTERNS
        ),
    )
    return SessionArchiveWriteRequest(
        archive_id=args.archive_id,
        repository_root=args.root,
        output_root=args.output_root,
        include_policy=include_policy,
        max_shard_size_bytes=args.max_shard_size_bytes,
        max_entries_per_shard=args.max_entries_per_shard,
        duckdb_snapshot_source_path=args.duckdb_snapshot_source_path,
        duckdb_snapshot_description=args.duckdb_snapshot_description,
        collision_policy=args.archive_collision_policy,
    )


def _include_groups(args: argparse.Namespace) -> tuple[SessionArchiveLogicalGroup, ...]:
    selected: list[SessionArchiveLogicalGroup] = []
    if args.include_features:
        selected.append(SessionArchiveLogicalGroup.FEATURES)
    if args.include_artifacts:
        selected.append(SessionArchiveLogicalGroup.ARTIFACTS)
    if args.include_configs:
        selected.append(SessionArchiveLogicalGroup.CONFIGS)
    if args.include_duckdb_snapshot:
        selected.append(SessionArchiveLogicalGroup.DUCKDB_SNAPSHOT)
    if selected:
        return tuple(selected)
    return (
        SessionArchiveLogicalGroup.FEATURES,
        SessionArchiveLogicalGroup.ARTIFACTS,
        SessionArchiveLogicalGroup.CONFIGS,
    )


def _copy_archive_pack(
    *,
    source_archive_root: Path,
    destination_archive_root: Path,
    copy_policy: str,
) -> tuple[str, int, int]:
    _validate_copy_destination(
        source_archive_root=source_archive_root,
        destination_archive_root=destination_archive_root,
    )
    source_files = _archive_files(source_archive_root)
    if copy_policy == "fail_if_exists":
        if _has_any_files(destination_archive_root):
            raise SessionArchiveError(
                "Destination archive root exists and contains files under fail_if_exists: "
                f"{destination_archive_root.as_posix()}."
            )
        copied = _copy_all_files(source_archive_root, destination_archive_root, source_files)
        return "copied", copied, 0

    if copy_policy == "skip_existing":
        if _has_any_files(destination_archive_root):
            return "skipped_existing", 0, len(source_files)
        copied = _copy_all_files(source_archive_root, destination_archive_root, source_files)
        return "copied", copied, 0

    if copy_policy == "overwrite_allowed":
        if destination_archive_root.exists():
            if destination_archive_root.is_dir():
                shutil.rmtree(destination_archive_root)
            else:
                destination_archive_root.unlink()
        copied = _copy_all_files(source_archive_root, destination_archive_root, source_files)
        return "overwritten", copied, 0

    raise SessionArchiveError(
        "Unsupported bootstrap copy policy; expected one of "
        f"{sorted(SUPPORTED_COPY_POLICIES)}."
    )


def _copy_all_files(
    source_archive_root: Path,
    destination_archive_root: Path,
    source_files: tuple[Path, ...],
) -> int:
    copied = 0
    for source in source_files:
        relative = source.relative_to(source_archive_root)
        destination = (destination_archive_root / relative).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied += 1
    return copied


def _validate_copy_destination(
    *,
    source_archive_root: Path,
    destination_archive_root: Path,
) -> None:
    source = source_archive_root.resolve()
    destination = destination_archive_root.resolve()
    if destination == source:
        raise SessionArchiveError("Destination archive root must differ from local archive root.")
    if _is_relative_to(destination, source):
        raise SessionArchiveError(
            "Destination archive root must not be inside local archive root."
        )
    if _is_relative_to(source, destination):
        raise SessionArchiveError(
            "Local archive root must not be inside destination archive root."
        )


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _archive_files(root: Path) -> tuple[Path, ...]:
    return tuple(sorted((path for path in root.rglob("*") if path.is_file()), key=lambda p: p.as_posix()))


def _has_any_files(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return any(candidate.is_file() for candidate in path.rglob("*"))


def _write_bootstrap_report(
    source_root: Path,
    archive_id: str,
    summary: Mapping[str, Any],
) -> Path:
    report_path = (
        source_root
        / "artifacts"
        / "_derived"
        / "session_archives"
        / archive_id
        / "bootstrap_report.json"
    )
    payload = {key: summary[key] for key in sorted(summary)}
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    return atomic_write_text(report_path, text)


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


def _destination_archive_root(args: argparse.Namespace) -> str | None:
    if args.drive_root is None:
        return None
    return (Path(args.drive_root).resolve() / args.archive_id).as_posix()


def _issue_lines(issues: Sequence[Any], severity: str) -> list[str]:
    return [f"{issue.code}: {issue.message}" for issue in issues if issue.severity == severity]


def _emit(args: argparse.Namespace, payload: Mapping[str, Any]) -> None:
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    else:
        _print_human(payload)


def _print_human(payload: Mapping[str, Any]) -> None:
    print("Session archive bootstrap")
    print(f"Archive ID: {payload['archive_id']}")
    print(f"Local archive root: {payload['local_archive_root']}")
    print(f"Archive collision policy: {payload['archive_collision_policy']}")
    print(f"Copy policy: {payload['copy_policy']}")
    destination_archive_root = payload.get("destination_archive_root")
    if destination_archive_root:
        print(f"Destination archive root: {destination_archive_root}")
    print(f"Included groups: {', '.join(payload['included_logical_groups'])}")
    print(f"Shards: {payload['shard_count']}")
    print(f"Files: {payload['file_count']}")
    print(f"Size bytes: {payload['size_bytes']}")
    print(f"Copy status: {payload['copy_status']}")
    if payload.get("validation_status") != "not_requested":
        print(f"Validation status: {payload['validation_status']}")
    if payload.get("inspection_status") != "not_requested":
        print(f"Inspection status: {payload['inspection_status']}")
    if payload.get("bootstrap_report_path"):
        print(f"Bootstrap report: {payload['bootstrap_report_path']}")


if __name__ == "__main__":
    raise SystemExit(main())