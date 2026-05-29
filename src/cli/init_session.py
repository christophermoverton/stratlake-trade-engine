from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from src.cli.init_notebook_workspace import initialize_notebook_workspace
from src.cli.notebook_config_bundle import (
    initialize_notebook_config_bundle,
    notebook_config_bundle_not_requested,
)
from src.session import create_notebook_project_session, write_session_files
from src.session.io import PATH_RESOLUTION_FILE_NAME, SESSION_DIR_NAME, SESSION_FILE_NAME


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Initialize a StratLake notebook workspace and write deterministic "
            "project-session metadata."
        )
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Target StratLake project root. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--project-name",
        default=None,
        help="Project name recorded in session metadata. Defaults to the root directory name.",
    )
    parser.add_argument(
        "--marketlake-root",
        default="data/curated",
        help="MarketLake curated-data root to record for this session.",
    )
    parser.add_argument(
        "--drive-root",
        default=None,
        help="Optional Google Drive or cloud persistence root to record as metadata only.",
    )
    parser.add_argument(
        "--enable-drive-persistence",
        action="store_true",
        help=(
            "Record Drive persistence intent in session metadata. This does not sync, "
            "export, import, or copy Drive files."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Overwrite known starter templates through the notebook bootstrap and refresh "
            "session metadata."
        ),
    )
    parser.add_argument(
        "--notebook-configs",
        action="store_true",
        help=(
            "Generate a notebook-ready config bundle: configs/paths.yml, "
            "configs/universe.yml, and configs/tickers_sample.txt."
        ),
    )
    parser.add_argument(
        "--force-notebook-configs",
        action="store_true",
        help=(
            "Overwrite existing notebook config bundle files. Requires "
            "--notebook-configs."
        ),
    )
    args = parser.parse_args(argv)
    if args.enable_drive_persistence and not args.drive_root:
        parser.error("--enable-drive-persistence requires --drive-root")
    if args.force_notebook_configs and not args.notebook_configs:
        parser.error("--force-notebook-configs requires --notebook-configs")
    return args


def run_cli(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    root = Path(args.root).expanduser().resolve()
    _preflight_session_metadata(root, force=args.force)

    bootstrap_summary = initialize_notebook_workspace(root, force=args.force)
    session = create_notebook_project_session(
        project_root=root,
        project_name=args.project_name,
        marketlake_root=args.marketlake_root,
        drive_root=args.drive_root,
        drive_persistence_enabled=args.enable_drive_persistence,
    )
    write_result = write_session_files(session, overwrite=args.force)
    notebook_configs = notebook_config_bundle_not_requested()
    if args.notebook_configs:
        notebook_configs = initialize_notebook_config_bundle(
            project_root=root,
            session=session,
            force=args.force_notebook_configs,
        )
    _write_init_session_diagnostics(
        write_result.path_resolution_path,
        session=session,
        notebook_config_summary=notebook_configs.to_dict(),
    )
    summary: dict[str, object] = {
        "bootstrap": bootstrap_summary,
        "session": session.to_dict(),
        "session_path": write_result.session_path.as_posix(),
        "path_resolution_path": write_result.path_resolution_path.as_posix(),
        "drive_persistence_enabled": args.enable_drive_persistence,
        "notebook_configs": notebook_configs.to_dict(),
    }
    print_summary(summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run_cli(argv)
    except (
        FileExistsError,
        FileNotFoundError,
        NotADirectoryError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def _preflight_session_metadata(root: Path, *, force: bool) -> None:
    session_dir = (root / SESSION_DIR_NAME).resolve()
    _ensure_under_root(session_dir, root)
    session_path = (session_dir / SESSION_FILE_NAME).resolve()
    resolution_path = (session_dir / PATH_RESOLUTION_FILE_NAME).resolve()
    _ensure_under_root(session_path, root)
    _ensure_under_root(resolution_path, root)
    if force:
        return

    existing = [path for path in (session_path, resolution_path) if path.exists()]
    if existing:
        joined = ", ".join(path.as_posix() for path in existing)
        raise FileExistsError(
            "Session metadata already exists. Re-run with --force to refresh known "
            f"session files: {joined}"
        )


def _ensure_under_root(path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Refusing to write session metadata outside project root: {path.as_posix()}"
        ) from exc


def print_summary(summary: dict[str, object]) -> None:
    bootstrap = summary["bootstrap"]
    session = summary["session"]

    print(f"Initialized StratLake notebook session at: {bootstrap['root']}")
    print(
        "Template status: "
        f"copied={len(bootstrap['copied'])}, "
        f"overwritten={len(bootstrap['overwritten'])}, "
        f"skipped={len(bootstrap['skipped'])}"
    )
    print("Resolved roots:")
    for key in (
        "project_root",
        "configs_root",
        "artifacts_root",
        "features_root",
        "marketlake_root",
        "drive_root",
    ):
        if key in session:
            value = session[key]
            print(f"  {key}: {value['path']} ({value['kind']}, {value['source']})")
    drive_persistence = session["drive_persistence"]
    print(
        "Drive persistence: "
        f"{'enabled' if drive_persistence['enabled'] else 'disabled'} "
        f"({drive_persistence['mode']})"
    )
    print("Session metadata:")
    print(f"  session: {summary['session_path']}")
    print(f"  path_resolution: {summary['path_resolution_path']}")
    notebook_configs = summary["notebook_configs"]
    print(
        "Notebook config bundle: "
        f"requested={'yes' if notebook_configs['requested'] else 'no'}, "
        f"generated={len(notebook_configs['generated'])}, "
        f"overwritten={len(notebook_configs['overwritten'])}, "
        f"skipped={len(notebook_configs['skipped'])}"
    )
    print("Next: open notebooks from this project root and pass explicit paths to StratLake APIs.")


def _write_init_session_diagnostics(
    resolution_path: Path,
    *,
    session,
    notebook_config_summary: dict[str, object],
) -> None:
    payload = json.loads(resolution_path.read_text(encoding="utf-8"))
    payload["notebook_config_bundle"] = {
        **notebook_config_summary,
        "session_root": session.project_root.resolved_path,
        "project_name": session.project_name,
        "marketlake_root": {
            "path": session.marketlake_root.path,
            "kind": session.marketlake_root.kind.value,
        },
        "drive_root": None
        if session.drive_root is None
        else {
            "path": session.drive_root.path,
            "kind": session.drive_root.kind.value,
        },
    }
    resolution_path.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
