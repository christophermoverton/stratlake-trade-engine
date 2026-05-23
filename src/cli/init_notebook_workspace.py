from __future__ import annotations

import argparse
from importlib import resources
import sys
from pathlib import Path
from typing import Sequence

STARTER_CONFIGS: tuple[str, ...] = (
    "features.yml",
    "strategies.yml",
    "evaluation.yml",
    "execution.yml",
    "portfolios.yml",
    "candidate_selection.yml",
    "review.yml",
    "profiles/notebook.yml",
    "pipelines/scenario_matrix_pipeline.yml",
)

STARTER_DOCS: tuple[str, ...] = (
    "getting_started.md",
    "notebook_integration.md",
    "notebook_execution_api.md",
    "pipeline_integration.md",
    "concurrency_and_idempotency.md",
    "cross_layer_validation.md",
    "catalog_notebook_ergonomics.md",
    "examples/notebook_execution_api_examples.py",
)

WORKSPACE_DIRS: tuple[str, ...] = ("notebooks", "configs", "docs", "contracts", "artifacts")
NOTEBOOK_WORKSPACE_RESOURCE_PACKAGE = "src.resources.notebook_workspace"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Initialize a local StratLake notebook workspace with starter configs, docs, and "
            "workspace directories."
        )
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Target workspace root directory. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite copied starter templates if they already exist.",
    )
    return parser.parse_args(argv)


def run_cli(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    summary = initialize_notebook_workspace(Path(args.root), force=args.force)
    print_summary(summary)
    return summary


def initialize_notebook_workspace(root: Path, *, force: bool = False) -> dict[str, object]:
    workspace_root = root.expanduser().resolve()
    resource_root = _resolve_resource_root()

    before_exists = workspace_root.exists()
    workspace_root.mkdir(parents=True, exist_ok=True)

    created_dirs: list[str] = []
    existing_dirs: list[str] = []
    for relative in WORKSPACE_DIRS:
        destination = _destination_path(workspace_root, relative)
        if destination.exists():
            existing_dirs.append(relative)
        else:
            destination.mkdir(parents=True, exist_ok=True)
            created_dirs.append(relative)

    copied: list[str] = []
    overwritten: list[str] = []
    skipped: list[str] = []

    copied_configs, overwritten_configs, skipped_configs = _copy_starters(
        workspace_root=workspace_root,
        resource_root=resource_root,
        source_folder="configs",
        destination_folder="configs",
        allowlist=STARTER_CONFIGS,
        force=force,
    )
    copied.extend(copied_configs)
    overwritten.extend(overwritten_configs)
    skipped.extend(skipped_configs)

    copied_docs, overwritten_docs, skipped_docs = _copy_starters(
        workspace_root=workspace_root,
        resource_root=resource_root,
        source_folder="docs",
        destination_folder="docs",
        allowlist=STARTER_DOCS,
        force=force,
    )
    copied.extend(copied_docs)
    overwritten.extend(overwritten_docs)
    skipped.extend(skipped_docs)

    return {
        "root": workspace_root.as_posix(),
        "workspace_preexisting": before_exists,
        "created_dirs": created_dirs,
        "existing_dirs": existing_dirs,
        "copied": copied,
        "overwritten": overwritten,
        "skipped": skipped,
        "force": force,
    }


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run_cli(argv)
    except (FileNotFoundError, NotADirectoryError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def _resolve_resource_root() -> resources.abc.Traversable:
    try:
        resource_root = resources.files(NOTEBOOK_WORKSPACE_RESOURCE_PACKAGE)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Notebook starter templates are unavailable in this installation. "
            f"Missing package resources: {NOTEBOOK_WORKSPACE_RESOURCE_PACKAGE}."
        ) from exc

    missing = []
    if not resource_root.joinpath("configs").is_dir():
        missing.append("configs")
    if not resource_root.joinpath("docs").is_dir():
        missing.append("docs")
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            "Notebook starter templates are unavailable in this installation. "
            f"Missing resource directories: {joined}."
        )
    return resource_root


def _copy_starters(
    *,
    workspace_root: Path,
    resource_root: resources.abc.Traversable,
    source_folder: str,
    destination_folder: str,
    allowlist: Sequence[str],
    force: bool,
) -> tuple[list[str], list[str], list[str]]:
    copied: list[str] = []
    overwritten: list[str] = []
    skipped: list[str] = []

    for relative in allowlist:
        source = _resource_path(resource_root, f"{source_folder}/{relative}")
        if not source.is_file():
            raise FileNotFoundError(f"Starter template not found: {source_folder}/{relative}")

        destination = _destination_path(workspace_root, f"{destination_folder}/{relative}")
        parent = destination.parent
        if parent.exists() and not parent.is_dir():
            raise NotADirectoryError(
                "Expected parent directory for starter template but found non-directory path: "
                f"{parent.as_posix()}"
            )
        parent.mkdir(parents=True, exist_ok=True)

        if destination.exists() and not destination.is_file():
            raise ValueError(
                f"Starter destination exists and is not a file: {destination.as_posix()}"
            )

        if destination.exists() and not force:
            skipped.append(f"{destination_folder}/{relative}")
            continue

        if destination.exists() and force:
            overwritten.append(f"{destination_folder}/{relative}")
        else:
            copied.append(f"{destination_folder}/{relative}")

        destination.write_bytes(source.read_bytes())

    return copied, overwritten, skipped


def _destination_path(workspace_root: Path, relative: str) -> Path:
    destination = (workspace_root / relative).resolve()
    if not destination.is_relative_to(workspace_root):
        raise ValueError(
            f"Refusing to write outside workspace root: {destination.as_posix()}"
        )
    return destination


def _resource_path(
    resource_root: resources.abc.Traversable,
    relative: str,
) -> resources.abc.Traversable:
    candidate = resource_root
    for part in relative.split("/"):
        candidate = candidate.joinpath(part)
    return candidate


def print_summary(summary: dict[str, object]) -> None:
    print(f"Initialized StratLake notebook workspace at: {summary['root']}")
    print(
        "Directory status: "
        f"created={len(summary['created_dirs'])}, "
        f"already_present={len(summary['existing_dirs'])}"
    )
    print(
        "Template status: "
        f"copied={len(summary['copied'])}, "
        f"overwritten={len(summary['overwritten'])}, "
        f"skipped={len(summary['skipped'])}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
