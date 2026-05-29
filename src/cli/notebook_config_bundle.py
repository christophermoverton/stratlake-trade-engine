from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from src.session.contracts import NotebookProjectSession

NOTEBOOK_CONFIG_TARGETS: tuple[str, ...] = (
    "configs/paths.yml",
    "configs/universe.yml",
    "configs/tickers_sample.txt",
)

SAMPLE_TICKERS: tuple[str, ...] = (
    "AAPL",
    "AMZN",
    "GOOGL",
    "JNJ",
    "JPM",
    "META",
    "MSFT",
    "NVDA",
    "TSLA",
    "WMT",
)


@dataclass(frozen=True)
class NotebookConfigBundleResult:
    requested: bool
    force: bool
    config_dir: str
    generated: tuple[str, ...]
    overwritten: tuple[str, ...]
    skipped: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "requested": self.requested,
            "force": self.force,
            "config_dir": self.config_dir,
            "generated": list(self.generated),
            "overwritten": list(self.overwritten),
            "skipped": list(self.skipped),
        }


def initialize_notebook_config_bundle(
    *,
    project_root: Path,
    session: NotebookProjectSession,
    force: bool,
) -> NotebookConfigBundleResult:
    root = project_root.expanduser().resolve()
    files = {
        "configs/paths.yml": _render_paths_yaml(session),
        "configs/universe.yml": _render_universe_yaml(),
        "configs/tickers_sample.txt": _render_tickers_sample(),
    }

    generated: list[str] = []
    overwritten: list[str] = []
    skipped: list[str] = []

    for relative, content in files.items():
        destination = (root / relative).resolve()
        if not destination.is_relative_to(root):
            raise ValueError(
                f"Refusing to write notebook configs outside project root: {destination.as_posix()}"
            )
        if destination.exists() and destination.is_dir():
            raise ValueError(
                "Notebook config destination exists and is not a file: "
                f"{destination.as_posix()}"
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not force:
            skipped.append(relative)
            continue
        if destination.exists():
            overwritten.append(relative)
        else:
            generated.append(relative)
        destination.write_text(content, encoding="utf-8", newline="\n")

    return NotebookConfigBundleResult(
        requested=True,
        force=force,
        config_dir="configs",
        generated=tuple(generated),
        overwritten=tuple(overwritten),
        skipped=tuple(skipped),
    )


def notebook_config_bundle_not_requested() -> NotebookConfigBundleResult:
    return NotebookConfigBundleResult(
        requested=False,
        force=False,
        config_dir="configs",
        generated=(),
        overwritten=(),
        skipped=NOTEBOOK_CONFIG_TARGETS,
    )


def _render_paths_yaml(session: NotebookProjectSession) -> str:
    drive_root = None if session.drive_root is None else session.drive_root.path
    payload: dict[str, object] = {
        "project_root": session.project_root.path,
        "configs_root": session.configs_root.path,
        "artifacts_root": session.artifacts_root.path,
        "features_root": session.features_root.path,
        "marketlake_root": session.marketlake_root.path,
        "drive_root": drive_root,
        "tickers_file": "configs/tickers_sample.txt",
        "path_kinds": {
            "project_root": session.project_root.kind.value,
            "configs_root": session.configs_root.kind.value,
            "artifacts_root": session.artifacts_root.kind.value,
            "features_root": session.features_root.kind.value,
            "marketlake_root": session.marketlake_root.kind.value,
            "drive_root": None if session.drive_root is None else session.drive_root.kind.value,
        },
    }
    return _yaml_dump(payload)


def _render_universe_yaml() -> str:
    payload: dict[str, object] = {
        "dataset": "features_daily",
        "name": "notebook_sample_universe",
        "tickers_file": "configs/tickers_sample.txt",
        "timeframe": "1D",
    }
    return _yaml_dump(payload)


def _render_tickers_sample() -> str:
    return "\n".join(SAMPLE_TICKERS) + "\n"


def _yaml_dump(payload: dict[str, object]) -> str:
    return yaml.safe_dump(
        payload,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=False,
    )
