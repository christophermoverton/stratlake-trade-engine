from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any, Sequence


DEFAULT_GUARDED_SURFACES: tuple[str, ...] = (
    "README.md",
    "docs/**/*.md",
    "docs/examples/**/*.md",
    "docs/examples/**/*.py",
    "examples/**/*.py",
)

_WINDOWS_ABSOLUTE_PATH = re.compile(r"(?P<path>(?:[A-Za-z]:[\\/]|/[A-Za-z]:/)[^\s\])\[\(\)\"'`<>]+)")
_UNIX_HOME_ABSOLUTE_PATH = re.compile(r"(?P<path>/(?:Users|home)/[^\s\])\[\(\)\"'`<>]+)")
_FILE_URI_PATH = re.compile(r"(?P<path>file://[^\s\])\[\(\)\"'`<>]+)")

_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("windows_absolute_path", _WINDOWS_ABSOLUTE_PATH),
    ("unix_home_absolute_path", _UNIX_HOME_ABSOLUTE_PATH),
    ("file_uri_path", _FILE_URI_PATH),
)


@dataclass(frozen=True)
class PathLintFinding:
    file_path: str
    line: int
    column: int
    rule: str
    matched_text: str
    line_text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def lint_guarded_surfaces(
    repo_root: str | Path,
    guarded_surfaces: Sequence[str] | None = None,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    surfaces = tuple(guarded_surfaces or DEFAULT_GUARDED_SURFACES)
    files = _resolve_guarded_files(root, surfaces)
    findings: list[PathLintFinding] = []
    for file_path in files:
        findings.extend(_lint_file(file_path=file_path, repo_root=root))

    payload = {
        "run_type": "docs_path_lint",
        "schema_version": 1,
        "status": "passed" if not findings else "failed",
        "guarded_surfaces": list(surfaces),
        "guarded_file_count": len(files),
        "finding_count": len(findings),
        "findings": [finding.to_dict() for finding in findings],
    }
    return payload


def write_docs_path_lint_report(report: dict[str, Any], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _resolve_guarded_files(repo_root: Path, guarded_surfaces: Sequence[str]) -> list[Path]:
    files: set[Path] = set()
    for pattern in guarded_surfaces:
        for candidate in repo_root.glob(pattern):
            if candidate.is_file():
                files.add(candidate)
    return sorted(files)


def _lint_file(*, file_path: Path, repo_root: Path) -> list[PathLintFinding]:
    findings: list[PathLintFinding] = []
    text = file_path.read_text(encoding="utf-8")
    relative = file_path.relative_to(repo_root).as_posix()
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        for rule_name, pattern in _RULES:
            for match in pattern.finditer(raw_line):
                matched_text = match.group("path")
                findings.append(
                    PathLintFinding(
                        file_path=relative,
                        line=line_number,
                        column=match.start("path") + 1,
                        rule=rule_name,
                        matched_text=matched_text,
                        line_text=raw_line.strip(),
                    )
                )
    return findings
