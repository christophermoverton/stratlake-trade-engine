from __future__ import annotations

from pathlib import Path

from src.validation.docs_path_lint import lint_guarded_surfaces


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_docs_path_lint_detects_absolute_path_leakage(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "leak.md").write_text(
        "Reference: C:/Users/example/local/path/file.md\n",
        encoding="utf-8",
    )

    report = lint_guarded_surfaces(tmp_path, guarded_surfaces=("docs/**/*.md",))
    assert report["status"] == "failed"
    assert report["finding_count"] >= 1
    assert any(finding["rule"] == "windows_absolute_path" for finding in report["findings"])


def test_repo_guarded_surfaces_have_no_absolute_path_leakage() -> None:
    report = lint_guarded_surfaces(REPO_ROOT)
    assert report["status"] == "passed", _render_findings(report)
    assert report["finding_count"] == 0


def _render_findings(report: dict[str, object]) -> str:
    findings = list(report.get("findings", []))
    if not findings:
        return ""
    preview = findings[:10]
    lines = [
        "Guarded surfaces contain absolute-path findings:",
    ]
    for finding in preview:
        lines.append(
            "- "
            + f"{finding['file_path']}:{finding['line']}:{finding['column']} "
            + f"[{finding['rule']}] {finding['matched_text']}"
        )
    return "\n".join(lines)
