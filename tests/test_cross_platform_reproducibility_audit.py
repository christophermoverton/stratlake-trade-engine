from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_MD = REPO_ROOT / "docs" / "architecture" / "cross_platform_reproducibility_audit.md"
AUDIT_JSON = REPO_ROOT / "docs" / "architecture" / "cross_platform_reproducibility_audit.json"

FORBIDDEN_LOCAL_PATH_PATTERNS = (
    re.compile(r"[A-Za-z]:[\\/]"),
    re.compile(r"/(?:Users|home)/"),
    re.compile(r"file://"),
)


def test_cross_platform_audit_artifacts_exist() -> None:
    assert AUDIT_MD.exists()
    assert AUDIT_JSON.exists()


def test_cross_platform_audit_json_is_deterministic() -> None:
    payload = _load_audit_json()
    expected_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    assert AUDIT_JSON.read_text(encoding="utf-8") == expected_text


def test_cross_platform_audit_json_has_expected_top_level_keys() -> None:
    payload = _load_audit_json()
    assert set(payload) == {
        "audit_scope",
        "categories",
        "findings",
        "issue",
        "out_of_scope",
        "schema_version",
    }
    assert payload["schema_version"] == 1
    assert payload["issue"] == "360"


def test_cross_platform_audit_paths_are_repository_relative() -> None:
    payload = _load_audit_json()
    for finding in payload["findings"]:
        path = finding["path"]
        assert isinstance(path, str)
        assert path
        assert Path(path).is_absolute() is False
        assert "\\" not in path
        assert path == Path(path).as_posix()


def test_cross_platform_audit_artifacts_do_not_leak_local_absolute_paths() -> None:
    for artifact in (AUDIT_MD, AUDIT_JSON):
        text = artifact.read_text(encoding="utf-8")
        for pattern in FORBIDDEN_LOCAL_PATH_PATTERNS:
            assert not pattern.search(text), f"{artifact.relative_to(REPO_ROOT)} leaked {pattern.pattern}"


def test_cross_platform_audit_covers_required_finding_groups() -> None:
    payload = _load_audit_json()
    findings = payload["findings"]
    assert _has_finding(findings, "ci")
    assert _has_finding(findings, "env-paths")
    assert _has_finding(findings, "line-endings")
    assert _has_finding(findings, "packaging")


def _load_audit_json() -> dict[str, Any]:
    payload = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _has_finding(findings: list[dict[str, Any]], id_fragment: str) -> bool:
    return any(id_fragment in finding["id"] for finding in findings)
