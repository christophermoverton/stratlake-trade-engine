from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
GITATTRIBUTES = REPO_ROOT / ".gitattributes"

EXPECTED_TEXT_POLICIES = {
    "*": "text=auto eol=lf",
    ".env.example": "text eol=lf",
    ".gitattributes": "text eol=lf",
    ".gitignore": "text eol=lf",
    "*.bat": "text eol=lf",
    "*.cfg": "text eol=lf",
    "*.cmd": "text eol=lf",
    "*.css": "text eol=lf",
    "*.csv": "text eol=lf",
    "*.html": "text eol=lf",
    "*.ini": "text eol=lf",
    "*.js": "text eol=lf",
    "*.json": "text eol=lf",
    "*.jsonl": "text eol=lf",
    "*.lock": "text eol=lf",
    "*.md": "text eol=lf",
    "*.ps1": "text eol=lf",
    "*.py": "text eol=lf",
    "*.sh": "text eol=lf",
    "*.sql": "text eol=lf",
    "*.toml": "text eol=lf",
    "*.ts": "text eol=lf",
    "*.txt": "text eol=lf",
    "*.yaml": "text eol=lf",
    "*.yml": "text eol=lf",
}

EXPECTED_BINARY_POLICIES = {
    "*.db",
    "*.duckdb",
    "*.gif",
    "*.gz",
    "*.ico",
    "*.jpeg",
    "*.jpg",
    "*.parquet",
    "*.pdf",
    "*.pickle",
    "*.pkl",
    "*.png",
    "*.sqlite",
    "*.tar",
    "*.webp",
    "*.zip",
}


def test_gitattributes_uses_lf_line_endings() -> None:
    payload = GITATTRIBUTES.read_bytes()

    assert b"\r\n" not in payload
    assert payload.endswith(b"\n")


def test_gitattributes_declares_expected_text_lf_policies() -> None:
    policies = _gitattributes_policies()

    for pattern, attributes in EXPECTED_TEXT_POLICIES.items():
        assert policies.get(pattern) == attributes


def test_gitattributes_declares_binary_artifacts() -> None:
    policies = _gitattributes_policies()

    for pattern in EXPECTED_BINARY_POLICIES:
        assert policies.get(pattern) == "binary"


def test_gitattributes_does_not_mark_binary_artifacts_as_text() -> None:
    policies = _gitattributes_policies()

    for pattern in EXPECTED_BINARY_POLICIES:
        assert "text" not in policies[pattern]
        assert "eol=lf" not in policies[pattern]


def _gitattributes_policies() -> dict[str, str]:
    policies: dict[str, str] = {}
    for raw_line in GITATTRIBUTES.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        pattern, attributes = line.split(maxsplit=1)
        policies[pattern] = attributes
    return policies
