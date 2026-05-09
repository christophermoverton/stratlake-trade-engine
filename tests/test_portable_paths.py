from __future__ import annotations

import json
from pathlib import Path
import re

from src.artifacts.safety import portable_path
from src.research.governance.writer import run_promotion_governance_report
from src.research.registry import append_registry_entry
from src.validation.docs_path_lint import lint_guarded_surfaces


REPO_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_LOCAL_PATH_PATTERNS = (
    re.compile(r"[A-Za-z]:[\\/]"),
    re.compile(r"/(?:Users|home)/"),
    re.compile(r"file://"),
)


def test_portable_path_normalizes_relative_windows_path() -> None:
    rendered = portable_path("docs\\examples\\output\\manifest.json")

    assert rendered == "docs/examples/output/manifest.json"
    assert "\\" not in rendered


def test_portable_path_prefers_repository_relative_path() -> None:
    rendered = portable_path(REPO_ROOT / "docs" / "architecture", roots=(REPO_ROOT,))

    assert rendered == "docs/architecture"
    assert "\\" not in rendered


def test_portable_path_handles_windows_absolute_path_under_root() -> None:
    rendered = portable_path(
        "C:\\repo\\stratlake-trade-engine\\artifacts\\run\\manifest.json",
        roots=("C:\\repo\\stratlake-trade-engine",),
    )

    assert rendered == "artifacts/run/manifest.json"
    assert "\\" not in rendered


def test_portable_path_does_not_leak_unrooted_local_absolute_paths() -> None:
    rendered_values = [
        portable_path("C:\\Users\\Example\\secret\\manifest.json"),
        portable_path("C:/Users/Example/secret/manifest.json"),
        portable_path("/Users/example/secret/manifest.json"),
        portable_path("/home/example/secret/manifest.json"),
        portable_path("file:///Users/example/secret/manifest.json"),
    ]

    for rendered in rendered_values:
        assert rendered
        assert "\\" not in rendered
        assert not any(pattern.search(rendered) for pattern in FORBIDDEN_LOCAL_PATH_PATTERNS)


def test_env_example_is_guarded_by_docs_path_lint() -> None:
    report = lint_guarded_surfaces(REPO_ROOT)

    assert ".env.example" in report["guarded_surfaces"]
    assert report["status"] == "passed"


def test_governance_manifest_uses_portable_paths_for_sources(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    strategy_dir = artifact_root / "strategies" / "portable_strategy"
    registry_path = artifact_root / "strategies" / "registry.jsonl"
    summary = {
        "promotion_status": "eligible",
        "evaluation_status": "pass",
        "decision_reason_codes": [],
        "gate_count": 1,
    }
    strategy_dir.mkdir(parents=True)
    (strategy_dir / "manifest.json").write_text(
        json.dumps({"run_id": "portable_strategy", "promotion_gate_summary": summary}),
        encoding="utf-8",
        newline="\n",
    )
    append_registry_entry(
        registry_path,
        {
            "run_id": "portable_strategy",
            "run_type": "strategy",
            "artifact_path": strategy_dir.as_posix(),
            "promotion_status": "eligible",
            "review_status": "candidate",
            "promotion_gate_summary": summary,
        },
    )

    result = run_promotion_governance_report(
        registry_path=registry_path,
        artifact_root=artifact_root,
        output_dir=tmp_path / "governance",
        report_id="portable_paths",
    )

    manifest_text = result.manifest_path.read_text(encoding="utf-8")
    assert "\\" not in manifest_text
    assert str(tmp_path) not in manifest_text
    assert tmp_path.as_posix() not in manifest_text
    assert not any(pattern.search(manifest_text) for pattern in FORBIDDEN_LOCAL_PATH_PATTERNS)
