from __future__ import annotations

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = REPO_ROOT / "docs" / "examples" / "m31_readiness_gated_promotion_case_study.py"


def _load_example_module():
    spec = spec_from_file_location(EXAMPLE_PATH.stem, EXAMPLE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _iter_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_strings(item)


def test_m31_readiness_gated_promotion_case_study_runs(tmp_path, monkeypatch, capsys) -> None:
    module = _load_example_module()

    monkeypatch.chdir(tmp_path)
    assert module.main() == 0
    captured = capsys.readouterr()

    output_dir = tmp_path / "docs" / "examples" / "output" / "m31_readiness_gated_promotion_case_study"
    campaign_summary = json.loads((output_dir / "campaign_summary.json").read_text(encoding="utf-8"))
    candidate_review_summary = json.loads(
        (output_dir / "candidate_review_summary.json").read_text(encoding="utf-8")
    )
    review_summary = json.loads((output_dir / "review_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    assert "M31 readiness-gated promotion case study" in captured.out
    assert review_summary["promotion_status_counts"] == {
        "blocked": 1,
        "eligible": 1,
        "needs_review": 1,
        "warn": 1,
    }
    assert campaign_summary["final_outcomes"]["review_promotion_status"] == "blocked"
    assert campaign_summary["final_outcomes"]["review_promotion_highest_severity"] == "block"
    assert campaign_summary["final_outcomes"]["review_promotion_decision_reason_codes"] == [
        "gate_failed_threshold",
        "severity_block",
        "severity_review",
        "severity_warn",
    ]
    assert candidate_review_summary["promotion_context"]["candidate_promotion_status_counts"] == {
        "blocked": 1,
        "eligible": 1,
        "needs_review": 1,
        "warn": 1,
    }
    assert candidate_review_summary["promotion_context"]["portfolio_promotion_gate_summary"][
        "highest_severity"
    ] == "warn"
    assert "runs/blocked/promotion_gates.json" in manifest["artifact_files"]

    warn_gates = json.loads((output_dir / "runs" / "warn" / "promotion_gates.json").read_text(encoding="utf-8"))
    assert warn_gates["promotion_status"] == "warn"
    assert warn_gates["highest_severity"] == "warn"
    assert warn_gates["severity_counts"] == {
        "block": 0,
        "reject": 0,
        "review": 0,
        "warn": 2,
    }


def test_m31_readiness_gated_promotion_case_study_outputs_are_relative(
    tmp_path,
    monkeypatch,
) -> None:
    module = _load_example_module()

    monkeypatch.chdir(tmp_path)
    assert module.main() == 0
    output_dir = tmp_path / "docs" / "examples" / "output" / "m31_readiness_gated_promotion_case_study"

    for path in output_dir.rglob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        strings = list(_iter_strings(payload))
        assert not any("C:/" in item or "C:\\\\" in item for item in strings)
        assert not any(str(tmp_path) in item for item in strings)


def test_m31_readiness_gated_promotion_case_study_source_uses_canonical_config() -> None:
    source = EXAMPLE_PATH.read_text(encoding="utf-8")

    assert "configs/statistical_readiness_promotion_gates_example.yml" in source
    assert "source: metrics" not in source
    assert "promotion_decision.json" not in source
    assert "promotion_readiness.json" not in source
