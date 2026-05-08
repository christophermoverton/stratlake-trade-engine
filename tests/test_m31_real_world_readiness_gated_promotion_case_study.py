from __future__ import annotations

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = REPO_ROOT / "docs" / "examples" / "m31_real_world_readiness_gated_promotion_case_study.py"


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


def test_m31_real_world_readiness_gated_promotion_case_study_runs(tmp_path, capsys) -> None:
    module = _load_example_module()

    artifacts = module.run_case_study(output_root=tmp_path / "m31_real_world_case_study", verbose=True)
    captured = capsys.readouterr()

    output_dir = artifacts.output_root
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    campaign_summary = json.loads((output_dir / "campaign_summary.json").read_text(encoding="utf-8"))
    candidate_review_summary = json.loads(
        (output_dir / "candidate_review_summary.json").read_text(encoding="utf-8")
    )
    review_summary = json.loads((output_dir / "review_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    assert "M31 real-world readiness-gated promotion case study" in captured.out
    assert summary["promotion_status_counts"] == {
        "blocked": 1,
        "eligible": 1,
        "needs_review": 1,
    }
    assert review_summary["review_status_counts"] == {
        "candidate": 1,
        "needs_review": 1,
        "rejected": 1,
    }
    assert campaign_summary["final_outcomes"]["review_promotion_status"] == "blocked"
    assert campaign_summary["final_outcomes"]["review_promotion_highest_severity"] == "block"
    assert candidate_review_summary["promotion_context"]["candidate_promotion_status_counts"] == {
        "blocked": 1,
        "eligible": 1,
        "needs_review": 1,
    }
    assert candidate_review_summary["promotion_context"]["portfolio_promotion_gate_summary"][
        "promotion_status"
    ] == "eligible"
    assert manifest["data_source"] == "pinned_market_shaped_fixture"
    assert manifest["live_market_data_required"] is False
    assert manifest["network_required"] is False
    assert "runs/broad_market_momentum/promotion_gates.json" in manifest["artifact_files"]

    eligible_gates = json.loads(
        (output_dir / "runs" / "broad_market_momentum" / "promotion_gates.json").read_text(encoding="utf-8")
    )
    blocked_gates = json.loads(
        (output_dir / "runs" / "short_history_breakout" / "promotion_gates.json").read_text(encoding="utf-8")
    )
    assert eligible_gates["promotion_status"] == "eligible"
    assert eligible_gates["decision_reason_codes"] == []
    assert blocked_gates["promotion_status"] == "blocked"
    assert blocked_gates["highest_severity"] == "block"
    assert blocked_gates["decision_reason_codes"] == [
        "gate_failed_threshold",
        "severity_block",
        "severity_review",
    ]


def test_m31_real_world_readiness_gated_promotion_case_study_is_deterministic(tmp_path) -> None:
    module = _load_example_module()

    first = module.run_case_study(output_root=tmp_path / "first", verbose=False)
    second = module.run_case_study(output_root=tmp_path / "second", verbose=False)

    assert first.summary == second.summary
    assert first.summary_path.read_text(encoding="utf-8") == second.summary_path.read_text(encoding="utf-8")


def test_m31_real_world_readiness_gated_promotion_case_study_outputs_are_portable(tmp_path) -> None:
    module = _load_example_module()

    artifacts = module.run_case_study(output_root=tmp_path / "portable", verbose=False)

    forbidden_artifact_names = {"promotion_decision.json", "promotion_readiness.json"}
    assert not any(path.name in forbidden_artifact_names for path in artifacts.output_root.rglob("*"))

    for path in artifacts.output_root.rglob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        strings = list(_iter_strings(payload))
        assert not any("C:/" in item or "C:\\\\" in item for item in strings)
        assert not any(str(tmp_path) in item for item in strings)


def test_m31_real_world_readiness_gated_promotion_case_study_documents_reused_pattern() -> None:
    source = EXAMPLE_PATH.read_text(encoding="utf-8")

    assert "real_world_campaign_case_study.py" in source
    assert "pinned market-shaped fixture" in source
    assert "configs/statistical_readiness_promotion_gates_example.yml" in source
    assert "promotion_decision.json" not in source
    assert "promotion_readiness.json" not in source
