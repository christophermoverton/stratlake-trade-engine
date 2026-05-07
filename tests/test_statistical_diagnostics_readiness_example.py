from __future__ import annotations

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = REPO_ROOT / "docs" / "examples" / "statistical_diagnostics_readiness_example.py"
EXPECTED_FIELDS = {
    "t_stat",
    "p_value",
    "hit_rate_p_value",
    "autocorr_lag1",
    "effective_n",
    "split_mean_diff",
    "split_mean_diff_p",
    "rolling_sharpe_mean",
    "rolling_sharpe_sd",
    "sharpe_stability_ratio",
}


def _load_example_module():
    spec = spec_from_file_location(EXAMPLE_PATH.stem, EXAMPLE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_statistical_diagnostics_readiness_example_runs(tmp_path, monkeypatch, capsys) -> None:
    module = _load_example_module()

    monkeypatch.chdir(tmp_path)
    assert module.main() == 0
    captured = capsys.readouterr()

    output_dir = tmp_path / "docs" / "examples" / "output" / "statistical_diagnostics_readiness_example"
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    readiness = json.loads((output_dir / "metrics_readiness.json").read_text(encoding="utf-8"))

    assert EXPECTED_FIELDS.issubset(metrics)
    assert readiness["schema_version"] == 1
    assert readiness["source_metrics_artifact"] == "metrics.json"
    assert readiness["diagnostics"]["return_inference"]["t_stat"] == metrics["t_stat"]
    assert readiness["diagnostics"]["hit_rate"]["hit_rate_p_value"] == metrics["hit_rate_p_value"]
    assert "M30 statistical diagnostics readiness example" in captured.out
    assert "metrics_readiness.json" in captured.out


def test_statistical_diagnostics_readiness_example_uses_relative_paths() -> None:
    source = EXAMPLE_PATH.read_text(encoding="utf-8")

    assert "C:/" not in source
    assert "C:\\" not in source
    assert "/Users/" not in source
    assert "docs/examples/output/statistical_diagnostics_readiness_example" in source
