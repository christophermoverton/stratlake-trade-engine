from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

_LEGACY_MODULE_PATH = Path(__file__).resolve().parent.parent / "robustness.py"


def _load_legacy_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_stratlake_legacy_research_robustness", _LEGACY_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load legacy robustness module from {_LEGACY_MODULE_PATH.as_posix()}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_legacy = _load_legacy_module()
for _name, _value in vars(_legacy).items():
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = _value

from .models import (  # noqa: E402
    MULTIPLE_TESTING_JSON_FIELDS,
    SAMPLE_SIZE_JSON_FIELDS,
    SENSITIVITY_SUMMARY_COLUMNS,
    WALK_FORWARD_EFFICIENCY_COLUMNS,
    ArtifactReference,
    MultipleTestingSummary,
    RobustnessFinding,
    RobustnessReport,
    RobustnessReportResult,
    SampleSizeValidation,
    SensitivitySummaryRow,
    UpstreamReferences,
    WalkForwardEfficiencyRow,
)
from .summary import build_robustness_summary  # noqa: E402
from .writer import (  # noqa: E402
    DEFAULT_ROBUSTNESS_ROOT,
    FINDINGS_FILENAME,
    MANIFEST_FILENAME,
    MULTIPLE_TESTING_FILENAME,
    REPORT_FILENAME,
    SAMPLE_SIZE_FILENAME,
    SENSITIVITY_FILENAME,
    SUMMARY_FILENAME,
    WALK_FORWARD_EFFICIENCY_FILENAME,
    write_robustness_report_bundle,
)

__all__ = [
    *(name for name in vars(_legacy) if not (name.startswith("__") and name.endswith("__"))),
    "ArtifactReference",
    "DEFAULT_ROBUSTNESS_ROOT",
    "FINDINGS_FILENAME",
    "MANIFEST_FILENAME",
    "MULTIPLE_TESTING_FILENAME",
    "MULTIPLE_TESTING_JSON_FIELDS",
    "MultipleTestingSummary",
    "REPORT_FILENAME",
    "RobustnessFinding",
    "RobustnessReport",
    "RobustnessReportResult",
    "SAMPLE_SIZE_FILENAME",
    "SAMPLE_SIZE_JSON_FIELDS",
    "SENSITIVITY_FILENAME",
    "SENSITIVITY_SUMMARY_COLUMNS",
    "SUMMARY_FILENAME",
    "SampleSizeValidation",
    "SensitivitySummaryRow",
    "UpstreamReferences",
    "WALK_FORWARD_EFFICIENCY_COLUMNS",
    "WALK_FORWARD_EFFICIENCY_FILENAME",
    "WalkForwardEfficiencyRow",
    "build_robustness_summary",
    "write_robustness_report_bundle",
]

del Any
