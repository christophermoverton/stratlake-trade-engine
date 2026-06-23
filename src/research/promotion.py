from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import yaml

from src.research.registry import canonicalize_value, serialize_canonical_json

DEFAULT_PROMOTION_ARTIFACT_FILENAME = "promotion_gates.json"
PROMOTION_STATE_SCHEMA_VERSION = 2
PROMOTION_STATE_ARTIFACT_TYPE = "promotion_state"
DEFAULT_STATUS_ON_PASS = "eligible"
DEFAULT_STATUS_ON_FAIL = "blocked"
SUPPORTED_PROMOTION_STATE_RUN_TYPES = frozenset(
    {
        "alpha_evaluation",
        "portfolio",
        "research_campaign",
        "research_campaign_orchestration",
        "research_campaign_scenario",
        "review",
        "strategy",
    }
)
_CANONICAL_MACHINE_PROMOTION_STATUSES = frozenset(
    {"eligible", "warn", "needs_review", "rejected", "blocked"}
)
_UNCONFIGURED_DECISION_REASON_CODES = ("promotion_policy_not_configured",)
_PROMOTION_STATE_GATE_COUNT_KEYS = frozenset(
    {
        "total",
        "passed",
        "failed",
        "missing",
        "skipped",
        "warning",
        "review",
        "rejected",
        "blocked",
    }
)
_SUPPORTED_SEVERITIES = frozenset({"warn", "review", "reject", "block"})
_SEVERITY_RANK = {
    "warn": 0,
    "review": 1,
    "reject": 2,
    "block": 3,
}
_SEVERITY_PROMOTION_STATUS = {
    "warn": "warn",
    "review": "needs_review",
    "reject": "rejected",
    "block": "blocked",
}
_SEVERITY_REASON_CODES = {
    "warn": "severity_warn",
    "review": "severity_review",
    "reject": "severity_reject",
    "block": "severity_block",
}
_SUPPORTED_SOURCES = frozenset(
    {
        "metrics",
        "qa_summary",
        "manifest",
        "config",
        "metadata",
        "split_metrics",
        "timeseries",
        "aggregate_metrics",
    }
)
_SUPPORTED_STATISTICS = frozenset(
    {
        "value",
        "count",
        "non_null_count",
        "min",
        "max",
        "mean",
        "median",
        "std",
        "range",
        "abs_max",
    }
)
_SUPPORTED_COMPARATORS = frozenset({"gt", "gte", "lt", "lte", "eq", "ne"})


class PromotionGateError(ValueError):
    """Raised when promotion gate configuration is malformed."""


@dataclass(frozen=True)
class PromotionGateDefinition:
    gate_id: str
    source: str
    metric_path: str
    comparator: str
    threshold: float
    statistic: str = "value"
    missing_behavior: str = "fail"
    description: str | None = None
    severity: str | None = None


@dataclass(frozen=True)
class PromotionGateResult:
    gate_id: str
    status: str
    source: str
    metric_path: str
    statistic: str
    comparator: str
    threshold: float
    actual_value: float | None
    missing_behavior: str
    description: str | None
    severity: str | None
    reason_codes: list[str]
    reason: str


@dataclass(frozen=True)
class PromotionGateEvaluation:
    configured: bool
    run_type: str
    evaluation_status: str
    promotion_status: str | None
    status_on_pass: str | None
    status_on_fail: str | None
    gate_count: int
    passed_gate_count: int
    failed_gate_count: int
    missing_gate_count: int
    highest_severity: str | None
    severity_counts: dict[str, int]
    warning_gate_count: int
    review_gate_count: int
    rejected_gate_count: int
    blocked_gate_count: int
    decision_reason_codes: list[str]
    artifact_filename: str
    definitions: list[PromotionGateDefinition]
    results: list[PromotionGateResult]

    def to_payload(self) -> dict[str, Any]:
        return canonicalize_value(
            {
                "configured": self.configured,
                "run_type": self.run_type,
                "evaluation_status": self.evaluation_status,
                "promotion_status": self.promotion_status,
                "status_on_pass": self.status_on_pass,
                "status_on_fail": self.status_on_fail,
                "gate_count": self.gate_count,
                "passed_gate_count": self.passed_gate_count,
                "failed_gate_count": self.failed_gate_count,
                "missing_gate_count": self.missing_gate_count,
                "highest_severity": self.highest_severity,
                "severity_counts": self.severity_counts,
                "warning_gate_count": self.warning_gate_count,
                "review_gate_count": self.review_gate_count,
                "rejected_gate_count": self.rejected_gate_count,
                "blocked_gate_count": self.blocked_gate_count,
                "decision_reason_codes": self.decision_reason_codes,
                "artifact_filename": self.artifact_filename,
                "definitions": [asdict(definition) for definition in self.definitions],
                "results": [asdict(result) for result in self.results],
            }
        )

    def summary(self) -> dict[str, Any]:
        return canonicalize_value(
            {
                "configured": self.configured,
                "evaluation_status": self.evaluation_status,
                "promotion_status": self.promotion_status,
                "status_on_pass": self.status_on_pass,
                "status_on_fail": self.status_on_fail,
                "gate_count": self.gate_count,
                "passed_gate_count": self.passed_gate_count,
                "failed_gate_count": self.failed_gate_count,
                "missing_gate_count": self.missing_gate_count,
                "highest_severity": self.highest_severity,
                "severity_counts": self.severity_counts,
                "warning_gate_count": self.warning_gate_count,
                "review_gate_count": self.review_gate_count,
                "rejected_gate_count": self.rejected_gate_count,
                "blocked_gate_count": self.blocked_gate_count,
                "decision_reason_codes": self.decision_reason_codes,
                "artifact_filename": self.artifact_filename,
            }
        )


@dataclass(frozen=True, init=False)
class PromotionState:
    """Canonical M45 promotion-state payload with construction-time validation."""

    _payload: dict[str, Any]
    _allow_legacy_promotion_status: bool

    def __init__(self, payload: Mapping[str, Any]) -> None:
        self._set_payload(payload, allow_legacy_promotion_status=False)

    @classmethod
    def _from_evaluation(
        cls,
        evaluation: PromotionGateEvaluation,
        *,
        provenance: Mapping[str, Any] | None = None,
        artifact_filename: str | None = None,
    ) -> PromotionState:
        if not isinstance(evaluation, PromotionGateEvaluation):
            raise PromotionGateError("evaluation must be a PromotionGateEvaluation.")
        payload = _promotion_state_payload_from_evaluation(
            evaluation,
            provenance=provenance,
            artifact_filename=artifact_filename,
        )
        state = cls.__new__(cls)
        state._set_payload(payload, allow_legacy_promotion_status=True)
        return state

    def _set_payload(
        self,
        payload: Mapping[str, Any],
        *,
        allow_legacy_promotion_status: bool,
    ) -> None:
        payload = canonicalize_value(deepcopy(dict(payload)))
        _validate_promotion_state_payload(
            payload,
            allow_legacy_promotion_status=allow_legacy_promotion_status,
        )
        object.__setattr__(self, "_payload", payload)
        object.__setattr__(
            self,
            "_allow_legacy_promotion_status",
            allow_legacy_promotion_status,
        )

    @property
    def payload(self) -> dict[str, Any]:
        """Return a defensive JSON-safe copy of the canonical payload."""

        return self.to_payload()

    def to_payload(self) -> dict[str, Any]:
        payload = canonicalize_value(deepcopy(self._payload))
        _validate_promotion_state_payload(
            payload,
            allow_legacy_promotion_status=self._allow_legacy_promotion_status,
        )
        return payload

    def _payload_with_artifact_filename(self, artifact_filename: str) -> dict[str, Any]:
        updated = canonicalize_value(deepcopy(self._payload))
        updated["artifact_metadata"] = _promotion_state_artifact_metadata(artifact_filename)
        if "artifact_filename" in updated:
            updated["artifact_filename"] = artifact_filename
        _validate_promotion_state_payload(
            updated,
            allow_legacy_promotion_status=self._allow_legacy_promotion_status,
        )
        return updated

    def summary(self) -> dict[str, Any]:
        payload = self.to_payload()
        summary = {
            "schema_version": payload["schema_version"],
            "artifact_type": payload["artifact_type"],
            "configured": payload["configured"],
            "configuration_state": payload["configuration_state"],
            "evaluation_status": payload["evaluation_status"],
            "promotion_status": payload["promotion_status"],
            "decision_authority": payload["decision_authority"],
            "decision_reason_codes": payload["decision_reason_codes"],
            "artifact_filename": payload["artifact_metadata"]["artifact_filename"],
        }
        legacy_keys = (
            "gate_count",
            "passed_gate_count",
            "failed_gate_count",
            "missing_gate_count",
            "highest_severity",
            "severity_counts",
            "warning_gate_count",
            "review_gate_count",
            "rejected_gate_count",
            "blocked_gate_count",
            "status_on_pass",
            "status_on_fail",
        )
        for key in legacy_keys:
            if key in payload:
                summary[key] = payload[key]
        return canonicalize_value(summary)


def load_promotion_gate_config(path: str | Path) -> dict[str, Any]:
    """Load one JSON or YAML promotion gate config."""

    resolved_path = Path(path)
    if not resolved_path.exists():
        raise PromotionGateError(f"Promotion gate config file does not exist: {resolved_path}")

    suffix = resolved_path.suffix.lower()
    with resolved_path.open("r", encoding="utf-8") as handle:
        if suffix == ".json":
            payload = json.load(handle)
        elif suffix in {".yaml", ".yml"}:
            payload = yaml.safe_load(handle) or {}
        else:
            raise PromotionGateError(
                f"Unsupported promotion gate config format {resolved_path.suffix!r}. Use JSON, YAML, or YML."
            )
    if not isinstance(payload, dict):
        raise PromotionGateError("Promotion gate config must deserialize to a mapping.")
    nested = payload.get("promotion_gates")
    if nested is None:
        return payload
    if not isinstance(nested, dict):
        raise PromotionGateError("promotion_gates config section must be a mapping.")
    return nested


def evaluate_promotion_gates(
    *,
    run_type: str,
    config: Mapping[str, Any] | None,
    sources: Mapping[str, Any],
    artifact_filename: str = DEFAULT_PROMOTION_ARTIFACT_FILENAME,
) -> PromotionGateEvaluation | None:
    """Evaluate one normalized promotion gate config against deterministic sources."""

    if config is None:
        return None

    normalized_config = _normalize_promotion_gate_config(config)
    definitions = normalized_config["definitions"]
    results = [
        _evaluate_definition(definition, sources=sources)
        for definition in definitions
    ]
    failed_gate_count = sum(result.status == "fail" for result in results)
    missing_gate_count = sum(result.status == "missing" for result in results)
    passed_gate_count = sum(result.status == "pass" for result in results)
    evaluation_status = "pass" if failed_gate_count == 0 and missing_gate_count == 0 else "fail"
    highest_severity = _highest_failing_severity(results)
    severity_counts = _severity_counts(results)
    return PromotionGateEvaluation(
        configured=True,
        run_type=run_type,
        evaluation_status=evaluation_status,
        promotion_status=_resolve_promotion_status(
            evaluation_status=evaluation_status,
            highest_severity=highest_severity,
            status_on_pass=normalized_config["status_on_pass"],
            status_on_fail=normalized_config["status_on_fail"],
        ),
        status_on_pass=normalized_config["status_on_pass"],
        status_on_fail=normalized_config["status_on_fail"],
        gate_count=len(results),
        passed_gate_count=passed_gate_count,
        failed_gate_count=failed_gate_count,
        missing_gate_count=missing_gate_count,
        highest_severity=highest_severity,
        severity_counts=severity_counts,
        warning_gate_count=severity_counts["warn"],
        review_gate_count=severity_counts["review"],
        rejected_gate_count=severity_counts["reject"],
        blocked_gate_count=severity_counts["block"],
        decision_reason_codes=_decision_reason_codes(results),
        artifact_filename=artifact_filename,
        definitions=definitions,
        results=results,
    )


def build_unconfigured_promotion_state(
    *,
    run_type: str,
    provenance: Mapping[str, Any] | None = None,
    artifact_filename: str = DEFAULT_PROMOTION_ARTIFACT_FILENAME,
) -> PromotionState:
    """Build a deterministic v2 state for completed runs with no promotion policy."""

    normalized_run_type = _normalize_promotion_state_run_type(run_type)
    payload = {
        "schema_version": PROMOTION_STATE_SCHEMA_VERSION,
        "artifact_type": PROMOTION_STATE_ARTIFACT_TYPE,
        "run_type": normalized_run_type,
        "configured": False,
        "configuration_state": "not_configured",
        "evaluation_status": "not_configured",
        "promotion_status": "not_reviewed",
        "decision_authority": "none",
        "human_decision": None,
        "decision_reason_codes": list(_UNCONFIGURED_DECISION_REASON_CODES),
        "gate_counts": _gate_counts_payload(
            total=0,
            passed=0,
            failed=0,
            missing=0,
            skipped=0,
            severity_counts=None,
        ),
        "gate_definitions": [],
        "gate_results": [],
        "provenance": _promotion_state_provenance(provenance, run_type=normalized_run_type),
        "artifact_metadata": _promotion_state_artifact_metadata(artifact_filename),
    }
    return PromotionState(canonicalize_value(payload))


def build_promotion_state_from_evaluation(
    evaluation: PromotionGateEvaluation,
    *,
    provenance: Mapping[str, Any] | None = None,
    artifact_filename: str | None = None,
) -> PromotionState:
    """Build a deterministic v2 state from an existing configured gate evaluation."""

    return PromotionState._from_evaluation(
        evaluation,
        provenance=provenance,
        artifact_filename=artifact_filename,
    )


def _promotion_state_payload_from_evaluation(
    evaluation: PromotionGateEvaluation,
    *,
    provenance: Mapping[str, Any] | None = None,
    artifact_filename: str | None = None,
) -> dict[str, Any]:
    if not isinstance(evaluation, PromotionGateEvaluation):
        raise PromotionGateError("evaluation must be a PromotionGateEvaluation.")
    if not evaluation.configured:
        raise PromotionGateError("Configured promotion state requires evaluation.configured to be true.")
    normalized_run_type = _normalize_promotion_state_run_type(evaluation.run_type)
    resolved_artifact_filename = artifact_filename or evaluation.artifact_filename
    legacy_payload = evaluation.to_payload()
    definitions = [asdict(definition) for definition in evaluation.definitions]
    results = [asdict(result) for result in evaluation.results]
    payload = {
        "schema_version": PROMOTION_STATE_SCHEMA_VERSION,
        "artifact_type": PROMOTION_STATE_ARTIFACT_TYPE,
        "run_type": normalized_run_type,
        "configured": True,
        "configuration_state": "configured",
        "evaluation_status": evaluation.evaluation_status,
        "promotion_status": evaluation.promotion_status,
        "decision_authority": "engine",
        "human_decision": None,
        "decision_reason_codes": list(evaluation.decision_reason_codes),
        "gate_counts": _gate_counts_payload(
            total=evaluation.gate_count,
            passed=evaluation.passed_gate_count,
            failed=evaluation.failed_gate_count,
            missing=evaluation.missing_gate_count,
            skipped=sum(
                result.status == "pass" and result.actual_value is None and result.missing_behavior == "skip"
                for result in evaluation.results
            ),
            severity_counts=evaluation.severity_counts,
        ),
        "gate_definitions": definitions,
        "gate_results": results,
        "provenance": _promotion_state_provenance(provenance, run_type=normalized_run_type),
        "artifact_metadata": _promotion_state_artifact_metadata(resolved_artifact_filename),
        # Legacy fields keep existing readers and summaries compatible while
        # new consumers transition to the v2 canonical names above.
        **legacy_payload,
        "artifact_filename": resolved_artifact_filename,
        "definitions": definitions,
        "results": results,
    }
    return canonicalize_value(payload)


def serialize_promotion_state(state: PromotionState) -> dict[str, Any]:
    """Return one canonical JSON-safe M45 promotion-state payload."""

    if not isinstance(state, PromotionState):
        raise PromotionGateError("state must be a PromotionState.")
    return state.to_payload()


def write_promotion_state_artifact(
    output_dir: str | Path,
    state: PromotionState,
    *,
    artifact_filename: str | None = None,
) -> Path:
    """Persist a canonical v2 promotion-state artifact as stable JSON."""

    payload = serialize_promotion_state(state)
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    resolved_filename = artifact_filename or payload["artifact_metadata"]["artifact_filename"]
    if resolved_filename != payload["artifact_metadata"]["artifact_filename"]:
        payload = state._payload_with_artifact_filename(resolved_filename)
    artifact_path = resolved_output_dir / resolved_filename
    with artifact_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return artifact_path


def write_promotion_gate_artifact(
    output_dir: str | Path,
    evaluation: PromotionGateEvaluation | None,
    *,
    artifact_filename: str = DEFAULT_PROMOTION_ARTIFACT_FILENAME,
) -> Path | None:
    """Persist a promotion gate evaluation as stable JSON when configured."""

    if evaluation is None:
        return None
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = resolved_output_dir / artifact_filename
    with artifact_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(evaluation.to_payload(), handle, indent=2, sort_keys=True)
    return artifact_path


def promotion_gate_config_digest(config: Mapping[str, Any] | None) -> str | None:
    if config is None:
        return None
    normalized_config = _normalize_promotion_gate_config(config)
    payload = {
        "status_on_pass": normalized_config["status_on_pass"],
        "status_on_fail": normalized_config["status_on_fail"],
        "definitions": [asdict(definition) for definition in normalized_config["definitions"]],
    }
    return serialize_canonical_json(payload)


def _normalize_promotion_gate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise PromotionGateError("promotion_gates must be a mapping when provided.")

    status_on_pass = _normalize_required_string(
        config.get("status_on_pass", DEFAULT_STATUS_ON_PASS),
        field_name="promotion_gates.status_on_pass",
    )
    status_on_fail = _normalize_required_string(
        config.get("status_on_fail", DEFAULT_STATUS_ON_FAIL),
        field_name="promotion_gates.status_on_fail",
    )
    raw_definitions = config.get("gates")
    if not isinstance(raw_definitions, list) or not raw_definitions:
        raise PromotionGateError("promotion_gates.gates must be a non-empty list.")
    definitions = [_normalize_definition(item, index=index) for index, item in enumerate(raw_definitions)]
    return {
        "status_on_pass": status_on_pass,
        "status_on_fail": status_on_fail,
        "definitions": definitions,
    }


def _gate_counts_payload(
    *,
    total: int,
    passed: int,
    failed: int,
    missing: int,
    skipped: int,
    severity_counts: Mapping[str, int] | None,
) -> dict[str, int]:
    severity_counts = dict(severity_counts or {})
    return canonicalize_value(
        {
            "total": total,
            "passed": passed,
            "failed": failed,
            "missing": missing,
            "skipped": skipped,
            "warning": int(severity_counts.get("warn", 0)),
            "review": int(severity_counts.get("review", 0)),
            "rejected": int(severity_counts.get("reject", 0)),
            "blocked": int(severity_counts.get("block", 0)),
        }
    )


def _promotion_state_provenance(
    provenance: Mapping[str, Any] | None,
    *,
    run_type: str,
) -> dict[str, Any]:
    if provenance is None:
        payload: dict[str, Any] = {}
    elif isinstance(provenance, Mapping):
        payload = dict(provenance)
    else:
        raise PromotionGateError("promotion state provenance must be a mapping when provided.")
    payload.setdefault("run_type", run_type)
    return canonicalize_value(payload)


def _promotion_state_artifact_metadata(artifact_filename: str) -> dict[str, Any]:
    normalized_filename = _normalize_required_string(
        artifact_filename,
        field_name="artifact_filename",
    )
    return canonicalize_value(
        {
            "artifact_filename": normalized_filename,
            "deterministic": True,
            "generated_by": "src.research.promotion",
            "writer": "engine",
        }
    )


def _normalize_promotion_state_run_type(value: object) -> str:
    run_type = _normalize_required_string(value, field_name="run_type")
    if run_type not in SUPPORTED_PROMOTION_STATE_RUN_TYPES:
        formatted = ", ".join(sorted(SUPPORTED_PROMOTION_STATE_RUN_TYPES))
        raise PromotionGateError(f"run_type must be one of: {formatted}.")
    return run_type


def _validate_promotion_state_payload(
    payload: Mapping[str, Any],
    *,
    allow_legacy_promotion_status: bool = False,
) -> None:
    if not isinstance(payload, Mapping):
        raise PromotionGateError("Promotion state payload must be a mapping.")
    if payload.get("schema_version") != PROMOTION_STATE_SCHEMA_VERSION:
        raise PromotionGateError("Promotion state schema_version must be 2.")
    if payload.get("artifact_type") != PROMOTION_STATE_ARTIFACT_TYPE:
        raise PromotionGateError("Promotion state artifact_type must be 'promotion_state'.")
    run_type = _normalize_promotion_state_run_type(payload.get("run_type"))
    _validate_promotion_state_provenance(payload, run_type=run_type)
    _validate_promotion_state_artifact_metadata_payload(payload)

    configured = payload.get("configured")
    if not isinstance(configured, bool):
        raise PromotionGateError("Promotion state configured must be a boolean.")
    if payload.get("human_decision") is not None:
        raise PromotionGateError("M45 promotion state factory output must not include human_decision.")

    gate_definitions = payload.get("gate_definitions")
    gate_results = payload.get("gate_results")
    if not isinstance(gate_definitions, list):
        raise PromotionGateError("Promotion state gate_definitions must be a list.")
    if not isinstance(gate_results, list):
        raise PromotionGateError("Promotion state gate_results must be a list.")
    gate_counts = payload.get("gate_counts")
    validated_gate_counts = _validate_promotion_state_gate_counts(gate_counts)

    if configured:
        _validate_configured_promotion_state(
            payload,
            gate_counts=validated_gate_counts,
            allow_legacy_promotion_status=allow_legacy_promotion_status,
        )
    else:
        _validate_unconfigured_promotion_state(payload, gate_counts=validated_gate_counts)


def _validate_configured_promotion_state(
    payload: Mapping[str, Any],
    *,
    gate_counts: Mapping[str, int],
    allow_legacy_promotion_status: bool,
) -> None:
    if payload.get("configuration_state") != "configured":
        raise PromotionGateError("Configured promotion state requires configuration_state='configured'.")
    if payload.get("decision_authority") != "engine":
        raise PromotionGateError("Configured promotion state requires decision_authority='engine'.")
    if payload.get("evaluation_status") not in {"pass", "fail"}:
        raise PromotionGateError("Configured promotion state evaluation_status must be pass or fail.")
    promotion_status = payload.get("promotion_status")
    if not isinstance(promotion_status, str) or not promotion_status.strip():
        raise PromotionGateError("Configured promotion state promotion_status must be a non-empty string.")
    if not allow_legacy_promotion_status and promotion_status not in _CANONICAL_MACHINE_PROMOTION_STATUSES:
        formatted = ", ".join(sorted(_CANONICAL_MACHINE_PROMOTION_STATUSES))
        raise PromotionGateError(f"Configured promotion state promotion_status must be one of: {formatted}.")
    gate_results = payload.get("gate_results")
    if not gate_results:
        raise PromotionGateError("Configured promotion state requires at least one gate result.")
    if gate_counts["total"] != len(gate_results):
        raise PromotionGateError("Configured promotion state gate_counts.total must equal gate_results length.")
    if gate_counts["passed"] + gate_counts["failed"] + gate_counts["missing"] != gate_counts["total"]:
        raise PromotionGateError("Configured promotion state pass/fail/missing gate counts must sum to total.")
    nonpassing_count = sum(
        isinstance(result, Mapping) and result.get("status") in {"fail", "missing"}
        for result in gate_results
    )
    for key in ("warning", "review", "rejected", "blocked"):
        if gate_counts[key] > nonpassing_count:
            raise PromotionGateError("Configured promotion state severity counts cannot exceed non-passing results.")
    if gate_counts["skipped"] > gate_counts["passed"]:
        raise PromotionGateError("Configured promotion state skipped gate count cannot exceed passed count.")


def _validate_unconfigured_promotion_state(
    payload: Mapping[str, Any],
    *,
    gate_counts: Mapping[str, int],
) -> None:
    if payload.get("configuration_state") != "not_configured":
        raise PromotionGateError("Unconfigured promotion state requires configuration_state='not_configured'.")
    if payload.get("evaluation_status") != "not_configured":
        raise PromotionGateError("Unconfigured promotion state requires evaluation_status='not_configured'.")
    if payload.get("promotion_status") != "not_reviewed":
        raise PromotionGateError("Unconfigured promotion state requires promotion_status='not_reviewed'.")
    if payload.get("decision_authority") != "none":
        raise PromotionGateError("Unconfigured promotion state requires decision_authority='none'.")
    if payload.get("gate_definitions") or payload.get("gate_results"):
        raise PromotionGateError("Unconfigured promotion state cannot include gate definitions or results.")
    if payload.get("decision_reason_codes") != list(_UNCONFIGURED_DECISION_REASON_CODES):
        raise PromotionGateError("Unconfigured promotion state requires promotion_policy_not_configured reason code.")
    if any(value != 0 for value in gate_counts.values()):
        raise PromotionGateError("Unconfigured promotion state requires zero gate counts.")


def _validate_promotion_state_provenance(payload: Mapping[str, Any], *, run_type: str) -> None:
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise PromotionGateError("Promotion state provenance must be a mapping.")
    provenance_run_type = provenance.get("run_type")
    if provenance_run_type != run_type:
        raise PromotionGateError("Promotion state provenance.run_type must match run_type.")


def _validate_promotion_state_artifact_metadata_payload(payload: Mapping[str, Any]) -> None:
    metadata = payload.get("artifact_metadata")
    if not isinstance(metadata, Mapping):
        raise PromotionGateError("Promotion state artifact_metadata must be a mapping.")
    required = {"artifact_filename", "writer", "generated_by", "deterministic"}
    missing = required - set(metadata)
    if missing:
        formatted = ", ".join(sorted(missing))
        raise PromotionGateError(f"Promotion state artifact_metadata missing required keys: {formatted}.")
    if set(metadata) != required:
        formatted = ", ".join(sorted(set(metadata) - required))
        raise PromotionGateError(f"Promotion state artifact_metadata has unexpected keys: {formatted}.")
    _normalize_required_string(metadata.get("artifact_filename"), field_name="artifact_metadata.artifact_filename")
    if metadata.get("writer") != "engine":
        raise PromotionGateError("Promotion state artifact_metadata.writer must be 'engine'.")
    if metadata.get("generated_by") != "src.research.promotion":
        raise PromotionGateError("Promotion state artifact_metadata.generated_by must be 'src.research.promotion'.")
    if metadata.get("deterministic") is not True:
        raise PromotionGateError("Promotion state artifact_metadata.deterministic must be True.")


def _validate_promotion_state_gate_counts(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise PromotionGateError("Promotion state gate_counts must be a mapping.")
    keys = set(value)
    missing = _PROMOTION_STATE_GATE_COUNT_KEYS - keys
    if missing:
        formatted = ", ".join(sorted(missing))
        raise PromotionGateError(f"Promotion state gate_counts missing required keys: {formatted}.")
    unexpected = keys - _PROMOTION_STATE_GATE_COUNT_KEYS
    if unexpected:
        formatted = ", ".join(sorted(unexpected))
        raise PromotionGateError(f"Promotion state gate_counts has unexpected keys: {formatted}.")
    counts: dict[str, int] = {}
    for key in sorted(_PROMOTION_STATE_GATE_COUNT_KEYS):
        count = value[key]
        if isinstance(count, bool) or not isinstance(count, int):
            raise PromotionGateError(f"Promotion state gate_counts.{key} must be a nonnegative integer.")
        if count < 0:
            raise PromotionGateError(f"Promotion state gate_counts.{key} must be a nonnegative integer.")
        counts[key] = count
    return counts


def _normalize_definition(raw_definition: Any, *, index: int) -> PromotionGateDefinition:
    if not isinstance(raw_definition, Mapping):
        raise PromotionGateError(f"promotion_gates.gates[{index}] must be a mapping.")

    gate_id = _normalize_required_string(
        raw_definition.get("gate_id"),
        field_name=f"promotion_gates.gates[{index}].gate_id",
    )
    source = _normalize_required_string(
        raw_definition.get("source", "metrics"),
        field_name=f"promotion_gates.gates[{index}].source",
    ).lower()
    if source not in _SUPPORTED_SOURCES:
        formatted = ", ".join(sorted(_SUPPORTED_SOURCES))
        raise PromotionGateError(
            f"promotion_gates.gates[{index}].source must be one of: {formatted}."
        )

    metric_path = raw_definition.get("metric_path", raw_definition.get("metric"))
    metric_path = _normalize_required_string(
        metric_path,
        field_name=f"promotion_gates.gates[{index}].metric_path",
    )
    comparator = _normalize_required_string(
        raw_definition.get("comparator"),
        field_name=f"promotion_gates.gates[{index}].comparator",
    ).lower()
    if comparator not in _SUPPORTED_COMPARATORS:
        formatted = ", ".join(sorted(_SUPPORTED_COMPARATORS))
        raise PromotionGateError(
            f"promotion_gates.gates[{index}].comparator must be one of: {formatted}."
        )
    threshold = _coerce_finite_float(
        raw_definition.get("threshold"),
        field_name=f"promotion_gates.gates[{index}].threshold",
    )
    statistic = _normalize_required_string(
        raw_definition.get("statistic", "value"),
        field_name=f"promotion_gates.gates[{index}].statistic",
    ).lower()
    if statistic not in _SUPPORTED_STATISTICS:
        formatted = ", ".join(sorted(_SUPPORTED_STATISTICS))
        raise PromotionGateError(
            f"promotion_gates.gates[{index}].statistic must be one of: {formatted}."
        )
    missing_behavior = _normalize_required_string(
        raw_definition.get("missing_behavior", "fail"),
        field_name=f"promotion_gates.gates[{index}].missing_behavior",
    ).lower()
    if missing_behavior not in {"fail", "skip"}:
        raise PromotionGateError(
            f"promotion_gates.gates[{index}].missing_behavior must be one of: fail, skip."
        )
    description = raw_definition.get("description")
    if description is not None:
        description = _normalize_required_string(
            description,
            field_name=f"promotion_gates.gates[{index}].description",
        )
    severity = _normalize_optional_severity(
        raw_definition.get("severity"),
        field_name=f"promotion_gates.gates[{index}].severity",
    )
    return PromotionGateDefinition(
        gate_id=gate_id,
        source=source,
        metric_path=metric_path,
        comparator=comparator,
        threshold=threshold,
        statistic=statistic,
        missing_behavior=missing_behavior,
        description=description,
        severity=severity,
    )


def _evaluate_definition(
    definition: PromotionGateDefinition,
    *,
    sources: Mapping[str, Any],
) -> PromotionGateResult:
    raw_source = sources.get(definition.source)
    actual_value = _extract_actual_value(
        raw_source,
        metric_path=definition.metric_path,
        statistic=definition.statistic,
    )
    if actual_value is None:
        status = "pass" if definition.missing_behavior == "skip" else "missing"
        reason = (
            "metric missing and gate skipped"
            if definition.missing_behavior == "skip"
            else "metric missing"
        )
        reason_codes = _result_reason_codes(
            status=status,
            severity=definition.severity,
            base_reason_code=(
                "gate_missing_skipped"
                if definition.missing_behavior == "skip"
                else "gate_missing"
            ),
        )
        return PromotionGateResult(
            gate_id=definition.gate_id,
            status=status,
            source=definition.source,
            metric_path=definition.metric_path,
            statistic=definition.statistic,
            comparator=definition.comparator,
            threshold=definition.threshold,
            actual_value=None,
            missing_behavior=definition.missing_behavior,
            description=definition.description,
            severity=definition.severity,
            reason_codes=reason_codes,
            reason=reason,
        )

    passed = _compare(
        actual_value=actual_value,
        comparator=definition.comparator,
        threshold=definition.threshold,
    )
    return PromotionGateResult(
        gate_id=definition.gate_id,
        status="pass" if passed else "fail",
        source=definition.source,
        metric_path=definition.metric_path,
        statistic=definition.statistic,
        comparator=definition.comparator,
        threshold=definition.threshold,
        actual_value=actual_value,
        missing_behavior=definition.missing_behavior,
        description=definition.description,
        severity=definition.severity,
        reason_codes=_result_reason_codes(
            status="pass" if passed else "fail",
            severity=definition.severity,
            base_reason_code="gate_passed" if passed else "gate_failed_threshold",
        ),
        reason=(
            f"{actual_value} {definition.comparator} {definition.threshold}"
            if passed
            else f"{actual_value} not {definition.comparator} {definition.threshold}"
        ),
    )


def _normalize_optional_severity(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    severity = _normalize_required_string(value, field_name=field_name).lower()
    if severity not in _SUPPORTED_SEVERITIES:
        formatted = ", ".join(sorted(_SUPPORTED_SEVERITIES))
        raise PromotionGateError(f"{field_name} must be one of: {formatted}.")
    return severity


def _result_reason_codes(*, status: str, severity: str | None, base_reason_code: str) -> list[str]:
    codes = [base_reason_code]
    if status in {"fail", "missing"} and severity is not None:
        codes.append(_SEVERITY_REASON_CODES[severity])
    return codes


def _nonpassing_results(results: Iterable[PromotionGateResult]) -> list[PromotionGateResult]:
    return [result for result in results if result.status in {"fail", "missing"}]


def _highest_failing_severity(results: Iterable[PromotionGateResult]) -> str | None:
    severities = [
        result.severity
        for result in _nonpassing_results(results)
        if result.severity is not None
    ]
    if not severities:
        return None
    return max(severities, key=lambda severity: _SEVERITY_RANK[severity])


def _severity_counts(results: Iterable[PromotionGateResult]) -> dict[str, int]:
    counts = {severity: 0 for severity in sorted(_SUPPORTED_SEVERITIES)}
    for result in _nonpassing_results(results):
        if result.severity is not None:
            counts[result.severity] += 1
    return counts


def _decision_reason_codes(results: Iterable[PromotionGateResult]) -> list[str]:
    codes: set[str] = set()
    for result in _nonpassing_results(results):
        codes.update(result.reason_codes)
    return sorted(codes)


def _resolve_promotion_status(
    *,
    evaluation_status: str,
    highest_severity: str | None,
    status_on_pass: str,
    status_on_fail: str,
) -> str:
    if evaluation_status == "pass":
        return status_on_pass
    if highest_severity is None:
        return status_on_fail
    return _SEVERITY_PROMOTION_STATUS[highest_severity]


def _extract_actual_value(
    raw_source: Any,
    *,
    metric_path: str,
    statistic: str,
) -> float | None:
    values = _extract_values(raw_source, metric_path=metric_path)
    if not values:
        return None
    if statistic == "value":
        return values[0] if len(values) == 1 else None
    if statistic == "count":
        return float(len(values))
    if statistic == "non_null_count":
        return float(len(values))

    series = pd.Series(values, dtype="float64")
    if series.empty:
        return None
    if statistic == "min":
        return float(series.min())
    if statistic == "max":
        return float(series.max())
    if statistic == "mean":
        return float(series.mean())
    if statistic == "median":
        return float(series.median())
    if statistic == "std":
        return float(series.std(ddof=0))
    if statistic == "range":
        return float(series.max() - series.min())
    if statistic == "abs_max":
        return float(series.abs().max())
    raise PromotionGateError(f"Unsupported statistic {statistic!r}.")


def _extract_values(raw_source: Any, *, metric_path: str) -> list[float]:
    if raw_source is None:
        return []
    if isinstance(raw_source, pd.DataFrame):
        if metric_path not in raw_source.columns:
            return []
        return _coerce_numeric_iterable(raw_source[metric_path].tolist())
    if isinstance(raw_source, Mapping):
        resolved = _lookup_mapping_value(raw_source, metric_path)
        if resolved is None:
            return []
        numeric = _coerce_numeric_value(resolved)
        return [] if numeric is None else [numeric]
    if isinstance(raw_source, list | tuple):
        collected: list[float] = []
        for item in raw_source:
            if isinstance(item, Mapping):
                resolved = _lookup_mapping_value(item, metric_path)
                numeric = _coerce_numeric_value(resolved)
                if numeric is not None:
                    collected.append(numeric)
        return collected
    return []


def _lookup_mapping_value(mapping: Mapping[str, Any], metric_path: str) -> Any:
    current: Any = mapping
    for part in metric_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _coerce_numeric_iterable(values: Iterable[Any]) -> list[float]:
    normalized: list[float] = []
    for value in values:
        numeric = _coerce_numeric_value(value)
        if numeric is not None:
            normalized.append(numeric)
    return normalized


def _coerce_numeric_value(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        if isinstance(value, bool):
            return float(value)
        return None
    if isinstance(value, int | float):
        numeric = float(value)
    else:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def _compare(*, actual_value: float, comparator: str, threshold: float) -> bool:
    if comparator == "gt":
        return actual_value > threshold
    if comparator == "gte":
        return actual_value >= threshold
    if comparator == "lt":
        return actual_value < threshold
    if comparator == "lte":
        return actual_value <= threshold
    if comparator == "eq":
        return actual_value == threshold
    if comparator == "ne":
        return actual_value != threshold
    raise PromotionGateError(f"Unsupported comparator {comparator!r}.")


def _normalize_required_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise PromotionGateError(f"{field_name} must be a non-empty string.")
    normalized = value.strip()
    if not normalized:
        raise PromotionGateError(f"{field_name} must be a non-empty string.")
    return normalized


def _coerce_finite_float(value: object, *, field_name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise PromotionGateError(f"{field_name} must be a finite float.") from exc
    if math.isnan(numeric) or math.isinf(numeric):
        raise PromotionGateError(f"{field_name} must be a finite float.")
    return numeric


__all__ = [
    "DEFAULT_PROMOTION_ARTIFACT_FILENAME",
    "PROMOTION_STATE_ARTIFACT_TYPE",
    "PROMOTION_STATE_SCHEMA_VERSION",
    "PromotionGateDefinition",
    "PromotionGateEvaluation",
    "PromotionGateError",
    "PromotionGateResult",
    "PromotionState",
    "SUPPORTED_PROMOTION_STATE_RUN_TYPES",
    "build_promotion_state_from_evaluation",
    "build_unconfigured_promotion_state",
    "evaluate_promotion_gates",
    "load_promotion_gate_config",
    "promotion_gate_config_digest",
    "serialize_promotion_state",
    "write_promotion_gate_artifact",
    "write_promotion_state_artifact",
]
