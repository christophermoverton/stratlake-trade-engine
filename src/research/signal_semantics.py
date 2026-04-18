from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.research.registry import canonicalize_value

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SIGNAL_TYPES_REGISTRY = REPO_ROOT / "artifacts" / "registry" / "signal_types.jsonl"
DEFAULT_SIGNAL_SCHEMA_VERSION = "1.0.0"
_STRUCTURAL_COLUMNS: tuple[str, ...] = ("symbol", "ts_utc")
_EXECUTABLE_POSITION_CONSTRUCTOR = "backtest_numeric_exposure"


class SignalSemanticsError(ValueError):
    """Raised when a signal violates the declared semantic contract."""


@dataclass(frozen=True)
class SignalTypeDefinition:
    """Registry-backed definition for one canonical signal type."""

    signal_type_id: str
    version: str
    status: str
    domain: str
    codomain: str
    description: str
    validation_rules: dict[str, Any]
    transformation_policies: dict[str, Any]
    compatible_position_constructors: tuple[str, ...]
    tags: tuple[str, ...] = ()
    semantic_meaning: str | None = None
    mathematical_definition: str | None = None
    directional: bool = False
    ordinal: bool = False
    probabilistic: bool = False
    executable: bool = False
    required_columns: tuple[str, ...] = _STRUCTURAL_COLUMNS
    cross_sectional: bool = False
    # Directional asymmetry metadata (M21.3 extension)
    directional_asymmetry_allowed: bool = False  # Can this signal support asymmetric long/short?
    long_short_preference: str = "balanced"  # {balanced, long_favored, short_favored}
    asymmetry_validation_rules: dict[str, Any] | None = None  # Additional validation for asymmetry use


@dataclass(frozen=True)
class Signal:
    """Typed signal container used across strategy, alpha, and backtest layers."""

    signal_type: str
    version: str
    data: pd.DataFrame
    value_column: str = "signal"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_metadata_payload(self) -> dict[str, Any]:
        payload = {
            "signal_type": self.signal_type,
            "version": self.version,
            "value_column": self.value_column,
            "metadata": dict(self.metadata),
        }
        return canonicalize_signal_payload(payload)


def default_signal_registry_path() -> Path:
    return DEFAULT_SIGNAL_TYPES_REGISTRY


def canonicalize_signal_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    canonical = canonicalize_value(dict(payload))
    if not isinstance(canonical, dict):
        raise SignalSemanticsError("Signal metadata must serialize to a JSON object.")
    return canonical


def _normalize_non_empty_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise SignalSemanticsError(f"{field_name} must be a non-empty string.")
    normalized = value.strip()
    if not normalized:
        raise SignalSemanticsError(f"{field_name} must be a non-empty string.")
    return normalized


def _coerce_optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _definition_from_payload(payload: Mapping[str, Any]) -> SignalTypeDefinition:
    signal_type_id = _normalize_non_empty_string(payload.get("signal_type_id"), field_name="signal_type_id")
    version = _normalize_non_empty_string(payload.get("version"), field_name="version")
    validation_rules = payload.get("validation_rules")
    if not isinstance(validation_rules, dict):
        raise SignalSemanticsError(f"Signal type {signal_type_id!r} must define object validation_rules.")
    transformation_policies = payload.get("transformation_policies")
    if not isinstance(transformation_policies, dict):
        transformation_policies = {}
    compatible = payload.get("compatible_position_constructors")
    if not isinstance(compatible, list):
        compatible = []
    required_columns = validation_rules.get("required_columns", list(_STRUCTURAL_COLUMNS))
    if not isinstance(required_columns, list) or not required_columns:
        raise SignalSemanticsError(f"Signal type {signal_type_id!r} must declare required_columns.")
    semantic_flags = payload.get("semantic_flags")
    if not isinstance(semantic_flags, dict):
        semantic_flags = {}
    return SignalTypeDefinition(
        signal_type_id=signal_type_id,
        version=version,
        status=_normalize_non_empty_string(payload.get("status"), field_name="status"),
        domain=_normalize_non_empty_string(payload.get("domain"), field_name="domain"),
        codomain=_normalize_non_empty_string(payload.get("codomain"), field_name="codomain"),
        description=_normalize_non_empty_string(payload.get("description"), field_name="description"),
        validation_rules=canonicalize_signal_payload(validation_rules),
        transformation_policies=canonicalize_signal_payload(transformation_policies),
        compatible_position_constructors=tuple(str(value) for value in compatible if str(value).strip()),
        tags=tuple(str(value) for value in payload.get("tags", []) if str(value).strip()),
        semantic_meaning=_coerce_optional_string(payload.get("semantic_meaning")),
        mathematical_definition=_coerce_optional_string(payload.get("mathematical_definition")),
        directional=bool(semantic_flags.get("directional", payload.get("directional", False))),
        ordinal=bool(semantic_flags.get("ordinal", payload.get("ordinal", False))),
        probabilistic=bool(semantic_flags.get("probabilistic", payload.get("probabilistic", False))),
        executable=bool(semantic_flags.get("executable", payload.get("executable", False))),
        required_columns=tuple(str(value) for value in required_columns),
        cross_sectional=bool(validation_rules.get("cross_sectional", False)),
    )


def load_signal_type_registry(path: str | Path = DEFAULT_SIGNAL_TYPES_REGISTRY) -> dict[str, SignalTypeDefinition]:
    """Load the canonical signal registry keyed by signal_type_id."""

    registry_path = Path(path)
    if not registry_path.exists():
        raise SignalSemanticsError(f"Signal type registry not found: {registry_path.as_posix()}")

    definitions: dict[str, SignalTypeDefinition] = {}
    with registry_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SignalSemanticsError(
                    f"Signal type registry contains invalid JSON on line {line_number}."
                ) from exc
            if not isinstance(payload, dict):
                raise SignalSemanticsError(
                    f"Signal type registry line {line_number} must deserialize to an object."
                )
            definition = _definition_from_payload(payload)
            definitions[definition.signal_type_id] = definition

    if not definitions:
        raise SignalSemanticsError("Signal type registry is empty.")
    return definitions


def resolve_signal_type_definition(
    signal_type: str,
    *,
    version: str = DEFAULT_SIGNAL_SCHEMA_VERSION,
    registry_path: str | Path = DEFAULT_SIGNAL_TYPES_REGISTRY,
) -> SignalTypeDefinition:
    """Resolve one declared signal type through the registry."""

    normalized_signal_type = _normalize_non_empty_string(signal_type, field_name="signal_type")
    definitions = load_signal_type_registry(registry_path)
    try:
        definition = definitions[normalized_signal_type]
    except KeyError as exc:
        available = ", ".join(sorted(definitions)) or "<none>"
        raise SignalSemanticsError(
            f"Undefined signal_type {normalized_signal_type!r}. Available registry entries: {available}."
        ) from exc
    if definition.version != version:
        raise SignalSemanticsError(
            f"Unsupported signal_type version for {normalized_signal_type!r}: {version!r}. "
            f"Registry defines version {definition.version!r}."
        )
    return definition


def _validate_unique_keys(df: pd.DataFrame) -> None:
    duplicate_mask = df.duplicated(subset=["symbol", "ts_utc"], keep=False)
    if duplicate_mask.any():
        duplicate_row = df.loc[duplicate_mask, ["symbol", "ts_utc"]].iloc[0]
        raise SignalSemanticsError(
            "Signal frames must not contain duplicate (symbol, ts_utc) rows. "
            f"First duplicate key: ({duplicate_row['symbol']!r}, {duplicate_row['ts_utc']})."
        )


def _validate_deterministic_order(df: pd.DataFrame) -> None:
    order_columns = [column for column in ("symbol", "ts_utc", "timeframe") if column in df.columns]
    if not order_columns:
        return
    ordering = df.loc[:, order_columns].copy()
    if "symbol" in ordering.columns:
        ordering["symbol"] = ordering["symbol"].astype("string")
    if "ts_utc" in ordering.columns:
        ordering["ts_utc"] = pd.to_datetime(ordering["ts_utc"], utc=True, errors="coerce")
    if "timeframe" in ordering.columns:
        ordering["timeframe"] = ordering["timeframe"].astype("string")
    expected = ordering.sort_values(order_columns, kind="stable")
    if ordering.index.equals(expected.index):
        return
    first_mismatch = next(
        position
        for position, (actual_index, expected_index) in enumerate(zip(ordering.index, expected.index, strict=False))
        if actual_index != expected_index
    )
    bad_row = ordering.iloc[first_mismatch].to_dict()
    raise SignalSemanticsError(
        "Signal frames must be sorted deterministically by "
        f"{order_columns}. First out-of-order row: {bad_row}."
    )


def _validate_value_domain(
    values: pd.Series,
    *,
    definition: SignalTypeDefinition,
    parameters: Mapping[str, Any],
) -> None:
    rules = definition.validation_rules
    allowed_values = rules.get("allowed_values")
    if isinstance(allowed_values, list):
        allowed = {float(value) for value in allowed_values}
        invalid = values.loc[~values.isin(list(allowed))]
        if not invalid.empty:
            raise SignalSemanticsError(
                f"Signal type {definition.signal_type_id!r} only allows values {sorted(allowed)}. "
                f"First invalid value: {float(invalid.iloc[0])}."
            )

    min_value = rules.get("min_value")
    max_value = rules.get("max_value")
    if min_value is None and "clip" in parameters and definition.signal_type_id == "signed_zscore":
        clip_value = parameters.get("clip")
        if clip_value is not None:
            min_value = -abs(float(clip_value))
            max_value = abs(float(clip_value))
    if min_value is not None and bool((values < float(min_value)).any()):
        bad_value = float(values.loc[values < float(min_value)].iloc[0])
        raise SignalSemanticsError(
            f"Signal type {definition.signal_type_id!r} requires values >= {float(min_value)}. "
            f"First invalid value: {bad_value}."
        )
    if max_value is not None and bool((values > float(max_value)).any()):
        bad_value = float(values.loc[values > float(max_value)].iloc[0])
        raise SignalSemanticsError(
            f"Signal type {definition.signal_type_id!r} requires values <= {float(max_value)}. "
            f"First invalid value: {bad_value}."
        )


def _validate_cross_section_requirements(
    df: pd.DataFrame,
    *,
    definition: SignalTypeDefinition,
    parameters: Mapping[str, Any],
) -> None:
    min_size = definition.validation_rules.get("min_cross_section_size")
    if min_size is None:
        min_size = parameters.get("min_cross_section_size")
    if min_size is None:
        min_size = 2
    min_size = int(min_size)
    if min_size <= 0:
        raise SignalSemanticsError("min_cross_section_size must be a positive integer when provided.")

    sizes = df.groupby("ts_utc", sort=False).size()
    if sizes.empty:
        return
    violating = sizes.loc[sizes < min_size]
    if not violating.empty:
        ts_utc = violating.index[0]
        size = int(violating.iloc[0])
        raise SignalSemanticsError(
            f"Signal type {definition.signal_type_id!r} requires cross-sections of at least {min_size} rows. "
            f"First failing ts_utc={pd.Timestamp(ts_utc).isoformat().replace('+00:00', 'Z')}, size={size}."
        )


def validate_signal_frame(
    df: pd.DataFrame,
    *,
    signal_type: str,
    value_column: str = "signal",
    version: str = DEFAULT_SIGNAL_SCHEMA_VERSION,
    registry_path: str | Path = DEFAULT_SIGNAL_TYPES_REGISTRY,
    parameters: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Validate a signal frame against its declared semantic contract."""

    if not isinstance(df, pd.DataFrame):
        raise SignalSemanticsError("Signal data must be provided as a pandas DataFrame.")

    definition = resolve_signal_type_definition(signal_type, version=version, registry_path=registry_path)
    normalized_value_column = _normalize_non_empty_string(value_column, field_name="value_column")
    resolved_parameters = dict(parameters or {})

    required_columns = [
        normalized_value_column if column == "value" else column
        for column in definition.required_columns
    ]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        formatted = ", ".join(repr(column) for column in missing)
        raise SignalSemanticsError(
            f"Signal type {definition.signal_type_id!r} requires columns: {formatted}."
        )

    normalized = df.copy(deep=True)
    normalized.attrs = {}
    if normalized.empty:
        return normalized

    normalized["symbol"] = normalized["symbol"].astype("string")
    if normalized["symbol"].isna().any() or normalized["symbol"].str.strip().eq("").any():
        raise SignalSemanticsError("Signal frames must contain non-empty symbol values.")
    normalized["symbol"] = normalized["symbol"].str.strip()
    normalized["ts_utc"] = pd.to_datetime(normalized["ts_utc"], utc=True, errors="coerce")
    if normalized["ts_utc"].isna().any():
        raise SignalSemanticsError("Signal frames must contain parseable UTC ts_utc values.")

    _validate_unique_keys(normalized)
    _validate_deterministic_order(normalized)

    try:
        values = pd.to_numeric(normalized[normalized_value_column], errors="raise").astype("float64")
    except (TypeError, ValueError) as exc:
        raise SignalSemanticsError(
            f"Signal type {definition.signal_type_id!r} requires numeric column {normalized_value_column!r}."
        ) from exc

    if values.isna().any():
        raise SignalSemanticsError(
            f"Signal column {normalized_value_column!r} must not contain NaN values."
        )
    finite_mask = values.map(math.isfinite)
    if not bool(finite_mask.all()):
        raise SignalSemanticsError(
            f"Signal column {normalized_value_column!r} must contain only finite numeric values."
        )

    _validate_value_domain(values, definition=definition, parameters=resolved_parameters)
    if definition.cross_sectional:
        _validate_cross_section_requirements(
            normalized,
            definition=definition,
            parameters=resolved_parameters,
        )

    normalized[normalized_value_column] = values
    return normalized


def attach_signal_metadata(df: pd.DataFrame, payload: Mapping[str, Any]) -> pd.DataFrame:
    normalized = canonicalize_signal_payload(dict(payload))
    df.attrs["signal_semantics"] = normalized
    return df


def extract_signal_metadata(df: pd.DataFrame | None) -> dict[str, Any] | None:
    if df is None or not isinstance(df, pd.DataFrame):
        return None
    payload = df.attrs.get("signal_semantics")
    if isinstance(payload, dict):
        return canonicalize_signal_payload(dict(payload))
    return None


def create_signal(
    df: pd.DataFrame,
    *,
    signal_type: str,
    value_column: str = "signal",
    version: str = DEFAULT_SIGNAL_SCHEMA_VERSION,
    registry_path: str | Path = DEFAULT_SIGNAL_TYPES_REGISTRY,
    parameters: Mapping[str, Any] | None = None,
    source: Mapping[str, Any] | None = None,
    transformation_history: list[dict[str, Any]] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Signal:
    """Validate one frame and wrap it in a typed signal container."""

    validated = validate_signal_frame(
        df,
        signal_type=signal_type,
        value_column=value_column,
        version=version,
        registry_path=registry_path,
        parameters=parameters,
    )
    definition = resolve_signal_type_definition(signal_type, version=version, registry_path=registry_path)
    payload = canonicalize_signal_payload(
        {
            "signal_type": definition.signal_type_id,
            "version": definition.version,
            "value_column": value_column,
            "source": dict(source or {}),
            "parameters": dict(parameters or {}),
            "timestamp_normalization": "UTC",
            "transformation_history": list(transformation_history or []),
            "definition": {
                "directional": definition.directional,
                "ordinal": definition.ordinal,
                "probabilistic": definition.probabilistic,
                "executable": definition.executable,
                "compatible_position_constructors": list(definition.compatible_position_constructors),
                "cross_sectional": definition.cross_sectional,
            },
            **dict(metadata or {}),
        }
    )
    attach_signal_metadata(validated, payload)
    return Signal(
        signal_type=definition.signal_type_id,
        version=definition.version,
        data=validated,
        value_column=value_column,
        metadata=payload,
    )


def ensure_signal_type_compatible(
    signal_type: str,
    *,
    position_constructor: str,
    version: str = DEFAULT_SIGNAL_SCHEMA_VERSION,
    registry_path: str | Path = DEFAULT_SIGNAL_TYPES_REGISTRY,
) -> SignalTypeDefinition:
    definition = resolve_signal_type_definition(signal_type, version=version, registry_path=registry_path)
    normalized_constructor = _normalize_non_empty_string(
        position_constructor,
        field_name="position_constructor",
    )
    if normalized_constructor not in definition.compatible_position_constructors:
        formatted = ", ".join(definition.compatible_position_constructors) or "<none>"
        raise SignalSemanticsError(
            f"Signal type {definition.signal_type_id!r} is not compatible with position constructor "
            f"{normalized_constructor!r}. Compatible constructors: {formatted}."
        )
    return definition


def executable_position_constructor_name() -> str:
    return _EXECUTABLE_POSITION_CONSTRUCTOR


def legacy_signal_type_from_values(signals: pd.Series) -> str:
    """Best-effort legacy classification for unmanaged signal frames."""

    values = pd.to_numeric(signals, errors="raise").astype("float64")
    unique_values = set(values.dropna().tolist())
    if unique_values.issubset({0.0, 1.0}):
        return "binary_signal"
    if unique_values.issubset({-1.0, 0.0, 1.0}):
        return "ternary_quantile"
    return "target_weight"


def _normalize_quantile(value: float, *, field_name: str, upper_bound: float) -> float:
    try:
        quantile = float(value)
    except (TypeError, ValueError) as exc:
        raise SignalSemanticsError(f"{field_name} must be numeric.") from exc
    if math.isnan(quantile) or math.isinf(quantile):
        raise SignalSemanticsError(f"{field_name} must be finite.")
    if quantile <= 0.0 or quantile > upper_bound:
        raise SignalSemanticsError(f"{field_name} must be in the interval (0, {upper_bound}].")
    return quantile


def _ordered_index(values: pd.Series, *, symbols: pd.Series, ascending_scores: bool) -> list[Any]:
    order_frame = pd.DataFrame(
        {
            "_index": values.index,
            "_value": values.astype("float64"),
            "_symbol": symbols.astype("string"),
        }
    )
    ordered = order_frame.sort_values(
        ["_value", "_symbol"],
        ascending=[ascending_scores, True],
        kind="stable",
    )
    return ordered["_index"].tolist()


def _zscore_values(values: pd.Series, *, clip: float | None) -> pd.Series:
    if len(values) <= 1:
        return pd.Series(0.0, index=values.index, dtype="float64")
    mean_value = float(values.mean())
    std_value = float(values.std(ddof=0))
    if std_value == 0.0:
        result = pd.Series(0.0, index=values.index, dtype="float64")
    else:
        result = ((values - mean_value) / std_value).astype("float64")
    if clip is None:
        return result
    clip_value = abs(float(clip))
    return result.clip(lower=-clip_value, upper=clip_value).astype("float64")


def _rank_values(values: pd.Series, symbols: pd.Series) -> pd.Series:
    count = int(values.shape[0])
    if count <= 1:
        return pd.Series(0.0, index=values.index, dtype="float64")
    ordered = _ordered_index(values, symbols=symbols, ascending_scores=False)
    denominator = float(count - 1)
    result = pd.Series(index=values.index, dtype="float64")
    for rank_position, row_index in enumerate(ordered, start=1):
        result.loc[row_index] = 1.0 - (2.0 * float(rank_position - 1) / denominator)
    return result.astype("float64")


def _top_bottom_quantile(values: pd.Series, *, symbols: pd.Series, quantile: float) -> pd.Series:
    count = int(values.shape[0])
    if count <= 1:
        return pd.Series(0.0, index=values.index, dtype="float64")
    selected_count = min(max(int(math.ceil(float(count) * quantile)), 1), count // 2)
    result = pd.Series(0.0, index=values.index, dtype="float64")
    if selected_count == 0:
        return result
    top_rows = _ordered_index(values, symbols=symbols, ascending_scores=False)[:selected_count]
    bottom_rows = _ordered_index(values, symbols=symbols, ascending_scores=True)[:selected_count]
    result.loc[top_rows] = 1.0
    result.loc[bottom_rows] = -1.0
    return result.astype("float64")


def _long_only_top_quantile(values: pd.Series, *, symbols: pd.Series, quantile: float) -> pd.Series:
    count = int(values.shape[0])
    if count == 0:
        return pd.Series(dtype="float64", index=values.index)
    selected_count = min(max(int(math.ceil(float(count) * quantile)), 1), count)
    result = pd.Series(0.0, index=values.index, dtype="float64")
    top_rows = _ordered_index(values, symbols=symbols, ascending_scores=False)[:selected_count]
    result.loc[top_rows] = 1.0
    return result.astype("float64")


def _require_signal_type(signal: Signal, *, expected: set[str]) -> None:
    if signal.signal_type not in expected:
        formatted = ", ".join(sorted(expected))
        raise SignalSemanticsError(
            f"Transformation requires signal_type in {{{formatted}}}; received {signal.signal_type!r}."
        )


def _finalize_transformed_signal(
    parent_signal: Signal,
    frame: pd.DataFrame,
    *,
    output_type: str,
    output_column: str,
    parameters: Mapping[str, Any],
    operation: str,
) -> Signal:
    history = list(parent_signal.metadata.get("transformation_history", []))
    history.append(
        canonicalize_signal_payload(
            {
                "operation": operation,
                "input_type": parent_signal.signal_type,
                "output_type": output_type,
                "parameters": dict(parameters),
            }
        )
    )
    metadata = dict(parent_signal.metadata)
    metadata["transformation_history"] = history
    metadata["parameters"] = dict(parameters)
    if parameters.get("clip") is not None:
        metadata["clipping"] = {"clip": float(parameters["clip"])}
    return create_signal(
        frame,
        signal_type=output_type,
        value_column=output_column,
        version=parent_signal.version,
        parameters=parameters,
        source=parent_signal.metadata.get("source") if isinstance(parent_signal.metadata.get("source"), Mapping) else {},
        transformation_history=history,
        metadata={
            key: value
            for key, value in metadata.items()
            if key not in {"signal_type", "version", "value_column", "source", "parameters", "transformation_history"}
        },
    )


def _transform_cross_sectional(
    signal: Signal,
    *,
    output_type: str,
    output_column: str,
    parameters: Mapping[str, Any],
    operation: str,
    transform_fn: Any,
) -> Signal:
    frame = signal.data.copy(deep=True)
    frame.attrs = {}
    result_values = pd.Series(index=frame.index, dtype="float64")
    for _, group in frame.groupby("ts_utc", sort=False):
        values = pd.to_numeric(group[signal.value_column], errors="raise").astype("float64")
        symbols = group["symbol"].astype("string")
        result_values.loc[group.index] = transform_fn(values, symbols).astype("float64")
    frame[output_column] = result_values.astype("float64")
    return _finalize_transformed_signal(
        signal,
        frame,
        output_type=output_type,
        output_column=output_column,
        parameters=parameters,
        operation=operation,
    )


def score_to_signed_zscore(
    signal: Signal,
    *,
    output_column: str = "signal",
    clip: float | None = None,
) -> Signal:
    """Map cross-sectional scores into deterministic signed z-scores."""

    _require_signal_type(signal, expected={"prediction_score"})
    return _transform_cross_sectional(
        signal,
        output_type="signed_zscore",
        output_column=output_column,
        parameters={"clip": clip},
        operation="score_to_zscore",
        transform_fn=lambda values, _symbols: _zscore_values(values, clip=clip),
    )


def score_to_cross_section_rank(
    signal: Signal,
    *,
    output_column: str = "signal",
) -> Signal:
    """Map cross-sectional scores into deterministic normalized rank exposures."""

    _require_signal_type(signal, expected={"prediction_score"})
    return _transform_cross_sectional(
        signal,
        output_type="cross_section_rank",
        output_column=output_column,
        parameters={},
        operation="score_to_rank",
        transform_fn=_rank_values,
    )


def rank_to_percentile(
    signal: Signal,
    *,
    output_column: str = "signal",
) -> Signal:
    """Convert normalized cross-sectional ranks into percentiles in [0, 1]."""

    _require_signal_type(signal, expected={"cross_section_rank"})
    frame = signal.data.copy(deep=True)
    frame.attrs = {}
    frame[output_column] = (
        (pd.to_numeric(frame[signal.value_column], errors="raise").astype("float64") + 1.0) / 2.0
    )
    return _finalize_transformed_signal(
        signal,
        frame,
        output_type="cross_section_percentile",
        output_column=output_column,
        parameters={},
        operation="rank_to_percentile",
    )


def percentile_to_quantile_bucket(
    signal: Signal,
    *,
    quantile: float,
    output_column: str = "signal",
    long_only: bool = False,
) -> Signal:
    """Convert cross-sectional percentiles into deterministic binary or ternary buckets."""

    _require_signal_type(signal, expected={"cross_section_percentile"})
    normalized_quantile = _normalize_quantile(quantile, field_name="quantile", upper_bound=1.0)
    if not long_only and normalized_quantile > 0.5:
        raise SignalSemanticsError("Ternary percentile bucketing requires quantile in the interval (0, 0.5].")

    frame = signal.data.copy(deep=True)
    frame.attrs = {}
    percentile = pd.to_numeric(frame[signal.value_column], errors="raise").astype("float64")
    output = pd.Series(0.0, index=frame.index, dtype="float64")
    if long_only:
        output.loc[percentile >= (1.0 - normalized_quantile)] = 1.0
        output_type = "binary_signal"
    else:
        output.loc[percentile >= (1.0 - normalized_quantile)] = 1.0
        output.loc[percentile <= normalized_quantile] = -1.0
        output_type = "ternary_quantile"
    frame[output_column] = output
    return _finalize_transformed_signal(
        signal,
        frame,
        output_type=output_type,
        output_column=output_column,
        parameters={"quantile": normalized_quantile, "long_only": long_only},
        operation="percentile_to_quantile_bucket",
    )


def score_to_ternary_long_short(
    signal: Signal,
    *,
    quantile: float,
    output_column: str = "signal",
) -> Signal:
    """Map scores directly into deterministic long/flat/short buckets."""

    _require_signal_type(signal, expected={"prediction_score"})
    normalized_quantile = _normalize_quantile(quantile, field_name="quantile", upper_bound=0.5)
    return _transform_cross_sectional(
        signal,
        output_type="ternary_quantile",
        output_column=output_column,
        parameters={"quantile": normalized_quantile},
        operation="score_to_ternary_long_short",
        transform_fn=lambda values, symbols: _top_bottom_quantile(values, symbols=symbols, quantile=normalized_quantile),
    )


def score_to_binary_long_only(
    signal: Signal,
    *,
    quantile: float,
    output_column: str = "signal",
) -> Signal:
    """Map scores into deterministic long-only binary buckets."""

    _require_signal_type(signal, expected={"prediction_score"})
    normalized_quantile = _normalize_quantile(quantile, field_name="quantile", upper_bound=1.0)
    return _transform_cross_sectional(
        signal,
        output_type="binary_signal",
        output_column=output_column,
        parameters={"quantile": normalized_quantile},
        operation="score_to_binary_long_only",
        transform_fn=lambda values, symbols: _long_only_top_quantile(values, symbols=symbols, quantile=normalized_quantile),
    )


def spread_to_zscore(
    signal: Signal,
    *,
    output_column: str = "signal",
    clip: float | None = None,
    group_columns: tuple[str, ...] = ("symbol",),
) -> Signal:
    """Normalize raw spread values into deterministic time-series z-scores."""

    _require_signal_type(signal, expected={"spread_zscore", "prediction_score"})
    frame = signal.data.copy(deep=True)
    frame.attrs = {}
    missing = [column for column in group_columns if column not in frame.columns]
    if missing:
        formatted = ", ".join(repr(column) for column in missing)
        raise SignalSemanticsError(f"spread_to_zscore requires group columns: {formatted}.")

    values = pd.to_numeric(frame[signal.value_column], errors="raise").astype("float64")
    zscores = values.groupby(
        [frame[column].astype("string") for column in group_columns],
        sort=False,
        dropna=False,
    ).transform(lambda series: _zscore_values(series, clip=clip))
    frame[output_column] = zscores.astype("float64")
    return _finalize_transformed_signal(
        signal,
        frame,
        output_type="spread_zscore",
        output_column=output_column,
        parameters={"clip": clip, "group_columns": list(group_columns)},
        operation="spread_to_zscore",
    )


def validate_directional_asymmetry_compatibility(
    signal_definition: SignalTypeDefinition,
    constructor_id: str,
    *,
    asymmetry_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Validate that a signal type is compatible with directional asymmetry and constructor.
    
    Args:
        signal_definition: The signal type definition
        constructor_id: Position constructor ID
        asymmetry_config: Optional asymmetry configuration to validate against
    
    Returns:
        Dictionary with validation status and details
    """
    result: dict[str, Any] = {
        "valid": True,
        "signal_type": signal_definition.signal_type_id,
        "constructor_id": constructor_id,
        "issues": [],
        "warnings": [],
    }
    
    # Check if signal supports asymmetry
    if not signal_definition.directional_asymmetry_allowed:
        result["warnings"].append(
            f"Signal type {signal_definition.signal_type_id!r} does not explicitly support directional asymmetry. "
            "Use with caution."
        )
    
    # Check if signal is directional
    if not signal_definition.directional:
        result["warnings"].append(
            f"Signal type {signal_definition.signal_type_id!r} is not directional. "
            "Asymmetry features may not be meaningful."
        )
    
    # Check constructor compatibility
    if constructor_id not in signal_definition.compatible_position_constructors:
        result["valid"] = False
        result["issues"].append(
            f"Constructor {constructor_id!r} not in compatible list for {signal_definition.signal_type_id!r}: "
            f"{signal_definition.compatible_position_constructors}"
        )
    
    # Validate asymmetry-specific rules if provided
    if asymmetry_config is not None and signal_definition.asymmetry_validation_rules:
        for rule_name, rule_value in signal_definition.asymmetry_validation_rules.items():
            if rule_name in asymmetry_config:
                configured_value = asymmetry_config[rule_name]
                if isinstance(rule_value, dict) and "range" in rule_value:
                    min_val, max_val = rule_value["range"]
                    if not (min_val <= configured_value <= max_val):
                        result["valid"] = False
                        result["issues"].append(
                            f"Asymmetry parameter {rule_name!r}={configured_value} outside "
                            f"allowed range [{min_val}, {max_val}] for signal {signal_definition.signal_type_id!r}"
                        )
    
    if result["issues"]:
        result["error_message"] = "; ".join(result["issues"])
    
    return result


__all__ = [
    "DEFAULT_SIGNAL_SCHEMA_VERSION",
    "DEFAULT_SIGNAL_TYPES_REGISTRY",
    "Signal",
    "SignalSemanticsError",
    "SignalTypeDefinition",
    "attach_signal_metadata",
    "canonicalize_signal_payload",
    "create_signal",
    "default_signal_registry_path",
    "ensure_signal_type_compatible",
    "executable_position_constructor_name",
    "extract_signal_metadata",
    "legacy_signal_type_from_values",
    "load_signal_type_registry",
    "percentile_to_quantile_bucket",
    "rank_to_percentile",
    "resolve_signal_type_definition",
    "validate_directional_asymmetry_compatibility",
    "score_to_binary_long_only",
    "score_to_cross_section_rank",
    "score_to_signed_zscore",
    "score_to_ternary_long_short",
    "spread_to_zscore",
    "validate_signal_frame",
]
