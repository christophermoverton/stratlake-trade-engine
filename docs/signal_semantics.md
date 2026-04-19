# Signal Semantics Layer

## Purpose

The signal semantics layer makes signal meaning explicit across research, alpha, backtest, and pipeline workflows.
Every typed signal answers three questions without reading implementation code:

- what the signal means
- how it was created
- what downstream components may consume it

The implementation lives in [src/research/signal_semantics.py](/C:/Users/christophermoverton/stratlake-trade-engine/src/research/signal_semantics.py:1) and is backed by the registry in [artifacts/registry/signal_types.jsonl](/C:/Users/christophermoverton/stratlake-trade-engine/artifacts/registry/signal_types.jsonl:1).

## Canonical Signal Types

The registry defines the canonical `signal_type` contract for:

- `prediction_score`
- `cross_section_rank`
- `cross_section_percentile`
- `signed_zscore`
- `ternary_quantile`
- `binary_signal`
- `spread_zscore`
- `target_weight`

Each entry records:

- semantic meaning and mathematical definition
- domain and codomain
- semantic flags such as directional, ordinal, probabilistic, and executable
- required columns and cross-sectional rules
- deterministic transformation policies
- compatible position constructors

## Data Contract

Signals are represented by the `Signal` container:

```python
Signal(
    signal_type="signed_zscore",
    version="1.0.0",
    data=signal_frame,
    value_column="signal",
    metadata={...},
)
```

The metadata payload is stored in `DataFrame.attrs["signal_semantics"]` and includes:

- `signal_type`
- `version`
- `value_column`
- `source`
- `parameters`
- `timestamp_normalization`
- `transformation_history`
- executable compatibility details from the registry

## Validation Rules

`validate_signal_frame(...)` enforces deterministic, fail-fast validation:

- required columns must be present
- `symbol` values must be non-empty
- `ts_utc` must parse as UTC timestamps
- `(symbol, ts_utc)` keys must be unique
- rows must be sorted deterministically
- signal values must be numeric, finite, and non-null
- per-type range and allowed-value constraints must hold
- cross-sectional signal types must satisfy minimum universe-size requirements

No silent coercion or implicit conversion is performed.

## Deterministic Transformations

Supported explicit transformations include:

- `score_to_signed_zscore`
- `score_to_cross_section_rank`
- `rank_to_percentile`
- `percentile_to_quantile_bucket`
- `score_to_ternary_long_short`
- `score_to_binary_long_only`
- `spread_to_zscore`

Every transformation appends one deterministic record to `transformation_history`, including:

- operation name
- input type
- output type
- parameters used

Tie-breaking uses stable sorting with `symbol` ascending as the final deterministic key.

## Integration Points

### Strategy Layer

Strategy definitions may declare:

```yaml
signal_type: ternary_quantile
signal_params: {}
```

`generate_signals(...)` now wraps strategy output in a typed `Signal` and attaches signal metadata to the returned frame.

### Alpha Layer

Alpha signal mapping resolves an explicit config:

```yaml
signal_mapping:
  policy: zscore_continuous
  signal_type: signed_zscore
  signal_params:
    clip: 3.0
```

The mapping result carries both the output DataFrame and the typed `Signal`.

### Backtest Layer

`run_backtest(...)` validates signal semantics before execution.

- typed signals are validated against the registry and compatibility rules
- canonical strategy, alpha-sleeve, walk-forward, robustness, and builder paths require managed typed signals
- legacy unmanaged inputs remain supported only for direct/manual backtest usage outside canonical workflows
- non-executable or undefined types are rejected

### Artifacts And Manifests

When signal metadata is present, strategy and alpha artifacts persist it in `signal_semantics.json` and surface it in `manifest.json`.

This gives downstream tooling a stable answer to:

> What is this signal, how was it created, and what can consume it?

## Migration Guidance

Preferred usage is explicit declaration in config.

- strategies should declare `signal_type` and `signal_params`
- alpha mappings should declare `policy`, `signal_type`, and `signal_params`
- downstream consumers should read signal semantics from manifests instead of inferring meaning from raw numeric values

Legacy numeric signal frames are still accepted by the backtest runner for compatibility, but they should be upgraded to typed signals over time.

Canonical workflows do not infer contracts from raw numeric frames. When a workflow persists `signals.parquet`, it also persists `signal_semantics.json`; reload that sidecar if you need to reconstruct typed-signal metadata after reading artifacts back from disk.
