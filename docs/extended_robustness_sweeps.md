# Extended Robustness Sweeps

StratLake now supports deterministic robustness sweeps across the full research stack:

- strategy selection and parameters
- signal semantics and transformation settings
- position constructors and constructor parameters
- long/short asymmetry controls
- bounded ensemble definitions

The sweep runner remains registry-driven and backward compatible with legacy parameter-only sweeps.

## Example Config

See [../configs/robustness_extended_example.yml](../configs/robustness_extended_example.yml).

The new `robustness.sweep` mapping supports these sections:

- `strategy`
  Strategy ids plus parameter and option grids.
- `signal`
  Signal type plus semantic transform parameters such as `quantile` or `clip`.
- `constructor`
  Position constructor ids plus constructor parameter grids.
- `asymmetry`
  Directional constraints such as `exclude_short` or short-availability policy.
- `ensemble`
  Explicit ensemble definitions with deterministic weighting methods.
- `group_by`
  Optional grouping metadata persisted into the aggregate summary.
- `batching`
  Optional deterministic batching controls for large research spaces.

## Execution

Run an extended sweep through the normal strategy CLI:

```bash
python -m src.cli.run_strategy --strategy cross_section_momentum --robustness configs/robustness_extended_example.yml
```

Run the same sweep as a normal pipeline step through the M20 runner:

```yaml
id: research_pipeline
steps:
  - id: robustness
    adapter: python_module
    module: src.cli.run_strategy
    argv:
      - --strategy
      - cross_section_momentum
      - --robustness
      - configs/robustness_extended_example.yml
```

## Validation Rules

Before execution, the runner resolves strategies, signal types, and position constructors through the registries and rejects incompatible combinations up front. Common invalid cases include:

- signal semantics incompatible with the requested constructor
- transformed signal types unsupported by the source strategy output
- asymmetry settings applied to unsupported signal and constructor pairs
- duplicate or non-composable ensemble definitions

Rejected configurations are recorded in `summary.json` so the sweep remains auditable.

## Artifacts

Each extended sweep writes:

- per-run child artifacts under `runs/<variant_id>/`
- `metrics_by_config.csv`
- `metrics_by_variant.csv`
- `ranked_configs.csv`
- `aggregate_metrics.json`
- `summary.json`
- `manifest.json`

These outputs preserve the strategy, signal, constructor, asymmetry, and ensemble metadata needed to explain why a configuration performed well and whether that performance was stable.
