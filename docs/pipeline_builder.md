# Pipeline Builder

## Overview

`src.pipeline.builder.PipelineBuilder` is a declarative authoring layer for
M20-compatible research pipelines.

It sits above the existing pipeline runner. It does not replace M20.

The builder:

* resolves strategies, signal semantics, and position constructors from the
  existing registries
* validates compatibility before any YAML is emitted
* writes deterministic, check-in-friendly pipeline/config files
* executes through the existing `python -m src.cli.run_pipeline` contract
* preserves normal strategy, robustness, portfolio, manifest, lineage, and
  registry artifacts

## When To Use It

Use the builder when you want to compose research workflows from named,
registry-backed components instead of hand-wiring low-level pipeline YAML.

Use raw M20 YAML when you already know the exact step graph you want and do not
need this higher-level composition layer.

## Python API

```python
from src.pipeline.builder import PipelineBuilder

builder = (
    PipelineBuilder("m21_research_pipeline")
    .strategy("cross_section_momentum", params={"lookback_days": 63})
    .signal("cross_section_rank")
    .construct_positions(
        "rank_dollar_neutral",
        params={"gross_long": 0.5, "gross_short": 0.5},
    )
    .portfolio("equal_weight", params={"timeframe": "1D"})
)

builder.to_yaml("configs/generated/m21_research_pipeline.yml")
result = builder.run("configs/generated/m21_research_pipeline.yml")
```

## Generated Files

Single-run pipelines generate:

* one M20 pipeline YAML file
* one declarative strategy config file consumed by `src.cli.run_builder_strategy`
* optionally one portfolio config file consumed by `src.cli.run_portfolio`

Sweep pipelines generate:

* one M20 pipeline YAML file
* one strategies config file consumed by `src.cli.run_strategy`
* one robustness config file consumed by `src.cli.run_strategy --robustness`

## Single Strategy Example

```python
builder = (
    PipelineBuilder("cross_section_rank_pipeline")
    .strategy("cross_section_momentum", params={"lookback_days": 20})
    .signal("cross_section_rank")
    .construct_positions(
        "rank_dollar_neutral",
        params={"gross_long": 0.5, "gross_short": 0.5},
    )
)
```

This emits a one-step M20 pipeline that runs a declarative strategy artifact
through the normal strategy artifact tracker.

## Signal + Constructor Example

```python
builder = (
    PipelineBuilder("rank_to_binary_pipeline")
    .strategy("cross_section_momentum", params={"lookback_days": 20})
    .signal("binary_signal", params={"quantile": 0.2})
    .construct_positions("identity_weights", params={"exclude_short": True})
)
```

The builder validates that:

* the strategy can transform from its native output signal to `binary_signal`
* `binary_signal` is compatible with `identity_weights`
* the supplied asymmetry or constructor params are valid for that constructor

## Asymmetry Example

```python
builder = (
    PipelineBuilder("asymmetric_rank_pipeline")
    .strategy("cross_section_momentum", params={"lookback_days": 20})
    .signal("cross_section_rank")
    .construct_positions(
        "rank_dollar_neutral",
        params={"gross_long": 0.6, "gross_short": 0.4},
    )
    .asymmetry(
        {
            "exclude_short": False,
            "max_short_positions": 10,
            "short_position_scale": 0.8,
        }
    )
)
```

## Portfolio Example

```python
builder = (
    PipelineBuilder("strategy_plus_portfolio")
    .strategy("cross_section_momentum", params={"lookback_days": 20})
    .signal("cross_section_rank")
    .construct_positions(
        "rank_dollar_neutral",
        params={"gross_long": 0.5, "gross_short": 0.5},
    )
    .portfolio(
        "equal_weight",
        params={
            "timeframe": "1D",
            "alignment_policy": "intersection",
            "initial_capital": 1.0,
        },
    )
)
```

The builder creates a pipeline-local strategy artifact name so the downstream
portfolio step can deterministically resolve the strategy run from the registry
without colliding with unrelated runs.

## Sweep Example

```python
builder = (
    PipelineBuilder("research_sweep")
    .strategy("cross_section_momentum", params={"lookback_days": 20})
    .signal("cross_section_rank")
    .construct_positions(
        "rank_dollar_neutral",
        params={"gross_long": 0.5, "gross_short": 0.5},
    )
    .asymmetry({"exclude_short": False})
    .sweep(
        {
            "signal": {
                "type": ["cross_section_rank", "binary_signal"],
                "params": {"quantile": [0.2]},
            },
            "constructor": {
                "name": ["rank_dollar_neutral", "identity_weights"],
            },
            "asymmetry": {
                "exclude_short": [False, True],
            },
            "ranking": {
                "primary_metric": "sharpe_ratio",
                "tie_breakers": ["total_return"],
            },
        }
    )
)
```

This emits a standard robustness pipeline step that still runs through
`src.cli.run_strategy --robustness`.

## CLI

Render builder-friendly config to M20 YAML:

```powershell
python -m src.cli.build_pipeline --config configs/pipeline_builder.yml --output configs/generated/pipeline.yml
```

Render and immediately execute:

```powershell
python -m src.cli.build_pipeline --config configs/pipeline_builder.yml --output configs/generated/pipeline.yml --run
```

## Builder Config Shape

```yaml
pipeline_id: builder_example
start: "2025-01-01"
end: "2025-03-31"
strategy:
  name: cross_section_momentum
  params:
    lookback_days: 20
signal:
  type: cross_section_rank
constructor:
  name: rank_dollar_neutral
  params:
    gross_long: 0.5
    gross_short: 0.5
portfolio:
  name: equal_weight
  params:
    timeframe: 1D
```

## Notes

* The builder stays registry-driven for strategies, signals, and position
  constructors.
* Portfolio authoring uses existing config-loader semantics rather than a new
  portfolio registry, because the repo does not currently expose one.
* Builder-generated single runs use `src.cli.run_builder_strategy` so signal
  transformations and constructor composition are real execution behavior, not
  just relabeled config.
* First pass limitation: sweep pipelines do not also build a downstream
  portfolio stage in the same declarative spec.
