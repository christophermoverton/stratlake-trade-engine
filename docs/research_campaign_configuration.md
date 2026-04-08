# Research Campaign Configuration

`src.config.research_campaign` defines one normalized contract for campaign-level
research settings that span alpha comparison, strategy comparison, candidate
selection, portfolio construction, and unified review.

## Contract

The campaign schema supports these sections:

* `dataset_selection`: shared dataset, timeframe, evaluation horizon,
  mapping name, and optional ticker file
* `time_windows`: shared outer, train, and predict windows
* `targets`: alpha, strategy, and portfolio names plus catalog/config paths
* `comparison`: registry-backed comparison settings and ranking controls
* `candidate_selection`: governed-candidate filters, thresholds, allocation,
  execution toggles, and artifact destinations
* `portfolio`: portfolio execution target and high-level run settings
* `review`: unified research review filters, ranking, outputs, and promotion
  gate wiring through the existing `ReviewConfig` contract
* `outputs`: common artifact roots and shared output destinations

The repository example lives at `configs/research_campaign.yml`.

## Precedence

Use `resolve_research_campaign_config(...)` to merge layers deterministically:

```text
repository defaults < config sources < CLI overrides
```

This matches the repository's existing config precedence pattern used by review,
runtime, and execution settings.

## Normalization

The loader applies a few shared defaults so one campaign file can stay concise:

* `candidate_selection.dataset`, `timeframe`, `evaluation_horizon`, and
  `mapping_name` inherit from `dataset_selection` when omitted
* `candidate_selection.alpha_name` inherits from `targets.alpha_names` when
  exactly one alpha target is listed
* `portfolio.portfolio_name` inherits from `targets.portfolio_names` when
  exactly one portfolio target is listed
* `review.filters.dataset`, `timeframe`, and single-target alpha/strategy/
  portfolio filters inherit from the campaign-wide sections
* candidate-selection and review output paths inherit from `outputs` when their
  local sections omit explicit paths
* string lists are trimmed, deduplicated, and preserved in input order
* path-like strings are normalized to forward-slash form for stable manifests

## Loading

```python
from src.config.research_campaign import resolve_research_campaign_config

config = resolve_research_campaign_config(
    {"research_campaign": {"comparison": {"enabled": True}}},
    cli_overrides={"portfolio": {"enabled": True}},
)
```

Use `load_research_campaign_config()` when you only want repository defaults or
one YAML/JSON file.
