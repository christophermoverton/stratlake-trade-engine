# Regime Sensitivity Matrix

## Overview

The regime sensitivity layer extends Milestone 25 Issue 1 from single-profile
calibration into multi-profile comparison.

It does not rebuild the taxonomy, rename labels, or introduce new modeling.
Instead, it runs multiple deterministic calibration profiles against the same
canonical regime-label input and makes the tradeoffs measurable.

Use it to answer questions such as:

* which profiles are too reactive?
* which profiles are too conservative?
* how do flip rate, duration, transition counts, and fallback usage change by
  profile?
* which profile is stable enough for downstream decisioning or later ML / policy
  work?
* how does regime-conditioned performance shift when returns or IC inputs are
  attached?

## How It Differs From Single-Profile Calibration

Single-profile calibration answers: "what does this one profile produce?"

Sensitivity analysis answers: "how do multiple valid profiles compare on the
same input?"

The sensitivity layer reuses:

* Issue 1 built-in profiles and profile resolution
* Issue 1 stability metrics and instability flags
* M24 alignment helpers
* M24 conditional evaluation and attribution utilities

It adds:

* one matrix row per profile
* deterministic profile ranking
* a recommended profile when one passes downstream eligibility gates
* optional performance comparison across the calibrated outputs

## Public API

```python
from src.research.regimes import (
    run_regime_calibration_sensitivity,
    write_regime_sensitivity_artifacts,
)

result = run_regime_calibration_sensitivity(
    regime_labels,
    profiles=["baseline", "conservative", "reactive", "crisis_sensitive"],
)

write_regime_sensitivity_artifacts(
    "artifacts/regimes/example_run/sensitivity",
    result,
    source_regime_artifact_references={"manifest_path": "../manifest.json"},
)
```

Optional performance-aware usage:

```python
result = run_regime_calibration_sensitivity(
    regime_labels,
    profiles=["baseline", "conservative"],
    performance_frame=strategy_returns,
    performance_surface="strategy",
    performance_value_column="strategy_return",
)
```

## Matrix Columns

`regime_sensitivity_matrix.csv` contains one row per profile.

Core columns include:

* `profile_name`
* `transition_count`
* `flip_rate`
* `single_day_flip_count`
* `single_day_flip_share`
* `average_regime_duration`
* `median_regime_duration`
* `minimum_regime_duration`
* `maximum_regime_duration`
* `unstable_regime_share`
* `low_confidence_share`
* `attribution_eligible_regime_count`
* `attribution_ineligible_regime_count`
* `warning_count`
* `fallback_rows_total`
* `unknown_fallback_rows`
* `low_confidence_fallback_rows`
* `unstable_profile_fallback_rows`
* `dominant_regime_label`
* `dominant_regime_share`
* `defined_observation_share`
* `stable_profile_rank`
* `stability_score`
* `eligible_for_downstream_decisioning`
* `is_recommended_profile`

## Ranking Logic

Ranking is intentionally simple and transparent.

Profiles are ordered by:

1. stable before unstable
2. lower `flip_rate`
3. lower `single_day_flip_share`
4. lower `unstable_regime_share`
5. higher `attribution_eligible_regime_count`
6. higher `defined_observation_share`
7. fewer warnings
8. caller-supplied profile order
9. profile name

`stable_profile_rank` is the deterministic position under those rules.

`eligible_for_downstream_decisioning` requires:

* profile not flagged unstable
* some defined observations remain
* at least one attribution-eligible regime remains

If at least one profile is eligible, the highest-ranked eligible row is marked
with `is_recommended_profile = true`.

## What Makes A Profile Unstable

Sensitivity analysis does not invent a new instability definition. It reuses
Issue 1 profile flags:

* `exceeds_max_flip_rate`
* `exceeds_max_single_day_flip_share`
* `exceeds_low_confidence_share`
* `has_unstable_runs`

If any of those are true, `is_unstable_profile` is true.

## Optional Performance Columns

When `performance_frame` is provided, the sensitivity result also includes
profile-level summaries built from the calibrated labels aligned back onto the
performance surface.

Return-based surfaces expose profile summaries such as:

* best regime by mean return
* worst regime by mean return
* profile mean return
* profile volatility
* profile cumulative return

These summaries are descriptive only. They help compare how calibration choices
change the distribution of observations across regime buckets.

## Artifact Contract

`write_regime_sensitivity_artifacts()` writes:

* `regime_sensitivity_matrix.csv`
* `regime_sensitivity_summary.json`
* `regime_stability_report.md`
* `calibration_profile_results.json`

`regime_sensitivity_summary.json` is the manifest-like summary payload. It
includes:

* `schema_version`
* `artifact_type`
* `taxonomy_version`
* `profile_count`
* `profile_names`
* `recommended_profile`
* `source_regime_artifact_references`
* `metadata`
* `file_inventory`

As with `regime_calibration.json`, the summary JSON is listed in its own
`file_inventory` by path only. It is not self-hashed because a self-hash would
change the file contents recursively.

## Interpretation Guidance

Use the matrix to spot three common patterns:

* overly reactive profiles: high flip rate, high single-day-flip share, many
  warnings
* overly conservative profiles: very low transition count, high dominant regime
  share, or reduced defined coverage after fallback application
* acceptable compromise profiles: low noise, meaningful defined coverage, and
  enough eligible regimes for downstream work

The recommendation is a deterministic heuristic, not an optimization routine.
It is meant to narrow review, not replace judgment.
