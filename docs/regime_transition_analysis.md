# Regime Transition Analysis (M24.4)

## Overview

M24.4 extends the regime stack from static regime-conditioned analysis into
deterministic transition analysis. The goal is descriptive auditability:

* what changed
* when it changed
* how the research surface behaved around that change

This layer does not infer hidden states, does not use macro overlays, and does
not adapt live trading behavior.

## Transition Event Detection

Transition events are detected directly from the sorted canonical regime labels
produced by M24.1 and validated by M24.2.

Supported transition dimensions:

* `composite`
* `volatility`
* `trend`
* `drawdown_recovery`
* `stress`

Detection contract:

* input must satisfy the canonical regime-label contract
* rows are inspected in sorted `ts_utc` order
* a transition event is emitted only when two adjacent rows are both
  `is_defined == true`
* undefined rows never count as transition evidence
* the event timestamp is the current row timestamp where the state changed
* event ordering is deterministic: `ts_utc`, then canonical dimension order

Each event records:

* transition timestamp
* dimension
* previous state
* current state
* stable transition label
* descriptive category
* directional tag
* stress-transition flag
* taxonomy version
* deterministic event ordering metadata

## Stress-Shift Categories

Transition categories are descriptive, not causal:

* `volatility_upshift`
* `volatility_downshift`
* `trend_breakdown`
* `drawdown_onset`
* `recovery_onset`
* `stress_onset`
* `stress_relief`
* `generic_transition`

Composite transitions inherit the first non-generic category found while
walking changed component dimensions in canonical taxonomy order.

## Transition Windows

Rows can be tagged around each event with deterministic row-based windows:

* `pre_window`
* `event_timestamp`
* `post_window`

Default configuration is row-count based:

* `pre_event_rows = 2`
* `post_event_rows = 2`

Tagged rows include:

* event id and event timestamp
* window role
* row offset from event
* timestamp distance from event
* overlap count
* overlap flag
* valid-evidence flag

If aligned regime columns are present, `transition_is_valid_evidence` is true
only for rows whose alignment status is `matched_defined`. Undefined or
unmatched rows may still appear in the window artifact for auditability, but
they do not contribute to summary metrics.

Overlap handling is explicit:

* if `allow_window_overlap = true`, overlapping rows appear once per event
  assignment and record overlap counts
* if `allow_window_overlap = false`, overlapping windows raise an error

## Transition-Aware Summaries

Supported surfaces:

* strategy returns
* alpha IC / Rank IC
* portfolio returns

Strategy and portfolio summaries include:

* `pre_transition_return`
* `event_window_return`
* `post_transition_return`
* `window_cumulative_return`
* `window_volatility`
* `window_max_drawdown`
* `window_win_rate`
* observation counts
* coverage status

Alpha summaries include:

* `pre_mean_ic`
* `event_mean_ic`
* `post_mean_ic`
* `window_mean_ic`
* `pre_mean_rank_ic`
* `event_mean_rank_ic`
* `post_mean_rank_ic`
* `window_mean_rank_ic`
* observation counts
* coverage status

Coverage semantics:

* `empty`: no valid matched-defined observations in the window
* `sparse`: at least one valid observation exists, but a required section is
  empty or total valid observations are below `min_observations`
* `sufficient`: pre, event, and post sections are all present and the window
  meets `min_observations`

## Artifact Outputs

Canonical persisted outputs:

* `regime_transition_events.csv`
* `regime_transition_windows.csv`
* `regime_transition_summary.json`
* `regime_transition_manifest.json`

The summary and manifest include:

* taxonomy version
* surface
* transition dimensions present
* event count
* stress-transition count
* window configuration
* coverage counts
* file inventory metadata

## Interpretation Boundaries

These outputs are descriptive. They show association around deterministic regime
changes; they do not prove causality and they are not adaptive trading signals.

Later reporting, notebook, comparison, and case-study layers can build on these
stable artifacts without changing the underlying transition contracts.
