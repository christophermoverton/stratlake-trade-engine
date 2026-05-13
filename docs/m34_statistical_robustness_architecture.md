# M34 Statistical Robustness Artifact Contract

Milestone 34 adds a statistical robustness evidence layer for research outputs. The first contract is intentionally small: it defines deterministic schemas, writer behavior, manifest conventions, and a Markdown skeleton that later diagnostics can populate without changing artifact names or downstream integration points.

## Purpose

The robustness bundle answers a different question than promotion governance. Governance reporting observes existing promotion outcomes, reason codes, and review metadata. Robustness validation records whether research evidence is statistically defensible enough to trust, including walk-forward efficiency, sample-size adequacy, sensitivity, multiple-testing metadata, and future temporal validation checks.

Issue 384 does not calculate those diagnostics. It creates the durable artifact interface that later M34 issues can extend.

## Canonical Bundle

Robustness reports are written under:

```text
artifacts/robustness/<report_id>/
```

Each bundle contains:

- `robustness_summary.json`
- `robustness_findings.json`
- `walk_forward_efficiency.csv`
- `sample_size_validation.json`
- `sensitivity_summary.csv`
- `multiple_testing_summary.json`
- `robustness_report.md`
- `manifest.json`

JSON files are emitted with sorted keys, two-space indentation, no non-finite floats, and LF line endings. CSV files use explicit column lists and LF line endings. Empty diagnostics are schema-valid: CSV files still contain headers and JSON files still contain their top-level arrays and field contracts.

## Finding Schema

Findings use stable fields:

- `check_id`
- `severity`
- `workflow_type`
- `run_id`
- `message`
- `details`

Severity values are normalized to:

- `info`
- `warning`
- `needs_review`
- `reject`
- `blocked`

`details` is reserved for structured evidence and is serialized through the same portable-path sanitizer as manifests.

## Summary Schema

`robustness_summary.json` rolls report evidence into deterministic counts:

- `report_id`
- `workflow_type_counts`
- `finding_count`
- `finding_count_by_severity`
- `highest_severity`
- `artifact_count`
- `source_run_count`
- `source_run_ids`
- `robustness_status_counts`
- `checks_present`
- `checks_missing`
- `generated_artifacts`

The summary is designed for CLI, notebook, API, and pipeline consumers that need a compact status view without parsing every artifact.

## Path Portability

Artifacts must not persist local absolute paths, file URIs, or machine-specific workspace roots. The writer uses the shared portable path helper from the artifact safety layer so references inside findings, upstream metadata, manifests, and writer config remain reproducible across machines.

`manifest.json` records portable artifact references, source artifact references, source run IDs, schema version, writer metadata, and the canonical generated artifact inventory.

## Walk-Forward Efficiency

Issue 385 adds the first concrete diagnostic: Walk-Forward Efficiency. WFE measures whether in-sample performance transfers to out-of-sample validation periods:

```text
WFE = Sharpe_OOS / Sharpe_IS
```

The default deterministic status bands are:

- `robust`: WFE is at least `0.75`
- `acceptable`: WFE is at least `0.50` and below `0.75`
- `weak`: WFE is at least `0.00` and below `0.50`
- `broken`: WFE is below `0.00`, or the in-sample Sharpe is negative
- `undefined`: WFE cannot be computed because an input is non-finite or the in-sample Sharpe is zero or near zero
- `missing`: required in-sample or out-of-sample Sharpe evidence is absent

Thresholds are represented by `WalkForwardEfficiencyThresholds` so later configuration work can tune the bands without changing artifact consumers. WFE rows are written to `walk_forward_efficiency.csv` using the Issue 384 column contract. Split dates, trade counts, threshold values, source run IDs, and edge-case reasons are included in the row `details` field and in matching robustness findings.

WFE findings use check IDs such as `walk_forward_efficiency.robust`, `walk_forward_efficiency.weak`, `walk_forward_efficiency.broken`, `walk_forward_efficiency.undefined`, and `walk_forward_efficiency.missing`. These findings are review evidence only. They do not automatically change promotion governance decisions in Issue 385; governance integration is deferred to later M34 work.

## Sample-Size And Trade-Count Guardrails

Issue 386 adds deterministic evidence-sufficiency guardrails. These checks flag research outputs that may look attractive but rest on too few observations, too few trades, too little out-of-sample evidence, insufficient split-level support, missing trade metadata, or thin regime coverage.

The default thresholds are represented by `SampleSizeThresholds` and are intentionally configurable. They are guardrails for review, not universal institutional requirements. The current checks include:

- `sample_size.minimum_total_samples`
- `sample_size.minimum_total_trades`
- `sample_size.minimum_oos_trades`
- `sample_size.minimum_trades_per_split`
- `sample_size.minimum_unique_periods`
- `sample_size.minimum_regime_coverage`
- `sample_size.minimum_trades_per_regime`
- `sample_size.missing_sample_count`
- `sample_size.missing_trade_count`
- `sample_size.missing_oos_trade_count`

Sample count and unique period checks focus on whether the observation base is broad enough to support inference. Trade-count checks focus on whether the realized decision set is thick enough to trust reported performance, especially out of sample. Split-level checks prevent one aggregate count from hiding fragile validation folds.

Regime-aware checks run only when regime trade-count metadata is supplied. If no regime metadata is available, the report degrades gracefully and does not fail regime coverage by assumption. When regime metadata is present, the guardrails check both the number of represented regimes and the minimum trades per regime.

Missing metadata is emitted as deterministic `missing` validations and findings rather than causing opaque failures. Non-finite values are treated as unavailable and are not serialized into canonical artifacts.

As with WFE, sample-size findings are robustness evidence for human or later governance review. Issue 386 does not change promotion governance decisions.

## Parameter Sensitivity And Fragility Analysis

Issue 387 adds deterministic parameter sensitivity diagnostics for local robustness testing around a selected configuration. The objective is to identify fragile optima where small perturbations materially degrade performance. The objective is not to introduce a new optimization loop.

For a base configuration $\theta$ and perturbation $\theta'$, the sensitivity layer compares a selected metric $M$:

$$
\Delta M = M(\theta') - M(\theta)
$$

Direction-aware deterioration is computed from metric direction metadata:

$$
D =
\begin{cases}
M(\theta) - M(\theta') & \text{if higher is better} \\
M(\theta') - M(\theta) & \text{if lower is better}
\end{cases}
$$

Relative deterioration is only computed when the base metric is finite and safely away from zero:

$$
\text{relative\_deterioration} = \frac{D}{|M(\theta)|}
$$

When $|M(\theta)|$ is below the configured near-zero guard, relative deterioration is marked undefined to avoid unstable ratios. Absolute delta and deterioration remain available when both metric values are finite.

Sensitivity statuses are deterministic and configurable:

- `improved`
- `stable`
- `mildly_sensitive`
- `fragile`
- `undefined`
- `missing`

The sensitivity module supports both dataclass and mapping inputs so it can consume precomputed candidate outputs and explicit sensitivity-grid metadata without re-running strategy engines. Findings include direction metadata (`higher_is_better`), perturbation metadata (`perturbation_type`, `perturbation_size`), parameter distance fields, and threshold values used for classification.

Numeric parameter perturbations emit:

- `parameter_distance`
- `normalized_parameter_distance`

Categorical perturbations do not force numeric distance calculations.

Rows are emitted through `sensitivity_summary.csv` using the existing Issue 384 column contract. Extended evidence (deterioration, relative deterioration, perturbation metadata, thresholds, reasons, and source references) is stored in deterministic `details` payloads. Findings are emitted as structured `RobustnessFinding` records with check IDs like `sensitivity.fragile`, `sensitivity.mildly_sensitive`, `sensitivity.undefined`, and `sensitivity.missing`.

Sensitivity findings are review evidence. They do not automatically reject runs or change promotion governance decisions.

## Multiple-Testing And Trial-Count Metadata

Issue 388 adds deterministic multiple-testing metadata. The goal is transparency: a high Sharpe ratio, hit rate, information coefficient, or other selected metric is less informative when it was chosen from a large search space than when it came from a small controlled experiment. Trying many candidates, configurations, parameters, factors, models, strategies, portfolios, campaigns, or scenarios increases the chance that one result looks strong by luck, data snooping, or overfitting.

The first implementation records the size of the search space without claiming a formal statistical correction. It accepts explicit metadata from robustness configs, campaign manifests, comparison leaderboards, registries, or manual mappings. The effective trial count is computed conservatively as the maximum available value across explicit trial-count fields:

- `candidate_count`
- `tested_configuration_count`
- `parameter_combination_count`
- `scenario_count`
- `factor_count`
- `model_count`
- `portfolio_count`
- `campaign_count`

Only present, finite, non-negative integer count values are used. Missing trial-count metadata is emitted as `multiple_testing.missing_trial_count_metadata`. Non-finite, negative, fractional, or otherwise invalid count metadata is emitted as `multiple_testing.undefined_trial_count_metadata`. These cases are findings, not opaque failures.

Trial-count statuses are deterministic and configurable through `MultipleTestingThresholds`:

- `low_risk`: controlled search space under the configured low-risk maximum
- `moderate_risk`: non-trivial search space that warrants context
- `high_risk`: large search space where selection bias and data-snooping concern are material
- `extreme_risk`: very large search space requiring strong caution
- `missing`: no explicit trial-count metadata is available
- `undefined`: trial-count metadata is present but invalid

The emitted `multiple_testing_summary.json` rows preserve the Issue 384 artifact contract. Additional evidence is stored in deterministic row `details`, including the effective trial count, all supplied count fields, trial-count source, selected rank, selection metric, selection metric value, thresholds, reason, and support flags for future statistical methods.

Selected-rank metadata is recorded when available. If rank `1` is selected from a large configured search space, the report emits `multiple_testing.selected_rank_warning` to make the search-order dependency visible. If candidate count is present but selected rank is missing, if selected rank exceeds candidate count, or if selected rank exists without a selection metric, the same structured check ID is emitted with a reason in details. Rank alone is not interpreted as statistical significance.

This issue does not implement Deflated Sharpe Ratio, Probability of Backtest Overfitting, Combinatorial Symmetric Cross-Validation, statistical haircuts, or any new optimization loop. The artifact may mark DSR, PBO, and haircut support as false so downstream readers can distinguish metadata-only evidence from formal corrections. Those methods remain future extension points and should only be claimed when their assumptions, inputs, implementation, and tests are represented.

Multiple-testing findings are review evidence. They do not automatically reject runs or change promotion governance decisions.

## Extension Points

Later M34 issues can populate the existing artifacts with real diagnostics:

- Additional walk-forward efficiency extraction sources in `walk_forward_efficiency.csv`
- Additional sample-size and trade-count extraction sources in `sample_size_validation.json`
- Sensitivity or fragility rows in `sensitivity_summary.csv`
- Additional multiple-testing extraction sources in `multiple_testing_summary.json`
- Purged or embargoed validation findings through the shared finding schema
- Governance integration through upstream governance artifact references

The contract supports optional strategy, alpha, portfolio, campaign, governance, and generic upstream artifact references. Missing references are valid and degrade to empty manifest sections.

## Non-Goals

This contract does not implement multiple-testing haircuts, DSR, PBO, purged validation, dashboards, external services, or promotion governance decision changes. Those belong to later M34 issues and should consume this bundle rather than redefining it.
