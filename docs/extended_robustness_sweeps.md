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

The existing `robustness.statistical_controls` block now drives first-pass
statistical-validity outputs for multidimensional sweeps:

- `primary_metric`
  Optional validity primary metric override. When omitted, the sweep ranking
  primary metric is reused.
- `multiple_testing_awareness`
  Enables split-level raw p-value estimation plus FDR-style q-value adjustment
  when the sweep has eligible split evidence.
- `deflated_sharpe_ratio`
  Enables Deflated Sharpe Ratio when the primary metric is `sharpe_ratio` and
  return-history requirements are met.
- `validity_ranking_method`
  Explicit validity-aware ranking path. Supported values are
  `adjusted_q_value`, `deflated_sharpe_ratio`, and `none`.

## Statistical Validity Methodology

The first pass is intentionally explicit and conservative:

- raw ranking is still preserved and written unchanged
- validity-aware ranking is written separately through `validity_rank`
- the multiple-testing family is the full evaluated sweep after invalid
  combinations are removed
- deterministic batching does not silently compute per-batch corrections
- when the full family is unavailable, the artifacts persist
  `correction_not_applicable` plus a reason instead of inventing a fallback

Current methods:

- `fdr_bh`
  Benjamini-Hochberg q-value adjustment over eligible raw p-values.
- `deflated_sharpe_ratio`
  Available only when the primary metric is `sharpe_ratio` and enough return
  history exists per configuration.

Deferred or not implemented in this first pass:

- `pbo_cscv`
- `white_reality_check`
- `spa`

Those methods are recorded under
`statistical_validity.methods_skipped_and_reason`.

## Split-Level Evidence

Extended sweeps can now reuse `stability.mode: subperiods` as the evidence layer
for first-pass inference. Raw p-values are only produced when:

- the sweep is not running as a partial deterministic batch
- split-level values exist for the selected primary metric
- each eligible configuration has at least the configured
  `min_splits_for_inference`
- the primary metric is one of the documented zero-centered metrics supported by
  this first pass

If those requirements are not met, q-values are left blank and the run records
why they were not applicable.

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

### Row-Level Validity Fields

`metrics_by_config.csv` and `ranked_configs.csv` now include:

- `raw_primary_metric`
  The metric used by the validity layer before any correction-aware ranking.
- `correction_method`
  The explicit multiple-testing method used for that row, or
  `not_applicable`.
- `correction_eligible`
  Whether the row had enough admissible evidence to participate in q-value
  correction.
- `raw_p_value`
  The documented raw p-value used before adjustment.
- `adjusted_q_value`
  The FDR-adjusted q-value when available.
- `deflated_sharpe_ratio`
  The DSR probability-style output when applicable.
- `validity_warning_codes`
  Stable `|`-delimited warning codes.
- `validity_rank`
  Rank under the requested validity-aware ordering when available.
- `validity_notes`
  Short deterministic notes explaining estimation or non-applicability.

### Sweep-Level Validity Metadata

`summary.json`, `aggregate_metrics.json`, and `manifest.json` now persist a
`statistical_validity` object with:

- `enabled`
- `primary_metric`
- `family_scope`
- `n_configs_evaluated`
- `n_configs_eligible`
- `correction_method_used`
- `methods_skipped_and_reason`
- `warning_thresholds`
- `warning_counts_by_code`
- `methodology_version`

### Warning Codes

The first pass surfaces these deterministic warning codes:

- `large_search_space`
- `insufficient_splits_for_inference`
- `correction_not_applicable`
- `fdr_nonpass`
- `low_neighbor_gap`
- `low_threshold_pass_rate`
- `rank_instability`
- `insufficient_history_for_dsr`
- `high_pbo`

`high_pbo` is reserved for future use and will only appear once a PBO method is
implemented.

## Interpretation Notes

- Raw ranking answers "which configuration scored highest on the selected
  metric?" It does not adjust for search breadth or multiple testing.
- Validity-aware ranking answers "which configurations remain strongest under
  the explicitly requested validity method?" It must not be treated as the same
  concept as raw rank.
- Q-values only correct the documented family of eligible tests. They do not
  make ineligible rows significant.
- DSR is only meaningful for Sharpe-based ranking with sufficient history. When
  that requirement is not met, the artifacts explicitly say so.
- No composite validity score is produced. Every persisted field maps to one
  named method or one named warning.
