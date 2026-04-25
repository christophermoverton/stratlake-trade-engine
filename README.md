# StratLake Trade Engine

StratLake Trade Engine is a deterministic research platform for running
systematic strategy and portfolio experiments on curated market data.

It is built for research review, not ingestion or live trading. The engine
consumes validated feature datasets, runs backtests with explicit execution
assumptions, applies layered validation, and writes auditable artifacts for
later comparison, portfolio construction, and registry-backed reuse.

## Notebook Execution API

Milestone 23 exposes the repository's deterministic execution workflows through
stable Python imports for notebooks and scripts. Notebook users can run
strategy, alpha, alpha-evaluation, portfolio, pipeline, campaign, validation,
and benchmark-pack workflows without shelling out, then inspect the same
artifact-first contracts used by the CLI.

```python
from src.execution import run_strategy

result = run_strategy("momentum_v1", start="2022-01-01", end="2023-01-01")
result.notebook_summary()
result.load_metrics_json()
result.load_manifest()
```

Use notebooks for exploration, inspection, comparisons, and interactive review.
Use the CLI for automation, CI, milestone validation, release bundles, and
operational runs.

Start with:

* [docs/notebook_execution_api.md](docs/notebook_execution_api.md)
* [docs/examples/notebook_execution_api_examples.md](docs/examples/notebook_execution_api_examples.md)
* [docs/examples/notebook_execution_api_examples.py](docs/examples/notebook_execution_api_examples.py)
* [docs/examples/ml_cross_sectional_xgb_2026_q1_notebook.ipynb](docs/examples/ml_cross_sectional_xgb_2026_q1_notebook.ipynb)

## Milestone 25 Summary

Milestone 25 extends the regime-aware evaluation layer with deterministic
regime calibration profiles and stability controls. The calibration layer works
on top of existing Milestone 24 regime outputs, adds profile-driven smoothing
and minimum-duration gating, computes stability metrics, and persists
calibration artifacts without redefining the taxonomy.

Milestone 25 also adds a deterministic Gaussian Mixture Model (GMM)
regime-shift classifier that complements taxonomy labels with posterior
probabilities, confidence, entropy, and cluster-transition events on canonical
regime feature columns.

Milestone 25 Issue 4 adds a deterministic regime-aware policy optimization
layer. It consumes the existing taxonomy, calibration profile and stability
metadata, optional ML confidence, and emits config-driven policy decisions plus
adaptive-vs-static comparison artifacts for strategy, alpha, and portfolio
research surfaces.

Start with:

* [docs/regime_calibration.md](docs/regime_calibration.md)
* [docs/regime_gmm_classifier.md](docs/regime_gmm_classifier.md)
* [docs/regime_policy_optimization.md](docs/regime_policy_optimization.md)
* [docs/examples/full_year_regime_calibration_case_study.md](docs/examples/full_year_regime_calibration_case_study.md)

## Milestone 24 Summary

Milestone 24 introduces a full regime-aware research interpretation surface
across strategy, alpha, and portfolio workflows. It adds deterministic regime
classification and taxonomy artifacts, conditional evaluation by regime,
transition and stress-state analysis, attribution and fragility detection,
cross-run regime comparison surfaces, notebook inspection helpers, and both
canonical and real-data Q1 2026 case studies.

Start with:

* [docs/regime_taxonomy.md](docs/regime_taxonomy.md)
* [docs/regime_conditional_evaluation.md](docs/regime_conditional_evaluation.md)
* [docs/regime_transition_analysis.md](docs/regime_transition_analysis.md)
* [docs/regime_attribution_and_comparison.md](docs/regime_attribution_and_comparison.md)
* [docs/examples/regime_notebook_review_examples.md](docs/examples/regime_notebook_review_examples.md)
* [docs/examples/regime_aware_case_study.md](docs/examples/regime_aware_case_study.md)
* [docs/examples/real_q1_2026_regime_aware_case_study.md](docs/examples/real_q1_2026_regime_aware_case_study.md)
* [docs/examples/real_q1_2026_regime_aware_case_study_report.md](docs/examples/real_q1_2026_regime_aware_case_study_report.md)

## Milestone 23 Summary

Milestone 23 makes the repository's deterministic execution workflows
notebook-addressable while preserving CLI parity for release, CI, and
automation paths. It adds a stable `src.execution` Python API for running and
inspecting strategy, alpha, alpha-evaluation, portfolio, pipeline, campaign,
validation, and benchmark-pack workflows from notebooks and scripts.

The repository now includes:

* shared execution-layer adapters under `src/execution/` so notebook and CLI
  entrypoints use the same underlying workflow behavior
* notebook-friendly `ExecutionResult` helpers for inspecting metrics,
  manifests, summaries, comparison reports, named outputs, and artifact paths
* importable execution APIs for strategy comparison, alpha evaluation, full
  alpha runs, portfolio construction, pipeline execution, campaign
  orchestration, docs/path linting, deterministic rerun validation, milestone
  validation bundles, and benchmark packs
* CLI/API parity tests that protect shared artifact contracts and reduce drift
  between process-oriented CLI behavior and notebook-oriented Python behavior
* canonical notebook execution documentation and import-safe examples for
  copying into interactive analysis workflows
* a Q1 2026 cross-sectional XGBoost notebook case study that demonstrates the
  alpha-to-portfolio workflow with notebook inspection helpers

Common Milestone 23 Python entrypoints:

```python
from src.execution import (
    compare_strategies,
    run_alpha,
    run_alpha_evaluation,
    run_benchmark_pack,
    run_docs_path_lint,
    run_deterministic_rerun_validation,
    run_milestone_validation,
    run_pipeline,
    run_portfolio,
    run_research_campaign,
    run_strategy,
)
```

Common Milestone 23 validation commands:

```powershell
python -m src.cli.run_docs_path_lint --output artifacts/qa/docs_path_lint.json
python -m src.cli.run_deterministic_rerun_validation --output artifacts/qa/deterministic_rerun.json
python -m src.cli.run_milestone_validation --bundle-dir artifacts/qa/milestone_validation_bundle --include-full-pytest
```

Start with:

* [docs/notebook_execution_api.md](docs/notebook_execution_api.md)
* [docs/examples/notebook_execution_api_examples.md](docs/examples/notebook_execution_api_examples.md)
* [docs/examples/notebook_execution_api_examples.py](docs/examples/notebook_execution_api_examples.py)
* [docs/examples/ml_cross_sectional_xgb_2026_q1_notebook.ipynb](docs/examples/ml_cross_sectional_xgb_2026_q1_notebook.ipynb)
* [docs/getting_started.md](docs/getting_started.md)

## Milestone 22 Summary

Milestone 22 extends the Milestone 21 signal, strategy, portfolio, and pipeline
foundation into a stricter release-ready research surface. It adds
deterministic validation bundles, typed-signal hardening, richer strategy and
portfolio composition, statistical-validity controls, side-aware execution
realism, and reproducible benchmark packs for scale and rerun review.

The repository now includes:

* milestone-grade validation and release traceability:
  `.github/workflows/milestone_validation.yml`,
  `python -m src.cli.run_docs_path_lint`,
  `python -m src.cli.run_deterministic_rerun_validation`, and
  `python -m src.cli.run_milestone_validation`
* docs/path linting for release-facing docs, examples, and README links, with
  machine-readable reports under `artifacts/qa/`
* deterministic rerun validation for canonical pipeline examples:
  baseline reference, robustness scenario sweep, and declarative builder
* standardized validation bundles under
  `artifacts/qa/milestone_validation_bundle/`, including docs/path lint,
  deterministic rerun, ruff, milestone pytest slice, and optional full pytest
* stricter typed-signal enforcement across canonical workflows, including
  explicit signal semantics, transformation metadata, and fail-fast validation
* portfolio-template registry support plus sweep-to-portfolio builder parity,
  allowing top-ranked robustness outputs to flow into portfolio construction
  without hand-wired artifact lookup
* statistical-validity controls for extended robustness sweeps, including
  split-based raw p-values, FDR-style q-values, Deflated Sharpe Ratio support,
  validity-aware ranking fields, and explicit non-applicability metadata
* side-aware execution realism for strategy and portfolio research, including
  long/short cost splits, short borrow costs, short-capacity limits,
  short-availability policies, and binding-event accounting
* expanded strategy archetypes with `volatility_regime_momentum` and
  `weighted_cross_section_ensemble`, both integrated with registry,
  PipelineBuilder, docs, examples, and deterministic validation coverage
* canonical benchmark-pack infrastructure through
  `python -m src.cli.run_benchmark_pack`, with deterministic batching,
  checkpoint/resume state, benchmark matrices, artifact inventories, and
  cross-rerun inventory comparison
* committed M22 example surfaces for benchmark-pack reproducibility and
  regime/ensemble strategy showcase workflows

Common Milestone 22 commands:

```powershell
python -m src.cli.run_docs_path_lint --output artifacts/qa/docs_path_lint.json
python -m src.cli.run_deterministic_rerun_validation --output artifacts/qa/deterministic_rerun.json
python -m src.cli.run_milestone_validation --bundle-dir artifacts/qa/milestone_validation_bundle --include-full-pytest
python -m src.cli.run_benchmark_pack --config configs/benchmark_packs/m22_scale_repro.yml
python docs/examples/scale_repro_benchmark_pack.py
```

Start with:

* [docs/milestone_22_merge_readiness.md](docs/milestone_22_merge_readiness.md)
* [docs/milestone_22_benchmark_packs.md](docs/milestone_22_benchmark_packs.md)
* [docs/signal_semantics.md](docs/signal_semantics.md)
* [docs/extended_robustness_sweeps.md](docs/extended_robustness_sweeps.md)
* [docs/execution_model.md](docs/execution_model.md)
* [docs/strategy_library.md](docs/strategy_library.md)
* [docs/pipeline_builder.md](docs/pipeline_builder.md)
* [docs/examples/scale_repro_benchmark_pack.md](docs/examples/scale_repro_benchmark_pack.md)
* [docs/examples/pipelines/regime_ensemble_showcase/README.md](docs/examples/pipelines/regime_ensemble_showcase/README.md)

## Milestone 19 Summary

Milestone 19 builds on the sweep expansion and campaign orchestration layers
already in place and documents the full scenario-sweep workflow end to end.

The repository now includes:

* deterministic scenario sweeps from one campaign spec through `matrix` and
  `include` expansion
* sweep guardrails through `max_scenarios`, `max_values_per_axis`, and hard-cap
  enforcement
* orchestration preflight sizing output in `expansion_preflight.json`
* orchestration-level scenario catalog and comparison outputs via
  `scenario_catalog.json`, `scenario_matrix.csv`, and `scenario_matrix.json`
* deterministic cross-scenario ranking with explicit metric-priority tie-breaks
* scenario-partitioned campaign artifacts under
  `artifacts/research_campaigns/<orchestration_run_id>/scenarios/<scenario_id>/`
* documented scenario checkpoint and reuse semantics, including scenario-aware
  provenance for deterministic rerun behavior
* a committed real-world sweep case study with reproducible artifacts and
  leaderboard interpretation guidance

## Milestone 20 Summary

Milestone 20 adds a deterministic YAML-driven pipeline runner and a lightweight
data platform orchestration layer for composing existing CLI modules into one
artifact-driven workflow.

The repository now includes:

* YAML pipeline execution through `python -m src.cli.run_pipeline --config <spec>`
* deterministic pipeline artifacts under `artifacts/pipelines/<pipeline_run_id>/`
* pipeline registry tracking via `artifacts/pipelines/registry.jsonl`
* explicit step dependency ordering, state passing, manifest, lineage, and
  timing metrics for reproducible pipeline workflows
* a scenario-matrix pipeline case study in
  `docs/examples/pipeline_scenario_matrix_case_study.md`
* a committed pipeline spec at `configs/pipelines/scenario_matrix_pipeline.yml`

Run a pipeline with:

```powershell
python -m src.cli.run_pipeline --config configs/pipelines/scenario_matrix_pipeline.yml
```

For details, see:

* [docs/milestone_20_data_platform_orchestration.md](docs/milestone_20_data_platform_orchestration.md)
* [docs/examples/pipeline_scenario_matrix_case_study.md](docs/examples/pipeline_scenario_matrix_case_study.md)

## Milestone 21 Summary

Milestone 21 adds canonical composition layers for signal semantics, position
constructors, strategy archetypes, asymmetry-aware controls, extended
robustness sweeps, and declarative pipeline authoring.

The repository now includes:

* strategy archetype definitions and validation integrated with typed signal
  semantics and constructor compatibility
* registry-driven position-constructor selection and directional asymmetry
  validation
* declarative pipeline generation through `python -m src.cli.build_pipeline`
* canonical reference pipeline library under `docs/examples/pipelines/` for
  end-to-end composition patterns

Start with:

* [docs/signal_semantics.md](docs/signal_semantics.md)
* [docs/strategy_library.md](docs/strategy_library.md)
* [docs/extended_robustness_sweeps.md](docs/extended_robustness_sweeps.md)
* [docs/pipeline_builder.md](docs/pipeline_builder.md)
* [docs/examples/pipelines/README.md](docs/examples/pipelines/README.md)

## Milestone 18 Summary

Milestone 18 keeps the campaign-orchestration and resume/reuse layers
introduced in Milestones 16 and 17, and adds deterministic milestone-review
artifacts derived from completed campaign flows. StratLake now supports a
deterministic path from evaluated alpha sleeves to candidate decisions,
redundancy control, governed allocation, candidate-driven portfolio
construction, and candidate-level review outputs. The repository now supports:

* alpha model registration through a deterministic `BaseAlphaModel` interface
* deterministic alpha training and prediction helpers with explicit half-open
  time windows
* time-aware alpha split utilities for fixed and rolling train/predict windows
* cross-sectional helpers for same-timestamp alpha inspection
* deterministic alpha evaluation with forward-return alignment before signal
  mapping
* per-period and aggregate alpha metrics including IC, Rank IC, coverage, and
  leaderboard-ready summaries
* registry-backed alpha-evaluation persistence, comparison, and reproducible
  artifact manifests
* governed candidate selection with deterministic ranking, eligibility gates,
  redundancy filtering, and allocation constraints
* candidate-selection artifact contracts under
  `artifacts/candidate_selection/<run_id>/` with manifest-backed reproducibility
* candidate-driven portfolio construction through
  `python -m src.cli.run_portfolio --from-candidate-selection ...`
* candidate review and explainability outputs under
  `artifacts/reviews/candidate_selection_<run_id>/`
* built-in alpha catalog/config support through `configs/alphas.yml` plus
  `python -m src.cli.run_alpha`
* real-data Q1 2026 ML alpha case studies on `features_daily`, including
  XGBoost, LightGBM, and Elastic Net baselines plus campaign-backed comparison,
  candidate selection, portfolio construction, and review
* default full alpha runs that persist evaluation artifacts, mapped signals,
  sleeve return streams, sleeve metrics, and `alpha_run_scaffold.json`
* continuous-signal backtesting where finite numeric exposures are interpreted
  literally after lagged execution
* alpha-sleeve portfolio integration through `artifact_type: alpha_sleeve`
  components in portfolio configs
* centralized portfolio optimization with `equal_weight`, `max_sharpe`, and
  `risk_parity`
* operational volatility targeting in portfolio workflows, separate from
  diagnostic risk summaries
* unified review workflows for ranking completed alpha, strategy, and
  portfolio runs together, including optional alpha sleeve and linked portfolio
  context in review outputs
* deterministic return simulation, robustness analysis, artifact manifests,
  and registry-backed reuse

Start with:

* [docs/alpha_workflow.md](docs/alpha_workflow.md)
* [docs/alpha_evaluation_workflow.md](docs/alpha_evaluation_workflow.md)
* [docs/milestone_11_portfolio_workflow.md](docs/milestone_11_portfolio_workflow.md)
* [docs/milestone_13_research_review_workflow.md](docs/milestone_13_research_review_workflow.md)
* [docs/milestone_15_candidate_selection_issue_1.md](docs/milestone_15_candidate_selection_issue_1.md)
* [docs/backfilled_2026_q1_research_workflow.md](docs/backfilled_2026_q1_research_workflow.md)
* [docs/backfilled_2026_q1_alpha_workflow.md](docs/backfilled_2026_q1_alpha_workflow.md)
* [docs/ml_cross_sectional_xgb_2026_q1.md](docs/ml_cross_sectional_xgb_2026_q1.md)
* [docs/examples/ml_cross_sectional_lgbm_2026_q1_candidate_driven_workflow.md](docs/examples/ml_cross_sectional_lgbm_2026_q1_candidate_driven_workflow.md)
* [docs/examples/real_alpha_workflow.md](docs/examples/real_alpha_workflow.md)
* [docs/examples/candidate_selection_portfolio_case_study.py](docs/examples/candidate_selection_portfolio_case_study.py)
* [docs/examples/real_world_candidate_selection_portfolio_case_study.md](docs/examples/real_world_candidate_selection_portfolio_case_study.md)
* [docs/examples/real_world_campaign_case_study.md](docs/examples/real_world_campaign_case_study.md)
* [docs/milestone_19_scenario_sweeps.md](docs/milestone_19_scenario_sweeps.md)
* [docs/milestone_20_data_platform_orchestration.md](docs/milestone_20_data_platform_orchestration.md)
* [docs/examples/pipeline_scenario_matrix_case_study.md](docs/examples/pipeline_scenario_matrix_case_study.md)
* [docs/examples/real_world_scenario_sweep_case_study.md](docs/examples/real_world_scenario_sweep_case_study.md)
* [docs/examples/real_world_scenario_sweep_case_study.py](docs/examples/real_world_scenario_sweep_case_study.py)
* [docs/examples/real_data_scenario_sweep_case_study.md](docs/examples/real_data_scenario_sweep_case_study.md)
* [docs/examples/real_data_scenario_sweep_case_study.py](docs/examples/real_data_scenario_sweep_case_study.py)
* [docs/examples/real_data_full_scenario_sweep_case_study.md](docs/examples/real_data_full_scenario_sweep_case_study.md)
* [docs/examples/real_data_full_scenario_sweep_case_study.py](docs/examples/real_data_full_scenario_sweep_case_study.py)
* [docs/milestone_17_resume_workflow.md](docs/milestone_17_resume_workflow.md)
* [docs/milestone_18_milestone_review_workflow.md](docs/milestone_18_milestone_review_workflow.md)
* [docs/examples/real_world_resume_workflow_case_study.md](docs/examples/real_world_resume_workflow_case_study.md)
* [docs/examples/milestone_11_5_alpha_portfolio_workflow.md](docs/examples/milestone_11_5_alpha_portfolio_workflow.md)
* [docs/examples/milestone_13_review_promotion_workflow.md](docs/examples/milestone_13_review_promotion_workflow.md)
* [docs/milestone_reporting_artifacts.md](docs/milestone_reporting_artifacts.md)

## Overview

StratLake helps answer three practical research questions:

* Did the strategy or portfolio respect temporal integrity and deterministic
  execution assumptions?
* How do optimizer choices, execution frictions, and risk diagnostics change
  the result?
* Can the output be trusted across metrics, QA summaries, manifests, and
  registry entries?

The repository currently supports:

* deterministic feature-dataset driven strategy and alpha research
* alpha model registration, training, prediction, and cross-sectional review
* single-run and walk-forward strategy evaluation
* continuous-signal or discrete-signal backtesting with lagged execution
* deterministic robustness analysis for strategy parameter sweeps
* execution realism through lag, transaction costs, fixed fees, and slippage
* portfolio construction from completed strategy artifacts or alpha-derived
  return sleeves
* centralized portfolio optimization, validation, risk summaries, and
  operational volatility targeting
* deterministic return simulation for strategy or portfolio outputs
* strict-mode enforcement across strategy and portfolio CLIs
* deterministic promotion gates for alpha, strategy, and portfolio review
* candidate-selection governance with deterministic eligibility, redundancy,
  and allocation stages
* candidate-driven portfolio wiring from approved candidate sets
* candidate review outputs that explain selection and contribution decisions
* manifest-backed unified research review artifacts with deterministic review summaries
* unified runtime configuration with auditable persisted settings
* unified research campaign configuration for shared dataset, target, comparison,
  candidate-selection, portfolio, review, and output-path settings
* campaign-level orchestration through
  `python -m src.cli.run_research_campaign --config ...`
* deterministic scenario expansion from one sweep-enabled campaign spec into
  stable concrete scenario IDs plus effective config snapshots for tooling and
  review
* orchestration-level scenario matrix outputs (`scenario_matrix.csv` and
  `scenario_matrix.json`) with deterministic ranking and sweep-key columns
* campaign preflight validation with persisted `preflight_summary.json` reports
  before expensive research execution starts
* operator-facing `reuse_policy` controls for explicit checkpoint reuse,
  forced reruns, reuse disablement, and downstream invalidation
* persisted `partial`, `failed`, `pending`, `completed`, and `reused` stage
  states with resumable checkpoint metadata
* stitched campaign retry/resume metadata through `stage_execution`,
  `execution_metadata`, `retry_stage_names`, `partial_stage_names`,
  `reused_stage_names`, and `resumable_stage_names`
* stitched campaign `manifest.json` and `summary.json` artifacts for
  automation and multi-stage auditability
* auto-generated milestone review packs under
  `artifacts/research_campaigns/<campaign_run_id>/milestone_report/`
  containing `summary.json`, `decision_log.json`, `manifest.json`, and
  optional `report.md`
* milestone decision logs with relative evidence links back to campaign,
  candidate-review, and research-review artifacts
* follow-on milestone generation from an existing saved campaign via
  `python -m src.cli.generate_milestone_report --campaign-artifact-path ...`
* a committed Milestone 17 resume case study with partial, resumed, and stable
  reused campaign snapshots
* deterministic artifacts, manifests, and registry-backed reuse

Feature naming note:

* canonical daily SMA features use underscore window names such as `feature_sma_20` and `feature_sma_50`
* legacy config aliases such as `feature_sma20` and `feature_sma50` remain accepted by alpha tooling for backward compatibility

## Architecture

At a high level, StratLake is an artifact-driven research pipeline:

```text
features
  ->
alpha
  ->
predict
  ->
align
  ->
validate
  ->
evaluate
  ->
aggregate
  ->
persist
  ->
register
  ->
compare
  ->
candidate selection
  ->
allocation
  ->
portfolio
  ->
risk
  ->
artifacts
```

In practice, the alpha and strategy layers can meet at different points in that
pipeline:

* traditional strategies can emit `signal` directly from a validated feature
  dataset
* alpha workflows can train on the same canonical frame, emit
  `prediction_score`, inspect cross-sections, then map those predictions into
  backtestable exposures
* completed strategy artifacts or alpha-derived sleeves can then flow into the
  portfolio layer for optimization, execution accounting, risk review, and
  persistence

This design keeps research and portfolio workflows deterministic, auditable,
and easy to review from saved files rather than only in-memory results.

## Alpha Modeling

The alpha layer lives under `src/research/alpha/` and provides a deterministic
interface for ML-style models that operate on canonical research frames sorted
by `(symbol, ts_utc)`.

### Alpha model interface

`BaseAlphaModel` enforces:

* stable `fit(df)` and `predict(df)` behavior
* no mutation of the caller's input frame
* prediction output aligned exactly to `df.index`
* numeric prediction scores with deterministic repeatability checks

Models are registered through `register_alpha_model(...)` and instantiated by
name through the alpha registry.

### Training workflow

`train_alpha_model(...)`:

* validates the canonical input contract
* resolves feature columns explicitly or from `feature_*` columns
* applies half-open training bounds `[train_start, train_end)`
* returns a `TrainedAlphaModel` with the fitted model plus metadata about the
  training slice

### Prediction workflow

`predict_alpha_model(...)`:

* validates the trained model contract
* applies half-open prediction bounds `[predict_start, predict_end)`
* preserves structural columns such as `symbol`, `ts_utc`, and optional
  `timeframe`
* returns a deterministic prediction frame with `prediction_score`

See [docs/alpha_workflow.md](docs/alpha_workflow.md) for the full workflow.

## Alpha Evaluation (Milestone 12)

Milestone 12 adds a deterministic alpha-evaluation layer for measuring whether
predictions have cross-sectional forecasting power before they are mapped into
signals, backtests, or portfolios.

The workflow is:

```text
Alpha -> Predict -> Align -> Validate -> Evaluate -> Aggregate -> Persist -> Register -> Compare
```

What it provides:

* forward-return alignment from either prices or realized returns
* cross-sectional IC and Rank IC evaluation per timestamp
* aggregated summary metrics including `mean_ic`, `ic_ir`, `mean_rank_ic`,
  and `rank_ic_ir`
* persisted alpha-evaluation artifacts under `artifacts/alpha/<run_id>/`
* registry-backed alpha leaderboards under `artifacts/alpha_comparisons/`

Start here:

* [docs/alpha_evaluation_workflow.md](docs/alpha_evaluation_workflow.md)
* [docs/examples/alpha_evaluation_end_to_end.py](docs/examples/alpha_evaluation_end_to_end.py)

Quick start:

```powershell
python docs/examples/alpha_evaluation_end_to_end.py
python -m src.cli.run_alpha --alpha-name cs_linear_ret_1d --mode evaluate --start 2025-01-01 --end 2025-03-01
python -m src.cli.run_alpha --alpha-name rank_composite_momentum --start 2025-01-01 --end 2025-03-01 --signal-policy top_bottom_quantile --signal-quantile 0.2
python -m src.cli.run_alpha --config configs/alphas_2026_q1.yml --alpha-name ml_cross_sectional_xgb_2026_q1
python -m src.cli.run_alpha_evaluation --alpha-model your_model --model-class path/to/model.py:YourModel --dataset features_daily --target-column target_ret_1d --price-column close
python -m src.cli.compare_alpha --from-registry
```

Notes:

* `python -m src.cli.run_alpha` is the first-class entrypoint for named built-in alpha configs from `configs/alphas.yml`
* the Q1 2026 ML catalog lives in `configs/alphas_2026_q1.yml` and includes `ml_cross_sectional_xgb_2026_q1`, `ml_cross_sectional_lgbm_2026_q1`, and `ml_cross_sectional_elastic_net_2026_q1`
* `--mode evaluate` runs only the evaluation stage; the default `full` mode also writes `signals.parquet`, sleeve artifacts, and `alpha_run_scaffold.json`
* pass exactly one of `--price-column` or `--realized-return-column`
* `--model-class` accepts either `module:Class` or `path.py:Class`
* built-in XGBoost, LightGBM, and Elastic Net configs require their underlying packages in the active environment
* the end-to-end example writes reproducible outputs under
  `docs/examples/output/alpha_evaluation_end_to_end/`

`python -m src.cli.run_alpha --mode full` is the merge-review baseline for
built-in configs. It resolves one named alpha from `configs/alphas.yml`,
evaluates forecast quality, maps predictions into explicit signals, generates a
deterministic sleeve return stream, and leaves one scaffold artifact that keeps
the downstream alpha-to-sleeve flow auditable.

## Candidate Selection And Governed Allocation (Milestone 15)

Milestone 15 introduces a governed layer between alpha comparison and portfolio
construction:

```text
Alpha -> Predict -> Align -> Validate -> Evaluate -> Aggregate -> Persist -> Register -> Compare -> Candidate Selection -> Allocation -> Portfolio
```

What it provides:

* deterministic candidate-universe loading from alpha-evaluation registry data
* explicit eligibility thresholds on forecast-quality and history metrics
* redundancy pruning via pairwise-correlation checks with overlap controls
* governed allocation with deterministic constraints and audited weight outputs
* candidate-selection manifests and registry-backed candidate review artifacts
* candidate-driven portfolio construction from approved candidates only

Start here:

* [docs/milestone_15_candidate_selection_issue_1.md](docs/milestone_15_candidate_selection_issue_1.md)
* [docs/examples/ml_cross_sectional_lgbm_2026_q1_candidate_driven_workflow.md](docs/examples/ml_cross_sectional_lgbm_2026_q1_candidate_driven_workflow.md)
* [docs/examples/real_world_candidate_selection_portfolio_case_study.md](docs/examples/real_world_candidate_selection_portfolio_case_study.md)

Quick start:

```powershell
python -m src.cli.run_candidate_selection --config configs/candidate_selection.yml
python -m src.cli.run_portfolio --from-candidate-selection artifacts/candidate_selection/<candidate_selection_run_id> --portfolio-name your_candidate_driven_portfolio --timeframe 1D
python docs/examples/candidate_selection_portfolio_case_study.py
python docs/examples/real_world_candidate_selection_portfolio_case_study.py
```

## Research Campaign Orchestration (Milestone 16)

Milestone 16 adds one campaign-level entrypoint above the existing CLIs:

```text
Preflight -> Research -> Comparison -> Candidate Selection -> Portfolio -> Review
```

What it provides:

* one normalized campaign config for shared dataset, targets, time windows, and
  output roots
* fail-fast campaign preflight before multi-stage execution begins
* deterministic stage ordering across alpha, strategy, comparison,
  candidate-selection, portfolio, candidate review, and unified review
* one stitched campaign artifact directory under
  `artifacts/research_campaigns/<campaign_run_id>/`
* canonical campaign `checkpoint.json` state for resumable orchestration and
  stage-level reuse tracking
* campaign-level `summary.json` and `manifest.json` files for automation and
  auditability
* per-stage audit metadata in stitched outputs covering resume, retry, reuse,
  skip, failure, checkpoint state, and fingerprint inputs/details
* a Milestone 17 resume case study showing `partial -> completed -> reused`
  stage transitions on one stable campaign root

Start here:

* [docs/milestone_16_campaign_workflow.md](docs/milestone_16_campaign_workflow.md)
* [docs/milestone_17_resume_workflow.md](docs/milestone_17_resume_workflow.md)
* [docs/research_campaign_configuration.md](docs/research_campaign_configuration.md)
* [docs/milestone_16_merge_readiness.md](docs/milestone_16_merge_readiness.md)
* [docs/examples/milestone_16_campaign_workflow.md](docs/examples/milestone_16_campaign_workflow.md)
* [docs/examples/real_world_campaign_case_study.md](docs/examples/real_world_campaign_case_study.md)
* [docs/examples/milestone_17_resume_workflow.md](docs/examples/milestone_17_resume_workflow.md)
* [docs/examples/real_world_resume_workflow_case_study.md](docs/examples/real_world_resume_workflow_case_study.md)

Quick start:

```powershell
python -m src.cli.run_research_campaign
python -m src.cli.run_research_campaign --config docs/examples/data/milestone_16_campaign_configs/full_campaign.yml
python docs/examples/real_world_campaign_case_study.py
python docs/examples/real_world_resume_workflow_case_study.py
python docs/examples/regime_aware_case_study.py
python docs/examples/real_q1_2026_regime_aware_case_study.py
python docs/examples/full_year_regime_calibration_case_study.py
```

## Cross-Sectional Utilities

Alpha workflows often need to inspect one same-timestamp asset slice before
mapping predictions into signals or downstream portfolios.

The cross-sectional helpers in `src/research/alpha/cross_section.py` provide:

* `list_cross_section_timestamps(...)` for deterministic timestamp discovery
* `get_cross_section(...)` for one timestamp-specific multi-symbol slice
* `iter_cross_sections(...)` for ordered `(timestamp, frame)` iteration

These utilities validate sorted, duplicate-free `(symbol, ts_utc)` inputs so
cross-sectional review stays deterministic and auditable.

## Research Validity, Runtime, And Strict Mode

StratLake uses shared runtime configuration and strict-mode enforcement across
strategy and portfolio workflows.

### Research-validity layers

* Temporal integrity checks validate ordering, uniqueness, signal alignment,
  and lagged execution assumptions before trusting results.
* Consistency validation checks saved artifacts against each other after
  persistence.
* Sanity checks flag suspicious return paths, implausible smoothness, extreme
  annualized metrics, and other outliers.
* Portfolio validation enforces weight-sum, exposure, leverage, sleeve-weight,
  and compounding constraints.

### Execution realism

* `execution_delay` controls how many bars signals are lagged before execution.
* `transaction_cost_bps`, `fixed_fee`, and `slippage_bps` apply deterministic
  execution drag to strategy and portfolio runs.
* Slippage models currently include `constant`, `turnover_scaled`, and
  `volatility_scaled`.
* Strategy and portfolio runs persist gross returns, net returns, turnover, and
  total execution friction in their artifacts.

### Strict mode

* `--strict` promotes flagged validation and sanity issues into fail-fast CLI
  errors.
* In strict mode, runs that fail pre-persistence validation do not write
  artifacts or registry entries.
* Non-strict mode still records warnings in metrics and QA artifacts so review
  remains auditable.

### Runtime configuration

* One normalized `runtime` contract resolves execution, sanity,
  portfolio-validation, risk, and strict-mode settings.
* Precedence is deterministic: repository defaults < config < CLI.
* Effective runtime settings are persisted with completed runs for auditability.

### Promotion gates

* Optional `promotion_gates` configs can be attached to alpha, strategy, and
  portfolio runs.
* Completed runs persist `promotion_gates.json` plus a compact promotion summary
  in the manifest and registry.
* The unified research review surface now exposes each run's promotion status
  alongside leaderboard metrics.
* Unified review runs persist `leaderboard.csv`, `review_summary.json`,
  optional `promotion_gates.json`, and `manifest.json` under one review id.

See:

* [docs/research_validity_framework.md](docs/research_validity_framework.md)
* [docs/execution_model.md](docs/execution_model.md)
* [docs/runtime_configuration.md](docs/runtime_configuration.md)
* [docs/research_campaign_configuration.md](docs/research_campaign_configuration.md)
* [docs/milestone_16_campaign_workflow.md](docs/milestone_16_campaign_workflow.md)
* [docs/strict_mode.md](docs/strict_mode.md)
* [docs/research_integrity_and_qa.md](docs/research_integrity_and_qa.md)

## Quick Start

### 1. Set up the environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
pip install -e ".[dev]"
Copy-Item .env.example .env
```

Set `MARKETLAKE_ROOT` in `.env` to the curated data root produced by the
upstream ingestion repository.

### 2. Build or verify features

```powershell
python -m cli.build_features --timeframe 1D --start 2022-01-01 --end 2024-01-01 --tickers configs/tickers_50.txt
```

### 3. Run a strategy

Single run:

```powershell
python -m src.cli.run_strategy --strategy momentum_v1
```

Walk-forward evaluation:

```powershell
python -m src.cli.run_strategy --strategy momentum_v1 --evaluation
```

Robustness analysis:

```powershell
python -m src.cli.run_strategy --robustness
```

Simulation-enabled single run:

```powershell
python -m src.cli.run_strategy --strategy momentum_v1 --simulation path/to/simulation.yml
```

### 4. Run the real alpha workflow example

```powershell
python docs/examples/real_alpha_workflow.py
```

This example demonstrates config-driven built-in alpha selection, deterministic
prediction and evaluation on `features_daily`, explicit signal mapping, alpha
sleeve generation, downstream portfolio integration, and unified review
artifacts.

### 4b. Run the Milestone 12 alpha-evaluation example

```powershell
python docs/examples/alpha_evaluation_end_to_end.py
```

This example demonstrates deterministic prediction, forward-return alignment,
IC and Rank IC evaluation, artifact persistence, registry entry creation, and
leaderboard generation.

### 4c. Run the Milestone 13 review-and-promotion example

```powershell
python docs/examples/milestone_13_review_promotion_workflow.py
```

This example demonstrates completed alpha, strategy, and portfolio artifacts
flowing into one registry-backed review output and one review-level promotion
decision.
The primary workflow guide lives in
[docs/milestone_13_research_review_workflow.md](docs/milestone_13_research_review_workflow.md).

### 4d. Run the Q1 2026 ML alpha case studies

```powershell
python -m src.cli.run_alpha --config configs/alphas_2026_q1.yml --alpha-name ml_cross_sectional_xgb_2026_q1
python -m src.cli.run_alpha --config configs/alphas_2026_q1.yml --alpha-name ml_cross_sectional_lgbm_2026_q1
python -m src.cli.run_alpha --config configs/alphas_2026_q1.yml --alpha-name ml_cross_sectional_elastic_net_2026_q1
python -m src.cli.run_alpha --config configs/alphas_2026_q1.yml --alpha-name rank_composite_momentum_2026_q1
python -m src.cli.compare_alpha --from-registry --dataset features_daily --timeframe 1D --evaluation-horizon 5 --mapping-name top_bottom_quantile_q20 --view combined
python -m src.cli.run_portfolio --portfolio-config configs/portfolios_alpha_2026_q1.yml --portfolio-name ml_cross_sectional_xgb_2026_q1_equal --timeframe 1D
python -m src.cli.compare_research --from-registry --run-types alpha_evaluation portfolio --dataset features_daily --timeframe 1D --alpha-name ml_cross_sectional_xgb_2026_q1 --portfolio-name ml_cross_sectional_xgb_2026_q1_equal --disable-plots
```

These case studies demonstrate built-in Q1 2026 ML alphas trained on
`2026-01-01 <= ts_utc < 2026-03-02`, predicted on
`2026-03-02 <= ts_utc < 2026-04-03`, mapped with
`top_bottom_quantile[q=0.2]`, and consumed downstream as
`artifact_type: alpha_sleeve` portfolio components. See
[docs/ml_cross_sectional_xgb_2026_q1.md](docs/ml_cross_sectional_xgb_2026_q1.md).

### 4e. Run the Milestone 15 candidate-selection and governed-allocation workflows

Deterministic synthetic example:

```powershell
python docs/examples/candidate_selection_portfolio_case_study.py
```

Real-data governed case study:

```powershell
python docs/examples/real_world_candidate_selection_portfolio_case_study.py
```

Production CLI path:

```powershell
python -m src.cli.run_candidate_selection --config configs/candidate_selection.yml
python -m src.cli.run_portfolio --from-candidate-selection artifacts/candidate_selection/<candidate_selection_run_id> --portfolio-name your_candidate_driven_portfolio --timeframe 1D
```

### 4f. Run the Milestone 16 real-world campaign case study

```powershell
python docs/examples/real_world_campaign_case_study.py
```

This case study runs the campaign orchestration flow end to end on repository
`features_daily` data using the Q1 2026 XGBoost, LightGBM, Elastic Net, and
rank-composite alphas, then writes stitched campaign, candidate-selection,
candidate-review, portfolio, and research-review summaries.

### 4g. Run the Milestone 17 resume workflow case study

```powershell
python docs/examples/real_world_resume_workflow_case_study.py
```

This case study intentionally interrupts the campaign during `comparison`,
then reruns on the same campaign root to demonstrate partial checkpoint state,
resume behavior, retry metadata, full-stage reuse, and committed stitched
summary/manifest/checkpoint snapshots.

### 5. Run a portfolio

Baseline registry-backed portfolio:

```powershell
python -m src.cli.run_portfolio --portfolio-config configs/portfolios.yml --portfolio-name momentum_meanrev_equal --from-registry --timeframe 1D
```

Optimizer-aware portfolio:

```powershell
python -m src.cli.run_portfolio --portfolio-config configs/portfolios.yml --portfolio-name momentum_meanrev_equal --from-registry --timeframe 1D --optimizer-method max_sharpe
```

Risk-aware and execution-aware portfolio:

```powershell
python -m src.cli.run_portfolio --portfolio-config configs/portfolios.yml --portfolio-name momentum_meanrev_equal --from-registry --timeframe 1D --risk-target-volatility 0.12 --risk-volatility-window 20 --execution-enabled --transaction-cost-bps 5 --fixed-fee 0.001 --slippage-bps 2 --slippage-model turnover_scaled
```

Operational volatility targeting:

```powershell
python -m src.cli.run_portfolio --portfolio-config configs/portfolios.yml --portfolio-name momentum_meanrev_targeted --from-registry --timeframe 1D
```

Walk-forward portfolio evaluation:

```powershell
python -m src.cli.run_portfolio --portfolio-config configs/portfolios.yml --portfolio-name strict_valid_builtin_pair --from-registry --evaluation configs/evaluation.yml --timeframe 1D --strict
```

The end-to-end Milestone 11 guide, including config snippets and output
interpretation, lives in
[docs/milestone_11_portfolio_workflow.md](docs/milestone_11_portfolio_workflow.md).

## Backtesting

The backtest layer accepts finite numeric `signal` values and interprets them
as literal lagged exposures.

That means:

* discrete signals such as `-1`, `0`, and `1` continue to work
* continuous signals such as alpha prediction scores are also supported
* return contribution scales proportionally with the executed exposure
* the runner does not clip or normalize exposure values implicitly

See [docs/backtest_runner.md](docs/backtest_runner.md).

## Portfolio Workflow

Portfolio research now includes:

* component selection from explicit run ids or the shared strategy registry
* aligned return-matrix construction with `intersection` alignment
* deterministic static allocation from `equal_weight`, `max_sharpe`, or
  `risk_parity`
* optional operational post-optimizer volatility targeting
* execution-friction accounting on turnover-driven rebalances
* portfolio metrics plus centralized risk summaries
* optional simulation artifacts for single-run portfolios
* walk-forward portfolio evaluation across deterministic splits
* manifest and registry rows that expose optimizer, execution, risk, and
  simulation metadata

### Volatility targeting

Portfolio config now supports:

```yaml
volatility_targeting:
  enabled: true
  target_volatility: 0.10
  lookback_periods: 20
  volatility_epsilon: 1e-8
```

Important distinction:

* `risk.target_volatility` is diagnostic and affects risk summaries
* top-level `volatility_targeting` is operational and scales base weights
  before execution accounting and portfolio evaluation

When enabled, the constructor computes a deterministic scaling factor from the
estimated pre-target portfolio volatility and applies it directly to the base
weights. When disabled, base optimizer weights flow through unchanged.

Start with:

* [docs/alpha_workflow.md](docs/alpha_workflow.md)
* [docs/milestone_11_portfolio_workflow.md](docs/milestone_11_portfolio_workflow.md)
* [docs/portfolio_configuration.md](docs/portfolio_configuration.md)
* [docs/portfolio_artifact_logging.md](docs/portfolio_artifact_logging.md)

## Example Workflow

The main end-to-end alpha example lives at
[docs/examples/real_alpha_workflow.py](docs/examples/real_alpha_workflow.py).

Run it with:

```powershell
python docs/examples/real_alpha_workflow.py
```

It demonstrates:

* config-driven selection of a built-in alpha from `configs/alphas.yml`
* deterministic alpha prediction, evaluation, and registry-backed artifacts
* explicit alpha-to-signal mapping
* sleeve generation under `artifacts/alpha/<run_id>/`
* portfolio construction from an `alpha_sleeve` component
* review artifact writing under `docs/examples/output/real_alpha_workflow/`

See the companion guide
[docs/examples/real_alpha_workflow.md](docs/examples/real_alpha_workflow.md).

For the latest real-data Milestone 16 campaign-backed case study on the Q1
2026 `features_daily` surface, see
[docs/examples/real_world_campaign_case_study.md](docs/examples/real_world_campaign_case_study.md).
For the Milestone 17 resume/retry/reuse case study, see
[docs/examples/real_world_resume_workflow_case_study.md](docs/examples/real_world_resume_workflow_case_study.md).
For the standalone XGBoost alpha workflow, see
[docs/ml_cross_sectional_xgb_2026_q1.md](docs/ml_cross_sectional_xgb_2026_q1.md).

The lower-level custom-model walkthrough remains available at
[docs/examples/milestone_11_5_alpha_portfolio_workflow.py](docs/examples/milestone_11_5_alpha_portfolio_workflow.py)
with notes in
[docs/examples/milestone_11_5_alpha_portfolio_workflow.md](docs/examples/milestone_11_5_alpha_portfolio_workflow.md).

The Milestone 12 alpha-evaluation example lives at
[docs/examples/alpha_evaluation_end_to_end.py](docs/examples/alpha_evaluation_end_to_end.py)
with workflow notes in
[docs/alpha_evaluation_workflow.md](docs/alpha_evaluation_workflow.md).

The Milestone 13 review-and-promotion example lives at
[docs/examples/milestone_13_review_promotion_workflow.py](docs/examples/milestone_13_review_promotion_workflow.py)
with workflow notes in
[docs/examples/milestone_13_review_promotion_workflow.md](docs/examples/milestone_13_review_promotion_workflow.md).
The primary workflow guide lives at
[docs/milestone_13_research_review_workflow.md](docs/milestone_13_research_review_workflow.md).
For the real-data 2026 Q1 backfill through gated-review path, see
[docs/backfilled_2026_q1_research_workflow.md](docs/backfilled_2026_q1_research_workflow.md).
For the real-data Q1 2026 alpha continuation on the same `features_daily`
surface, see
[docs/backfilled_2026_q1_alpha_workflow.md](docs/backfilled_2026_q1_alpha_workflow.md).

## Artifact Overview

### Strategy artifacts

Successful strategy runs write under `artifacts/strategies/<run_id>/`.

Core files:

* `config.json`
* `metrics.json`
* `signal_diagnostics.json`
* `qa_summary.json`
* `promotion_gates.json` when promotion gates are configured
* `equity_curve.csv`
* `signals.parquet`
* `manifest.json`

Optional Milestone 11 additions:

* `simulation/` for single-run simulation artifacts
* robustness runs under `artifacts/strategies/robustness/<run_id>/`

Walk-forward runs also include:

* `metrics_by_split.csv`
* `splits/<split_id>/...`

### Portfolio artifacts

Successful portfolio runs write under `artifacts/portfolios/<run_id>/`.

Core files:

* `config.json`
* `components.json`
* `weights.csv`
* `portfolio_returns.csv`
* `portfolio_equity_curve.csv`
* `metrics.json`
* `qa_summary.json`
* `promotion_gates.json` when promotion gates are configured
* `manifest.json`

Optional Milestone 11 additions:

* manifest metadata for optimizer, risk, and execution-friction summaries
* `simulation/` for single-run simulation artifacts

Walk-forward portfolio runs also include:

* `aggregate_metrics.json`
* `metrics_by_split.csv`
* `splits/<split_id>/...`

### Candidate-selection artifacts

Successful candidate-selection runs write under
`artifacts/candidate_selection/<run_id>/`.

Core files:

* `candidate_universe.csv`
* `eligibility_filter_results.csv`
* `correlation_matrix.csv`
* `selected_candidates.csv`
* `rejected_candidates.csv`
* `allocation_weights.csv`
* `selection_summary.json`
* `manifest.json`

Optional review outputs from candidate-selection review runs write under
`artifacts/reviews/candidate_selection_<run_id>/` and include:

* `candidate_decisions.csv`
* `candidate_summary.csv`
* `candidate_contributions.csv`
* `diversification_summary.json`
* `candidate_review_summary.json`
* `candidate_review_report.md`
* `manifest.json`

See:

* [docs/experiment_artifact_logging.md](docs/experiment_artifact_logging.md)
* [docs/portfolio_artifact_logging.md](docs/portfolio_artifact_logging.md)

### Unified review artifacts

Successful unified review runs write under `artifacts/reviews/<review_id>/`.

Core files:

* `leaderboard.csv`
* `review_summary.json`
* `manifest.json`
* `promotion_gates.json` when review-level promotion gates are configured

### Research campaign artifacts

Successful or failed campaign preflight runs write under
`artifacts/research_campaigns/<campaign_run_id>/`.

Core files:

* `campaign_config.json`
* `checkpoint.json`
* `preflight_summary.json`
* `manifest.json`
* `summary.json`

The stitched `manifest.json` and `summary.json` files expose one
`stage_execution` mapping plus per-stage `execution_metadata` blocks. Those
surfaces are intended to answer operational questions deterministically:

* whether a stage is resumable from the persisted checkpoint
* whether a retry occurred and which prior failure or partial state triggered it
* whether the stage was reused, skipped, or failed
* which checkpoint state/source/fingerprint inputs justified that outcome

### Alpha-evaluation artifacts

Successful alpha-evaluation runs write under `artifacts/alpha/<run_id>/`.

Core files:

* `predictions.parquet`
* `training_summary.json`
* `coefficients.json`
* `cross_section_diagnostics.json`
* `qa_summary.json`
* `alpha_metrics.json`
* `ic_timeseries.csv`
* `manifest.json`
* `promotion_gates.json` when alpha promotion gates are configured

Full built-in alpha runs from `python -m src.cli.run_alpha --mode full` also
write:

* `signals.parquet`
* `signal_mapping.json`
* `sleeve_returns.csv`
* `sleeve_equity_curve.csv`
* `sleeve_metrics.json`
* `alpha_run_scaffold.json`

The `manifest.json` and `training_summary.json` files now also carry model
metadata such as feature columns, model type, and persisted hyperparameters,
which is especially useful for built-in ML alpha case studies like
`ml_cross_sectional_xgb_2026_q1`.

`qa_summary.json` is the practical alpha QA surface. It records usable
timestamp coverage, cross-section breadth, post-warmup null rates, and, when
signals are present, tradability diagnostics such as implied turnover,
concentration, and net exposure. Example thresholds live in
`configs/alpha_promotion_gates.yml`.

## Documentation Map

Start here:

* [docs/getting_started.md](docs/getting_started.md)
* [docs/alpha_workflow.md](docs/alpha_workflow.md)
* [docs/alpha_evaluation_workflow.md](docs/alpha_evaluation_workflow.md)
* [docs/strategy_evaluation_workflow.md](docs/strategy_evaluation_workflow.md)
* [docs/milestone_11_portfolio_workflow.md](docs/milestone_11_portfolio_workflow.md)
* [docs/milestone_13_research_review_workflow.md](docs/milestone_13_research_review_workflow.md)
* [docs/milestone_15_candidate_selection_issue_1.md](docs/milestone_15_candidate_selection_issue_1.md)
* [docs/milestone_16_campaign_workflow.md](docs/milestone_16_campaign_workflow.md)
* [docs/milestone_17_resume_workflow.md](docs/milestone_17_resume_workflow.md)
* [docs/milestone_20_data_platform_orchestration.md](docs/milestone_20_data_platform_orchestration.md)
* [docs/signal_semantics.md](docs/signal_semantics.md)
* [docs/strategy_library.md](docs/strategy_library.md)
* [docs/extended_robustness_sweeps.md](docs/extended_robustness_sweeps.md)
* [docs/pipeline_builder.md](docs/pipeline_builder.md)
* [docs/milestone_22_benchmark_packs.md](docs/milestone_22_benchmark_packs.md)
* [docs/milestone_16_merge_readiness.md](docs/milestone_16_merge_readiness.md)
* [docs/milestone_22_merge_readiness.md](docs/milestone_22_merge_readiness.md)
* [docs/backfilled_2026_q1_research_workflow.md](docs/backfilled_2026_q1_research_workflow.md)
* [docs/backfilled_2026_q1_alpha_workflow.md](docs/backfilled_2026_q1_alpha_workflow.md)
* [docs/ml_cross_sectional_xgb_2026_q1.md](docs/ml_cross_sectional_xgb_2026_q1.md)

Portfolio references:

* [docs/portfolio_construction_workflow.md](docs/portfolio_construction_workflow.md)
* [docs/portfolio_configuration.md](docs/portfolio_configuration.md)
* [docs/portfolio_artifact_logging.md](docs/portfolio_artifact_logging.md)

Research integrity and execution references:

* [docs/research_validity_framework.md](docs/research_validity_framework.md)
* [docs/execution_model.md](docs/execution_model.md)
* [docs/runtime_configuration.md](docs/runtime_configuration.md)
* [docs/research_campaign_configuration.md](docs/research_campaign_configuration.md)
* [docs/strict_mode.md](docs/strict_mode.md)
* [docs/research_integrity_and_qa.md](docs/research_integrity_and_qa.md)

Examples:

* [docs/examples/real_alpha_workflow.md](docs/examples/real_alpha_workflow.md)
* [docs/examples/milestone_11_5_alpha_portfolio_workflow.md](docs/examples/milestone_11_5_alpha_portfolio_workflow.md)
* [docs/examples/alpha_evaluation_end_to_end.py](docs/examples/alpha_evaluation_end_to_end.py)
* [docs/examples/milestone_13_review_promotion_workflow.md](docs/examples/milestone_13_review_promotion_workflow.md)
* [docs/examples/milestone_16_campaign_workflow.md](docs/examples/milestone_16_campaign_workflow.md)
* [docs/examples/real_world_campaign_case_study.md](docs/examples/real_world_campaign_case_study.md)
* [docs/examples/regime_aware_case_study.md](docs/examples/regime_aware_case_study.md)
* [docs/examples/real_q1_2026_regime_aware_case_study.md](docs/examples/real_q1_2026_regime_aware_case_study.md)
* [docs/examples/milestone_17_resume_workflow.md](docs/examples/milestone_17_resume_workflow.md)
* [docs/examples/real_world_resume_workflow_case_study.md](docs/examples/real_world_resume_workflow_case_study.md)
* [docs/examples/ml_cross_sectional_lgbm_2026_q1_candidate_driven_workflow.md](docs/examples/ml_cross_sectional_lgbm_2026_q1_candidate_driven_workflow.md)
* [docs/examples/real_world_candidate_selection_portfolio_case_study.md](docs/examples/real_world_candidate_selection_portfolio_case_study.md)
* [docs/examples/scale_repro_benchmark_pack.md](docs/examples/scale_repro_benchmark_pack.md)
* [docs/examples/pipelines/regime_ensemble_showcase/README.md](docs/examples/pipelines/regime_ensemble_showcase/README.md)
* [docs/ml_cross_sectional_xgb_2026_q1.md](docs/ml_cross_sectional_xgb_2026_q1.md)
* [docs/backfilled_2026_q1_research_workflow.md](docs/backfilled_2026_q1_research_workflow.md)

Merge-readiness notes:

* [docs/milestone_10_merge_readiness.md](docs/milestone_10_merge_readiness.md)
* [docs/milestone_11_merge_readiness.md](docs/milestone_11_merge_readiness.md)
* [docs/milestone_13_merge_readiness.md](docs/milestone_13_merge_readiness.md)
* [docs/milestone_16_merge_readiness.md](docs/milestone_16_merge_readiness.md)
* [docs/milestone_22_merge_readiness.md](docs/milestone_22_merge_readiness.md)

## Repository Layout

```text
src/
  cli/        command-line entrypoints
  config/     execution, runtime, evaluation, robustness, and simulation config
  portfolio/  construction, optimization, risk, validation, QA, and artifacts
  research/   alpha, strategy execution, integrity checks, robustness, simulation, and reporting

configs/
  evaluation.yml
  execution.yml
  portfolios.yml
  robustness.yml
  sanity.yml
  strategies.yml

artifacts/
  strategies/
  portfolios/
```
