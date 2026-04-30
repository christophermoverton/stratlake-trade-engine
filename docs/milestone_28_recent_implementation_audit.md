# Milestone 28 Recent Implementation Audit and Methodology Soundness Review

## Purpose

This document audits recent StratLake milestone implementations, focusing on
Milestones 22 through 27, and establishes the baseline for Milestone 28.

The audit reviews methodology soundness, deterministic behavior, artifact
compatibility, pipeline and execution integration readiness, CLI/API parity,
notebook friendliness, concurrency and idempotency risks, known limitations,
and follow-up recommendations.

This is an audit and planning document. It does not introduce a new execution
framework, a new pipeline abstraction, broker integration, live-trading
behavior, or a replacement for existing workflow logic.

## Executive Summary

StratLake already has a strong deterministic, artifact-driven workflow
foundation. Milestone 28 should harden and expose existing execution surfaces
rather than introducing a parallel pipeline system.

Verified repository evidence shows mature surfaces for CLI execution,
`src.execution` notebook/script APIs, `src.pipeline` execution, benchmark-pack
workflows, research-campaign orchestration, deterministic rerun validation,
milestone validation bundles, checkpoint/resume behavior, artifact inventories,
and regime-aware research review. The main Milestone 28 opportunity is to make
these surfaces safer and easier to reuse across CLI, API, pipeline,
orchestrator, validation, and notebook contexts.

The highest-priority follow-up is idempotency and concurrency hardening around
shared output roots, repeated runs, partial writes, manifest completeness, and
external orchestrators that may launch duplicate jobs.

## Scope

This audit covers at least these milestone areas:

* Milestone 22 validation, benchmark packs, deterministic reruns, artifact
  inventories, and release traceability.
* Milestone 23 shared `src.execution` APIs, CLI/API parity, and notebook
  execution patterns.
* Milestone 24 regime-aware research interpretation across strategy, alpha,
  and portfolio surfaces.
* Milestone 25 calibration profiles, GMM-assisted regime confidence, regime
  sensitivity, and adaptive policy optimization inputs.
* Milestone 26 governed adaptive regime-policy workflow, promotion gates,
  review packs, candidate selection, stress tests, and case-study stitching.
* Milestone 27 market simulation stress-testing case study and optional bridge
  into Milestone 26 adaptive policy stress evidence.

The audit also reviews existing pipeline and execution infrastructure:

* CLI entrypoints under `src/cli/`.
* Shared Python APIs under `src/execution/`.
* Existing `run_pipeline` behavior through `src.cli.run_pipeline`,
  `src.execution.run_pipeline`, and `src.pipeline`.
* Benchmark-pack workflows in `src.research.benchmark_pack`,
  `src.cli.run_benchmark_pack`, and `src.execution.run_benchmark_pack`.
* Research-campaign orchestration through `src.cli.run_research_campaign`,
  `src.config.research_campaign`, and `src.execution.run_research_campaign`.
* Deterministic rerun validation through
  `src.validation.deterministic_rerun`,
  `src.cli.run_deterministic_rerun_validation`, and
  `src.execution.run_deterministic_rerun_validation`.
* Milestone-validation bundles through `src.validation.milestone_bundle`,
  `src.cli.run_milestone_validation`, and
  `src.execution.run_milestone_validation`.
* Checkpoint/resume behavior in research campaigns and benchmark packs.
* Artifact manifests and inventories across milestone surfaces.
* Notebook execution API patterns in `docs/notebook_execution_api.md`,
  `docs/examples/notebook_execution_api_examples.md`, and
  `docs/examples/notebook_execution_api_examples.py`.

## Methodology

The audit was performed by reviewing repository files and implementation
surfaces that exist in the current tree:

* Repository overview and milestone summaries in `README.md`.
* Milestone readiness and workflow documentation in `docs/`.
* Notebook execution documentation and examples in `docs/notebook_execution_api.md`
  and `docs/examples/`.
* CLI entrypoints under `src/cli/`.
* Importable execution APIs under `src/execution/`.
* Pipeline implementation under `src/pipeline/`.
* Research, regime, regime ML, benchmark-pack, and market-simulation modules
  under `src/research/`.
* Portfolio execution and artifact surfaces under `src/portfolio/`.
* Validation modules under `src/validation/`.
* Config contracts under `configs/` and `src/config/`.
* Test coverage names and relevant slices under `tests/`.

The review focused on:

* Documentation consistency and stated limitations.
* CLI/API surface alignment.
* Artifact contract consistency, manifests, inventories, and relative-path
  expectations.
* Deterministic seeds, stable IDs, stable ordering, and reproducible reports.
* Pipeline and notebook readiness.
* Checkpoint/resume behavior.
* Concurrency and idempotency risk assessment.

Where behavior could not be verified from repository files, this document marks
it as not verified or requires follow-up.

## Milestone-by-Milestone Audit Table

| Milestone | Primary Capability | Relevant Files / Surfaces | Determinism Posture | Artifact Contract Posture | Pipeline / Execution Integration Posture | Notebook Integration Posture | Methodology Soundness Notes | Risks / Follow-Ups |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M22 | Validation hardening, release traceability, deterministic reruns, benchmark packs. | `docs/milestone_22_merge_readiness.md`, `docs/milestone_22_benchmark_packs.md`, `src/validation/`, `src/cli/run_docs_path_lint.py`, `src/cli/run_deterministic_rerun_validation.py`, `src/cli/run_milestone_validation.py`, `src/cli/run_benchmark_pack.py`, `src/research/benchmark_pack.py`, `configs/benchmark_packs/m22_scale_repro.yml`, `tests/test_m22_deterministic_rerun_validation.py`, `tests/test_benchmark_pack.py`. | Strong. Deterministic rerun validation runs canonical pipeline examples twice and compares normalized summaries. Benchmark packs support deterministic scenario batching, resume, and inventory comparison. | Strong. Validation bundles produce `summary.json` and check reports. Benchmark packs produce `manifest.json`, `summary.json`, `checkpoint.json`, `inventory.json`, batch plans, and benchmark matrices. | Strong. Benchmark packs sit above existing research-campaign orchestration and explicitly do not replace it. Validation CLIs are reusable release surfaces. | Moderate to strong after M23. M22 itself is primarily CLI/release oriented; M23 exposes validation and benchmark-pack wrappers to notebooks. | Sound as release validation and reproducibility evidence. The benchmark-pack layer is a wrapper above existing orchestration, not a new research semantics layer. | Concurrent runs sharing the same output root can collide. Partial writes and reused checkpoints require stronger idempotency guarantees before external orchestrator examples are promoted. |
| M23 | Shared `src.execution` APIs and notebook-friendly execution results. | `docs/notebook_execution_api.md`, `docs/examples/notebook_execution_api_examples.md`, `docs/examples/notebook_execution_api_examples.py`, `src/execution/__init__.py`, `src/execution/result.py`, `src/execution/pipeline.py`, `src/execution/orchestration.py`, `src/execution/validation.py`, `src/execution/benchmark.py`, `tests/test_execution_api.py`, `tests/test_cli_api_parity.py`. | Strong by design. Wrappers delegate to existing deterministic CLI/workflow implementations and expose `ExecutionResult` summaries. | Strong. `ExecutionResult` exposes named output paths, manifests, metrics, summaries, registries, and JSON-loading helpers without creating alternate persistence contracts. | Strong. Verified wrappers exist for strategy, alpha, alpha evaluation, portfolio, pipeline, campaign, validation, and benchmark-pack workflows. | Strong. This is the main notebook-facing surface, with inspection helpers and examples. | Sound because it preserves CLI contracts and treats notebooks as inspection/exploration surfaces, while CLI remains preferred for CI and release automation. | API parity is protected by tests, but full parity across every optional workflow/configuration remains not verified. Notebook reruns still need clearer guidance around output-root isolation. |
| M24 | Regime-aware interpretation across strategy, alpha, and portfolio research. | `docs/regime_taxonomy.md`, `docs/regime_conditional_evaluation.md`, `docs/regime_transition_analysis.md`, `docs/regime_attribution_and_comparison.md`, `docs/examples/regime_aware_case_study.md`, `docs/examples/regime_notebook_review_examples.md`, `src/research/regimes/`, `tests/test_regime_classification.py`, `tests/test_regime_conditional_evaluation.py`, `tests/test_regime_transition_analysis.py`, `tests/test_regime_notebook_helpers.py`. | Strong for the verified taxonomy layer. Regime classification is documented as rule based, deterministic, UTC-normalized, sorted, and exact-timestamp aligned. | Strong. Regime artifacts include `regime_labels.csv`, `regime_summary.json`, and `manifest.json`, with canonical schemas and relative inventory expectations. | Moderate to strong. Regime outputs attach to existing strategy, alpha, and portfolio artifacts and are used by later M25-M27 flows. Direct pipeline parity is implied through existing artifacts, but not every pipeline combination is verified. | Strong for inspection. Notebook helpers and examples exist for regime review. | Sound safeguards include no hidden-state inference, no macro/live inputs, exact timestamp alignment, explicit undefined handling, and deferred taxonomy migration. | Risk of overstating regime interpretation if users treat labels as causal market truths. Follow-up should keep taxonomy versioning stable and avoid redefining labels inside M28. |
| M25 | Calibration profiles, GMM confidence, sensitivity, and adaptive policy optimization inputs. | `docs/regime_calibration.md`, `docs/regime_gmm_classifier.md`, `docs/regime_policy_optimization.md`, `docs/regime_sensitivity_matrix.md`, `src/research/regimes/calibration.py`, `src/research/regimes/gmm_classifier.py`, `src/research/regimes/policy.py`, `src/research/regime_ml/`, `tests/test_regime_calibration.py`, `tests/test_regime_gmm_classifier.py`, `tests/test_regime_policy_optimization.py`, `tests/test_regime_ml.py`. | Strong where documented. Calibration is causal and deterministic. GMM requires `random_state`, stable timestamp sorting, stable cluster ordering, and deterministic persistence. | Strong. Calibration and GMM write JSON/CSV artifacts with manifests or inventory payloads and relative path expectations. | Moderate to strong. M25 builds on M24 outputs and feeds M26 policy governance, but it is not a standalone pipeline framework. | Moderate to strong. Notebook-friendly review flows can inspect emitted artifacts; direct M23 wrappers for every M25-specific function are not verified. | Sound if treated as confidence/stability evidence, not taxonomy replacement. Calibration explicitly does not redefine labels or use lookahead-sensitive smoothing. GMM complements taxonomy labels rather than replacing them. | Requires ongoing guardrails around calibration stability assumptions, minimum observations, ML confidence interpretation, and avoiding claims that GMM clusters are definitive regimes. |
| M26 | Governed adaptive regime-policy workflow with benchmark packs, promotion gates, review packs, candidate selection, stress tests, and full-year case study. | `docs/milestone_26_merge_readiness.md`, `docs/regime_benchmark_packs.md`, `docs/regime_promotion_gates.md`, `docs/regime_review_packs.md`, `docs/regime_aware_candidate_selection.md`, `docs/regime_policy_stress_testing.md`, `docs/examples/full_year_regime_policy_benchmark_case_study.md`, `src/cli/run_regime_benchmark_pack.py`, `src/cli/run_regime_promotion_gates.py`, `src/cli/generate_regime_review_pack.py`, `src/cli/run_regime_aware_candidate_selection.py`, `src/cli/run_regime_policy_stress_tests.py`, `src/research/regime_*`, `tests/test_full_year_regime_policy_benchmark_case_study.py`. | Strong for fixture-backed and config-driven workflows. M26 readiness records focused and full-suite validation. Stress transforms are deterministic diagnostics. | Strong. Regime benchmark, promotion, review, candidate-selection, stress, and case-study roots produce manifests, summaries, evidence indexes, decision logs, and comparison tables. | Strong for CLI workflow surfaces. Integration through `src.execution` exists for several M26 regime surfaces in `src/execution/regime_*`, and M26 consumes M24/M25 artifacts. | Moderate. Case-study and review artifacts are inspectable, but notebook-first wrappers/examples for every M26-specific surface are not fully verified. | Sound as a research-governance layer. Documentation explicitly says it does not introduce live trading, production deployment, or new research semantics. | External orchestration of multi-step M26 flows should reuse existing CLIs/APIs. Follow-up should harden shared output roots, stage reuse, and manifest completion before adding Airflow/Prefect/Dagster-style examples. |
| M27 | Market simulation stress-testing case study and optional M26 bridge. | `docs/milestone_27_merge_readiness.md`, `docs/market_simulation_stress_testing.md`, `docs/market_simulation_models_and_integrations.md`, `docs/examples/m27_market_simulation_case_study.md`, `src/cli/run_market_simulation_scenarios.py`, `src/execution/market_simulation.py`, `src/research/market_simulation/`, `tests/test_m27_market_simulation_case_study.py`, `tests/test_market_simulation_artifacts.py`, `tests/test_simulation_stress_metrics.py`, `tests/test_market_simulation_policy_stress_integration.py`. | Strong for configured simulations. The framework resolves seeds, stable scenario/path identifiers, deterministic catalogs, and stable manifests. Monte Carlo is deterministic when configured through the framework. | Strong. M27 emits scenario catalogs, inventories, normalized configs, manifests, simulation summaries, leaderboards, and policy failure summaries. | Strong as a research evidence layer. It integrates optionally with M26 stress summaries and reuses the M27 metrics layer rather than reimplementing simulation logic. | Moderate. Artifacts are notebook-inspectable; dedicated notebook examples through `src.execution` for M27 are not verified. | Sound boundaries are documented: simulations are diagnostics, not forecasts or trading recommendations. Regime-transition Monte Carlo remains regime-only unless return or policy replay artifacts exist. | Follow-up should avoid presenting M27 outputs as empirical forecasts. External reruns need idempotent output-root guidance and safe handling of generated source artifacts. |

## Existing Pipeline and Execution Infrastructure Review

Verified repository evidence shows that StratLake already has a layered
execution foundation:

* `src/cli/` contains process-oriented entrypoints for strategy, alpha,
  portfolio, pipeline, research campaign, benchmark-pack, validation, regime
  governance, and market-simulation workflows.
* `src/execution/` exposes notebook/script-friendly APIs that delegate to the
  same underlying workflow implementations used by CLI entrypoints.
* `src/execution/result.py` defines `ExecutionResult`, which provides
  notebook-friendly access to workflow identity, metrics, artifact roots,
  manifests, named output paths, summaries, and JSON artifact loading.
* `src/execution/pipeline.py` runs `PipelineSpec.from_yaml(...)` through
  `PipelineRunner.run()`, the same implementation family used by
  `python -m src.cli.run_pipeline`.
* `src/pipeline/` contains the existing pipeline runner, builder, registry,
  CLI adapter, testing helpers, and feature-pipeline support.
* `docs/pipeline_builder.md` states that `PipelineBuilder` sits above the
  existing M20-compatible runner and does not replace it.
* `src/execution/orchestration.py` delegates research campaigns to the existing
  campaign runner, preserving preflight checks, checkpoints, stage ordering,
  resume/reuse decisions, manifests, milestone reports, and scenario
  orchestration.
* `src/execution/benchmark.py` delegates benchmark packs to the existing
  benchmark runner, preserving deterministic output layout, checkpoint/resume
  behavior, manifests, inventories, benchmark matrices, and optional inventory
  comparisons.
* `src/execution/validation.py` exposes docs/path lint, deterministic rerun
  validation, and milestone-validation bundle creation without changing their
  CLI contracts.
* `docs/research_campaign_configuration.md` documents campaign preflight,
  checkpoint reuse policy, scenario expansion, stage states, manifests, and
  summaries.
* `docs/milestone_22_benchmark_packs.md` documents benchmark-pack checkpoint,
  resume, inventory, matrix, and comparison behavior above research-campaign
  orchestration.
* `docs/notebook_execution_api.md` documents the intended CLI/API relationship:
  notebooks are for exploration and inspection, while CLI remains the release,
  automation, and CI interface.

Milestone 28 should reuse and harden existing surfaces such as:

* `src.execution` APIs.
* CLI entrypoints.
* Benchmark-pack execution.
* Research-campaign orchestration.
* Deterministic rerun validation.
* Milestone-validation bundles.
* Checkpoint/resume behavior.
* Artifact manifests and inventories.
* Notebook execution API patterns.

Milestone 28 should provide thin wrappers, examples, validation, and
documentation around these surfaces. It should not create a second workflow
system.

## Pipeline and Notebook Friendliness Assessment

The current system is pipeline-friendly because it already has:

* YAML-driven pipeline execution through `src.pipeline` and
  `src.cli.run_pipeline`.
* Declarative pipeline authoring through `PipelineBuilder`, documented as a
  layer above the existing runner.
* Research-campaign orchestration with preflight, scenario expansion,
  checkpoint/reuse policy, stage summaries, and manifests.
* Benchmark packs that batch scenario campaigns and emit matrix, inventory,
  checkpoint, manifest, and comparison artifacts.
* Validation commands suitable for CI and release checks.

The current system is notebook-friendly because it already has:

* Importable `src.execution` functions for major workflows.
* `ExecutionResult` helpers for artifact inspection.
* Notebook execution documentation and import-safe examples.
* Regime notebook review helpers and examples.
* A documented distinction between notebook exploration and CLI release
  automation.

The current system is artifact-driven because major workflows persist JSON,
CSV, Parquet, Markdown, manifests, summaries, inventories, decision logs,
leaderboards, matrices, and validation reports with stable named paths.

The current system is reproducible across entry points for representative
workflows. This is verified by CLI/API parity tests, deterministic rerun
validation, milestone validation, documented benchmark-pack inventory
comparison, and repeated references to deterministic IDs, sorted ordering,
seeded simulation/ML behavior, and relative persisted paths.

Practical hardening needed before external orchestration examples are added:

* Define output-root isolation guidance for orchestrated and notebook reruns.
* Make repeated runs explicit about reuse, overwrite, append, or fail-fast
  behavior.
* Strengthen partial-write safety and manifest completion guarantees.
* Document how external schedulers should pass unique run/output identifiers.
* Validate CLI/API/pipeline/notebook parity for the exact M28 examples.
* Keep examples as thin calls into existing CLIs/APIs rather than new workflow
  implementations.

## Concurrency and Idempotency Risk Review

This section identifies risks for Issue M28.2. It does not implement fixes.

Repeated runs:

* A rerun against a stable output root may reuse checkpoints or overwrite
  generated files depending on the workflow. The intended behavior is
  documented for benchmark packs and campaigns, but not every workflow has the
  same explicit contract.
* Notebook users may rerun cells and unintentionally write into the same
  artifact root.

Concurrent runs:

* Two processes targeting the same output root can collide on manifest,
  checkpoint, summary, registry, inventory, or leaderboard files.
* The repository evidence does not verify distributed locking or atomic
  multi-process safety across all artifact writers.

Output directory collisions:

* Deterministic IDs are valuable for reproducibility, but they can increase
  collision risk when multiple orchestrators run identical configs at the same
  time.
* Shared roots such as `artifacts/qa/`, `artifacts/benchmark_packs/`,
  `artifacts/research_campaigns/`, and `docs/examples/output/` require clearer
  isolation rules for automated launchers.

Partial artifact writes:

* If a run stops while writing a manifest, inventory, summary, or checkpoint,
  downstream readers may observe incomplete state unless writers use atomic
  temp-file replacement consistently. This is not verified across all surfaces.

Incomplete manifests:

* Manifests and inventories are central review surfaces. M28.2 should verify
  that failure paths either emit intentionally partial status or avoid exposing
  complete-looking manifests for incomplete runs.

Shared output roots:

* Research campaigns and benchmark packs intentionally share nested artifact
  structures. External orchestrators should use per-run output roots or
  documented partitioning to avoid cross-run contamination.

Notebook reruns:

* `ExecutionResult` helpers are read-oriented, but workflow calls can write
  artifacts. Notebook examples should show explicit output roots for rerunnable
  cells where supported.

External orchestrators launching duplicate runs:

* Airflow, Prefect, Dagster, CI, or scheduler retries may invoke the same
  command more than once. Existing checkpoint/reuse behavior is useful, but
  M28.2 should define which workflows are safe to retry, which require unique
  roots, and which should fail fast when an active/incomplete run is detected.

## Methodology Soundness Review

Recent regime research work has a generally sound methodology posture when
used as documented:

* Deterministic seeds are documented where ML or simulation introduces
  stochastic behavior. The GMM classifier requires `random_state`; M27 resolves
  seeds and stable scenario/path identifiers.
* M24 defines `regime_taxonomy_v1` as the canonical rule-based taxonomy.
  M25 calibration and GMM documentation explicitly state they do not redefine
  or replace that taxonomy.
* Calibration uses causal trailing-window smoothing and minimum-duration
  stabilization. Documentation states it does not use centered smoothing,
  forward-fill, or lookahead-sensitive logic.
* Regime alignment uses exact timestamp equality and does not silently
  forward-fill, backfill, interpolate, or drop target rows.
* Artifact-based reproducibility is central across M22-M27. Manifests,
  inventories, summaries, configs, matrices, and validation reports make
  review repeatable.
* Human review gates are represented in promotion gates, review packs,
  decision logs, evidence indexes, and milestone reports.
* M26 and M27 documentation correctly avoids production or live-trading claims.
  M27 simulation outputs are framed as diagnostics, not forecasts or trading
  recommendations.

Soundness concerns to keep visible:

* Regime labels are deterministic classifications, not proof of causal market
  states.
* GMM clusters and posterior confidence scores complement taxonomy evidence;
  they should not be treated as definitive hidden regimes.
* Calibration stability assumptions may be profile-sensitive and should be
  reviewed under sensitivity analysis before downstream use.
* Stress tests and simulations validate behavior under configured scenarios;
  they do not prove future performance or live risk control.
* Fixture-backed case studies verify workflow and artifact contracts, not
  empirical trading edge.
* No-lookahead and temporal alignment expectations should remain explicit in
  docs, examples, and validation.

## Findings

### Verified Strengths

* StratLake already exposes reusable execution surfaces through CLI entrypoints
  and `src.execution`.
* Representative CLI/API parity is covered by tests.
* Notebook execution is documented as artifact-first and complementary to CLI
  automation.
* Benchmark packs and campaigns already support checkpoint/resume behavior and
  explicit manifests/summaries.
* M22 validation provides docs/path linting, deterministic rerun validation,
  and milestone-validation bundles.
* M24-M27 regime and simulation work preserves deterministic framing,
  artifact traceability, and research-only boundaries.
* Existing pipeline documentation explicitly avoids replacing the established
  pipeline runner.

### Risks / Gaps

* Distributed or concurrent writer safety is not verified across all workflows.
* Output-root collision behavior is not uniformly documented for notebooks,
  external orchestrators, and duplicate scheduler launches.
* Partial-write and incomplete-manifest behavior needs a cross-surface audit.
* Notebook examples may need stronger explicit output-root patterns for
  rerunnable cells.
* Full CLI/API/pipeline/notebook parity is verified for representative slices,
  not every optional configuration.
* Some M26/M27 specialized surfaces are inspectable from notebooks, but
  notebook-first examples for every specialized regime workflow are not
  verified.

### Recommended Follow-Ups

* M28.2 should define and validate idempotency, retry, collision, and
  partial-write behavior for existing artifact and execution surfaces.
* M28.3 should provide Airflow/Prefect/Dagster-style examples as thin wrappers
  around existing CLIs or `src.execution` functions.
* M28.4 should route notebook examples through `src.execution` or
  CLI-equivalent code paths and avoid custom workflow logic.
* M28.5 should validate parity across CLI, API, pipeline, and notebook entry
  points for selected M28 workflows.
* M28.6 should stitch a capstone from existing execution surfaces instead of
  introducing a separate case-study pipeline.

## Milestone 28 Design Implications

Milestone 28 should harden the repository as a reproducible research platform
by reinforcing the execution system already present.

M28.2 should harden existing artifact and execution surfaces for
idempotency/concurrency. It should focus on output-root collision rules,
repeated-run semantics, partial-write safety, manifest completion, checkpoint
state clarity, and retry guidance.

M28.3 should wrap existing execution surfaces for Airflow/Prefect/Dagster-style
examples. Those examples should call CLIs or `src.execution` functions and
should not add a new orchestration framework dependency or second workflow
system.

M28.4 should route notebooks through `src.execution` or CLI-equivalent paths.
Notebook guidance should emphasize `ExecutionResult`, named outputs, manifests,
and explicit artifact roots for rerunnable examples.

M28.5 should validate CLI/API/pipeline/notebook parity using focused,
representative workflows. Parity should compare stable machine-readable
contracts, not stdout, object identity, absolute prefixes, or transient logs.

M28.6 should be a capstone over the existing execution system, not a separate
case-study pipeline. It should demonstrate reproducible research behavior
across existing CLI, API, pipeline, validation, and notebook surfaces.

## Limitations

This audit does not prove production trading readiness.

This audit does not prove distributed concurrency safety unless that behavior
is separately implemented and validated.

This audit does not validate every possible user configuration.

This audit does not replace unit tests, integration tests, deterministic rerun
validation, benchmark-pack comparison, milestone-validation bundles, or manual
research review.

This audit is grounded in repository files available during review. Any
behavior not visible from those files is marked as not verified or requires
follow-up.

## Recommended Validation Commands

Run these commands from the repository root:

```powershell
python -m src.cli.run_docs_path_lint
python -m src.cli.run_deterministic_rerun_validation
python -m src.cli.run_milestone_validation
pytest
```

Focused practical alternatives when full validation is too expensive:

```powershell
pytest tests/test_docs_path_portability.py tests/test_execution_api.py tests/test_cli_api_parity.py
pytest tests/test_deterministic_reruns.py tests/test_m22_deterministic_rerun_validation.py
pytest tests/test_benchmark_pack.py tests/test_cli_run_research_campaign.py tests/test_pipeline_runner.py
```

Optional deeper slices for regime and simulation surfaces:

```powershell
pytest tests/test_regime_calibration.py tests/test_regime_gmm_classifier.py tests/test_regime_policy_optimization.py
pytest tests/test_regime_policy_stress_tests.py tests/test_market_simulation_policy_stress_integration.py tests/test_m27_market_simulation_case_study.py
```

`python -m src.cli.run_milestone_validation --include-full-pytest` and full
`pytest` are recommended before merging broad execution changes, but they may
be more expensive than a documentation-only audit requires.
