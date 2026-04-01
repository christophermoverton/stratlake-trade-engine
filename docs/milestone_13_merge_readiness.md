# Milestone 13 Merge Readiness

This checklist defines the validation and release-prep workflow for merging the
Milestone 13 branch into `main`.

The goal is to validate existing behavior, not to introduce new functionality.
If any step fails, stop the process, fix the issue on the milestone branch, and
rerun the workflow from the beginning.

## Scope Delivered

Milestone 13 merge readiness covers:

* unified registry-backed review across alpha-evaluation, strategy, and
  portfolio runs
* deterministic review config resolution from repository defaults, config file,
  and CLI overrides
* review-level promotion-gate evaluation and persisted promotion summaries
* deterministic review artifacts under `artifacts/reviews/<review_id>/`
* review manifest, leaderboard, and summary persistence aligned with the
  repository's artifact-first design
* reporting and visualization workflow alignment for post-run review assets

## Docs Updated

Review these Milestone 13 docs together before merge:

* [../README.md](../README.md)
* [milestone_13_research_review_workflow.md](milestone_13_research_review_workflow.md)
* [review_configuration.md](review_configuration.md)
* [strategy_comparison_cli.md](strategy_comparison_cli.md)
* [research_visualization_workflow.md](research_visualization_workflow.md)
* [examples/milestone_13_review_promotion_workflow.md](examples/milestone_13_review_promotion_workflow.md)
* [milestone_13_merge_readiness.md](milestone_13_merge_readiness.md)

## Validation Performed

Milestone 13 readiness should be grounded in repository behavior already
covered by code and committed examples:

* review config parsing and precedence in `src/config/review.py`
* unified review CLI parsing in `src/cli/compare_research.py`
* registry-backed ranking, manifest writing, and review promotion-gate
  persistence in `src/research/review.py`
* deterministic rerun and registry-dedup behavior in the existing rerun and
  registry tests
* reporting and plot generation from saved strategy artifacts in
  `src/cli/generate_report.py` and `src/cli/plot_strategy_run.py`

## Known Limits Intentionally Deferred

Current Milestone 13 limits that should remain explicit at merge time:

* unified review is registry-backed only; it does not execute fresh research
  runs
* ranking remains within run type; there is no cross-type normalized score
* review-level promotion gates evaluate saved deterministic artifacts, not
  discretionary analyst approvals
* review CLI smoke against the repository root requires populated
  `artifacts/alpha/registry.jsonl`, `artifacts/strategies/registry.jsonl`, and
  `artifacts/portfolios/registry.jsonl`
* there is no committed repository-level `configs/review_gates.yml`; review
  promotion-gate smoke uses the committed Milestone 13 example workflow

## Pre-Merge Validation

Run all commands from the repository root with the project virtual environment
activated.

### 1. Full automated test suite

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Pass criteria:

* all tests pass
* no new warning classes appear
* no Milestone 13 tests are skipped unexpectedly

### 2. Targeted Milestone 13 validation slice

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests\test_research_review.py `
  tests\test_review_config.py `
  tests\test_milestone_13_review_promotion_workflow.py `
  tests\test_cli_alpha_eval.py `
  tests\test_cli_reporting.py `
  tests\test_alpha_eval_comparison.py `
  tests\test_alpha_eval_registry.py `
  tests\test_deterministic_reruns.py `
  tests\test_experiment_registry.py `
  tests\test_promotion_gates.py
```

Pass criteria:

* unified review, review config, promotion-gate, rerun, registry, alpha-review,
  and reporting coverage all pass together
* no targeted test requires code or docs changes outside Milestone 13 scope

### 3. Alpha-evaluation review smoke

```powershell
.\.venv\Scripts\python.exe docs\examples\alpha_evaluation_end_to_end.py
.\.venv\Scripts\python.exe -m src.cli.compare_alpha `
  --from-registry `
  --artifacts-root docs/examples/output/alpha_evaluation_end_to_end/artifacts/alpha `
  --output-path docs/examples/output/alpha_evaluation_end_to_end/comparisons/manual_smoke
```

Pass criteria:

* the example writes deterministic alpha artifacts and registry entries under
  `docs/examples/output/alpha_evaluation_end_to_end/`
* `compare_alpha` succeeds against the example artifacts without extra flags
* leaderboard outputs are written under the requested comparison directory

### 4. Unified review, promotion, and reporting smoke

Run the committed Milestone 13 review-and-promotion example:

```powershell
.\.venv\Scripts\python.exe docs\examples\milestone_13_review_promotion_workflow.py
```

Then run registry-backed strategy comparison plus plot/report generation from a
saved strategy artifact:

```powershell
.\.venv\Scripts\python.exe -m src.cli.compare_strategies --strategies momentum_v1 mean_reversion_v1 --from-registry
$runDir = (Get-Content artifacts/strategies/registry.jsonl | ForEach-Object { $_ | ConvertFrom-Json } | Where-Object { $_.run_type -eq "strategy" -and $_.strategy_name -eq "momentum_v1" } | Sort-Object timestamp, run_id | Select-Object -Last 1).artifact_path
.\.venv\Scripts\python.exe -m src.cli.plot_strategy_run --run-dir $runDir
.\.venv\Scripts\python.exe -m src.cli.generate_report --run-dir $runDir
```

Pass criteria:

* the Milestone 13 example writes `leaderboard.csv`, `review_summary.json`,
  `manifest.json`, `promotion_gates.json`, and `summary.json`
* the example review promotion status remains `review_ready`
* `compare_strategies` succeeds in registry mode
* plot generation writes supported files under `<run_dir>/plots/`
* report generation writes `<run_dir>/report.md`

### 5. Unified review CLI smoke against live registries

Run this only when all three live registries are populated:

```powershell
.\.venv\Scripts\python.exe -m src.cli.compare_research `
  --from-registry `
  --run-types alpha_evaluation strategy portfolio `
  --timeframe 1D `
  --top-k 3
```

If `artifacts/alpha/registry.jsonl` is absent, do not invent a local smoke
environment inside `artifacts/`; record this step as manual or precondition
not met and rely on the committed example plus automated tests for merge
evidence.

Pass criteria when the command is runnable:

* the command exits successfully
* one deterministic review directory is written under `artifacts/reviews/`
* `manifest.json` and `review_summary.json` persist the effective review config
* leaderboard rows point back to existing source artifact directories

### 6. Deterministic rerun checks

Rerun both example commands from steps 3 and 4.

Pass criteria:

* generated run ids and review ids remain stable for identical inputs
* rerun artifact contents remain byte-stable
* registry state does not accumulate duplicate run ids

### 7. Artifact, manifest, and registry validation

Inspect the produced review and source artifacts with these checks:

```powershell
Get-Content docs/examples/output/milestone_13_review_promotion_workflow/manifest.json | ConvertFrom-Json
Get-Content docs/examples/output/milestone_13_review_promotion_workflow/review_summary.json | ConvertFrom-Json
Import-Csv docs/examples/output/milestone_13_review_promotion_workflow/leaderboard.csv
Get-Content docs/examples/data/milestone_13_review_inputs/strategies/registry.jsonl | ForEach-Object { $_ | ConvertFrom-Json } | Group-Object run_id | Where-Object Count -gt 1
Get-Content docs/examples/data/milestone_13_review_inputs/portfolios/registry.jsonl | ForEach-Object { $_ | ConvertFrom-Json } | Group-Object run_id | Where-Object Count -gt 1
Get-Content docs/examples/data/milestone_13_review_inputs/alpha/registry.jsonl | ForEach-Object { $_ | ConvertFrom-Json } | Group-Object run_id | Where-Object Count -gt 1
```

Pass criteria:

* manifest `artifact_files` matches the files actually written
* `review_summary.json` counts agree with `leaderboard.csv`
* source registries contain no duplicate run ids
* saved promotion status and gate counts agree between leaderboard rows and the
  referenced source artifacts

### 8. Documentation alignment checks

Review the Milestone 13 docs against the implemented CLI surface in:

* `src/cli/compare_research.py`
* `src/cli/compare_alpha.py`
* `src/cli/compare_strategies.py`
* `src/cli/plot_strategy_run.py`
* `src/cli/generate_report.py`

Check that:

* documented flags exist and use the current names
* no doc relies on a committed `configs/review_gates.yml`
* review outputs are described as deterministic, manifest-backed, and
  registry-backed
* links remain relative and resolve inside the repository

## Review Checklist

### Code Review Focus

Review these areas against the intended Milestone 13 design:

* `src/config/review.py`
* `src/research/review.py`
* `src/research/promotion.py`
* `src/research/comparison_plots.py`
* `src/cli/compare_research.py`
* `src/cli/compare_alpha.py`
* `src/cli/generate_report.py`
* `src/cli/plot_strategy_run.py`
* `tests/test_research_review.py`
* `tests/test_milestone_13_review_promotion_workflow.py`

Review for:

* deterministic config precedence and stable review ids
* registry-backed selection of latest matching runs without duplicate review
  rows
* manifest and summary payloads that match the files actually written
* promotion status resolution that preserves auditability
* no hidden dependence on mutable notebook state or ad hoc filesystem layout

### Documentation Review Focus

Review:

* [../README.md](../README.md)
* [milestone_13_research_review_workflow.md](milestone_13_research_review_workflow.md)
* [review_configuration.md](review_configuration.md)
* [strategy_comparison_cli.md](strategy_comparison_cli.md)
* [research_visualization_workflow.md](research_visualization_workflow.md)
* [examples/milestone_13_review_promotion_workflow.md](examples/milestone_13_review_promotion_workflow.md)

Check that:

* Milestone 13 terminology is consistent across alpha, strategy, portfolio,
  review, and promotion docs
* the docs distinguish fixture-backed examples from live-registry smoke checks
* report and plot commands are described as post-run artifact workflows, not
  fresh execution workflows

## Merge Procedure

Only merge after the pre-merge validation and review pass is clean.

Replace `<milestone-13-branch>` with the actual branch name before running the
merge.

```powershell
git checkout main
git merge --no-ff <milestone-13-branch>
```

Merge criteria:

* no conflicts
* merge commit completes cleanly
* no unrelated files are introduced during conflict resolution

## Post-Merge Verification

After the merge, rerun:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe docs\examples\milestone_13_review_promotion_workflow.py
git status --short
```

If the live alpha, strategy, and portfolio registries are populated on `main`,
also rerun:

```powershell
.\.venv\Scripts\python.exe -m src.cli.compare_research --from-registry --run-types alpha_evaluation strategy portfolio --timeframe 1D --top-k 3
```

Pass criteria:

* the full test suite still passes on `main`
* the committed Milestone 13 example still produces the expected deterministic
  review artifact set
* live registry review smoke still succeeds when its prerequisites are present
* the working tree is clean except for intentionally generated local artifacts

## Failure Handling

If any step fails:

1. stop the merge flow immediately
2. keep `main` unchanged
3. fix the issue on the Milestone 13 branch
4. rerun the workflow from the beginning

Do not accept partial validation, partial merges, or uncertain registry,
manifest, or promotion state.
