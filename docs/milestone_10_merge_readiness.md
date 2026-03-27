# Milestone 10 Merge Readiness

This checklist defines the deterministic validation workflow for merging the
Milestone 10 branch into `main`.

The goal is to validate existing behavior, not to introduce new functionality.
If any step fails, stop the process, fix the failure on the milestone branch,
and rerun the workflow from the beginning.

## Scope

Milestone 10 merge readiness covers:

* temporal integrity and lagged execution enforcement
* execution realism, turnover, and cost attribution
* strict-mode behavior across strategy and portfolio CLIs
* runtime configuration persistence
* artifact integrity and registry correctness
* portfolio validation, sanity checks, and cross-layer consistency
* documentation alignment for the released workflow

## Pre-Merge Validation

Run all commands from the repository root with the project virtual environment
activated.

### 1. Full automated test suite

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Pass criteria:

* all tests pass
* only expected, already-understood warnings appear
* no new failures or warning classes are introduced

### 2. Strict-mode strategy smoke

```powershell
.\.venv\Scripts\python.exe -m src.cli.run_strategy --strategy momentum_v1 --strict
.\.venv\Scripts\python.exe -m src.cli.run_strategy --strategy mean_reversion_v1 --strict
```

Pass criteria:

* both commands exit successfully
* artifacts are written for completed runs
* `config.json`, `metrics.json`, `manifest.json`, `qa_summary.json`, and
  `signal_diagnostics.json` are present
* persisted config includes `runtime` and `strict_mode`
* `metrics.json` reports a non-failing sanity status

### 3. Strict-mode portfolio integration smoke

```powershell
.\.venv\Scripts\python.exe -m src.cli.run_portfolio --portfolio-config configs/portfolios.yml --portfolio-name strict_valid_builtin_pair --from-registry --timeframe 1D --strict
```

Pass criteria:

* command exits successfully
* `config.json`, `components.json`, `weights.csv`, `portfolio_returns.csv`,
  `portfolio_equity_curve.csv`, `metrics.json`, `qa_summary.json`, and
  `manifest.json` are present
* persisted config includes `runtime`, `validation`, and `strict_mode`
* manifest records `strict_mode`
* registry entry exists and points to the written artifact directory

### 4. Determinism checks

Rerun the exact same strategy and portfolio commands from steps 2 and 3.

Pass criteria:

* deterministic run ids are unchanged
* artifact contents are byte-stable across reruns
* registry state does not accumulate duplicate run ids

### 5. Artifact and registry validation

Validate the generated strategy and portfolio run directories by checking:

* manifests enumerate the files actually written
* metrics agree with the equity curves
* QA summaries and diagnostics agree with metrics
* registry entries exist only for successful runs
* persisted runtime config matches the effective CLI/config resolution

## Review Checklist

### Code review focus

Review these areas against the intended Milestone 10 design:

* `src/config/runtime.py`
* `src/research/strict_mode.py`
* `src/research/integrity.py`
* `src/research/consistency.py`
* `src/research/sanity.py`
* `src/portfolio/constructor.py`
* `src/portfolio/validation.py`
* `src/portfolio/artifacts.py`
* `src/cli/run_strategy.py`
* `src/cli/run_portfolio.py`

Review for:

* clear separation of runtime config, policy, validation, and execution
* no silent failures before persistence
* validation before artifact or registry writes
* consistent naming and error semantics across research and portfolio layers
* deterministic behavior and auditable persisted state

### Documentation review focus

Review:

* `README.md`
* `docs/research_validity_framework.md`
* `docs/execution_model.md`
* `docs/strict_mode.md`
* `docs/runtime_configuration.md`
* `docs/portfolio_construction_workflow.md`
* `docs/cli_strategy_runner.md`
* `docs/getting_started.md`

Check that:

* Milestone 10 terminology is consistent
* CLI examples still match the implemented flags
* runtime configuration precedence is documented correctly
* strict mode is described as fail-fast and pre-persistence
* links are relative and resolve inside the repository

## Merge Procedure

Only merge after the pre-merge validation and review pass is clean.

```powershell
git checkout main
git merge --no-ff m10-research-validity-execution-realism
```

Merge criteria:

* no conflicts
* merge commit completes cleanly
* no unrelated files are changed during resolution

## Post-Merge Verification

After the merge, rerun:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m src.cli.run_strategy --strategy momentum_v1 --strict
.\.venv\Scripts\python.exe -m src.cli.run_portfolio --portfolio-config configs/portfolios.yml --portfolio-name strict_valid_builtin_pair --from-registry --timeframe 1D --strict
git status --short
```

Pass criteria:

* full test suite still passes on `main`
* strategy and portfolio smoke commands still succeed on `main`
* artifact and registry validation remain correct
* working tree is clean except for ignored artifacts

## Failure Handling

If any step fails:

1. stop the merge flow immediately
2. keep `main` unchanged
3. fix the issue on the milestone branch
4. rerun the workflow from the start

Do not accept partial validation, partial merges, or uncertain registry and
artifact state.
