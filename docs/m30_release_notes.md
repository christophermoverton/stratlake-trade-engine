# v0.30.0 - Statistical Diagnostics & Metrics Readiness

Milestone 30 adds deterministic statistical diagnostics and advisory readiness
artifacts across StratLake strategy and portfolio workflows.

## Highlights

* Added SciPy-backed return inference diagnostics:
  * `t_stat`
  * `p_value`
  * `conf_int_lower`
  * `conf_int_upper`
* Added trade-level binomial hit-rate significance:
  * `hit_rate_p_value`
* Added serial-dependence diagnostics:
  * `autocorr_lag1`
  * `effective_n`
* Added split-period consistency diagnostics:
  * `split_mean_diff`
  * `split_mean_diff_p`
* Added rolling Sharpe stability diagnostics:
  * `rolling_sharpe_mean`
  * `rolling_sharpe_sd`
  * `sharpe_stability_ratio`
* Added advisory readiness manifests:
  * `metrics_readiness.json`
* Added documentation and a runnable statistical diagnostics readiness example.

## Research Integrity Notes

* Student-t return inference assumes independent period returns.
* Serial-dependence diagnostics inform interpretation but do not yet adjust p-values.
* Split-period and rolling Sharpe diagnostics are lightweight stability checks,
  not replacements for walk-forward evaluation.
* Readiness manifests are advisory review artifacts, not hard promotion gates.

## Validation

Final validation for Milestone 30 Issue #338:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_metrics.py tests\test_portfolio_metrics.py
```

Result: `68 passed`.

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_experiment_tracker.py tests\test_experiment_registry.py tests\test_consistency.py
```

Result: `34 passed`.

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_walk_forward.py tests\test_portfolio_walk_forward.py
```

Result: `13 passed`.

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_cli_run_strategy.py tests\test_baseline_strategies.py
```

Result: `35 passed`.

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_catalog_indexer.py tests\test_catalog_lineage.py tests\test_catalog_query.py tests\test_catalog_validation.py
```

Result: `68 passed`.

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_docs_path_portability.py tests\test_statistical_diagnostics_readiness_example.py
```

Result: `4 passed`.

```powershell
.\.venv\Scripts\python.exe docs\examples\statistical_diagnostics_readiness_example.py
```

Result: example completed successfully and wrote `metrics.json` plus
`metrics_readiness.json` under its relative output directory.

```powershell
.\.venv\Scripts\ruff.exe check .
```

Result: all checks passed.

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Result: `1750 passed`, with expected fixture and diagnostic warnings.

Additional M30 artifact and determinism probes confirmed:

* repeated metric computation over the same deterministic fixture returns
  identical dictionaries
* repeated readiness manifest generation over the same metrics payload returns
  identical dictionaries
* sorted JSON serialization with `allow_nan=False` succeeds for metrics and
  readiness payloads
* strategy artifacts include `metrics.json` and `metrics_readiness.json`
* strategy `metrics.json` includes every Milestone 30 metric field
* `metrics_readiness.json` includes `schema_version`, `status`, `run_id`,
  `source_metrics_artifact`, grouped `diagnostics`, `checks`, and `summary`
* the docs example can run repeatedly with stable output file content
* walk-forward strategy and portfolio split readiness artifacts are covered by
  `tests\test_walk_forward.py` and `tests\test_portfolio_walk_forward.py`
* generated JSON artifacts reject non-finite values at serialization boundaries
* docs source path hygiene found no absolute local user-directory paths

Release title:

```text
v0.30.0-statistical-diagnostics-readiness
```
