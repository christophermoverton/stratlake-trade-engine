# Milestone 12 Alpha & Portfolio Workflow Example

Run the example with:

```bash
python docs/examples/milestone_12_alpha_portfolio_workflow.py
```

This example uses the committed deterministic dataset at
`docs/examples/data/milestone_12_alpha_portfolio_dataset.csv` and exercises:

* alpha training with `train_alpha_model(...)`
* alpha prediction with `predict_alpha_model(...)`
* fixed and rolling alpha time splits
* cross-section inspection with `get_cross_section(...)` and `list_cross_section_timestamps(...)`
* temporary sign mapping with `signal = np.sign(prediction_score)`
* continuous-signal backtesting through `run_backtest(...)`
* portfolio construction with and without volatility targeting
* portfolio artifact writing through `write_portfolio_artifacts(...)`

Outputs are written to:

```text
docs/examples/output/milestone_12_alpha_portfolio_workflow/
```

Key files:

* `summary.json`
* `predictions.csv`
* `cross_section.csv`
* `single_symbol_backtest.csv`
* `portfolio_returns_matrix.csv`
* `artifacts/baseline/`
* `artifacts/targeted/`

The example keeps the signal mapping intentionally simple for Milestone 12:

```python
signal = np.sign(prediction_score)
```

It also passes the raw `prediction_score` into the backtest as a continuous
exposure so the example demonstrates proportional return scaling alongside the
requested sign-based mapping.
