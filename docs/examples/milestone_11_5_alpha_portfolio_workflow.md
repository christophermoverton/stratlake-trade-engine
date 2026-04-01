# Milestone 11.5 Alpha & Portfolio Workflow Example

Run the example with:

```bash
python docs/examples/milestone_11_5_alpha_portfolio_workflow.py
```

This example uses the committed deterministic dataset at
`docs/examples/data/milestone_12_alpha_portfolio_dataset.csv` and exercises:

* alpha model registration through `register_alpha_model(...)`
* alpha training with `train_alpha_model(...)`
* alpha prediction with `predict_alpha_model(...)`
* fixed and rolling alpha time splits
* cross-section inspection with `get_cross_section(...)` and `list_cross_section_timestamps(...)`
* explicit signal mapping with `signal = np.sign(prediction_score)`
* continuous-signal backtesting by passing raw `prediction_score` as exposure
  into `run_backtest(...)`
* portfolio construction with and without volatility targeting
* portfolio artifact writing through `write_portfolio_artifacts(...)`

Outputs are written to:

```text
docs/examples/output/milestone_11_5_alpha_portfolio_workflow/
```

Key files:

* `summary.json`
* `predictions.csv`
* `cross_section.csv`
* `single_symbol_backtest.csv`
* `portfolio_returns_matrix.csv`
* `artifacts/baseline/`
* `artifacts/targeted/`

What the outputs show:

* `predictions.csv` contains `prediction_score`, sign-mapped `signal`, and
  `continuous_signal`
* `cross_section.csv` shows one deterministic same-timestamp asset slice
* `single_symbol_backtest.csv` demonstrates proportional return scaling from
  continuous exposure
* `artifacts/baseline/` keeps equal-weight portfolio outputs without
  operational targeting
* `artifacts/targeted/` shows the same portfolio after deterministic
  volatility-target scaling

This example is Milestone 11.5 material because it focuses on alpha modeling,
cross-sectional review, continuous-signal backtesting, and portfolio
construction rather than alpha-evaluation metrics or registry-backed
leaderboards.

```python
signal = np.sign(prediction_score)
```

It also passes the raw `prediction_score` into the backtest as a continuous
exposure so the example demonstrates proportional return scaling alongside the
requested sign-based mapping.
