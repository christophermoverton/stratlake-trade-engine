# Real Alpha Workflow Example

Run the example with:

```bash
python docs/examples/real_alpha_workflow.py
```

This is the primary end-to-end alpha example in the repository. It stays small
enough to read in one sitting, but it follows the real built-in research path:

* generates a deterministic `features_daily` fixture inside the example workspace
* selects the built-in `rank_composite_momentum` alpha from `configs/alphas.yml`
* trains, predicts, evaluates, and persists registry-backed alpha artifacts via
  `python -m src.cli.run_alpha`
* maps predictions into tradable signals with `top_bottom_quantile`
* writes sleeve artifacts with `sleeve_returns.csv`, `sleeve_equity_curve.csv`,
  and `sleeve_metrics.json`
* optionally constructs a downstream portfolio that consumes the alpha sleeve as
  an `artifact_type: alpha_sleeve` component
* writes unified review artifacts that rank the alpha run and linked portfolio

Outputs are written to:

```text
docs/examples/output/real_alpha_workflow/
```

The example summary lives at:

```text
docs/examples/output/real_alpha_workflow/summary.json
```

Key artifact groups:

* `workspace/artifacts/alpha/<run_id>/`
* `workspace/artifacts/portfolios/<run_id>/`
* `review/`

What to look at:

* `summary.json` for the full stitched workflow
* `workspace/artifacts/alpha/<run_id>/alpha_metrics.json` for evaluation
* `workspace/artifacts/alpha/<run_id>/signals.parquet` for alpha-to-signal mapping
* `workspace/artifacts/alpha/<run_id>/signal_semantics.json` for the explicit typed-signal contract
* `workspace/artifacts/alpha/<run_id>/sleeve_returns.csv` for the alpha sleeve
* `workspace/artifacts/portfolios/<run_id>/portfolio_returns.csv` for portfolio integration
* `review/leaderboard.csv` and `review/review_summary.json` for downstream review outputs

Workflow sketch:

```text
features_daily
    ->
built-in alpha catalog
    ->
predict + evaluate
    ->
signal mapping
    ->
alpha sleeve
    ->
optional portfolio
    ->
review artifacts
```

The example uses the same first-class entrypoints you would use in normal
research work:

```bash
python -m src.cli.run_alpha --alpha-name rank_composite_momentum --start 2025-01-01 --end 2025-01-30 --train-start 2025-01-01 --train-end 2025-01-18 --predict-start 2025-01-18 --predict-end 2025-01-25 --signal-policy top_bottom_quantile --signal-quantile 0.34
python -m src.cli.run_portfolio --portfolio-config path/to/portfolio_from_alpha_sleeve.yml --timeframe 1D
```

That makes it a better starting point than the older custom-model example when
you want to understand how a built-in alpha moves through evaluation, sleeve
generation, portfolio consumption, and review.
