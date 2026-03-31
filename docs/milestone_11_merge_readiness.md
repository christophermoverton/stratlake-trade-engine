# Milestone 11 Merge Readiness

## Scope Delivered

Milestone 11 documentation now reflects the repository's current deterministic
research and portfolio workflow, including:

* portfolio optimization
* portfolio risk summaries and volatility-targeting diagnostics
* deterministic return simulation
* execution-friction modeling with fixed fees and slippage models
* robustness workflow context
* artifact and registry behavior for portfolio runs
* CLI examples for baseline, optimizer-aware, risk-aware, simulation-enabled,
  execution-aware, and walk-forward portfolio workflows

## Docs Updated

Updated:

* [../README.md](../README.md)
* [portfolio_configuration.md](portfolio_configuration.md)
* [portfolio_construction_workflow.md](portfolio_construction_workflow.md)

Added:

* [milestone_11_portfolio_workflow.md](milestone_11_portfolio_workflow.md)
* [milestone_11_merge_readiness.md](milestone_11_merge_readiness.md)

## Validation Performed

The documentation was checked for:

* relative link portability in the updated Milestone 11 docs
* absence of Windows absolute-path Markdown links in the updated docs
* CLI example alignment with `src.cli.run_portfolio` and `src.cli.run_strategy`
  argument parsing
* referenced config and documentation file existence
* consistency with optimizer, risk, simulation, execution, artifact, and
  registry behavior currently implemented in code

## Known Limits Intentionally Left For Future Milestones

Documented current limits include:

* optimizer supports long-only allocation only
* simulation does not ship with a default `configs/simulation.yml`
* simulation is not supported for strategy walk-forward, strategy robustness,
  or portfolio walk-forward runs
* volatility targeting is now available as an optional post-optimizer scaling
  step; leverage caps and regime-aware targeting remain future work
* fixed-fee support currently exposes only the `per_rebalance` model
