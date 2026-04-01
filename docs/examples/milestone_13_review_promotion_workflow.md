# Milestone 13 Review And Promotion Workflow Example

This example shows how small completed alpha, strategy, and portfolio artifacts
flow into one deterministic review surface and one review-level promotion
decision.

Use this as the small committed companion to the main workflow guide:
[../milestone_13_research_review_workflow.md](../milestone_13_research_review_workflow.md).

## Run It

```powershell
python docs/examples/milestone_13_review_promotion_workflow.py
```

The example reads checked-in fixture registries and artifact manifests from:

```text
docs/examples/data/milestone_13_review_inputs/
```

It writes deterministic outputs under:

```text
docs/examples/output/milestone_13_review_promotion_workflow/
```

## What It Demonstrates

* loading completed alpha, strategy, and portfolio runs from registry fixtures
* carrying forward per-run review or promotion status into the unified review
  leaderboard
* writing stable review artifacts:
  * `leaderboard.csv`
  * `review_summary.json`
  * `manifest.json`
  * `promotion_gates.json`
  * `summary.json`
* evaluating review-level promotion gates from aggregate review metrics

## Why This Example Is Small

The inputs are intentionally tiny and text-based so they are easy to inspect in
git, docs, and tests. Plot generation is disabled to keep the committed output
set compact and deterministic.

Intentional limits in this example:

* registry-backed review only; it does not execute fresh research runs
* one selected row per run type because `top_k_per_type=1`
* no plots because `emit_plots` is disabled
