# M34 Release Notes

Milestone title: `M34 - Statistical Robustness & Overfitting Guardrails`

Milestone 34 adds deterministic robustness evidence for research review. The
framework makes statistical trust questions visible through portable artifacts:
does in-sample performance transfer out of sample, is the sample large enough,
is the selected configuration fragile, how large was the search space, and were
temporal validation splits protected against leakage?

The milestone extends StratLake's artifact-first research architecture. It does
not add a competing execution engine and does not silently change promotion
decisions.

## Major Capabilities

* Issue 384: canonical robustness report bundle with schema-stable JSON, CSV,
  Markdown, and manifest artifacts.
* Issue 385: Walk-Forward Efficiency diagnostics for in-sample to
  out-of-sample transfer.
* Issue 386: sample-size and trade-count guardrails, including missing metadata
  findings.
* Issue 387: parameter sensitivity and fragility analysis for local robustness
  around selected configurations.
* Issue 388: multiple-testing and trial-count metadata so reviewers can see
  when results were selected from large search spaces.
* Issue 389: deterministic purged and embargoed temporal-validation primitives
  with split-plan and leakage-validation artifacts.
* Issue 390: read-only promotion governance integration that surfaces
  robustness evidence as context.
* Issue 391: documentation, CI-safe examples, and release-readiness validation.

## Artifact Inventory

Canonical robustness reports contain:

* `robustness_summary.json`
* `robustness_findings.json`
* `walk_forward_efficiency.csv`
* `sample_size_validation.json`
* `sensitivity_summary.csv`
* `multiple_testing_summary.json`
* `robustness_report.md`
* `manifest.json`

Temporal-validation companion artifacts contain:

* `purged_split_plan.json`
* `purged_split_summary.csv`
* `leakage_validation.json`

## Governance Boundary

Promotion governance asks what decision or outcome was recorded. Statistical
robustness asks whether the research evidence is trustworthy. M34 connects
those views by adding governance-visible robustness fields and reason codes, but
robustness findings remain review evidence unless a future explicit policy
layer chooses to enforce them.

M34 does not replay promotion gates, recompute `promotion_status`, or reject
candidates automatically.

## Validation Commands

Focused M34 validation:

```bash
python -m pytest tests/test_robustness_schema.py
python -m pytest tests/test_robustness_walk_forward_efficiency.py
python -m pytest tests/test_robustness_sample_size.py
python -m pytest tests/test_robustness_sensitivity.py
python -m pytest tests/test_robustness_multiple_testing.py
python -m pytest tests/test_robustness_purged_splits.py
python -m pytest tests/test_robustness_governance_integration.py
python -m pytest tests/test_robustness_example.py
```

Example and portability validation:

```bash
python docs/examples/robustness_report_example.py
python -m pytest tests/test_docs_path_portability.py
```

Regression validation:

```bash
python -m pytest tests/test_promotion_governance.py tests/test_promotion_governance_integration.py
python -m pytest tests/test_research_robustness.py tests/test_extended_robustness.py
ruff check src/research/robustness tests docs/examples
python -m pytest
```

## Preserved M33 Guarantees

M34 keeps the M33 portability posture intact:

* no local absolute paths in guarded docs or generated robustness artifacts
* deterministic JSON and CSV output
* LF-normalized generated text artifacts
* CI-safe examples with no external data or services
* compatibility with Windows, Ubuntu, and macOS smoke validation
* no package metadata or release workflow weakening

## Limitations And Non-Goals

M34 does not implement:

* Deflated Sharpe Ratio
* Probability of Backtest Overfitting
* full Combinatorial Purged Cross-Validation
* statistical haircuts
* LLM black-swan or synthetic macro simulation
* a new optimization loop
* automatic promotion rejection based on robustness findings

Those remain future or optional extension points. They should only be claimed
after their assumptions, inputs, deterministic implementation, artifact
contracts, and tests are represented.

## Further Reading

* `docs/m34_statistical_robustness_architecture.md`
* `docs/m34_robustness_validation_examples.md`
* `docs/m34_release_validation_checklist.md`
* `docs/examples/robustness_report_example.py`
