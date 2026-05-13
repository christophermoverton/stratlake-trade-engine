# M34 Release Validation Checklist

This checklist documents the release-readiness checks for Milestone 34. It does
not replace existing CI or release automation.

## Pre-Merge Focused Checks

Run the robustness unit slice:

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

Run the CI-safe example and docs/path portability checks:

```bash
python docs/examples/robustness_report_example.py
python -m pytest tests/test_docs_path_portability.py
```

Run governance regressions:

```bash
python -m pytest tests/test_promotion_governance.py tests/test_promotion_governance_integration.py
```

Run existing robustness runner regressions:

```bash
python -m pytest tests/test_research_robustness.py tests/test_extended_robustness.py
```

Run lint over the changed robustness, test, and example surfaces:

```bash
ruff check src/research/robustness tests docs/examples
```

## Full Validation

Run the complete suite before merge:

```bash
python -m pytest
```

Known non-blocking warnings may appear in the full pytest run from legacy
fixture scenarios, including low sample-size warnings, degenerate signal
warnings, or open matplotlib figure warnings. These warnings do not block M34
release validation when the full suite is green, but any new warning class
introduced by M34 should be investigated before release.

If package or release metadata changes are included in the same branch, also run
the existing M33 packaging and cross-platform validation workflow rather than
duplicating it locally.

## Cross-Platform CI Smoke

Confirm the M33 cross-platform smoke posture remains green on supported CI
targets:

* installability on Windows, Ubuntu, and macOS
* import smoke
* docs/path lint
* line-ending policy
* package metadata validation
* local wheel and source distribution build where the release workflow runs it
* promotion-governance reporting smoke
* campaign milestone reporting smoke

## Artifact Review

Inspect the M34 example output directory after running the example:

```text
docs/examples/output/robustness_report_example/
```

Expected files:

* `robustness_summary.json`
* `robustness_findings.json`
* `walk_forward_efficiency.csv`
* `sample_size_validation.json`
* `sensitivity_summary.csv`
* `multiple_testing_summary.json`
* `robustness_report.md`
* `manifest.json`
* `purged_split_plan.json`
* `purged_split_summary.csv`
* `leakage_validation.json`

Confirm generated artifacts contain portable paths, deterministic timestamps or
explicit fixed timestamps, no non-finite JSON values, and no external data
dependencies.

## Post-Merge Checks

After merge:

* verify primary CI is green
* verify cross-platform smoke CI is green
* run the M34 example from a clean checkout if release notes reference it
* confirm docs/path portability still passes
* confirm no generated machine-specific paths are committed

## Release Tag Checklist

Before creating a release tag:

* confirm `docs/m34_release_notes.md` reflects the merged feature set
* confirm advanced methods remain marked future unless separately implemented
* use a single space after the issue prefix in final release commits, for
  example: `Issue 391: docs(robustness): add M34 examples validation and release notes`
* confirm release automation remains tag-driven and least-privilege
* confirm package/build validation matches the existing M33 release workflow
* confirm no promotion governance behavior changed without an explicit policy
  issue
