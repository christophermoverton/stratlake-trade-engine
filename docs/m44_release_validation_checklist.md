# M44 Release Validation Checklist

This checklist documents release-readiness checks for Milestone 44. It does
not replace existing CI, milestone validation, or release automation.

Milestone title:
`M44 - Colab Notebook Session Ergonomics and MarketLake Restore Handoff`

M44 branch:
`feature/m44-unified-colab-session-drive-persistence`

Target branch:
`main`

Issue range covered:
`#483` through `#491`

Candidate milestone release tag:
`v0.44.0-colab-notebook-session-ergonomics-marketlake-handoff`

Package/build version:
`0.44.0`

## M44 Scope Recap

M44 adds release-ready notebook ergonomics for:

* `stratlake-init-session --notebook-configs`
* unified Colab profile variables
* `stratlake-validate-marketlake-handoff`
* restore-first Colab archive workflow documentation
* `stratlake-notebook-doctor`
* notebook-native `src.execution` examples after restore/init
* unified persistence guidance across fintech and StratLake
* fintech restore-to-local handoff guidance
* M44 branch validation workflow coverage

## Architecture Boundaries

Confirm validation and docs preserve:

* local runtime roots as active working state
* mounted Drive paths as filesystem persistence/transport only
* archive and backup packs as derived and non-canonical
* notebook/session metadata as diagnostic/session state
* MarketLake handoff reports as diagnostic
* notebook `src.execution` APIs as interfaces over existing deterministic
  workflows
* no second execution engine
* no hidden sync
* no Google API or OAuth behavior
* no restore-on-import behavior
* no background persistence
* no fintech ingestion behavior added to StratLake

## Required Validation Commands

Focused M44 tests:

```bash
python -m pytest tests/test_m44_colab_session_profile_docs.py -q
python -m pytest tests/test_m44_marketlake_handoff_docs.py tests/test_validate_marketlake_handoff.py -q
python -m pytest tests/test_m44_restore_first_colab_docs.py -q
python -m pytest tests/test_notebook_doctor.py tests/test_m44_notebook_doctor_docs.py -q
python -m pytest tests/test_m44_notebook_execution_examples_docs.py -q
python -m pytest tests/test_m44_unified_persistence_guide_docs.py -q
python -m pytest tests/test_m44_fintech_handoff_persistence_docs.py -q
```

M44 workflow-equivalent pytest slice:

```bash
python -m pytest tests/test_validate_marketlake_handoff.py tests/test_notebook_doctor.py tests/test_m44_marketlake_handoff_docs.py tests/test_m44_notebook_doctor_docs.py tests/test_m44_colab_session_profile_docs.py tests/test_m44_restore_first_colab_docs.py tests/test_m44_notebook_execution_examples_docs.py tests/test_m44_unified_persistence_guide_docs.py tests/test_m44_fintech_handoff_persistence_docs.py tests/test_init_session_cli.py tests/test_m44_release_readiness_docs.py -q
```

Focused Ruff checks:

```bash
python -m ruff check src/validation/marketlake_handoff.py src/validation/notebook_doctor.py src/cli/validate_marketlake_handoff.py src/cli/notebook_doctor.py tests/test_validate_marketlake_handoff.py tests/test_notebook_doctor.py tests/test_m44_marketlake_handoff_docs.py tests/test_m44_notebook_doctor_docs.py tests/test_m44_colab_session_profile_docs.py tests/test_m44_restore_first_colab_docs.py tests/test_m44_notebook_execution_examples_docs.py tests/test_m44_unified_persistence_guide_docs.py tests/test_m44_fintech_handoff_persistence_docs.py tests/test_m44_release_readiness_docs.py

python -m ruff format --check src/validation/marketlake_handoff.py src/validation/notebook_doctor.py src/cli/validate_marketlake_handoff.py src/cli/notebook_doctor.py tests/test_validate_marketlake_handoff.py tests/test_notebook_doctor.py tests/test_m44_marketlake_handoff_docs.py tests/test_m44_notebook_doctor_docs.py tests/test_m44_colab_session_profile_docs.py tests/test_m44_restore_first_colab_docs.py tests/test_m44_notebook_execution_examples_docs.py tests/test_m44_unified_persistence_guide_docs.py tests/test_m44_fintech_handoff_persistence_docs.py tests/test_m44_release_readiness_docs.py
```

Docs/path lint:

```bash
python -m src.cli.run_docs_path_lint
```

Broader checks:

```bash
python -m ruff check src tests examples
python -m pytest -q
python -m build
git diff --check
```

Generated-output cleanup checks:

```bash
git status --short
git ls-files docs/examples/output/m38_static_evidence_review_pack_example
```

If full pytest recreates
`docs/examples/output/m38_static_evidence_review_pack_example/`, remove it
locally unless that output is part of a separate intentional change.

## Local Validation Results

Issue #491 release-readiness validation results from
`feature/m44-unified-colab-session-drive-persistence`:

* refreshed editable install:
  `stratlake-trade-engine==0.44.0`
* focused release-readiness metadata tests:
  `16 passed`
* M44 workflow-equivalent pytest slice:
  `63 passed`
* focused M44 Ruff check:
  `All checks passed!`
* focused M44 Ruff format check:
  `15 files already formatted`
* docs/path lint:
  `docs_path_lint_status: passed`, `finding_count: 0`
* repository-wide Ruff:
  `All checks passed!`
* full pytest:
  `2527 passed, 6 skipped, 348 warnings`
* package build:
  built `stratlake_trade_engine-0.44.0.tar.gz` and
  `stratlake_trade_engine-0.44.0-py3-none-any.whl`
* generated-output cleanup:
  removed untracked
  `docs/examples/output/m38_static_evidence_review_pack_example/`;
  `git ls-files docs/examples/output/m38_static_evidence_review_pack_example`
  returned no tracked files

Build caveat:
`python -m build` emitted the existing setuptools deprecation warning for
`project.license` table metadata. The build completed successfully.

## PR-Ready Metadata

Recommended PR title:
`M44: Colab notebook session ergonomics and MarketLake handoff`

Recommended PR description summary:

```markdown
Completes M44 - Colab Notebook Session Ergonomics and MarketLake Restore
Handoff. Adds notebook config bundle generation, unified Colab profile
guidance, MarketLake handoff validation, notebook doctor diagnostics,
restore-first Colab archive documentation, notebook-native execution examples,
unified persistence guidance, fintech restore-to-local handoff guidance, and
branch validation coverage while preserving Drive/archive/non-canonical
boundaries.
```

## Post-Merge Validation Checklist

* confirm the merged branch targets `main`
* confirm package/build version `0.44.0`
* confirm release tag candidate
  `v0.44.0-colab-notebook-session-ergonomics-marketlake-handoff`
* run the M44 workflow-equivalent pytest slice
* run focused Ruff and format checks
* run docs/path lint and confirm `finding_count: 0`
* run full pytest or document unrelated failures
* confirm no unrelated generated output is included
