# Milestone 16 Merge Readiness

This checklist defines the validation, documentation, and release-tag
preparation work required before merging the Milestone 16 campaign
orchestration branch into `main`.

The goal is to confirm that the repository tells one coherent post-merge story:
campaign orchestration sits above the existing alpha, comparison,
candidate-selection, portfolio, and review flows without changing their
deterministic artifact contracts.

## Scope Delivered

This merge-readiness pass covers:

* unified research campaign config loading in `src/config/research_campaign.py`
* campaign orchestration in `python -m src.cli.run_research_campaign`
* fail-fast preflight validation and persisted `preflight_summary.json`
* stitched campaign `campaign_config.json`, `manifest.json`, and `summary.json`
* registry-backed downstream chaining for candidate-selection and portfolio
  artifacts
* campaign-aware candidate review and unified review handoff
* committed Milestone 16 example configs and the real-world campaign case study
* README and docs alignment with the current campaign workflow, outputs, and
  release story

## Docs Updated

Review these docs together before merge:

* [../README.md](../README.md)
* [milestone_16_campaign_workflow.md](milestone_16_campaign_workflow.md)
* [research_campaign_configuration.md](research_campaign_configuration.md)
* [examples/milestone_16_campaign_workflow.md](examples/milestone_16_campaign_workflow.md)
* [examples/real_world_campaign_case_study.md](examples/real_world_campaign_case_study.md)
* [milestone_15_candidate_selection_issue_1.md](milestone_15_candidate_selection_issue_1.md)
* [milestone_13_research_review_workflow.md](milestone_13_research_review_workflow.md)
* [milestone_16_merge_readiness.md](milestone_16_merge_readiness.md)

## Validation Performed

Merge readiness should be grounded in repository behavior already covered by
code and committed examples:

* campaign config resolution and inheritance in `src/config/research_campaign.py`
* campaign stage orchestration, preflight, and stitched artifacts in
  `src/cli/run_research_campaign.py`
* alpha, comparison, candidate-selection, portfolio, and review handoff through
  the existing CLI entrypoints
* registry-backed dependency resolution for downstream campaign stages
* deterministic example reruns and stable stitched artifact payloads in the
  Milestone 16 test coverage

## Known Limits Intentionally Deferred

Current Milestone 16 limits that should remain explicit at merge time:

* campaign orchestration coordinates existing CLIs; it does not replace
  stage-local artifact contracts
* review remains registry-backed; the campaign runner does not invent a new
  review schema
* strategy execution inside campaigns still depends on valid strategy configs
  and available feature datasets
* registry-chained downstream workflows require unique matching entries; the
  runner now fails on ambiguous or incomplete registry state instead of picking
  a latest run heuristically
* the default repository campaign config is a template, not a guaranteed
  runnable production campaign

## Pre-Merge Validation

Run all commands from the repository root with the project virtual environment
activated.

### 1. Milestone 16 targeted automated validation

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests\test_research_campaign_config.py `
  tests\test_cli_run_research_campaign.py `
  tests\test_candidate_selection_portfolio_case_study_example.py `
  tests\test_real_alpha_workflow_example.py `
  tests\test_milestone_13_review_promotion_workflow.py
```

Pass criteria:

* campaign config inheritance and explicit validation pass
* campaign stage ordering, preflight, and stitched artifact coverage pass
* downstream candidate-selection and portfolio chaining remains deterministic
* existing Milestone 13 and 15 downstream example coverage still passes

### 2. Real-world campaign case study smoke

```powershell
.\.venv\Scripts\python.exe docs\examples\real_world_campaign_case_study.py
```

Pass criteria:

* the example writes one stitched summary under
  `docs/examples/output/real_world_campaign_case_study/summary.json`
* the nested campaign artifact directory includes `campaign_config.json`,
  `preflight_summary.json`, `manifest.json`, and `summary.json`
* the campaign completes end to end across alpha evaluation, comparison,
  candidate selection, portfolio, candidate review, and unified review

### 3. Documentation alignment checks

Review the Milestone 16 docs against:

* `src/cli/run_research_campaign.py`
* `src/config/research_campaign.py`
* `docs/examples/data/milestone_16_campaign_configs/full_campaign.yml`
* `docs/examples/data/milestone_16_campaign_configs/registry_chained_campaign.yml`
* `docs/examples/data/milestone_16_campaign_configs/real_world_campaign.yml`

Check that:

* documented config keys use the current canonical names
* campaign stage order and optional-stage behavior match the implementation
* committed example-config links remain relative and resolve inside the repo
* README references the Milestone 16 workflow, case study, artifacts, and merge
  readiness checklist consistently

### 4. Branch integration validation

Before merge, confirm the branch still integrates cleanly with `main`:

```powershell
git checkout main
git merge --no-ff --no-commit codex/milestone-16-research-campaign-orchestration
git merge --abort
git checkout codex/milestone-16-research-campaign-orchestration
```

Pass criteria:

* no merge conflicts
* no unrelated files appear during the merge simulation
* abort returns `main` to a clean pre-merge state

If local policy prefers a read-only check, use `git diff --stat main...HEAD`
plus an equivalent merge simulation in CI before the final merge.

## Release Tag Preparation

Recommended Milestone 16 tag name:

* `v0.16.0-research-campaign-orchestration`

Tag-readiness criteria:

* targeted Milestone 16 validation is green
* the real-world campaign case study has been rerun successfully on the branch
* branch integration validation against `main` is clean
* README and supporting docs describe the merged campaign workflow accurately
* no unresolved documentation or artifact-contract drift remains

Create the tag only after the merge commit is complete and post-merge smoke
passes on `main`.

## Post-Merge Verification

After the merge, rerun:

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests\test_research_campaign_config.py `
  tests\test_cli_run_research_campaign.py
.\.venv\Scripts\python.exe docs\examples\real_world_campaign_case_study.py
git status --short
```

Pass criteria:

* the Milestone 16 validation slice still passes on `main`
* the real-world campaign case study still produces the expected stitched
  outputs
* the working tree is clean except for intentionally generated local artifacts

## Failure Handling

If any step fails:

1. stop the merge flow immediately
2. keep `main` unchanged
3. fix the issue on the Milestone 16 branch
4. rerun the workflow from the beginning

Do not tag or merge on partial validation, ambiguous registry state, or docs
that no longer match the actual CLI and artifact surface.
