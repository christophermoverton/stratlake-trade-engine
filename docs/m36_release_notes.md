# M36 Release Notes - Scalable Evidence Interoperability and Release Hardening

Milestone title: `M36 - Scalable Evidence Interoperability and Release Hardening`

M36 branch:
`feature/m36-scalable-evidence-interoperability-release-hardening`

Candidate milestone release tag:
`v0.36.0-scalable-evidence-interoperability-release-hardening`

## Milestone Principle

Scalable evidence systems should make research artifacts faster to find, easier
to export, and safer to release without weakening deterministic artifact
provenance.

## Issue #402 Scope

Issue #402 is a release and process hardening pass. It prepares M36 for later
catalog interoperability work by clarifying version policy, milestone branch
naming, and milestone-validation workflow coverage.

This issue does not add catalog features, derived indexes, lineage export,
dataset or feature fingerprints, artifact contract migrations, promotion policy
changes, or governance decision changes.

## Version Policy

Package version and milestone release tags are intentionally separate.

The package version in `pyproject.toml` is Python distribution metadata. It is
used by editable installs, package metadata checks, wheel/sdist build
validation, and installed import metadata. For M36 it remains `0.1.0` because
Issue #402 does not change package distribution semantics or publish the package
to PyPI/TestPyPI.

Milestone release tags identify repository snapshots and release-validation
evidence. M36 milestone release tags should use the `v0.36.0-<slug>` form, with
`v0.36.0-scalable-evidence-interoperability-release-hardening` as the candidate
tag. These tags drive `.github/workflows/release.yml`, create GitHub Releases,
and attach release-validation evidence. They do not imply that the Python
package version has changed.

Future milestones should update milestone release notes, validation checklists,
and candidate tag names for the milestone. They should preserve
`pyproject.toml` unless the milestone intentionally changes package metadata,
install behavior, or distribution compatibility.

## Branch Naming

M36 uses:

```text
feature/m36-scalable-evidence-interoperability-release-hardening
```

Future milestone branches should use:

```text
feature/m<NUMBER>-<short-kebab-description>
```

The milestone validation workflow covers this `feature/m*` convention while
preserving legacy `milestone/**` and `m22/**` branch support.

## Validation Workflow Coherence

`.github/workflows/milestone_validation.yml` keeps manual dispatch support and
continues to run the existing milestone validation bundle job. Its branch
trigger coverage now includes:

- `milestone/**`
- `m22/**`
- `feature/m*`

The pull request job guard also accepts `feature/m` source branches, so pull
requests from
`feature/m36-scalable-evidence-interoperability-release-hardening` into `main`
can run milestone validation.

## GitHub Actions Supply-Chain Hardening

Issue #403 pins every external GitHub Action reference in `.github/workflows/`
to a full commit SHA. This reduces exposure to mutable tag movement while
keeping the human-readable upstream tag context in workflow comments.

Current workflow action inventory:

| Action | Classification | Pinned SHA | Upstream tag represented |
| --- | --- | --- | --- |
| `actions/checkout` | GitHub-maintained | `34e114876b0b11c390a56381ad16ebd13914f8d5` | `v4` |
| `actions/setup-python` | GitHub-maintained | `a26af69be951a213d495a4c3e4e4022e16d87065` | `v5` |
| `actions/upload-artifact` | GitHub-maintained | `ea165f8d65b6e75b540449e92b4886f43607fa02` | `v4` |
| `softprops/action-gh-release` | Third-party | `3bb12739c298aeb8a4eeaf626c5b8d85266b0e65` | `v2` |

There are no local reusable actions in the current workflow set. There are no
intentionally unpinned external action references in the current workflow set.

To refresh a pinned action safely:

1. Review the upstream release/tag page and the exact commit represented by the
   desired tag.
2. Replace the full SHA and update the nearby tag-to-SHA comment in every
   affected workflow.
3. Re-run workflow YAML parsing, workflow pinning tests, focused workflow tests,
   docs/path lint, package build validation, and full pytest when practical.
4. Re-check that workflow names, job names, matrices, install commands, release
   artifacts, and milestone trigger coverage remain unchanged.

SHA pinning is part of M36 release hardening. It strengthens workflow provenance
without changing package publication scope, release tag semantics, catalog
behavior, artifact contracts, or governance decisions.

## Catalog Scale Baselines

Issue #404 adds deterministic scale baselines before any optional derived
metadata index work. The new synthetic test fixture builds a compact temporary
artifact history spanning strategy, alpha, portfolio, campaign/scenario,
robustness, governance, milestone-validation, release-validation, sparse, and
registry-only cases.

The required scale tests measure the existing direct-scan workflow only:

- catalog indexing over the synthetic artifact root
- evidence query filters
- lineage edge extraction
- explorer JSON, Markdown, and table rendering
- notebook/API helper views and renderers

Deterministic assertions cover record counts, family counts, query counts,
lineage edge counts, ordering, path portability, and source immutability. A
single intentionally broad elapsed-time ceiling protects against accidental
pathological regressions while avoiding brittle micro-benchmark behavior; exact
timings are environment-dependent and are not treated as deterministic output.

Issue #404 does not add a derived metadata index, database, cache, graph store,
search backend, or alternate canonical registry. Later M36 optimization work
should compare against these baselines rather than guessing at scale behavior.

## Release Notes Semantics

Human milestone release notes live in milestone docs such as this file. The
Release workflow renders deterministic GitHub Release body text from the pushed
tag name and validation steps. Both are release evidence, but neither changes
catalog records, artifact contracts, package publication scope, promotion
outcomes, or governance decisions.

## Further Reading

- `docs/m36_release_validation_checklist.md`
- `docs/m35_release_notes.md`
- `docs/m35_release_validation_checklist.md`
- `.github/workflows/milestone_validation.yml`
- `.github/workflows/release.yml`
- `pyproject.toml`
