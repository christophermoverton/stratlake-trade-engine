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
