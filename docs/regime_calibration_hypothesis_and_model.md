# Regime Calibration Hypothesis And Model

## Purpose

This document explains the conceptual layer behind Milestone 25 regime
calibration.

It is separate from the taxonomy definition in
`docs/regime_taxonomy.md` and separate from the implementation contract in
`docs/regime_calibration.md`.

Use this document when the question is not "what does the code do" but
instead:

* what is the calibration layer assuming?
* what problem is it trying to solve?
* what is the operating model behind the profile settings?

## Core Hypothesis

The core hypothesis of regime calibration is:

> the first-pass regime classifier is useful as a descriptive market-state
> surface, but many downstream decisions benefit from a second deterministic
> layer that controls sensitivity, persistence, stability, and evidence
> sufficiency.

That hypothesis has several parts.

### Hypothesis 1: Raw Regime Labels Can Be Too Reactive

The Milestone 24 classifier is designed to be auditable and deterministic. It
assigns labels directly from observable rolling metrics.

That is valuable for interpretation, but direct metric-threshold labeling can
produce short-lived changes when the market moves near a boundary. In other
words, the taxonomy may be correct while the observed path is still too noisy
for some consumers.

Calibration assumes that downstream users often care about the stability of the
label stream, not just the local correctness of each single timestamp.

### Hypothesis 2: Stability Is A Decision Variable

The right amount of sensitivity depends on the workflow.

Examples:

* an attribution report may prefer fewer, more persistent regimes.
* a stress monitor may prefer earlier detection, even at the cost of more
  reversals.
* a portfolio optimizer may want unstable segments explicitly downgraded.

Calibration therefore treats stability as configurable policy rather than as a
fixed property of the taxonomy itself.

### Hypothesis 3: Sparse And Unstable Regimes Should Be Explicitly Gated

A label can be valid in the taxonomy but still weak for downstream analysis.

Examples:

* a regime appears only a few times and is too small for attribution.
* the sequence flips repeatedly and should not drive scaling or optimization.
* confidence is low and the label stream should be treated cautiously.

Calibration assumes these cases should be surfaced explicitly through metrics,
flags, and optional fallback behavior instead of being left implicit.

### Hypothesis 4: Calibration Must Remain Deterministic And Causal

Calibration is not intended to be a hidden-state model, a Bayesian smoother, or
an ML confidence estimator.

The working hypothesis is that the second layer should remain:

* deterministic
* auditable
* dependency-light
* causal with no lookahead leakage
* compatible with the existing taxonomy contract

This is why the implementation uses profile settings, trailing smoothing,
minimum-duration rules, and explicit threshold gating rather than probabilistic
inference.

## Calibration Model

The calibration model is a deterministic policy model applied to an existing
regime-label time series.

It is easiest to think of it as a four-part model.

## 1. Input Model

The input is not raw market data. The input is already-classified canonical
regime output.

That means the model consumes:

* timestamp-ordered regime labels
* canonical dimension states
* the composite regime label
* `is_defined`
* optional confidence values supplied by the caller

This is important because calibration is not trying to learn the market state.
It is trying to shape how the already-defined state stream should be used.

## 2. State-Persistence Model

The model assumes that regime usefulness depends partly on persistence.

Two mechanisms encode that idea.

### Trailing Smoothing

Trailing smoothing asks whether the most recent local history supports the
current state choice. The smoothing window is causal: it only looks at the
current row and prior rows.

Interpretation:

* short windows preserve responsiveness.
* longer windows prefer local consensus over instantaneous change.

### Minimum Duration Gate

The minimum-duration rule asks whether a candidate change has persisted long
enough to be accepted as the new active regime.

Interpretation:

* lower duration thresholds accept faster shifts.
* higher duration thresholds reject brief reversals and absorb noise.

Together, these rules form the persistence model: a regime is more actionable
when it is not merely observed, but observed with enough temporal support.

## 3. Stability-Risk Model

The model treats certain sequence patterns as instability risk.

Instability is measured through summary statistics such as:

* transition count
* flip rate
* single-day flip count
* single-day flip share
* minimum, median, and average regime duration
* unstable regime share
* low-confidence share, when available

These are not taxonomy facts. They are behavioral properties of the regime
stream.

The model assumption is:

> if the calibrated sequence exceeds configured instability thresholds, then
> downstream consumers should be warned or optionally moved to a fallback state.

This is what makes the output decision-ready rather than purely descriptive.

## 4. Evidence-Sufficiency Model

The model also assumes that a regime should not automatically qualify for every
downstream use merely because it exists.

Two related questions matter:

* does the regime have enough total observations to be reviewable?
* does it have enough observations, and enough stability, to support
  attribution?

This is why calibration profiles include:

* `min_observations_per_regime`
* `min_observations_for_attribution`
* `require_stability_for_attribution`

The working model is simple: evidence sufficiency is part of the regime usage
decision, not part of the taxonomy definition.

## Fallback Model

Fallback behavior is the explicit action layer of calibration.

The model does not invent new labels. Instead, it routes weak or unsafe states
to an existing supported composite label when the profile says that is the
right operational behavior.

Fallbacks can be used for:

* undefined rows
* low-confidence rows
* globally unstable calibrated outputs

Conceptually, fallback says:

> when the label stream fails the configured reliability test, the system
> should degrade to an explicit known state rather than silently proceed as if
> nothing is wrong.

In many workflows that fallback will be the all-`undefined` composite label,
because that preserves honesty about uncertainty while remaining inside the
existing taxonomy contract.

## Profile Model

Profiles are packaged assumptions about the tradeoff between responsiveness and
stability.

### Baseline

Hypothesis:
Most workflows need moderate stabilization without hiding meaningful regime
movement.

Model intent:
Balanced default for review, sensitivity analysis, and general research.

### Conservative

Hypothesis:
For attribution, scaling, and optimization inputs, false instability is often
more damaging than delayed recognition.

Model intent:
Prefer persistence, higher evidence thresholds, and stricter instability
controls.

### Reactive

Hypothesis:
Some workflows value timeliness more than smoothness and can tolerate higher
flip rates.

Model intent:
Retain fast shifts with lighter gating and less restrictive attribution rules.

### Crisis Sensitive

Hypothesis:
Stress episodes can emerge quickly, and over-smoothing them may hide the signal
that matters most.

Model intent:
Remain responsive to abrupt stress transitions while still emitting stability
metrics and warnings.

## Relationship To Taxonomy

The taxonomy answers:

* what labels exist?
* what does each label mean?
* how are labels assigned from market metrics?

Calibration answers:

* how sensitive should the sequence be?
* how much persistence is required?
* when is the sequence too unstable for downstream use?
* when should fallback or attribution gating apply?

That separation is deliberate.

If taxonomy and calibration are mixed together, the system becomes harder to
audit because label meaning and label usability become entangled.

## Practical Reading Guide

If you are evaluating a calibration profile, ask these questions in order:

1. Is the raw regime stream descriptively plausible?
2. Is the calibrated stream stable enough for the target workflow?
3. Are the instability metrics within acceptable bounds?
4. Do the regimes have enough observations for the intended analysis?
5. Is fallback behavior appropriate, or should the workflow stop and escalate?

## Summary

The hypothesis behind regime calibration is that regime correctness alone is
not enough for downstream decisions. A second deterministic layer is needed to
control responsiveness, persistence, evidence sufficiency, and operational
safety.

The model behind regime calibration is therefore not a predictive market model.
It is a policy model over an existing regime sequence: smooth when necessary,
require persistence when appropriate, measure instability explicitly, gate weak
evidence, and degrade safely when configured thresholds are violated.

> **Operational note:** Calibrated labels should be interpreted as an
> operationally stabilized regime stream, not as a replacement for the raw
> descriptive regime stream. The raw stream remains the auditable record of
> what the classifier observed. The calibrated stream represents what the
> configured profile considers stable enough for downstream use.