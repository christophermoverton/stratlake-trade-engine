# Regime ML Confidence Layer

`src/research/regime_ml/` adds a deterministic, taxonomy-compatible ML layer on top of M24 regime labels.

## Scope

- Trains a supervised classifier against existing `regime_label` values.
- Calibrates confidence with Platt scaling on the validation window only.
- Produces cluster diagnostics with KMeans without changing taxonomy labels.
- Emits policy-friendly fallback routing fields for low-confidence or unsupported rows.

## Current Model Set

- `LogisticRegressionRegimeModel`

The baseline is intentionally interpretable and deterministic. Predicted labels remain members of the trained taxonomy label set only.

## Main Entry Point

Use `run_regime_ml_pipeline(...)` with:

- a feature frame containing `symbol`, `ts_utc`, `regime_label`, and model features
- an explicit `feature_columns` list
- an optional output directory for artifact persistence

The pipeline uses contiguous chronological splits:

- train
- validation
- test

## Calibration Choice

The first pass uses Platt scaling.

- Raw model probabilities are generated on the validation slice.
- One-vs-rest sigmoid calibrators are fitted on validation-only scores.
- Calibrated probabilities are renormalized to sum to 1.
- Pre- and post-calibration multiclass Brier scores are persisted.

## Artifact Set

When `output_dir` is provided, the pipeline writes:

- `regime_confidence.csv`
- `regime_ml_diagnostics.json`
- `regime_model_manifest.json`
- `regime_cluster_map.csv`
- `regime_cluster_diagnostics.json`
- `regime_label_mapping.json`

## Policy Interpretation

- `confidence_score` is the calibrated probability of the assigned `regime_label`.
- `confidence_bucket` uses:
  - `high`: `>= 0.70`
  - `medium`: `>= 0.40` and `< 0.70`
  - `low`: `< 0.40`
- `fallback_flag` is raised when the row is:
  - `low_confidence`
  - `ambiguous`
  - `unsupported`

`unsupported` currently covers missing-feature rows and evaluation labels not seen during training.

## Auditability

- Fixed random seeds are required.
- Feature columns are explicit and persisted in the model manifest.
- Label mappings are persisted without silent remapping.
- Cluster diagnostics are descriptive only and must not replace taxonomy labels.
