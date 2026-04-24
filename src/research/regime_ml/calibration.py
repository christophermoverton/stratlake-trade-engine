from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def multiclass_brier_score(probabilities: pd.DataFrame, labels: pd.Series) -> float:
    ordered = labels.astype("string")
    matrix = probabilities.loc[:, list(probabilities.columns)].to_numpy(dtype="float64", copy=True)
    truth = np.zeros_like(matrix, dtype="float64")
    class_index = {str(label): index for index, label in enumerate(probabilities.columns)}
    for row_index, label in enumerate(ordered.tolist()):
        label_index = class_index.get(str(label))
        if label_index is not None:
            truth[row_index, label_index] = 1.0
    squared_error = np.square(matrix - truth).sum(axis=1)
    return float(np.mean(squared_error))


@dataclass
class _BinaryPlattScaler:
    estimator: LogisticRegression | None

    def transform(self, scores: np.ndarray) -> np.ndarray:
        clipped = np.clip(scores.astype("float64"), 1.0e-9, 1.0 - 1.0e-9)
        logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
        if self.estimator is None:
            return clipped
        return self.estimator.predict_proba(logits)[:, 1].astype("float64")


@dataclass
class PlattCalibrator:
    classes_: list[str]
    scalers_: dict[str, _BinaryPlattScaler]

    @classmethod
    def fit(cls, probabilities: pd.DataFrame, labels: pd.Series) -> "PlattCalibrator":
        scalers: dict[str, _BinaryPlattScaler] = {}
        normalized_labels = labels.astype("string")
        for label in probabilities.columns:
            targets = normalized_labels.eq(str(label)).astype("int64")
            if int(targets.nunique()) < 2:
                scalers[str(label)] = _BinaryPlattScaler(estimator=None)
                continue
            clipped = np.clip(probabilities[str(label)].to_numpy(dtype="float64", copy=True), 1.0e-9, 1.0 - 1.0e-9)
            logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
            estimator = LogisticRegression(max_iter=1000, solver="lbfgs")
            estimator.fit(logits, targets)
            scalers[str(label)] = _BinaryPlattScaler(estimator=estimator)
        return cls(classes_=[str(label) for label in probabilities.columns], scalers_=scalers)

    def transform(self, probabilities: pd.DataFrame) -> pd.DataFrame:
        calibrated_columns: dict[str, np.ndarray] = {}
        for label in self.classes_:
            calibrated_columns[label] = self.scalers_[label].transform(
                probabilities[label].to_numpy(dtype="float64", copy=True)
            )
        frame = pd.DataFrame(calibrated_columns, index=probabilities.index, dtype="float64")
        totals = frame.sum(axis=1)
        zero_total_mask = totals.le(0.0) | totals.map(lambda value: not math.isfinite(float(value)))
        if zero_total_mask.any():
            uniform_value = 1.0 / float(len(self.classes_))
            frame.loc[zero_total_mask, :] = uniform_value
            totals = frame.sum(axis=1)
        return frame.div(totals, axis=0).astype("float64")
