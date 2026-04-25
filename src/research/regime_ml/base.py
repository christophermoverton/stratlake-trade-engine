from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class RegimeMLError(ValueError):
    """Raised when deterministic regime ML workflows cannot complete safely."""


class BaseRegimeModel(ABC):
    """Deterministic contract for taxonomy-compatible regime models."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, metadata: dict[str, Any] | None = None) -> "BaseRegimeModel":
        """Fit the model on one feature frame and canonical labels."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict canonical regime labels."""

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return class probabilities indexed like X and labeled by canonical classes."""
