from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Any

import pandas as pd

from src.research.regime_ml.base import RegimeMLError

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
except ImportError as exc:  # pragma: no cover - exercised only in missing-dependency environments
    KMeans = None  # type: ignore[assignment]
    adjusted_rand_score = None  # type: ignore[assignment]
    normalized_mutual_info_score = None  # type: ignore[assignment]
    _SKLEARN_IMPORT_ERROR = exc
else:
    _SKLEARN_IMPORT_ERROR = None

from src.research.registry import canonicalize_value


@dataclass(frozen=True)
class ClusterDiagnosticsResult:
    assignments: pd.Series
    cluster_map: pd.DataFrame
    diagnostics: dict[str, Any]


def compute_cluster_diagnostics(
    X: pd.DataFrame,
    labels: pd.Series,
    *,
    random_seed: int,
    n_clusters: int,
    weak_alignment_purity_threshold: float = 0.60,
) -> ClusterDiagnosticsResult:
    _require_sklearn()
    resolved_clusters = max(1, min(int(n_clusters), int(len(X)), int(labels.astype("string").nunique())))
    estimator = KMeans(n_clusters=resolved_clusters, n_init=10, random_state=random_seed)
    assignments = pd.Series(estimator.fit_predict(X), index=X.index, dtype="int64", name="cluster_id")
    normalized_labels = labels.astype("string")

    rows: list[dict[str, Any]] = []
    purity_by_cluster: dict[int, float] = {}
    entropy_by_cluster: dict[int, float] = {}
    weakly_aligned_clusters: list[int] = []
    for cluster_id in sorted(assignments.unique().tolist()):
        cluster_labels = normalized_labels.loc[assignments.eq(cluster_id)]
        counts = Counter(cluster_labels.tolist())
        total = int(sum(counts.values()))
        probabilities = [count / total for count in counts.values()]
        entropy = float(-sum(probability * math.log(probability, 2.0) for probability in probabilities if probability > 0.0))
        purity = float(max(counts.values()) / total)
        entropy_by_cluster[int(cluster_id)] = entropy
        purity_by_cluster[int(cluster_id)] = purity
        if purity < weak_alignment_purity_threshold:
            weakly_aligned_clusters.append(int(cluster_id))
        for regime_label in sorted(counts):
            count = int(counts[regime_label])
            rows.append(
                {
                    "cluster_id": int(cluster_id),
                    "regime_label": regime_label,
                    "observation_count": count,
                    "proportion": float(count / total),
                    "entropy": entropy,
                    "purity": purity,
                    "is_weakly_aligned": bool(purity < weak_alignment_purity_threshold),
                }
            )

    cluster_map = pd.DataFrame(rows).sort_values(["cluster_id", "regime_label"], kind="stable").reset_index(drop=True)
    diagnostics = canonicalize_value(
        {
            "n_clusters": resolved_clusters,
            "purity_mean": float(sum(purity_by_cluster.values()) / len(purity_by_cluster)),
            "nmi": float(normalized_mutual_info_score(normalized_labels, assignments.astype("string"))),
            "ari": float(adjusted_rand_score(normalized_labels, assignments.astype("string"))),
            "weakly_aligned_clusters": weakly_aligned_clusters,
            "cluster_entropy": {str(key): value for key, value in sorted(entropy_by_cluster.items())},
            "cluster_purity": {str(key): value for key, value in sorted(purity_by_cluster.items())},
        }
    )
    return ClusterDiagnosticsResult(assignments=assignments, cluster_map=cluster_map, diagnostics=diagnostics)


def _require_sklearn() -> None:
    if KMeans is None or adjusted_rand_score is None or normalized_mutual_info_score is None:
        raise RegimeMLError(
            "scikit-learn is required for regime ML. Install project ML dependencies before running this pipeline."
        ) from _SKLEARN_IMPORT_ERROR
