from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import pandas as pd

from src.config.robustness import RobustnessConfig
from src.research.metrics import aggregate_strategy_returns

METHODOLOGY_VERSION = "m22.4.v1"
FAMILY_SCOPE = "full_evaluated_sweep"
WARNING_CODES = (
    "correction_not_applicable",
    "fdr_nonpass",
    "high_pbo",
    "insufficient_history_for_dsr",
    "insufficient_splits_for_inference",
    "large_search_space",
    "low_neighbor_gap",
    "low_threshold_pass_rate",
    "rank_instability",
)
_SUPPORTED_ZERO_NULL_METRICS = {
    "annualized_return",
    "cumulative_return",
    "excess_return",
    "sharpe_ratio",
    "total_return",
}
_EULER_MASCHERONI = 0.5772156649015329


@dataclass(frozen=True)
class StatisticalValidityResult:
    variant_metrics: pd.DataFrame
    summary: dict[str, Any]


def apply_statistical_validity_controls(
    *,
    variant_metrics: pd.DataFrame,
    stability_metrics: pd.DataFrame,
    neighbor_metrics: pd.DataFrame,
    child_payloads: Sequence[Mapping[str, Any]],
    robustness_config: RobustnessConfig,
    higher_is_better: bool,
    summary: Mapping[str, Any],
) -> StatisticalValidityResult:
    controls = robustness_config.statistical_controls
    primary_metric = controls.primary_metric or robustness_config.ranking.primary_metric
    if primary_metric not in variant_metrics.columns:
        raise ValueError(
            f"Statistical-validity primary metric '{primary_metric}' is missing from variant metrics."
        )

    frame = variant_metrics.copy()
    frame["raw_primary_metric"] = pd.to_numeric(frame[primary_metric], errors="coerce").astype("float64")
    if frame["raw_primary_metric"].isna().any():
        raise ValueError(
            f"Statistical-validity primary metric '{primary_metric}' must be finite for all variants."
        )

    frame["correction_method"] = pd.Series(["not_applicable"] * len(frame), index=frame.index, dtype="string")
    frame["correction_eligible"] = pd.Series([False] * len(frame), index=frame.index, dtype="bool")
    frame["raw_p_value"] = pd.Series([None] * len(frame), index=frame.index, dtype="object")
    frame["adjusted_q_value"] = pd.Series([None] * len(frame), index=frame.index, dtype="object")
    frame["deflated_sharpe_ratio"] = pd.Series([None] * len(frame), index=frame.index, dtype="object")
    frame["validity_warning_codes"] = pd.Series([""] * len(frame), index=frame.index, dtype="string")
    frame["validity_rank"] = pd.Series([None] * len(frame), index=frame.index, dtype="object")
    frame["validity_notes"] = pd.Series([""] * len(frame), index=frame.index, dtype="string")

    enabled = bool(
        controls.multiple_testing_awareness
        or controls.deflated_sharpe_ratio
        or controls.validity_ranking_method != "none"
    )
    variant_ids = frame["variant_id"].astype("string")
    row_warnings: dict[str, set[str]] = {str(variant_id): set() for variant_id in variant_ids}
    row_notes: dict[str, set[str]] = {str(variant_id): set() for variant_id in variant_ids}
    global_warnings: set[str] = set()
    methods_skipped: list[dict[str, str]] = []
    batching_reassembled = robustness_config.research_sweep.batching.batch_size is None

    if len(frame) >= controls.overfitting_warning_search_space:
        global_warnings.add("large_search_space")

    threshold_pass_rate = summary.get("threshold_pass_rate")
    if (
        threshold_pass_rate is not None
        and float(threshold_pass_rate) < controls.low_threshold_pass_rate_threshold
    ):
        global_warnings.add("low_threshold_pass_rate")

    rank_consistency_mean = summary.get("rank_consistency_mean")
    if (
        rank_consistency_mean is not None
        and float(rank_consistency_mean) < controls.rank_instability_correlation_threshold
    ):
        global_warnings.add("rank_instability")

    for variant_id, gap in _neighbor_gap_by_variant(neighbor_metrics).items():
        if gap is not None and gap < controls.low_neighbor_gap_threshold:
            row_warnings.setdefault(str(variant_id), set()).add("low_neighbor_gap")

    returns_by_variant = _returns_by_variant(child_payloads)
    fdr_method_used = "disabled"
    fdr_unavailable_reason: str | None = None
    eligible_count = 0

    if controls.multiple_testing_awareness:
        raw_p_values, correction_eligible, p_value_reason, p_value_reason_code = _split_level_p_values(
            frame=frame,
            stability_metrics=stability_metrics,
            primary_metric=primary_metric,
            higher_is_better=higher_is_better,
            min_splits=controls.min_splits_for_inference,
            full_family_available=batching_reassembled,
        )
        eligible_count = int(sum(1 for value in correction_eligible.values() if value))
        if raw_p_values:
            q_values = _benjamini_hochberg(raw_p_values, frame=frame)
            for variant_id, raw_p_value in raw_p_values.items():
                index = frame.index[frame["variant_id"].astype("string") == str(variant_id)]
                frame.loc[index, "correction_method"] = "fdr_bh"
                frame.loc[index, "correction_eligible"] = True
                frame.loc[index, "raw_p_value"] = float(raw_p_value)
                frame.loc[index, "adjusted_q_value"] = float(q_values[str(variant_id)])
                if float(q_values[str(variant_id)]) > controls.fdr_alpha:
                    row_warnings[str(variant_id)].add("fdr_nonpass")
                row_notes[str(variant_id)].add(
                    f"raw p-value estimated from split-level {primary_metric} values using a one-sided normal approximation"
                )
            for variant_id, is_eligible in correction_eligible.items():
                if not is_eligible:
                    row_warnings[str(variant_id)].add("insufficient_splits_for_inference")
                    row_notes[str(variant_id)].add(p_value_reason)
            fdr_method_used = "fdr_bh"
        else:
            fdr_method_used = "not_applicable"
            fdr_unavailable_reason = p_value_reason
            methods_skipped.append({"method": "fdr_bh", "reason": p_value_reason})
            global_warnings.add("correction_not_applicable")
            for variant_id in row_warnings:
                row_warnings[variant_id].add("correction_not_applicable")
                if p_value_reason:
                    row_notes[variant_id].add(p_value_reason)
                if p_value_reason_code == "insufficient_splits" and correction_eligible.get(variant_id) is False:
                    row_warnings[variant_id].add("insufficient_splits_for_inference")
    else:
        methods_skipped.append({"method": "fdr_bh", "reason": "disabled by configuration"})

    dsr_unavailable_reason: str | None = None
    if controls.deflated_sharpe_ratio:
        dsr_values, dsr_reason_by_variant, dsr_reason = _deflated_sharpe_ratios(
            frame=frame,
            primary_metric=primary_metric,
            returns_by_variant=returns_by_variant,
            min_observations=controls.dsr_min_observations,
            full_family_available=batching_reassembled,
        )
        if dsr_values:
            for variant_id, dsr_value in dsr_values.items():
                index = frame.index[frame["variant_id"].astype("string") == str(variant_id)]
                frame.loc[index, "deflated_sharpe_ratio"] = float(dsr_value)
                row_notes[str(variant_id)].add(
                    "deflated_sharpe_ratio uses observed Sharpe, higher-moment variance adjustment, and full-sweep trial count"
                )
        else:
            dsr_unavailable_reason = dsr_reason
            methods_skipped.append({"method": "deflated_sharpe_ratio", "reason": dsr_reason})

        for variant_id, reason in dsr_reason_by_variant.items():
            row_warnings[str(variant_id)].add("insufficient_history_for_dsr")
            row_notes[str(variant_id)].add(reason)
    else:
        methods_skipped.append({"method": "deflated_sharpe_ratio", "reason": "disabled by configuration"})

    methods_skipped.extend(
        [
            {"method": "pbo_cscv", "reason": "not implemented in first pass"},
            {
                "method": "spa",
                "reason": "deferred in first pass because benchmark-testing infrastructure is not present",
            },
            {
                "method": "white_reality_check",
                "reason": "deferred in first pass because benchmark-testing infrastructure is not present",
            },
        ]
    )
    if controls.reality_check_placeholder:
        methods_skipped.append(
            {
                "method": "reality_check_placeholder",
                "reason": "placeholder flag present but no White Reality Check or SPA implementation exists",
            }
        )

    for variant_id in row_warnings:
        row_warnings[variant_id].update(global_warnings)

    requested_ranking_method = controls.validity_ranking_method
    validity_ranking_used: str | None = None
    validity_ranking_unavailable_reason: str | None = None
    if requested_ranking_method == "adjusted_q_value":
        if frame["adjusted_q_value"].notna().any():
            validity_ranking_used = "adjusted_q_value"
            validity_ranks = _validity_ranks(
                frame,
                sort_metric="adjusted_q_value",
                metric_ascending=True,
                ranking=robustness_config.ranking,
            )
        else:
            validity_ranks = {}
            validity_ranking_unavailable_reason = fdr_unavailable_reason or "adjusted q-values were unavailable"
    elif requested_ranking_method == "deflated_sharpe_ratio":
        if frame["deflated_sharpe_ratio"].notna().any():
            validity_ranking_used = "deflated_sharpe_ratio"
            validity_ranks = _validity_ranks(
                frame,
                sort_metric="deflated_sharpe_ratio",
                metric_ascending=False,
                ranking=robustness_config.ranking,
            )
        else:
            validity_ranks = {}
            validity_ranking_unavailable_reason = (
                dsr_unavailable_reason or "deflated Sharpe ratio was unavailable"
            )
    else:
        validity_ranks = {}
        validity_ranking_unavailable_reason = "validity-aware ranking disabled by configuration"

    if validity_ranks:
        frame["validity_rank"] = (
            frame["variant_id"]
            .astype("string")
            .map(lambda value: validity_ranks.get(str(value)))
            .astype("Int64")
        )
    for variant_id, warnings in row_warnings.items():
        index = frame.index[frame["variant_id"].astype("string") == str(variant_id)]
        frame.loc[index, "validity_warning_codes"] = "|".join(sorted(warnings))
        frame.loc[index, "validity_notes"] = "; ".join(sorted(row_notes[variant_id]))

    warning_counts_by_code = {
        code: int(sum(code in warnings for warnings in row_warnings.values()))
        for code in WARNING_CODES
        if code in set().union(*row_warnings.values())
    }

    statistical_validity_summary = {
        "enabled": enabled,
        "primary_metric": primary_metric,
        "family_scope": FAMILY_SCOPE,
        "n_configs_evaluated": int(len(frame)),
        "n_configs_eligible": eligible_count,
        "correction_method_used": fdr_method_used,
        "methods_skipped_and_reason": sorted(
            methods_skipped,
            key=lambda item: (item["method"], item["reason"]),
        ),
        "warning_thresholds": {
            "fdr_alpha": controls.fdr_alpha,
            "low_neighbor_gap_threshold": controls.low_neighbor_gap_threshold,
            "low_threshold_pass_rate_threshold": controls.low_threshold_pass_rate_threshold,
            "min_splits_for_inference": controls.min_splits_for_inference,
            "overfitting_warning_search_space": controls.overfitting_warning_search_space,
            "rank_instability_correlation_threshold": controls.rank_instability_correlation_threshold,
            "dsr_min_observations": controls.dsr_min_observations,
        },
        "warning_counts_by_code": warning_counts_by_code,
        "methodology_version": METHODOLOGY_VERSION,
        "validity_ranking_method_requested": requested_ranking_method,
        "validity_ranking_method_used": validity_ranking_used,
        "validity_ranking_available": validity_ranking_used is not None,
        "validity_ranking_unavailable_reason": validity_ranking_unavailable_reason,
    }
    return StatisticalValidityResult(variant_metrics=frame, summary=statistical_validity_summary)


def _split_level_p_values(
    *,
    frame: pd.DataFrame,
    stability_metrics: pd.DataFrame,
    primary_metric: str,
    higher_is_better: bool,
    min_splits: int,
    full_family_available: bool,
) -> tuple[dict[str, float], dict[str, bool], str, str]:
    if not full_family_available:
        return {}, {str(variant_id): False for variant_id in frame["variant_id"].astype("string")}, (
            "multiple-testing correction requires the full evaluated sweep after deterministic batch reassembly"
        ), "full_family_unavailable"
    if stability_metrics.empty:
        return {}, {str(variant_id): False for variant_id in frame["variant_id"].astype("string")}, (
            "multiple-testing correction requires split-level evidence, but stability metrics were not available"
        ), "stability_metrics_missing"
    if primary_metric not in stability_metrics.columns:
        return {}, {str(variant_id): False for variant_id in frame["variant_id"].astype("string")}, (
            f"multiple-testing correction is only available when split-level '{primary_metric}' values are present"
        ), "stability_metric_missing"
    if primary_metric not in _SUPPORTED_ZERO_NULL_METRICS:
        return {}, {str(variant_id): False for variant_id in frame["variant_id"].astype("string")}, (
            f"multiple-testing correction only supports zero-centered primary metrics; '{primary_metric}' is not supported"
        ), "unsupported_primary_metric"

    sign = 1.0 if higher_is_better else -1.0
    raw_p_values: dict[str, float] = {}
    eligible: dict[str, bool] = {}
    for variant_id, group in stability_metrics.groupby("variant_id", sort=False):
        values = pd.to_numeric(group[primary_metric], errors="coerce").dropna().astype("float64") * sign
        eligible[str(variant_id)] = len(values) >= min_splits
        if len(values) < min_splits:
            continue
        raw_p_values[str(variant_id)] = _one_sided_normal_p_value(values)

    for variant_id in frame["variant_id"].astype("string"):
        eligible.setdefault(str(variant_id), False)
    return raw_p_values, eligible, (
        f"split-level inference requires at least {min_splits} finite '{primary_metric}' observations per configuration"
    ), "insufficient_splits"


def _deflated_sharpe_ratios(
    *,
    frame: pd.DataFrame,
    primary_metric: str,
    returns_by_variant: Mapping[str, pd.Series],
    min_observations: int,
    full_family_available: bool,
) -> tuple[dict[str, float], dict[str, str], str]:
    if not full_family_available:
        return {}, {}, (
            "deflated Sharpe ratio requires the full evaluated sweep after deterministic batch reassembly"
        )
    if primary_metric != "sharpe_ratio":
        return {}, {}, "deflated Sharpe ratio is only applicable when the primary metric is sharpe_ratio"

    n_trials = int(len(frame))
    dsr_values: dict[str, float] = {}
    reason_by_variant: dict[str, str] = {}
    for _, row in frame.iterrows():
        variant_id = str(row["variant_id"])
        returns = pd.to_numeric(returns_by_variant.get(variant_id, pd.Series(dtype="float64")), errors="coerce").dropna()
        if len(returns) < min_observations:
            reason_by_variant[variant_id] = (
                f"DSR requires at least {min_observations} return observations; observed {len(returns)}"
            )
            continue
        dsr_value = _compute_deflated_sharpe_ratio(
            sharpe_ratio=float(row["raw_primary_metric"]),
            returns=returns.astype("float64"),
            n_trials=n_trials,
        )
        if dsr_value is None:
            reason_by_variant[variant_id] = "DSR variance adjustment was not well-defined for this return history"
            continue
        dsr_values[variant_id] = dsr_value

    return dsr_values, reason_by_variant, (
        f"deflated Sharpe ratio requires at least {min_observations} return observations per configuration"
    )


def _compute_deflated_sharpe_ratio(*, sharpe_ratio: float, returns: pd.Series, n_trials: int) -> float | None:
    if len(returns) < 2:
        return None
    skew = float(returns.skew()) if len(returns) >= 3 else 0.0
    kurtosis = float(returns.kurt()) + 3.0 if len(returns) >= 4 else 3.0
    denominator_term = (
        1.0
        - skew * sharpe_ratio
        + ((kurtosis - 1.0) / 4.0) * (sharpe_ratio ** 2)
    ) / float(len(returns) - 1)
    if denominator_term <= 0.0:
        return None
    sigma_sr = math.sqrt(denominator_term)
    max_sr_benchmark = sigma_sr * (
        (1.0 - _EULER_MASCHERONI) * _inverse_normal_cdf(1.0 - (1.0 / float(n_trials)))
        + _EULER_MASCHERONI * _inverse_normal_cdf(1.0 - (1.0 / (float(n_trials) * math.e)))
    )
    return _normal_cdf((sharpe_ratio - max_sr_benchmark) / sigma_sr)


def _validity_ranks(
    frame: pd.DataFrame,
    *,
    sort_metric: str,
    metric_ascending: bool,
    ranking: Any,
) -> dict[str, int]:
    ordered = frame.copy()
    ordered["_validity_missing"] = ordered[sort_metric].isna()
    ordered["_validity_metric"] = pd.to_numeric(ordered[sort_metric], errors="coerce")
    ordered["_raw_primary_metric"] = pd.to_numeric(ordered["raw_primary_metric"], errors="coerce")
    sort_columns = ["_validity_missing", "_validity_metric", "_raw_primary_metric"]
    ascending = [True, metric_ascending, False]
    for tie_breaker in ranking.tie_breakers:
        if tie_breaker not in ordered.columns:
            continue
        helper = f"_validity_tie_{tie_breaker}"
        ordered[helper] = pd.to_numeric(ordered[tie_breaker], errors="coerce")
        sort_columns.append(helper)
        ascending.append(False)
    sort_columns.extend(["variant_order", "variant_id"])
    ascending.extend([True, True])
    ordered = ordered.sort_values(sort_columns, ascending=ascending, kind="mergesort").reset_index(drop=True)
    ordered["validity_rank"] = pd.Series(range(1, len(ordered) + 1), dtype="int64")
    return {
        str(record["variant_id"]): int(record["validity_rank"])
        for record in ordered.loc[:, ["variant_id", "validity_rank"]].to_dict(orient="records")
    }


def _neighbor_gap_by_variant(neighbor_metrics: pd.DataFrame) -> dict[str, float | None]:
    if neighbor_metrics.empty:
        return {}
    rows: dict[str, list[float]] = {}
    for record in neighbor_metrics.to_dict(orient="records"):
        gap = record.get("metric_gap")
        if gap is None or pd.isna(gap):
            continue
        rows.setdefault(str(record["left_variant_id"]), []).append(float(gap))
        rows.setdefault(str(record["right_variant_id"]), []).append(float(gap))
    return {
        variant_id: (None if not gaps else float(min(gaps)))
        for variant_id, gaps in rows.items()
    }


def _returns_by_variant(child_payloads: Sequence[Mapping[str, Any]]) -> dict[str, pd.Series]:
    rows: dict[str, pd.Series] = {}
    for payload in child_payloads:
        variant = payload.get("variant")
        variant_id = getattr(variant, "variant_id", None)
        if variant_id is None:
            continue
        executed = payload.get("executed", {})
        artifact_kind = executed.get("artifact_kind")
        if artifact_kind == "strategy":
            results_df = executed.get("results_df")
            if isinstance(results_df, pd.DataFrame):
                rows[str(variant_id)] = aggregate_strategy_returns(results_df)["strategy_return"].astype("float64")
        elif artifact_kind == "portfolio":
            portfolio_output = executed.get("portfolio_output")
            if isinstance(portfolio_output, pd.DataFrame) and "portfolio_return" in portfolio_output.columns:
                rows[str(variant_id)] = pd.to_numeric(
                    portfolio_output["portfolio_return"],
                    errors="coerce",
                ).dropna().astype("float64")
    return rows


def _benjamini_hochberg(raw_p_values: Mapping[str, float], *, frame: pd.DataFrame) -> dict[str, float]:
    ordering = (
        frame.loc[frame["variant_id"].astype("string").isin(list(raw_p_values)), ["variant_id", "variant_order"]]
        .copy()
    )
    ordering["variant_id"] = ordering["variant_id"].astype("string")
    ordering["raw_p_value"] = ordering["variant_id"].map(lambda value: float(raw_p_values[str(value)]))
    ordering = ordering.sort_values(
        ["raw_p_value", "variant_order", "variant_id"],
        ascending=[True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    m = len(ordering)
    adjusted = [0.0] * m
    running_min = 1.0
    for index in range(m - 1, -1, -1):
        rank = index + 1
        candidate = float(ordering.loc[index, "raw_p_value"]) * float(m) / float(rank)
        running_min = min(running_min, candidate)
        adjusted[index] = min(1.0, running_min)
    ordering["adjusted_q_value"] = adjusted
    return {
        str(record["variant_id"]): float(record["adjusted_q_value"])
        for record in ordering.to_dict(orient="records")
    }


def _one_sided_normal_p_value(values: pd.Series) -> float:
    if len(values) < 2:
        return 1.0
    std = float(values.std(ddof=1))
    mean = float(values.mean())
    if std == 0.0:
        return 0.0 if mean > 0.0 else 1.0
    z_score = mean / (std / math.sqrt(len(values)))
    return 1.0 - _normal_cdf(z_score)


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _inverse_normal_cdf(probability: float) -> float:
    if not 0.0 < probability < 1.0:
        raise ValueError("Probability must be strictly between 0 and 1.")

    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    p_low = 0.02425
    p_high = 1.0 - p_low

    if probability < p_low:
        q = math.sqrt(-2.0 * math.log(probability))
        return (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    if probability <= p_high:
        q = probability - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )
    q = math.sqrt(-2.0 * math.log(1.0 - probability))
    return -(
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    )
