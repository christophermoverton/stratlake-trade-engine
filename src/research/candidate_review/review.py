from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


class CandidateReviewError(ValueError):
    """Raised when candidate review inputs are malformed or missing."""


@dataclass(frozen=True)
class CandidateReviewArtifacts:
    """Output paths and summary counts for one review run."""

    candidate_selection_run_id: str
    portfolio_run_id: str
    review_dir: Path
    candidate_decisions_csv: Path
    candidate_summary_csv: Path
    candidate_contributions_csv: Path
    diversification_summary_json: Path
    candidate_review_summary_json: Path
    candidate_review_report_md: Path | None
    manifest_json: Path
    total_candidates: int
    selected_candidates: int
    rejected_candidates: int


def review_candidate_selection(
    *,
    candidate_selection_artifact_dir: str | Path,
    portfolio_artifact_dir: str | Path,
    output_dir: str | Path | None = None,
    include_markdown_report: bool = True,
    top_n: int = 10,
) -> CandidateReviewArtifacts:
    """Generate deterministic candidate explainability artifacts from existing runs."""

    candidate_dir = Path(candidate_selection_artifact_dir)
    portfolio_dir = Path(portfolio_artifact_dir)
    if not candidate_dir.exists() or not candidate_dir.is_dir():
        raise CandidateReviewError(f"Candidate selection artifact dir does not exist: {candidate_dir}")
    if not portfolio_dir.exists() or not portfolio_dir.is_dir():
        raise CandidateReviewError(f"Portfolio artifact dir does not exist: {portfolio_dir}")

    candidate_manifest = _read_json(candidate_dir / "manifest.json")
    candidate_summary_payload = _read_json(candidate_dir / "selection_summary.json")
    portfolio_manifest = _read_json(portfolio_dir / "manifest.json")
    portfolio_metrics = _read_json(portfolio_dir / "metrics.json")
    components_payload = _read_json(portfolio_dir / "components.json")

    components = components_payload.get("components")
    if not isinstance(components, list) or not components:
        raise CandidateReviewError("Portfolio components.json must contain a non-empty 'components' list.")

    universe = _read_csv(candidate_dir / "candidate_universe.csv")
    eligibility = _read_csv(candidate_dir / "eligibility_filter_results.csv")
    rejected = _read_csv(candidate_dir / "rejected_candidates.csv")
    selected = _read_csv(candidate_dir / "selected_candidates.csv")
    allocation = _read_csv(candidate_dir / "allocation_weights.csv")
    correlation = _read_csv(candidate_dir / "correlation_matrix.csv")
    portfolio_returns = _read_csv(portfolio_dir / "portfolio_returns.csv")

    candidate_selection_run_id = str(candidate_manifest.get("run_id") or candidate_dir.name)
    portfolio_run_id = str(portfolio_manifest.get("run_id") or portfolio_dir.name)
    resolved_output_dir = candidate_dir / "review" if output_dir is None else Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    candidate_decisions = _build_candidate_decisions(
        universe=universe,
        eligibility=eligibility,
        rejected=rejected,
        selected=selected,
        candidate_selection_run_id=candidate_selection_run_id,
        portfolio_run_id=portfolio_run_id,
    )
    candidate_summary = _build_candidate_summary(
        universe=universe,
        eligibility=eligibility,
        rejected=rejected,
        selected=selected,
        allocation=allocation,
        components=components,
        candidate_selection_run_id=candidate_selection_run_id,
        portfolio_run_id=portfolio_run_id,
    )
    candidate_contributions = _build_candidate_contributions(
        selected=selected,
        components=components,
        portfolio_returns=portfolio_returns,
        candidate_selection_run_id=candidate_selection_run_id,
        portfolio_run_id=portfolio_run_id,
    )
    diversification_summary = _build_diversification_summary(
        correlation=correlation,
        candidate_summary=candidate_summary,
        allocation=allocation,
    )
    review_summary = _build_review_summary(
        candidate_decisions=candidate_decisions,
        candidate_summary=candidate_summary,
        candidate_contributions=candidate_contributions,
        diversification_summary=diversification_summary,
        candidate_selection_summary=candidate_summary_payload,
        portfolio_metrics=portfolio_metrics,
        candidate_selection_run_id=candidate_selection_run_id,
        portfolio_run_id=portfolio_run_id,
    )

    decisions_csv = resolved_output_dir / "candidate_decisions.csv"
    summary_csv = resolved_output_dir / "candidate_summary.csv"
    contributions_csv = resolved_output_dir / "candidate_contributions.csv"
    diversification_json = resolved_output_dir / "diversification_summary.json"
    review_summary_json = resolved_output_dir / "candidate_review_summary.json"
    report_md = resolved_output_dir / "candidate_review_report.md"
    manifest_json = resolved_output_dir / "manifest.json"

    _write_csv(decisions_csv, candidate_decisions)
    _write_csv(summary_csv, candidate_summary)
    _write_csv(contributions_csv, candidate_contributions)
    _write_json(diversification_json, diversification_summary)
    _write_json(review_summary_json, review_summary)
    if include_markdown_report:
        report_md.write_text(
            _build_markdown_report(
                candidate_decisions=candidate_decisions,
                candidate_contributions=candidate_contributions,
                review_summary=review_summary,
                top_n=max(1, int(top_n)),
            ),
            encoding="utf-8",
            newline="\n",
        )

    manifest_payload = {
        "run_type": "candidate_selection_review",
        "candidate_selection_run_id": candidate_selection_run_id,
        "portfolio_run_id": portfolio_run_id,
        "input_artifacts": {
            "candidate_selection_artifact_dir": candidate_dir.as_posix(),
            "portfolio_artifact_dir": portfolio_dir.as_posix(),
        },
        "artifact_files": sorted(
            [
                "candidate_contributions.csv",
                "candidate_decisions.csv",
                "candidate_review_summary.json",
                "candidate_summary.csv",
                "diversification_summary.json",
                "manifest.json",
                *(["candidate_review_report.md"] if include_markdown_report else []),
            ]
        ),
        "row_counts": {
            "candidate_decisions": int(len(candidate_decisions)),
            "candidate_summary": int(len(candidate_summary)),
            "candidate_contributions": int(len(candidate_contributions)),
        },
    }
    _write_json(manifest_json, manifest_payload)

    return CandidateReviewArtifacts(
        candidate_selection_run_id=candidate_selection_run_id,
        portfolio_run_id=portfolio_run_id,
        review_dir=resolved_output_dir,
        candidate_decisions_csv=decisions_csv,
        candidate_summary_csv=summary_csv,
        candidate_contributions_csv=contributions_csv,
        diversification_summary_json=diversification_json,
        candidate_review_summary_json=review_summary_json,
        candidate_review_report_md=(report_md if include_markdown_report else None),
        manifest_json=manifest_json,
        total_candidates=int(len(candidate_decisions)),
        selected_candidates=int((candidate_decisions["selection_status"] == "selected").sum()),
        rejected_candidates=int((candidate_decisions["selection_status"] != "selected").sum()),
    )


def _build_candidate_decisions(
    *,
    universe: pd.DataFrame,
    eligibility: pd.DataFrame,
    rejected: pd.DataFrame,
    selected: pd.DataFrame,
    candidate_selection_run_id: str,
    portfolio_run_id: str,
) -> pd.DataFrame:
    eligibility_by_id = _indexed(eligibility, "candidate_id")
    rejected_by_id = _indexed(rejected, "candidate_id")
    selected_ids = {str(value) for value in selected.get("candidate_id", pd.Series(dtype="string")).astype("string")}

    rows: list[dict[str, Any]] = []
    for _, row in _stable_candidates(universe).iterrows():
        candidate_id = str(row.get("candidate_id"))
        rejected_row = rejected_by_id.get(candidate_id)
        selection_status = "selected" if candidate_id in selected_ids else _status_from_rejection_row(rejected_row)
        eligibility_row = eligibility_by_id.get(candidate_id)
        rejected_stage = None if rejected_row is None else rejected_row.get("rejected_stage")
        rejection_reason = None if rejected_row is None else rejected_row.get("rejection_reason")
        failed_checks = None if rejected_row is None else rejected_row.get("failed_checks")

        rows.append(
            {
                "candidate_id": candidate_id,
                "alpha_name": row.get("alpha_name"),
                "alpha_run_id": row.get("alpha_run_id"),
                "sleeve_run_id": row.get("sleeve_run_id"),
                "selection_status": selection_status,
                "rejected_stage": rejected_stage,
                "rejection_reason": rejection_reason,
                "failed_checks": failed_checks,
                "rejection_reasons": _joined_reasons(rejected_row),
                "selection_rank": _as_int(row.get("selection_rank")),
                "mean_ic": _as_float(row.get("mean_ic")),
                "ic_ir": _as_float(row.get("ic_ir")),
                "mean_rank_ic": _as_float(row.get("mean_rank_ic")),
                "stability": _as_float(row.get("n_periods")),
                "mean_ic_threshold": _as_float(None if eligibility_row is None else eligibility_row.get("threshold_min_mean_ic")),
                "ic_ir_threshold": _as_float(None if eligibility_row is None else eligibility_row.get("threshold_min_ic_ir")),
                "mean_rank_ic_threshold": _as_float(None if eligibility_row is None else eligibility_row.get("threshold_min_mean_rank_ic")),
                "stability_threshold": _as_float(None if eligibility_row is None else eligibility_row.get("threshold_min_history_length")),
                "mean_ic_vs_threshold": _compare_threshold(
                    value=_as_float(row.get("mean_ic")),
                    threshold=_as_float(None if eligibility_row is None else eligibility_row.get("threshold_min_mean_ic")),
                ),
                "ic_ir_vs_threshold": _compare_threshold(
                    value=_as_float(row.get("ic_ir")),
                    threshold=_as_float(None if eligibility_row is None else eligibility_row.get("threshold_min_ic_ir")),
                ),
                "mean_rank_ic_vs_threshold": _compare_threshold(
                    value=_as_float(row.get("mean_rank_ic")),
                    threshold=_as_float(None if eligibility_row is None else eligibility_row.get("threshold_min_mean_rank_ic")),
                ),
                "stability_vs_threshold": _compare_threshold(
                    value=_as_float(row.get("n_periods")),
                    threshold=_as_float(None if eligibility_row is None else eligibility_row.get("threshold_min_history_length")),
                ),
                "candidate_selection_run_id": candidate_selection_run_id,
                "portfolio_run_id": portfolio_run_id,
            }
        )

    out = pd.DataFrame(rows)
    sort_cols = ["selection_rank", "candidate_id"]
    out = out.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    return out


def _build_candidate_summary(
    *,
    universe: pd.DataFrame,
    eligibility: pd.DataFrame,
    rejected: pd.DataFrame,
    selected: pd.DataFrame,
    allocation: pd.DataFrame,
    components: list[dict[str, Any]],
    candidate_selection_run_id: str,
    portfolio_run_id: str,
) -> pd.DataFrame:
    eligibility_by_id = _indexed(eligibility, "candidate_id")
    rejected_by_id = _indexed(rejected, "candidate_id")
    selected_ids = {str(value) for value in selected.get("candidate_id", pd.Series(dtype="string")).astype("string")}
    allocation_by_id = _indexed(allocation, "candidate_id")
    component_by_candidate_id = _component_candidate_map(components)

    rows: list[dict[str, Any]] = []
    for _, row in _stable_candidates(universe).iterrows():
        candidate_id = str(row.get("candidate_id"))
        eligibility_row = eligibility_by_id.get(candidate_id)
        rejected_row = rejected_by_id.get(candidate_id)
        allocation_row = allocation_by_id.get(candidate_id)
        component_row = component_by_candidate_id.get(candidate_id)
        selection_status = "selected" if candidate_id in selected_ids else _status_from_rejection_row(rejected_row)

        rows.append(
            {
                **{column: row.get(column) for column in universe.columns.tolist()},
                "selection_status": selection_status,
                "is_eligible": None if eligibility_row is None else bool(eligibility_row.get("is_eligible")),
                "failed_checks": None if eligibility_row is None else eligibility_row.get("failed_checks"),
                "rejected_stage": None if rejected_row is None else rejected_row.get("rejected_stage"),
                "rejection_reason": None if rejected_row is None else rejected_row.get("rejection_reason"),
                "rejected_against_candidate_id": None if rejected_row is None else rejected_row.get("rejected_against_candidate_id"),
                "observed_correlation": _as_float(None if rejected_row is None else rejected_row.get("observed_correlation")),
                "allocation_weight": _as_float(None if allocation_row is None else allocation_row.get("allocation_weight")),
                "allocation_method": None if allocation_row is None else allocation_row.get("allocation_method"),
                "component_strategy_name": None if component_row is None else component_row.get("strategy_name"),
                "component_run_id": None if component_row is None else component_row.get("run_id"),
                "candidate_selection_run_id": candidate_selection_run_id,
                "portfolio_run_id": portfolio_run_id,
            }
        )

    out = pd.DataFrame(rows)
    sort_cols = ["selection_rank", "candidate_id"] if "selection_rank" in out.columns else ["candidate_id"]
    out = out.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    return out


def _build_candidate_contributions(
    *,
    selected: pd.DataFrame,
    components: list[dict[str, Any]],
    portfolio_returns: pd.DataFrame,
    candidate_selection_run_id: str,
    portfolio_run_id: str,
) -> pd.DataFrame:
    selected_by_id = _indexed(selected, "candidate_id")
    component_map = _component_candidate_map(components, strict=True)
    if portfolio_returns.empty:
        return pd.DataFrame(
            columns=[
                "candidate_id",
                "alpha_name",
                "alpha_run_id",
                "sleeve_run_id",
                "component_strategy_name",
                "component_run_id",
                "average_weight",
                "return_contribution",
                "return_contribution_share",
                "volatility_contribution",
                "drawdown_contribution",
                "turnover_contribution",
                "candidate_selection_run_id",
                "portfolio_run_id",
            ]
        )

    portfolio_series = pd.to_numeric(portfolio_returns.get("portfolio_return"), errors="coerce").fillna(0.0)
    total_return_contribution = 0.0
    raw_rows: list[dict[str, Any]] = []

    for candidate_id, component in sorted(component_map.items(), key=lambda item: item[0]):
        strategy_name = str(component.get("strategy_name"))
        strategy_col = f"strategy_return__{strategy_name}"
        weight_col = f"weight__{strategy_name}"
        if strategy_col not in portfolio_returns.columns or weight_col not in portfolio_returns.columns:
            raise CandidateReviewError(
                f"Portfolio returns are missing required component columns for strategy '{strategy_name}'."
            )

        strategy_returns = pd.to_numeric(portfolio_returns[strategy_col], errors="coerce").fillna(0.0)
        weights = pd.to_numeric(portfolio_returns[weight_col], errors="coerce").fillna(0.0)
        period_contribution = (strategy_returns * weights).astype("float64")
        return_contribution = float(period_contribution.sum())
        total_return_contribution += return_contribution

        numerator = float(period_contribution.std(ddof=0))
        denominator = float(portfolio_series.std(ddof=0))
        volatility_contribution = None if denominator == 0.0 else float(numerator / denominator)

        component_equity = (1.0 + period_contribution).cumprod()
        running_max = component_equity.cummax()
        drawdown = (component_equity / running_max) - 1.0
        drawdown_contribution = float(drawdown.min()) if len(drawdown) > 0 else None

        abs_weight_change = weights.diff().abs().fillna(0.0)
        turnover_contribution = float(abs_weight_change.sum())

        selected_row = selected_by_id.get(candidate_id, {})
        raw_rows.append(
            {
                "candidate_id": candidate_id,
                "alpha_name": selected_row.get("alpha_name"),
                "alpha_run_id": selected_row.get("alpha_run_id"),
                "sleeve_run_id": selected_row.get("sleeve_run_id"),
                "component_strategy_name": strategy_name,
                "component_run_id": component.get("run_id"),
                "average_weight": float(weights.mean()),
                "return_contribution": return_contribution,
                "volatility_contribution": volatility_contribution,
                "drawdown_contribution": drawdown_contribution,
                "turnover_contribution": turnover_contribution,
                "candidate_selection_run_id": candidate_selection_run_id,
                "portfolio_run_id": portfolio_run_id,
            }
        )

    total_abs_turnover = sum(float(row["turnover_contribution"] or 0.0) for row in raw_rows)
    rows: list[dict[str, Any]] = []
    for row in raw_rows:
        contribution_share = None
        if total_return_contribution != 0.0:
            contribution_share = float(row["return_contribution"] / total_return_contribution)
        turnover_share = None
        if total_abs_turnover > 0.0:
            turnover_share = float(row["turnover_contribution"] / total_abs_turnover)
        rows.append(
            {
                **row,
                "return_contribution_share": contribution_share,
                "turnover_contribution": turnover_share,
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(["return_contribution", "candidate_id"], ascending=[False, True], kind="stable")
    out = out.reset_index(drop=True)
    return out


def _build_diversification_summary(
    *,
    correlation: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    allocation: pd.DataFrame,
) -> dict[str, Any]:
    selected_ids = [
        str(candidate_id)
        for candidate_id in candidate_summary.loc[
            candidate_summary["selection_status"].astype("string") == "selected", "candidate_id"
        ].tolist()
    ]
    rejected_ids = [
        str(candidate_id)
        for candidate_id in candidate_summary.loc[
            candidate_summary["selection_status"].astype("string") != "selected", "candidate_id"
        ].tolist()
    ]

    corr_matrix = _normalize_correlation_matrix(correlation)
    selected_pairwise = _pairwise_values(corr_matrix, selected_ids, selected_ids, skip_diagonal=True)
    selected_to_rejected = _pairwise_values(corr_matrix, selected_ids, rejected_ids, skip_diagonal=False)

    weights = pd.to_numeric(allocation.get("allocation_weight", pd.Series(dtype="float64")), errors="coerce").dropna()
    hhi = None if weights.empty else float((weights ** 2).sum())
    concentration_ratio_top3 = None if weights.empty else float(weights.sort_values(ascending=False).head(3).sum())

    return {
        "selected_candidate_count": len(selected_ids),
        "rejected_candidate_count": len(rejected_ids),
        "average_pairwise_correlation_selected": _safe_mean(selected_pairwise),
        "average_pairwise_abs_correlation_selected": _safe_mean([abs(value) for value in selected_pairwise]),
        "average_abs_correlation_selected_vs_rejected": _safe_mean([abs(value) for value in selected_to_rejected]),
        "concentration": {
            "herfindahl_index": hhi,
            "top3_concentration_ratio": concentration_ratio_top3,
        },
    }


def _build_review_summary(
    *,
    candidate_decisions: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    candidate_contributions: pd.DataFrame,
    diversification_summary: dict[str, Any],
    candidate_selection_summary: dict[str, Any],
    portfolio_metrics: dict[str, Any],
    candidate_selection_run_id: str,
    portfolio_run_id: str,
) -> dict[str, Any]:
    total = int(len(candidate_decisions))
    selected = int((candidate_decisions["selection_status"] == "selected").sum())
    rejected_eligibility = int((candidate_decisions["selection_status"] == "rejected_eligibility").sum())
    rejected_redundancy = int((candidate_decisions["selection_status"] == "rejected_redundancy").sum())

    top_contributors = _top_bottom_contributors(candidate_contributions, top=True, n=3)
    weakest_contributors = _top_bottom_contributors(candidate_contributions, top=False, n=3)

    allocation_weight_sum = _as_float(candidate_summary.get("allocation_weight", pd.Series(dtype="float64")).sum())
    allocation_hhi = None
    if "allocation_weight" in candidate_summary.columns:
        weight_series = pd.to_numeric(candidate_summary["allocation_weight"], errors="coerce").dropna()
        if len(weight_series) > 0:
            allocation_hhi = float((weight_series ** 2).sum())

    return {
        "run_type": "candidate_selection_review",
        "candidate_selection_run_id": candidate_selection_run_id,
        "portfolio_run_id": portfolio_run_id,
        "counts": {
            "total_candidates": total,
            "selected_candidates": selected,
            "rejected_candidates": int(total - selected),
            "rejected_by_stage": {
                "eligibility": rejected_eligibility,
                "redundancy": rejected_redundancy,
            },
        },
        "allocation_summary": {
            "weight_sum": allocation_weight_sum,
            "concentration_hhi": allocation_hhi,
            "configured_allocation_summary": candidate_selection_summary.get("allocation_summary", {}),
        },
        "top_contributors": top_contributors,
        "weakest_contributors": weakest_contributors,
        "diversification_summary": diversification_summary,
        "portfolio_metrics": portfolio_metrics,
        "key_observations": [
            {
                "name": "selection_rate",
                "value": None if total == 0 else float(selected / total),
            },
            {
                "name": "eligibility_rejection_rate",
                "value": None if total == 0 else float(rejected_eligibility / total),
            },
            {
                "name": "redundancy_rejection_rate",
                "value": None if total == 0 else float(rejected_redundancy / total),
            },
            {
                "name": "average_selected_allocation_weight",
                "value": _safe_mean(
                    pd.to_numeric(
                        candidate_summary.loc[
                            candidate_summary["selection_status"].astype("string") == "selected",
                            "allocation_weight",
                        ],
                        errors="coerce",
                    ).dropna().tolist()
                ),
            },
            {
                "name": "selected_avg_abs_pairwise_correlation",
                "value": diversification_summary.get("average_pairwise_abs_correlation_selected"),
            },
        ],
    }


def _build_markdown_report(
    *,
    candidate_decisions: pd.DataFrame,
    candidate_contributions: pd.DataFrame,
    review_summary: dict[str, Any],
    top_n: int,
) -> str:
    lines = [
        "# Candidate Selection Review",
        "",
        "## Summary",
        f"- Candidate Selection Run: {review_summary.get('candidate_selection_run_id')}",
        f"- Portfolio Run: {review_summary.get('portfolio_run_id')}",
        f"- Total Candidates: {review_summary.get('counts', {}).get('total_candidates')}",
        f"- Selected Candidates: {review_summary.get('counts', {}).get('selected_candidates')}",
        f"- Rejected (Eligibility): {review_summary.get('counts', {}).get('rejected_by_stage', {}).get('eligibility')}",
        f"- Rejected (Redundancy): {review_summary.get('counts', {}).get('rejected_by_stage', {}).get('redundancy')}",
        "",
        "## Top Candidate Decisions",
        _markdown_table(
            candidate_decisions.head(top_n),
            columns=[
                "candidate_id",
                "selection_status",
                "selection_rank",
                "mean_ic",
                "ic_ir",
                "mean_rank_ic",
                "rejection_reasons",
            ],
        ),
        "",
        "## Contribution Highlights",
        _markdown_table(
            candidate_contributions.head(top_n),
            columns=[
                "candidate_id",
                "average_weight",
                "return_contribution",
                "return_contribution_share",
                "volatility_contribution",
                "turnover_contribution",
            ],
        ),
    ]
    return "\n".join(lines).rstrip() + "\n"


def _markdown_table(frame: pd.DataFrame, *, columns: list[str]) -> str:
    if frame.empty:
        return "No rows"
    subset = frame.loc[:, [column for column in columns if column in frame.columns]].copy()
    header = "| " + " | ".join(subset.columns) + " |"
    sep = "| " + " | ".join(["---" for _ in subset.columns]) + " |"
    rows = [header, sep]
    for _, row in subset.iterrows():
        rows.append("| " + " | ".join(_render_markdown_value(row[column]) for column in subset.columns) + " |")
    return "\n".join(rows)


def _render_markdown_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _top_bottom_contributors(frame: pd.DataFrame, *, top: bool, n: int) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    sorted_frame = frame.sort_values(["return_contribution", "candidate_id"], ascending=[not top, True], kind="stable")
    rows = []
    for _, row in sorted_frame.head(max(0, int(n))).iterrows():
        rows.append(
            {
                "candidate_id": row.get("candidate_id"),
                "return_contribution": _as_float(row.get("return_contribution")),
                "average_weight": _as_float(row.get("average_weight")),
                "return_contribution_share": _as_float(row.get("return_contribution_share")),
            }
        )
    return rows


def _safe_mean(values: list[float]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return None
    return float(sum(finite) / len(finite))


def _pairwise_values(
    matrix: pd.DataFrame,
    left_ids: list[str],
    right_ids: list[str],
    *,
    skip_diagonal: bool,
) -> list[float]:
    if matrix.empty:
        return []
    values: list[float] = []
    for left_id in left_ids:
        if left_id not in matrix.index:
            continue
        for right_id in right_ids:
            if right_id not in matrix.columns:
                continue
            if skip_diagonal and left_id == right_id:
                continue
            value = _as_float(matrix.loc[left_id, right_id])
            if value is None:
                continue
            values.append(value)
    return values


def _normalize_correlation_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    if "candidate_id" not in frame.columns:
        return pd.DataFrame()
    ids = frame["candidate_id"].astype("string").tolist()
    value_columns = [column for column in frame.columns if column != "candidate_id"]
    matrix = frame.set_index("candidate_id").loc[:, value_columns]
    matrix.index = matrix.index.astype("string")
    matrix.columns = [str(column) for column in matrix.columns]
    for column in matrix.columns:
        matrix[column] = pd.to_numeric(matrix[column], errors="coerce")
    ordered_ids = [str(value) for value in ids if str(value) in matrix.index]
    if not ordered_ids:
        return pd.DataFrame()
    keep_columns = [column for column in ordered_ids if column in matrix.columns]
    if not keep_columns:
        return pd.DataFrame()
    return matrix.loc[ordered_ids, keep_columns]


def _component_candidate_map(
    components: list[dict[str, Any]],
    *,
    strict: bool = False,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for component in components:
        provenance = component.get("provenance") if isinstance(component, dict) else None
        candidate_id = None
        if isinstance(provenance, dict) and provenance.get("candidate_id") is not None:
            candidate_id = str(provenance.get("candidate_id"))
        if candidate_id is None:
            if strict:
                raise CandidateReviewError("Portfolio components must include provenance.candidate_id for attribution.")
            continue
        out[candidate_id] = {
            "strategy_name": component.get("strategy_name"),
            "run_id": component.get("run_id"),
        }
    return out


def _status_from_rejection_row(rejected_row: dict[str, Any] | None) -> str:
    if rejected_row is None:
        return "selected"
    if str(rejected_row.get("rejected_stage") or "") == "eligibility_gate":
        return "rejected_eligibility"
    return "rejected_redundancy"


def _joined_reasons(rejected_row: dict[str, Any] | None) -> str:
    if rejected_row is None:
        return ""
    reasons: list[str] = []
    failed_checks = str(rejected_row.get("failed_checks") or "").strip()
    rejection_reason = str(rejected_row.get("rejection_reason") or "").strip()
    if failed_checks:
        for check in failed_checks.split("|"):
            normalized = check.strip()
            if normalized:
                reasons.append(normalized)
    if rejection_reason:
        reasons.append(rejection_reason)
    return "|".join(sorted(set(reasons)))


def _compare_threshold(*, value: float | None, threshold: float | None) -> str:
    if threshold is None:
        return "not_applicable"
    if value is None:
        return "missing"
    return "pass" if value >= threshold else "fail"


def _stable_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    working = frame.copy()
    if "selection_rank" in working.columns:
        working["selection_rank"] = pd.to_numeric(working["selection_rank"], errors="coerce").fillna(10**9)
    if "candidate_id" in working.columns:
        working["candidate_id"] = working["candidate_id"].astype("string")
    sort_columns = [column for column in ["selection_rank", "candidate_id", "alpha_name"] if column in working.columns]
    return working.sort_values(sort_columns, kind="stable").reset_index(drop=True)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise CandidateReviewError(f"Required artifact missing: {path}")
    try:
        return pd.read_csv(path)
    except (OSError, ValueError) as exc:
        raise CandidateReviewError(f"Failed to load CSV artifact {path}: {exc}") from exc


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise CandidateReviewError(f"Required artifact missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateReviewError(f"Failed to load JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CandidateReviewError(f"Expected JSON object in {path}.")
    return payload


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        frame.to_csv(handle, index=False, lineterminator="\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8", newline="\n")


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    cast = float(value)
    if not math.isfinite(cast):
        return None
    return cast


def _as_int(value: Any) -> int | None:
    numeric = _as_float(value)
    if numeric is None:
        return None
    return int(numeric)


def _indexed(frame: pd.DataFrame, key: str) -> dict[str, dict[str, Any]]:
    if frame.empty or key not in frame.columns:
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for _, row in frame.iterrows():
        row_key = str(row.get(key))
        if row_key:
            rows[row_key] = row.to_dict()
    return rows
