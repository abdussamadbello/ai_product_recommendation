from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from kargo_reco.schemas import NearestAlternative, RecommendationItem, RecommendationRequest, SelectionReason


@dataclass
class RecommendationComputation:
    pre_filter_count: int
    post_vertical_count: int
    post_budget_count: int
    ranking_table: pd.DataFrame
    eligible: pd.DataFrame
    selected: RecommendationItem | None
    nearest_alternative: NearestAlternative | None
    tie_break_notes: list[str]
    no_match_reason: str | None


def _row_to_recommendation(row: pd.Series, request: RecommendationRequest, rank: int = 1) -> RecommendationItem:
    return RecommendationItem(
        creative_name=str(row["creative_name"]),
        vertical=str(row["vertical"]),
        minimum_budget=float(row["minimum_budget"]),
        click_through_rate=float(row["click_through_rate"]),
        in_view_rate=float(row["in_view_rate"]),
        rank=rank,
        selection_reason=SelectionReason(
            matched_vertical=True,
            within_budget=True,
            optimized_kpi=request.kpi,
        ),
    )


def _row_to_alternative(
    row: pd.Series, *, block_reason: str, budget_shortfall: float | None = None
) -> NearestAlternative:
    return NearestAlternative(
        creative_name=str(row["creative_name"]),
        vertical=str(row["vertical"]),
        minimum_budget=float(row["minimum_budget"]),
        click_through_rate=float(row["click_through_rate"]),
        in_view_rate=float(row["in_view_rate"]),
        block_reason=block_reason,
        budget_shortfall=budget_shortfall,
    )


def _sorted_candidates(frame: pd.DataFrame, request: RecommendationRequest) -> pd.DataFrame:
    other_kpi = "in_view_rate" if request.kpi == "click_through_rate" else "click_through_rate"
    sorted_frame = frame.sort_values(
        by=[request.kpi, "minimum_budget", other_kpi, "creative_name"],
        ascending=[False, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    if not sorted_frame.empty:
        sorted_frame["rank"] = sorted_frame.index + 1
    return sorted_frame


def _tie_break_notes(sorted_frame: pd.DataFrame, request: RecommendationRequest) -> list[str]:
    if len(sorted_frame) < 2:
        return []

    requested_kpi = request.kpi
    other_kpi = "in_view_rate" if requested_kpi == "click_through_rate" else "click_through_rate"
    top = sorted_frame.iloc[0]
    same_primary = sorted_frame[sorted_frame[requested_kpi] == top[requested_kpi]]
    notes: list[str] = []

    if len(same_primary) > 1:
        if same_primary["minimum_budget"].nunique() > 1:
            notes.append(
                f"Tie on {requested_kpi} resolved by lower minimum_budget for {top['creative_name']}."
            )
            return notes
        same_budget = same_primary[same_primary["minimum_budget"] == top["minimum_budget"]]
        if len(same_budget) > 1 and same_budget[other_kpi].nunique() > 1:
            notes.append(
                f"Tie on {requested_kpi} and minimum_budget resolved by higher {other_kpi} for {top['creative_name']}."
            )
            return notes
        same_other = same_budget[same_budget[other_kpi] == top[other_kpi]]
        if len(same_other) > 1:
            notes.append(
                f"Tie on both KPI metrics and minimum_budget resolved alphabetically by creative_name for {top['creative_name']}."
            )
    return notes


def _nearest_alternative(
    all_candidates: pd.DataFrame,
    vertical_candidates: pd.DataFrame,
    request: RecommendationRequest,
) -> tuple[NearestAlternative | None, str | None]:
    if not vertical_candidates.empty:
        shortfalls = vertical_candidates.copy()
        shortfalls["budget_shortfall"] = shortfalls["minimum_budget"] - request.budget
        shortfalls = shortfalls[shortfalls["budget_shortfall"] > 0]
        if not shortfalls.empty:
            shortfalls = shortfalls.sort_values(
                by=["budget_shortfall", request.kpi, "creative_name"],
                ascending=[True, False, True],
                kind="mergesort",
            )
            row = shortfalls.iloc[0]
            return (
                _row_to_alternative(
                    row,
                    block_reason="budget_shortfall",
                    budget_shortfall=float(row["budget_shortfall"]),
                ),
                "No candidate matched the requested vertical within budget. Closest same-vertical option exceeds budget.",
            )

    if all_candidates.empty:
        return None, "Benchmark data is empty."

    global_best = _sorted_candidates(all_candidates, request).iloc[0]
    shortfall = max(float(global_best["minimum_budget"]) - request.budget, 0.0)
    reason = "vertical_mismatch"
    return (
        _row_to_alternative(
            global_best,
            block_reason=reason,
            budget_shortfall=shortfall or None,
        ),
        "No candidate matched the requested vertical. Showing the strongest global alternative instead.",
    )


def compute_recommendation(
    frame: pd.DataFrame,
    request: RecommendationRequest,
) -> RecommendationComputation:
    working = frame.copy()
    pre_filter_count = len(working)
    requested_vertical = request.client_vertical.strip().lower()

    vertical_candidates = working[working["vertical_normalized"] == requested_vertical].copy()
    post_vertical_count = len(vertical_candidates)

    budget_candidates = vertical_candidates[vertical_candidates["minimum_budget"] <= request.budget].copy()
    ranking_table = _sorted_candidates(budget_candidates, request)
    post_budget_count = len(ranking_table)

    if not ranking_table.empty:
        selected_row = ranking_table.iloc[0]
        selected = _row_to_recommendation(selected_row, request, rank=1)
        no_match_reason = None
        nearest_alternative = None
    else:
        selected = None
        nearest_alternative, no_match_reason = _nearest_alternative(
            working, vertical_candidates, request
        )

    return RecommendationComputation(
        pre_filter_count=pre_filter_count,
        post_vertical_count=post_vertical_count,
        post_budget_count=post_budget_count,
        ranking_table=ranking_table,
        eligible=budget_candidates,
        selected=selected,
        nearest_alternative=nearest_alternative,
        tie_break_notes=_tie_break_notes(ranking_table, request),
        no_match_reason=no_match_reason,
    )
