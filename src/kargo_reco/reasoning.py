from __future__ import annotations

from typing import Any

from kargo_reco.recommender import RecommendationComputation
from kargo_reco.schemas import RecommendationRequest


def build_reasoning_facts(
    request: RecommendationRequest,
    computation: RecommendationComputation,
) -> dict[str, Any]:
    alternatives = []
    if not computation.ranking_table.empty:
        alternatives = computation.ranking_table.iloc[1:4].to_dict(orient="records")

    facts: dict[str, Any] = {
        "request": request.model_dump(),
        "decision_status": "success" if computation.selected else "no_match",
        "candidate_counts": {
            "pre_filter_count": computation.pre_filter_count,
            "post_vertical_count": computation.post_vertical_count,
            "post_budget_count": computation.post_budget_count,
        },
        "tie_break_notes": computation.tie_break_notes,
        "selected_product": (
            computation.selected.model_dump() if computation.selected else None
        ),
        "eligible_alternatives": alternatives,
        "nearest_alternative": (
            computation.nearest_alternative.model_dump()
            if computation.nearest_alternative
            else None
        ),
        "no_match_reason": computation.no_match_reason,
    }

    if computation.selected:
        facts["structured_reasons"] = [
            "Matched client vertical",
            "Fit within budget",
            f"Highest eligible {request.kpi}",
        ]
        if computation.tie_break_notes:
            facts["structured_reasons"].append(computation.tie_break_notes[0])
    else:
        facts["structured_reasons"] = [
            "No product satisfied both the requested vertical and budget constraints.",
        ]
        if computation.no_match_reason:
            facts["structured_reasons"].append(computation.no_match_reason)

    return facts


def build_fallback_summary(facts: dict[str, Any]) -> tuple[str, list[str], list[str] | None]:
    if facts["decision_status"] == "success":
        selected = facts["selected_product"]
        short_text = (
            f"{selected['creative_name']} was selected because it matched the "
            f"{facts['request']['client_vertical']} vertical, fit within the "
            f"{facts['request']['budget']:.0f} budget, and led eligible options on "
            f"{facts['request']['kpi']}."
        )
        alternative_notes = [
            f"{item['creative_name']} ranked below the selected product on the deterministic sort order."
            for item in facts["eligible_alternatives"]
        ] or None
        return short_text, list(facts["structured_reasons"]), alternative_notes

    nearest = facts.get("nearest_alternative")
    short_text = "No eligible product matched the request."
    alternative_notes = None
    if nearest:
        short_text += f" Closest alternative: {nearest['creative_name']} ({nearest['block_reason']})."
        alternative_notes = [
            f"{nearest['creative_name']} was the nearest alternative but was blocked by {nearest['block_reason']}."
        ]
    return short_text, list(facts["structured_reasons"]), alternative_notes
