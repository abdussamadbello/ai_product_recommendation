from __future__ import annotations

import pandas as pd

from kargo_reco.recommender import compute_recommendation
from kargo_reco.schemas import RecommendationRequest


def _frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "creative_name": "Alpha",
                "click_through_rate": 0.42,
                "in_view_rate": 0.70,
                "vertical": "Retail",
                "minimum_budget": 20000,
                "vertical_normalized": "retail",
            },
            {
                "creative_name": "Beta",
                "click_through_rate": 0.42,
                "in_view_rate": 0.65,
                "vertical": "Retail",
                "minimum_budget": 18000,
                "vertical_normalized": "retail",
            },
            {
                "creative_name": "Gamma",
                "click_through_rate": 0.42,
                "in_view_rate": 0.65,
                "vertical": "Retail",
                "minimum_budget": 18000,
                "vertical_normalized": "retail",
            },
            {
                "creative_name": "Auto One",
                "click_through_rate": 0.60,
                "in_view_rate": 0.90,
                "vertical": "Automotive",
                "minimum_budget": 50000,
                "vertical_normalized": "automotive",
            },
        ]
    )
    return frame


def test_compute_recommendation_uses_budget_tiebreak() -> None:
    request = RecommendationRequest(
        client_name="Acme",
        kpi="click_through_rate",
        client_vertical="Retail",
        budget=25000,
    )

    result = compute_recommendation(_frame(), request)

    assert result.selected is not None
    assert result.selected.creative_name == "Beta"
    assert "lower minimum_budget" in result.tie_break_notes[0]
    assert result.post_vertical_count == 3
    assert result.post_budget_count == 3


def test_compute_recommendation_uses_alphabetical_tiebreak_as_last_resort() -> None:
    frame = _frame()
    frame.loc[frame["creative_name"] == "Beta", "minimum_budget"] = 18000
    frame.loc[frame["creative_name"] == "Beta", "in_view_rate"] = 0.65
    request = RecommendationRequest(
        client_name="Acme",
        kpi="click_through_rate",
        client_vertical="Retail",
        budget=25000,
    )

    result = compute_recommendation(frame, request)

    assert result.selected is not None
    assert result.selected.creative_name == "Beta"


def test_compute_recommendation_returns_nearest_same_vertical_alternative() -> None:
    request = RecommendationRequest(
        client_name="Acme",
        kpi="click_through_rate",
        client_vertical="Retail",
        budget=10000,
    )

    result = compute_recommendation(_frame(), request)

    assert result.selected is None
    assert result.nearest_alternative is not None
    assert result.nearest_alternative.creative_name == "Beta"
    assert result.nearest_alternative.block_reason == "budget_shortfall"


def test_compute_recommendation_returns_vertical_mismatch_when_no_vertical_candidates() -> None:
    request = RecommendationRequest(
        client_name="Acme",
        kpi="in_view_rate",
        client_vertical="Finance",
        budget=100000,
    )

    result = compute_recommendation(_frame(), request)

    assert result.selected is None
    assert result.nearest_alternative is not None
    assert result.nearest_alternative.block_reason == "vertical_mismatch"
