from __future__ import annotations

import pytest
from pydantic import ValidationError

from kargo_reco.schemas import RecommendationRequest


def test_request_normalizes_kpi_and_text() -> None:
    request = RecommendationRequest(
        client_name="  Acme Shoes  ",
        kpi="CLICK_THROUGH_RATE",
        client_vertical="  Retail  ",
        budget="25000",
    )

    assert request.client_name == "Acme Shoes"
    assert request.kpi == "click_through_rate"
    assert request.client_vertical == "Retail"
    assert request.budget == 25000.0


def test_request_accepts_minimum_budget_alias() -> None:
    request = RecommendationRequest(
        client_name="Acme Shoes",
        kpi="IN_VIEW_RATE",
        client_vertical="Retail",
        minimum_budget="18000",
    )

    assert request.kpi == "in_view_rate"
    assert request.budget == 18000.0


def test_request_rejects_invalid_budget() -> None:
    with pytest.raises(ValidationError):
        RecommendationRequest(
            client_name="Acme",
            kpi="in_view_rate",
            client_vertical="Retail",
            budget=0,
        )


def test_request_rejects_unknown_kpi() -> None:
    with pytest.raises(ValidationError):
        RecommendationRequest(
            client_name="Acme",
            kpi="conversions",
            client_vertical="Retail",
            budget=100,
        )
