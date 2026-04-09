from __future__ import annotations

import pandas as pd
import pytest

from kargo_reco.guardrails import validate_agent_output, deterministic_fallback
from kargo_reco.schemas import RecommendationRequest


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"creative_name": "Alpha", "click_through_rate": 0.42, "in_view_rate": 0.70, "vertical": "Retail", "minimum_budget": 20000, "vertical_normalized": "retail"},
            {"creative_name": "Beta", "click_through_rate": 0.38, "in_view_rate": 0.82, "vertical": "Retail", "minimum_budget": 18000, "vertical_normalized": "retail"},
            {"creative_name": "Delta", "click_through_rate": 0.60, "in_view_rate": 0.90, "vertical": "Finance", "minimum_budget": 50000, "vertical_normalized": "finance"},
        ]
    )


@pytest.fixture
def retail_request() -> RecommendationRequest:
    return RecommendationRequest(
        client_name="Acme", kpi="click_through_rate", client_vertical="Retail", budget=25000
    )


def test_validate_passes_correct_output(sample_df, retail_request):
    agent_output = {
        "recommendations": [
            {"creative_name": "Alpha", "vertical": "Retail", "minimum_budget": 20000, "click_through_rate": 0.42, "in_view_rate": 0.70, "rank": 1},
        ],
        "reasoning": "Alpha is the best CTR pick.",
    }
    violations = validate_agent_output(agent_output, sample_df, retail_request)
    assert violations == []


def test_validate_catches_hallucinated_product(sample_df, retail_request):
    agent_output = {
        "recommendations": [
            {"creative_name": "FakeProduct", "vertical": "Retail", "minimum_budget": 20000, "click_through_rate": 0.99, "in_view_rate": 0.99, "rank": 1},
        ],
        "reasoning": "FakeProduct is great.",
    }
    violations = validate_agent_output(agent_output, sample_df, retail_request)
    assert any("not found" in v for v in violations)


def test_validate_catches_vertical_mismatch(sample_df, retail_request):
    agent_output = {
        "recommendations": [
            {"creative_name": "Delta", "vertical": "Finance", "minimum_budget": 50000, "click_through_rate": 0.60, "in_view_rate": 0.90, "rank": 1},
        ],
        "reasoning": "Delta has the best CTR.",
    }
    violations = validate_agent_output(agent_output, sample_df, retail_request)
    assert any("vertical" in v.lower() for v in violations)


def test_validate_catches_budget_overrun(sample_df, retail_request):
    agent_output = {
        "recommendations": [
            {"creative_name": "Alpha", "vertical": "Retail", "minimum_budget": 20000, "click_through_rate": 0.42, "in_view_rate": 0.70, "rank": 1},
            {"creative_name": "Beta", "vertical": "Retail", "minimum_budget": 18000, "click_through_rate": 0.38, "in_view_rate": 0.82, "rank": 2},
        ],
        "reasoning": "Bundle both.",
    }
    violations = validate_agent_output(agent_output, sample_df, retail_request)
    assert any("budget" in v.lower() for v in violations)


def test_validate_passes_bundled_within_budget(sample_df):
    request = RecommendationRequest(
        client_name="Acme", kpi="click_through_rate", client_vertical="Retail", budget=50000
    )
    agent_output = {
        "recommendations": [
            {"creative_name": "Alpha", "vertical": "Retail", "minimum_budget": 20000, "click_through_rate": 0.42, "in_view_rate": 0.70, "rank": 1},
            {"creative_name": "Beta", "vertical": "Retail", "minimum_budget": 18000, "click_through_rate": 0.38, "in_view_rate": 0.82, "rank": 2},
        ],
        "reasoning": "Both fit within budget.",
    }
    violations = validate_agent_output(agent_output, sample_df, request)
    assert violations == []


def test_deterministic_fallback_returns_best_by_kpi(sample_df, retail_request):
    result = deterministic_fallback(sample_df, retail_request)
    assert result["status"] == "success"
    assert len(result["recommendations"]) == 1
    assert result["recommendations"][0]["creative_name"] == "Alpha"


def test_deterministic_fallback_no_match(sample_df):
    request = RecommendationRequest(
        client_name="Acme", kpi="click_through_rate", client_vertical="Travel", budget=100000
    )
    result = deterministic_fallback(sample_df, request)
    assert result["status"] == "no_match"
    assert result["recommendations"] == []


def test_deterministic_fallback_budget_too_low(sample_df):
    request = RecommendationRequest(
        client_name="Acme", kpi="click_through_rate", client_vertical="Retail", budget=1000
    )
    result = deterministic_fallback(sample_df, request)
    assert result["status"] == "no_match"
