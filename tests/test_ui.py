from __future__ import annotations

from unittest.mock import MagicMock, patch

from kargo_reco.ui import (
    _format_currency,
    _format_rate,
    _render_recommendation_cards,
    _render_summary,
    _render_agent_steps,
    _build_request_preview,
)
from kargo_reco.schemas import (
    AgentStep,
    RecommendationItem,
    RecommendationRequest,
    RecommendationResponse,
    ResponseMeta,
)


def _sample_response() -> RecommendationResponse:
    return RecommendationResponse(
        request=RecommendationRequest(
            client_name="Acme", kpi="click_through_rate", client_vertical="Retail", budget=25000
        ),
        recommendations=[
            RecommendationItem(
                creative_name="Alpha", vertical="Retail", minimum_budget=20000,
                click_through_rate=0.42, in_view_rate=0.79, rank=1,
            ),
        ],
        summary="Alpha is the best pick for Retail CTR.",
        agent_trace=[
            AgentStep(step_number=1, tool_name="filter_by_vertical", tool_input={"vertical": "Retail"}, tool_output={"products": []}, agent_reasoning="Start by filtering.", latency_ms=50),
            AgentStep(step_number=2, tool_name="finalize_recommendation", tool_input={"products": ["Alpha"]}, tool_output={}, agent_reasoning=None, latency_ms=30),
        ],
        meta=ResponseMeta(
            status="success", request_id="abc-123", model="gpt-4.1-mini",
            total_tokens=100, agent_steps=2, latency_ms=500, source="llm",
        ),
    )


def test_format_currency() -> None:
    assert _format_currency(25000) == "$25,000"
    assert _format_currency(None) == "N/A"


def test_format_rate() -> None:
    assert _format_rate(0.42) == "42%"
    assert _format_rate(None) == "N/A"


def test_render_recommendation_cards_success() -> None:
    html = _render_recommendation_cards(_sample_response())
    assert "Alpha" in html
    assert "Retail" in html


def test_render_summary() -> None:
    html = _render_summary(_sample_response())
    assert "Alpha is the best pick" in html


def test_render_agent_steps() -> None:
    html = _render_agent_steps(_sample_response())
    assert "filter_by_vertical" in html
    assert "finalize_recommendation" in html
    assert "Start by filtering" in html


def test_build_request_preview() -> None:
    preview = _build_request_preview("Acme", "click_through_rate", "Retail", 25000)
    assert preview["client_name"] == "Acme"
    assert preview["budget"] == 25000
