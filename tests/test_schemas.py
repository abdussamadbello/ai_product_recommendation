from __future__ import annotations

import pytest
from pydantic import ValidationError

from kargo_reco.schemas import (
    AgentStep,
    RecommendationItem,
    RecommendationRequest,
    RecommendationResponse,
    ResponseMeta,
    TraceArtifact,
)


def test_recommendation_request_normalizes_ctr_alias() -> None:
    req = RecommendationRequest(
        client_name="Acme", kpi="CTR", client_vertical="Retail", budget=25000
    )
    assert req.kpi == "click_through_rate"


def test_recommendation_request_normalizes_ivr_alias() -> None:
    req = RecommendationRequest(
        client_name="Acme", kpi="IVR", client_vertical="Retail", budget=25000
    )
    assert req.kpi == "in_view_rate"


def test_recommendation_request_rejects_empty_client_name() -> None:
    with pytest.raises(ValidationError):
        RecommendationRequest(
            client_name="", kpi="click_through_rate", client_vertical="Retail", budget=25000
        )


def test_recommendation_request_rejects_zero_budget() -> None:
    with pytest.raises(ValidationError):
        RecommendationRequest(
            client_name="Acme", kpi="click_through_rate", client_vertical="Retail", budget=0
        )


def test_recommendation_request_coerces_string_budget() -> None:
    req = RecommendationRequest(
        client_name="Acme", kpi="click_through_rate", client_vertical="Retail", budget="25000"
    )
    assert req.budget == 25000.0


def test_agent_step_stores_tool_call_data() -> None:
    step = AgentStep(
        step_number=1,
        tool_name="filter_by_vertical",
        tool_input={"vertical": "Retail"},
        tool_output={"products": [{"creative_name": "Alpha"}]},
        agent_reasoning="I need to find Retail products first.",
        latency_ms=42,
    )
    assert step.tool_name == "filter_by_vertical"
    assert step.agent_reasoning is not None


def test_agent_step_allows_none_reasoning() -> None:
    step = AgentStep(
        step_number=1,
        tool_name="filter_by_vertical",
        tool_input={},
        tool_output={},
        agent_reasoning=None,
        latency_ms=10,
    )
    assert step.agent_reasoning is None


def test_recommendation_item_stores_product_data() -> None:
    item = RecommendationItem(
        creative_name="Alpha",
        vertical="Retail",
        minimum_budget=20000,
        click_through_rate=0.42,
        in_view_rate=0.79,
        rank=1,
    )
    assert item.creative_name == "Alpha"
    assert item.rank == 1


def test_response_meta_accepts_all_status_values() -> None:
    for status in ("success", "no_match", "guardrail_fallback"):
        meta = ResponseMeta(
            status=status,
            request_id="abc-123",
            model="gpt-4.1-mini",
            total_tokens=None,
            agent_steps=3,
            latency_ms=500,
            source="llm",
        )
        assert meta.status == status


def test_recommendation_response_assembles_correctly() -> None:
    response = RecommendationResponse(
        request=RecommendationRequest(
            client_name="Acme", kpi="click_through_rate", client_vertical="Retail", budget=25000
        ),
        recommendations=[
            RecommendationItem(
                creative_name="Alpha",
                vertical="Retail",
                minimum_budget=20000,
                click_through_rate=0.42,
                in_view_rate=0.79,
                rank=1,
            )
        ],
        summary="Alpha is the best pick for Retail CTR.",
        agent_trace=[],
        meta=ResponseMeta(
            status="success",
            request_id="abc-123",
            model="gpt-4.1-mini",
            total_tokens=100,
            agent_steps=4,
            latency_ms=1200,
            source="llm",
        ),
    )
    assert len(response.recommendations) == 1
    assert response.summary.startswith("Alpha")


def test_trace_artifact_generates_request_id() -> None:
    trace = TraceArtifact(source="test", raw_request={"client_name": "Acme"})
    assert trace.request_id is not None
    assert len(trace.request_id) > 0
