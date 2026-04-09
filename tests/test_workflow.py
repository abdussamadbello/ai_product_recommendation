from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from langchain_core.messages import AIMessage, ToolMessage

from kargo_reco.schemas import RecommendationRequest


def _mock_agent_response(tool_calls_sequence: list[tuple[str, dict, str]]) -> list:
    """Build a mock message history for the agent.

    Each tuple is (tool_name, tool_args, tool_result_json).
    """
    messages = []
    for tool_name, tool_args, tool_result in tool_calls_sequence:
        call_id = f"call_{tool_name}"
        messages.append(AIMessage(
            content=f"I'll use {tool_name}.",
            tool_calls=[{"id": call_id, "name": tool_name, "args": tool_args}],
        ))
        messages.append(ToolMessage(
            content=tool_result,
            tool_call_id=call_id,
        ))
    # Final AI message with no tool calls (agent done)
    messages.append(AIMessage(content="Done."))
    return messages


def test_workflow_success_with_mocked_agent(workflow_runner) -> None:
    mock_messages = _mock_agent_response([
        ("filter_by_vertical", {"vertical": "Retail"}, json.dumps([
            {"creative_name": "Retail Rocket", "click_through_rate": 0.42, "in_view_rate": 0.79, "vertical": "Retail", "minimum_budget": 20000},
            {"creative_name": "Retail Spotlight", "click_through_rate": 0.42, "in_view_rate": 0.75, "vertical": "Retail", "minimum_budget": 18000},
        ])),
        ("filter_by_budget", {"budget": 25000, "products": []}, json.dumps([
            {"creative_name": "Retail Rocket", "click_through_rate": 0.42, "in_view_rate": 0.79, "vertical": "Retail", "minimum_budget": 20000},
            {"creative_name": "Retail Spotlight", "click_through_rate": 0.42, "in_view_rate": 0.75, "vertical": "Retail", "minimum_budget": 18000},
        ])),
        ("sort_by_kpi", {"kpi": "click_through_rate", "products": [], "limit": 3}, json.dumps([
            {"creative_name": "Retail Rocket", "click_through_rate": 0.42, "in_view_rate": 0.79, "vertical": "Retail", "minimum_budget": 20000},
        ])),
        ("finalize_recommendation", {"products": ["Retail Rocket"], "reasoning": "Retail Rocket has the highest CTR at 42% within the $25k budget."}, json.dumps({
            "recommendations": [{"creative_name": "Retail Rocket", "vertical": "Retail", "minimum_budget": 20000, "click_through_rate": 0.42, "in_view_rate": 0.79, "rank": 1}],
            "reasoning": "Retail Rocket has the highest CTR at 42% within the $25k budget.",
        })),
    ])

    with patch.object(workflow_runner, "_run_agent", return_value=mock_messages):
        response = workflow_runner.run_from_payload(
            {"client_name": "Acme", "kpi": "CLICK_THROUGH_RATE", "client_vertical": "Retail", "budget": 25000},
            source="test",
        )

    assert response.meta.status == "success"
    assert response.recommendations[0].creative_name == "Retail Rocket"
    assert response.meta.source == "llm"
    assert len(response.agent_trace) > 0
    assert response.meta.trace_path is not None


def test_workflow_guardrail_fallback_on_hallucination(workflow_runner) -> None:
    mock_messages = _mock_agent_response([
        ("finalize_recommendation", {"products": ["FakeProduct"], "reasoning": "Best pick."}, json.dumps({
            "recommendations": [{"creative_name": "FakeProduct", "vertical": "Retail", "minimum_budget": 10000, "click_through_rate": 0.99, "in_view_rate": 0.99, "rank": 1}],
            "reasoning": "Best pick.",
        })),
    ])

    with patch.object(workflow_runner, "_run_agent", return_value=mock_messages):
        response = workflow_runner.run_from_payload(
            {"client_name": "Acme", "kpi": "CLICK_THROUGH_RATE", "client_vertical": "Retail", "budget": 25000},
            source="test",
        )

    assert response.meta.status in ("success", "guardrail_fallback")
    assert response.meta.source == "guardrail_fallback"
    assert response.recommendations[0].creative_name in ("Retail Rocket", "Retail Spotlight")


def test_workflow_no_match_vertical(workflow_runner) -> None:
    mock_messages = _mock_agent_response([
        ("filter_by_vertical", {"vertical": "Travel"}, json.dumps([])),
        ("finalize_recommendation", {"products": [], "reasoning": "No Travel products found."}, json.dumps({
            "recommendations": [],
            "reasoning": "No Travel products found.",
        })),
    ])

    with patch.object(workflow_runner, "_run_agent", return_value=mock_messages):
        response = workflow_runner.run_from_payload(
            {"client_name": "Explorer", "kpi": "in_view_rate", "client_vertical": "Travel", "budget": 50000},
            source="test",
        )

    assert response.meta.status == "no_match"


def test_reload_benchmarks_returns_metadata(workflow_runner) -> None:
    metadata = workflow_runner.reload_benchmarks()
    assert metadata["row_count"] == 7
    assert metadata["schema_valid"] is True


def test_upload_benchmarks_switches_active_dataset(workflow_runner, tmp_path: Path) -> None:
    uploaded_csv = tmp_path / "uploaded.csv"
    uploaded_csv.write_text(
        "\n".join(
            [
                "creative_name,click_through_rate,in_view_rate,vertical,minimum_budget",
                "Travel Lift,0.55,0.77,Travel,10000",
            ]
        ),
        encoding="utf-8",
    )

    metadata = workflow_runner.upload_benchmarks(str(uploaded_csv))
    assert metadata["row_count"] == 1
    assert "uploads" in metadata["file"]
