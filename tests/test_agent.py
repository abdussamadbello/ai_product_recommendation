from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from kargo_reco.agent import build_agent, build_tools, SYSTEM_PROMPT


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"creative_name": "Alpha", "click_through_rate": 0.42, "in_view_rate": 0.70, "vertical": "Retail", "minimum_budget": 20000, "vertical_normalized": "retail"},
            {"creative_name": "Beta", "click_through_rate": 0.38, "in_view_rate": 0.82, "vertical": "Retail", "minimum_budget": 18000, "vertical_normalized": "retail"},
        ]
    )


def test_build_tools_returns_five_tools(sample_df: pd.DataFrame) -> None:
    tools = build_tools(sample_df, vertical="Retail")
    assert len(tools) == 5
    tool_names = {t.name for t in tools}
    assert tool_names == {
        "get_eligible_products",
        "score_products",
        "get_product_details",
        "check_budget_remaining",
        "finalize_recommendation",
    }


def test_system_prompt_mentions_strategy() -> None:
    assert "vertical" in SYSTEM_PROMPT.lower()
    assert "budget" in SYSTEM_PROMPT.lower()
    assert "kpi" in SYSTEM_PROMPT.lower()


def test_build_agent_returns_compiled_graph(sample_df: pd.DataFrame) -> None:
    agent = build_agent(
        model_name="gpt-4.1-mini",
        api_key="test-key",
        base_url=None,
        timeout_s=10,
        df=sample_df,
        vertical="Retail",
    )
    # create_react_agent returns a CompiledStateGraph
    assert hasattr(agent, "invoke")
