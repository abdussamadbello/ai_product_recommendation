from __future__ import annotations

from typing import Any

import pandas as pd
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from kargo_reco import tools as tool_fns


SYSTEM_PROMPT = """You are a media strategist assistant for Kargo. Given a client request, use the available tools to explore the product catalog, find the best product(s) for the client's vertical and KPI, and stay within budget.

You MUST call finalize_recommendation as your last tool call to commit your selection.

Suggested strategy:
1. Use filter_by_vertical to find products in the client's vertical.
2. Use filter_by_budget to narrow to affordable products.
3. Use score_products to create a weighted ranking formula that reflects the client's priorities. For example, if the KPI is click_through_rate, you might weight CTR at 0.8 and IVR at 0.2 — but you decide the weights based on the client's context.
4. Use get_product_details to inspect the top candidate if needed.
5. Use check_budget_remaining to see if a second product can be added.
6. If budget allows, pick the next-best product that fits.
7. Call finalize_recommendation with your selected product(s) and reasoning.

You can also use sort_by_kpi for a simple single-KPI sort if a weighted formula isn't needed.

When reasoning, explain trade-offs: why you picked this product over alternatives, what weights you chose and why, how budget utilization factored in, and whether bundling adds value.
"""


def build_tools(df: pd.DataFrame, vertical: str) -> list[StructuredTool]:
    """Create the 7 LangChain tools that the agent can call, bound to a DataFrame."""

    def _filter_by_vertical(vertical: str) -> list[dict[str, Any]]:
        """Find all products available in a given vertical."""
        return tool_fns.filter_by_vertical(df, vertical=vertical)

    def _filter_by_budget(budget: float, products: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Filter a list of products to only those within the given budget."""
        return tool_fns.filter_by_budget(budget=budget, products=products)

    def _sort_by_kpi(kpi: str, products: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
        """Sort products by a KPI (click_through_rate or in_view_rate) descending. Returns top N."""
        return tool_fns.sort_by_kpi(kpi=kpi, products=products, limit=limit)

    def _score_products(products: list[dict[str, Any]], weights: dict[str, float], limit: int = 5) -> list[dict[str, Any]]:
        """Score and rank products using your own weighted formula. Set weights for click_through_rate and/or in_view_rate to reflect the client's priorities. Example: {"click_through_rate": 0.8, "in_view_rate": 0.2}."""
        return tool_fns.score_products(products=products, weights=weights, limit=limit)

    def _get_product_details(product_name: str, vertical: str) -> dict[str, Any] | None:
        """Get full details for a specific product by name and vertical."""
        return tool_fns.get_product_details(df, product_name=product_name, vertical=vertical)

    def _check_budget_remaining(budget: float, selected: list[str]) -> dict[str, Any]:
        """Check how much budget remains after selecting products, and what else is affordable."""
        return tool_fns.check_budget_remaining(df, budget=budget, selected=selected, vertical=vertical)

    def _finalize_recommendation(products: list[str], reasoning: str) -> dict[str, Any]:
        """Commit your final product selection with reasoning. This MUST be your last tool call."""
        return tool_fns.finalize_recommendation(df, products=products, reasoning=reasoning, vertical=vertical)

    return [
        StructuredTool.from_function(_filter_by_vertical, name="filter_by_vertical"),
        StructuredTool.from_function(_filter_by_budget, name="filter_by_budget"),
        StructuredTool.from_function(_sort_by_kpi, name="sort_by_kpi"),
        StructuredTool.from_function(_score_products, name="score_products"),
        StructuredTool.from_function(_get_product_details, name="get_product_details"),
        StructuredTool.from_function(_check_budget_remaining, name="check_budget_remaining"),
        StructuredTool.from_function(_finalize_recommendation, name="finalize_recommendation"),
    ]


def build_agent(
    *,
    model_name: str,
    api_key: str | None,
    base_url: str | None,
    timeout_s: float,
    df: pd.DataFrame,
    vertical: str,
):
    """Construct a LangGraph ReAct agent with the 7 product tools."""
    effective_api_key = api_key or ("ollama" if base_url else "not-set")
    llm = ChatOpenAI(
        model=model_name,
        api_key=effective_api_key,
        base_url=base_url,
        temperature=0,
        timeout=timeout_s,
    )
    tools = build_tools(df, vertical)
    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=SYSTEM_PROMPT,
        name="product_recommendation_agent",
    )
