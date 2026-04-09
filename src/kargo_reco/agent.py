from __future__ import annotations

from typing import Any

import pandas as pd
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from kargo_reco import tools as tool_fns


SYSTEM_PROMPT = """You are a media strategist assistant for Kargo. Given a client request, use the available tools to find the best product(s) for the client's vertical and KPI within budget.

You MUST call finalize_recommendation as your last tool call to commit your selection.

Strategy:
1. Call get_eligible_products to see all products in the client's vertical that fit the budget.
2. Call score_products with a weighted formula reflecting the client's priorities. For example, if the KPI is click_through_rate, you might weight CTR at 0.8 and IVR at 0.2 — but you decide the weights.
3. Call check_budget_remaining to see if a second product can be added within the remaining budget.
4. If budget allows, consider bundling the next-best product.
5. Call finalize_recommendation with your selected product(s) and a detailed reasoning.

In your reasoning, explain: what weights you chose and why, why you picked this product over alternatives, how budget utilization factored in, and whether bundling adds value.
"""


def build_tools(df: pd.DataFrame, vertical: str) -> list[StructuredTool]:
    """Create the 5 LangChain tools that the agent can call, bound to a DataFrame."""

    def _get_eligible_products(vertical: str, budget: float) -> dict[str, Any]:
        """Get all products in a vertical that fit within the budget. Returns product count and full list."""
        return tool_fns.get_eligible_products(df, vertical=vertical, budget=budget)

    def _score_products(weights: dict[str, float], budget: float | None = None, limit: int = 5) -> list[dict[str, Any]]:
        """Score and rank products using your own weighted formula. Set weights for click_through_rate and/or in_view_rate. Example: {"click_through_rate": 0.8, "in_view_rate": 0.2}. Pass budget to auto-filter. Returns scored products sorted by your formula."""
        products = tool_fns.filter_by_vertical(df, vertical=vertical)
        if budget is not None:
            products = tool_fns.filter_by_budget(budget=budget, products=products)
        return tool_fns.score_products(products=products, weights=weights, limit=limit)

    def _check_budget_remaining(budget: float, selected: list[str]) -> dict[str, Any]:
        """Check how much budget remains after selecting products, and what else is affordable in the vertical."""
        return tool_fns.check_budget_remaining(df, budget=budget, selected=selected, vertical=vertical)

    def _get_product_details(product_name: str, vertical: str) -> dict[str, Any] | None:
        """Get full details for a specific product by name and vertical."""
        return tool_fns.get_product_details(df, product_name=product_name, vertical=vertical)

    def _finalize_recommendation(products: list[str], reasoning: str) -> dict[str, Any]:
        """Commit your final product selection with reasoning. This MUST be your last tool call."""
        return tool_fns.finalize_recommendation(df, products=products, reasoning=reasoning, vertical=vertical)

    return [
        StructuredTool.from_function(_get_eligible_products, name="get_eligible_products"),
        StructuredTool.from_function(_score_products, name="score_products"),
        StructuredTool.from_function(_check_budget_remaining, name="check_budget_remaining"),
        StructuredTool.from_function(_get_product_details, name="get_product_details"),
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
    """Construct a LangGraph ReAct agent with the 5 product tools."""
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
