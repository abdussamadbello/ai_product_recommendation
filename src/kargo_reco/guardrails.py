from __future__ import annotations

from typing import Any

import pandas as pd

from kargo_reco.schemas import RecommendationRequest


def validate_agent_output(
    agent_output: dict[str, Any],
    df: pd.DataFrame,
    request: RecommendationRequest,
) -> list[str]:
    """Validate the agent's recommendation against hard constraints.

    Returns a list of violation descriptions. Empty list means valid.
    """
    violations: list[str] = []
    recommendations = agent_output.get("recommendations", [])

    if not recommendations:
        return violations

    requested_vertical = request.client_vertical.strip().lower()
    total_budget_used = 0.0

    for rec in recommendations:
        name = rec.get("creative_name", "")
        vertical = rec.get("vertical", "").strip().lower()

        # Check product exists in CSV
        matches = df[
            (df["creative_name"] == name) & (df["vertical_normalized"] == requested_vertical)
        ]
        if matches.empty:
            violations.append(f"Product '{name}' not found in {request.client_vertical} vertical")
            continue

        # Check vertical matches request
        if vertical != requested_vertical:
            violations.append(
                f"Product '{name}' vertical '{rec.get('vertical')}' does not match requested '{request.client_vertical}'"
            )

        total_budget_used += float(matches.iloc[0]["minimum_budget"])

    # Check total budget
    if total_budget_used > request.budget:
        violations.append(
            f"Total minimum budget ${total_budget_used:,.0f} exceeds client budget ${request.budget:,.0f}"
        )

    return violations


def deterministic_fallback(
    df: pd.DataFrame,
    request: RecommendationRequest,
) -> dict[str, Any]:
    """Produce a correct recommendation using deterministic logic.

    Used when the agent's output fails post-validation.
    """
    requested_vertical = request.client_vertical.strip().lower()

    # Filter by vertical
    vertical_matches = df[df["vertical_normalized"] == requested_vertical].copy()
    if vertical_matches.empty:
        return {
            "status": "no_match",
            "recommendations": [],
            "reasoning": f"No products found for vertical '{request.client_vertical}'.",
        }

    # Filter by budget
    affordable = vertical_matches[vertical_matches["minimum_budget"] <= request.budget].copy()
    if affordable.empty:
        return {
            "status": "no_match",
            "recommendations": [],
            "reasoning": (
                f"No products in '{request.client_vertical}' fit within "
                f"budget ${request.budget:,.0f}."
            ),
        }

    # Sort by requested KPI descending, budget ascending as tiebreak
    sorted_df = affordable.sort_values(
        by=[request.kpi, "minimum_budget"],
        ascending=[False, True],
    ).reset_index(drop=True)

    best = sorted_df.iloc[0]
    return {
        "status": "success",
        "recommendations": [
            {
                "creative_name": best["creative_name"],
                "vertical": best["vertical"],
                "minimum_budget": float(best["minimum_budget"]),
                "click_through_rate": float(best["click_through_rate"]),
                "in_view_rate": float(best["in_view_rate"]),
                "rank": 1,
            }
        ],
        "reasoning": (
            f"{best['creative_name']} was selected as the deterministic fallback: "
            f"highest {request.kpi} in {request.client_vertical} within budget."
        ),
    }
