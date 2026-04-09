from __future__ import annotations

from typing import Any

import pandas as pd


def filter_by_vertical(df: pd.DataFrame, *, vertical: str) -> list[dict[str, Any]]:
    """Return all products matching the given vertical (case-insensitive)."""
    normalized = vertical.strip().lower()
    matches = df[df["vertical_normalized"] == normalized]
    return matches[["creative_name", "click_through_rate", "in_view_rate", "vertical", "minimum_budget"]].to_dict(orient="records")


def filter_by_budget(*, budget: float, products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return products whose minimum_budget is at or below the given budget."""
    return [p for p in products if p["minimum_budget"] <= budget]


def sort_by_kpi(*, kpi: str, products: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    """Sort products by the given KPI descending and return top N."""
    sorted_products = sorted(products, key=lambda p: p.get(kpi, 0), reverse=True)
    return sorted_products[:limit]


def get_product_details(df: pd.DataFrame, *, product_name: str, vertical: str) -> dict[str, Any] | None:
    """Return a single product's full details by name and vertical."""
    normalized_vertical = vertical.strip().lower()
    matches = df[
        (df["creative_name"] == product_name) & (df["vertical_normalized"] == normalized_vertical)
    ]
    if matches.empty:
        return None
    row = matches.iloc[0]
    return {
        "creative_name": row["creative_name"],
        "vertical": row["vertical"],
        "minimum_budget": float(row["minimum_budget"]),
        "click_through_rate": float(row["click_through_rate"]),
        "in_view_rate": float(row["in_view_rate"]),
    }


def check_budget_remaining(
    df: pd.DataFrame,
    *,
    budget: float,
    selected: list[str],
    vertical: str,
) -> dict[str, Any]:
    """Calculate remaining budget after selected products and list affordable additions."""
    normalized_vertical = vertical.strip().lower()
    total_spent = 0.0
    for name in selected:
        matches = df[
            (df["creative_name"] == name) & (df["vertical_normalized"] == normalized_vertical)
        ]
        if matches.empty:
            return {"error": f"Product '{name}' not found in {vertical} vertical"}
        total_spent += float(matches.iloc[0]["minimum_budget"])

    remaining = budget - total_spent
    vertical_products = df[df["vertical_normalized"] == normalized_vertical]
    not_selected = vertical_products[~vertical_products["creative_name"].isin(selected)]
    affordable = not_selected[not_selected["minimum_budget"] <= remaining]
    affordable_list = affordable[["creative_name", "click_through_rate", "in_view_rate", "minimum_budget"]].to_dict(orient="records")

    return {
        "remaining": max(remaining, 0.0),
        "can_add_more": len(affordable_list) > 0,
        "affordable": affordable_list,
    }


def score_products(
    *,
    products: list[dict[str, Any]],
    weights: dict[str, float],
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Score and rank products using a weighted combination of KPIs.

    The agent generates the weights to express its strategy. Only known
    numeric columns (click_through_rate, in_view_rate) contribute to the
    score; unknown keys are silently ignored.
    """
    known_metrics = {"click_through_rate", "in_view_rate"}
    safe_weights = {k: v for k, v in weights.items() if k in known_metrics}

    scored = []
    for product in products:
        score = sum(
            product.get(metric, 0.0) * weight
            for metric, weight in safe_weights.items()
        )
        scored.append({**product, "score": round(score, 6)})

    scored.sort(key=lambda p: p["score"], reverse=True)
    return scored[:limit]


def finalize_recommendation(
    df: pd.DataFrame,
    *,
    products: list[str],
    reasoning: str,
    vertical: str,
) -> dict[str, Any]:
    """Build the structured recommendation output from agent's final selection."""
    normalized_vertical = vertical.strip().lower()
    recommendations = []
    for rank, name in enumerate(products, start=1):
        matches = df[
            (df["creative_name"] == name) & (df["vertical_normalized"] == normalized_vertical)
        ]
        if matches.empty:
            return {
                "recommendations": [],
                "reasoning": reasoning,
                "error": f"Product '{name}' not found in {vertical} vertical",
            }
        row = matches.iloc[0]
        recommendations.append({
            "creative_name": row["creative_name"],
            "vertical": row["vertical"],
            "minimum_budget": float(row["minimum_budget"]),
            "click_through_rate": float(row["click_through_rate"]),
            "in_view_rate": float(row["in_view_rate"]),
            "rank": rank,
        })

    return {"recommendations": recommendations, "reasoning": reasoning}
