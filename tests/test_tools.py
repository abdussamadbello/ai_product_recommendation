from __future__ import annotations

import pandas as pd
import pytest

from kargo_reco.tools import (
    check_budget_remaining,
    filter_by_budget,
    filter_by_vertical,
    finalize_recommendation,
    get_product_details,
    score_products,
    sort_by_kpi,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {"creative_name": "Alpha", "click_through_rate": 0.42, "in_view_rate": 0.70, "vertical": "Retail", "minimum_budget": 20000, "vertical_normalized": "retail"},
            {"creative_name": "Beta", "click_through_rate": 0.38, "in_view_rate": 0.82, "vertical": "Retail", "minimum_budget": 18000, "vertical_normalized": "retail"},
            {"creative_name": "Gamma", "click_through_rate": 0.50, "in_view_rate": 0.65, "vertical": "Retail", "minimum_budget": 30000, "vertical_normalized": "retail"},
            {"creative_name": "Delta", "click_through_rate": 0.60, "in_view_rate": 0.90, "vertical": "Finance", "minimum_budget": 50000, "vertical_normalized": "finance"},
        ]
    )
    return frame


def test_filter_by_vertical_returns_matching_products(sample_df: pd.DataFrame) -> None:
    result = filter_by_vertical(sample_df, vertical="Retail")
    assert len(result) == 3
    assert all(p["vertical"] == "Retail" for p in result)


def test_filter_by_vertical_case_insensitive(sample_df: pd.DataFrame) -> None:
    result = filter_by_vertical(sample_df, vertical="retail")
    assert len(result) == 3


def test_filter_by_vertical_unknown_returns_empty(sample_df: pd.DataFrame) -> None:
    result = filter_by_vertical(sample_df, vertical="Travel")
    assert result == []


def test_filter_by_budget_excludes_expensive(sample_df: pd.DataFrame) -> None:
    products = [
        {"creative_name": "Alpha", "minimum_budget": 20000, "vertical": "Retail", "click_through_rate": 0.42, "in_view_rate": 0.70},
        {"creative_name": "Gamma", "minimum_budget": 30000, "vertical": "Retail", "click_through_rate": 0.50, "in_view_rate": 0.65},
    ]
    result = filter_by_budget(budget=25000.0, products=products)
    assert len(result) == 1
    assert result[0]["creative_name"] == "Alpha"


def test_filter_by_budget_includes_exact_match() -> None:
    products = [{"creative_name": "X", "minimum_budget": 20000, "vertical": "Retail", "click_through_rate": 0.1, "in_view_rate": 0.1}]
    result = filter_by_budget(budget=20000.0, products=products)
    assert len(result) == 1


def test_filter_by_budget_empty_input() -> None:
    result = filter_by_budget(budget=50000.0, products=[])
    assert result == []


def test_sort_by_kpi_returns_top_n_descending() -> None:
    products = [
        {"creative_name": "A", "click_through_rate": 0.30, "in_view_rate": 0.70, "minimum_budget": 10000, "vertical": "Retail"},
        {"creative_name": "B", "click_through_rate": 0.50, "in_view_rate": 0.60, "minimum_budget": 10000, "vertical": "Retail"},
        {"creative_name": "C", "click_through_rate": 0.40, "in_view_rate": 0.80, "minimum_budget": 10000, "vertical": "Retail"},
    ]
    result = sort_by_kpi(kpi="click_through_rate", products=products, limit=2)
    assert len(result) == 2
    assert result[0]["creative_name"] == "B"
    assert result[1]["creative_name"] == "C"


def test_sort_by_kpi_with_in_view_rate() -> None:
    products = [
        {"creative_name": "A", "click_through_rate": 0.30, "in_view_rate": 0.70, "minimum_budget": 10000, "vertical": "Retail"},
        {"creative_name": "B", "click_through_rate": 0.50, "in_view_rate": 0.90, "minimum_budget": 10000, "vertical": "Retail"},
    ]
    result = sort_by_kpi(kpi="in_view_rate", products=products, limit=2)
    assert result[0]["creative_name"] == "B"


def test_sort_by_kpi_limit_exceeds_list_size() -> None:
    products = [{"creative_name": "A", "click_through_rate": 0.30, "in_view_rate": 0.70, "minimum_budget": 10000, "vertical": "Retail"}]
    result = sort_by_kpi(kpi="click_through_rate", products=products, limit=10)
    assert len(result) == 1


def test_get_product_details_returns_matching_product(sample_df: pd.DataFrame) -> None:
    result = get_product_details(sample_df, product_name="Alpha", vertical="Retail")
    assert result is not None
    assert result["creative_name"] == "Alpha"
    assert result["click_through_rate"] == 0.42


def test_get_product_details_not_found(sample_df: pd.DataFrame) -> None:
    result = get_product_details(sample_df, product_name="Nonexistent", vertical="Retail")
    assert result is None


def test_get_product_details_wrong_vertical(sample_df: pd.DataFrame) -> None:
    result = get_product_details(sample_df, product_name="Alpha", vertical="Finance")
    assert result is None


def test_check_budget_remaining_with_room_for_more(sample_df: pd.DataFrame) -> None:
    result = check_budget_remaining(sample_df, budget=50000.0, selected=["Alpha"], vertical="Retail")
    assert result["remaining"] == 30000.0
    assert result["can_add_more"] is True
    assert len(result["affordable"]) > 0


def test_check_budget_remaining_no_room(sample_df: pd.DataFrame) -> None:
    result = check_budget_remaining(sample_df, budget=20000.0, selected=["Alpha"], vertical="Retail")
    assert result["remaining"] == 0.0
    assert result["can_add_more"] is False
    assert result["affordable"] == []


def test_check_budget_remaining_unknown_product(sample_df: pd.DataFrame) -> None:
    result = check_budget_remaining(sample_df, budget=50000.0, selected=["Nonexistent"], vertical="Retail")
    assert "error" in result


def test_score_products_weighted_ranking() -> None:
    products = [
        {"creative_name": "A", "click_through_rate": 0.50, "in_view_rate": 0.60, "minimum_budget": 10000, "vertical": "Retail"},
        {"creative_name": "B", "click_through_rate": 0.30, "in_view_rate": 0.90, "minimum_budget": 10000, "vertical": "Retail"},
        {"creative_name": "C", "click_through_rate": 0.40, "in_view_rate": 0.80, "minimum_budget": 10000, "vertical": "Retail"},
    ]
    result = score_products(
        products=products,
        weights={"click_through_rate": 0.3, "in_view_rate": 0.7},
        limit=3,
    )
    assert len(result) == 3
    assert result[0]["creative_name"] == "B"
    assert "score" in result[0]


def test_score_products_single_kpi_weight() -> None:
    products = [
        {"creative_name": "A", "click_through_rate": 0.50, "in_view_rate": 0.60, "minimum_budget": 10000, "vertical": "Retail"},
        {"creative_name": "B", "click_through_rate": 0.30, "in_view_rate": 0.90, "minimum_budget": 10000, "vertical": "Retail"},
    ]
    result = score_products(products=products, weights={"click_through_rate": 1.0}, limit=2)
    assert result[0]["creative_name"] == "A"


def test_score_products_empty_input() -> None:
    result = score_products(products=[], weights={"click_through_rate": 1.0}, limit=5)
    assert result == []


def test_score_products_ignores_unknown_weight_keys() -> None:
    products = [
        {"creative_name": "A", "click_through_rate": 0.50, "in_view_rate": 0.60, "minimum_budget": 10000, "vertical": "Retail"},
    ]
    result = score_products(
        products=products,
        weights={"click_through_rate": 0.5, "in_view_rate": 0.5, "fake_metric": 0.5},
        limit=5,
    )
    assert len(result) == 1
    assert "score" in result[0]


def test_finalize_recommendation_structures_output(sample_df: pd.DataFrame) -> None:
    result = finalize_recommendation(
        sample_df,
        products=["Alpha", "Beta"],
        reasoning="Alpha has the highest CTR, Beta adds IVR coverage within budget.",
        vertical="Retail",
    )
    assert len(result["recommendations"]) == 2
    assert result["recommendations"][0]["rank"] == 1
    assert result["recommendations"][1]["rank"] == 2
    assert result["reasoning"] == "Alpha has the highest CTR, Beta adds IVR coverage within budget."


def test_finalize_recommendation_unknown_product(sample_df: pd.DataFrame) -> None:
    result = finalize_recommendation(
        sample_df, products=["Nonexistent"], reasoning="Best pick.", vertical="Retail"
    )
    assert len(result["recommendations"]) == 0
    assert "error" in result
