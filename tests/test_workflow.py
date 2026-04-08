from __future__ import annotations

import json
from pathlib import Path

from kargo_reco.schemas import RecommendationRequest


def test_run_recommendation_success_creates_trace(workflow_runner, tmp_path: Path) -> None:
    response = workflow_runner.run_recommendation(
        RecommendationRequest(
            client_name="Acme Shoes",
            kpi="CLICK_THROUGH_RATE",
            client_vertical="Retail",
            budget=25000,
        ),
        source="test",
    )

    assert response.meta.status == "success"
    assert response.recommendations[0].creative_name == "Retail Spotlight"
    assert response.meta.summary_source == "fallback_template"
    assert response.meta.trace_path is not None
    assert Path(response.meta.trace_path).exists()

    trace_payload = json.loads(Path(response.meta.trace_path).read_text(encoding="utf-8"))
    assert trace_payload["decision_trace"]["post_budget_count"] == 3
    assert trace_payload["final_response"]["meta"]["status"] == "success"


def test_run_from_payload_returns_no_match_with_nearest_alternative(workflow_runner) -> None:
    response = workflow_runner.run_from_payload(
        {
            "client_name": "Budget Buyer",
            "kpi": "in_view_rate",
            "client_vertical": "Retail",
            "budget": 1000,
        },
        source="test",
    )

    assert response.meta.status == "no_match"
    assert response.nearest_alternative is not None
    assert response.nearest_alternative.block_reason == "budget_shortfall"


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
                "Travel View,0.43,0.91,Travel,9000",
            ]
        ),
        encoding="utf-8",
    )

    metadata = workflow_runner.upload_benchmarks(str(uploaded_csv))
    response = workflow_runner.run_from_payload(
        {
            "client_name": "Traveler",
            "kpi": "CLICK_THROUGH_RATE",
            "client_vertical": "Travel",
            "budget": 15000,
        },
        source="test",
    )

    assert metadata["row_count"] == 2
    assert "uploads" in metadata["file"]
    assert response.meta.status == "success"
    assert response.recommendations[0].creative_name == "Travel Lift"
