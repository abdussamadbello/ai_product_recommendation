from __future__ import annotations

import json
from pathlib import Path

from kargo_reco.ui import (
    _apply_sample_request,
    _build_request_outputs,
    _build_request_preview,
    _handle_payload,
    _load_sample_requests,
    _sample_request_choices,
    build_app,
)


def test_build_app_returns_blocks(workflow_runner) -> None:
    app = build_app(workflow_runner)

    assert app is not None


def test_build_request_preview_returns_request_json() -> None:
    preview = _build_request_preview("Acme Shoes", "click_through_rate", "Retail", 25000)

    assert preview == {
        "client_name": "Acme Shoes",
        "kpi": "click_through_rate",
        "client_vertical": "Retail",
        "budget": 25000,
    }


def test_build_request_outputs_returns_preview_and_json_text() -> None:
    preview, json_text = _build_request_outputs(
        "Acme Shoes", "click_through_rate", "Retail", 25000
    )

    assert preview["client_name"] == "Acme Shoes"
    assert '"budget": 25000' in json_text


def test_load_sample_requests_and_choices(tmp_path: Path) -> None:
    path = tmp_path / "client_requests.json"
    path.write_text(
        json.dumps(
            [
                {
                    "client_name": "Acme Shoes",
                    "kpi": "CLICK_THROUGH_RATE",
                    "client_vertical": "Retail",
                    "budget": 25000,
                }
            ]
        ),
        encoding="utf-8",
    )

    samples = _load_sample_requests(path)
    choices = _sample_request_choices(samples)

    assert len(samples) == 1
    assert choices == ["Acme Shoes | Retail | CLICK_THROUGH_RATE | 25000"]


def test_apply_sample_request_populates_editable_fields() -> None:
    sample = {
        "client_name": "Stellar Bank",
        "kpi": "IN_VIEW_RATE",
        "client_vertical": "Finance",
        "budget": 80000,
    }
    label = "Stellar Bank | Finance | IN_VIEW_RATE | 80000"

    applied = _apply_sample_request(label, {label: sample})

    assert applied[0] == "Stellar Bank"
    assert applied[1] == "in_view_rate"
    assert applied[2] == "Finance"
    assert applied[3] == 80000
    assert applied[4]["client_name"] == "Stellar Bank"
    assert '"client_vertical": "Finance"' in applied[5]


def test_ui_handler_returns_renderable_outputs(workflow_runner) -> None:
    outputs = _handle_payload(
        workflow_runner,
        {
            "client_name": "Acme Shoes",
            "kpi": "CLICK_THROUGH_RATE",
            "client_vertical": "Retail",
            "budget": 25000,
        },
        "ui_test",
    )

    recommendation_card, summary_block, response_json, trace_json, ranking_table, trace_path = outputs

    assert "Retail Spotlight" in recommendation_card
    assert isinstance(summary_block, str)
    assert response_json["meta"]["status"] == "success"
    assert trace_json["decision_trace"]["post_budget_count"] == 3
    assert not ranking_table.empty
    assert trace_path is not None
