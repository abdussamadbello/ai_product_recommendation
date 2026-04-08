from __future__ import annotations

import importlib
import types

from kargo_reco.llm import SummaryGenerator


def _facts() -> dict:
    return {
        "request": {
            "client_name": "Acme Shoes",
            "kpi": "click_through_rate",
            "client_vertical": "Retail",
            "budget": 25000.0,
        },
        "decision_status": "success",
        "selected_product": {"creative_name": "Retail Rocket"},
        "structured_reasons": [
            "Matched client vertical",
            "Fit within budget",
            "Highest eligible click_through_rate",
        ],
        "eligible_alternatives": [],
        "nearest_alternative": None,
        "no_match_reason": None,
        "candidate_counts": {},
        "tie_break_notes": [],
    }


def test_summary_generator_falls_back_without_api_key() -> None:
    generator = SummaryGenerator(api_key=None, model="gpt-test", prompt_version="test")

    summary, trace = generator.generate(_facts())

    assert summary.summary_source == "fallback_template"
    assert trace.parse_status == "fallback"


def test_summary_generator_uses_llm_when_output_is_valid(monkeypatch) -> None:
    class FakeResponses:
        @staticmethod
        def create(**_: object):
            return types.SimpleNamespace(
                output_text='{"short_text":"Retail Rocket is the best fit for this request.","structured_reasoning":["Matched client vertical","Fit within budget"],"alternative_notes":["Retail Spotlight ranked lower."]}'
            )

    class FakeClient:
        def __init__(self, **_: object) -> None:
            self.responses = FakeResponses()

    fake_module = types.SimpleNamespace(OpenAI=FakeClient)
    monkeypatch.setattr(importlib, "import_module", lambda _: fake_module)

    generator = SummaryGenerator(
        api_key="test-key", model="gpt-test", prompt_version="test"
    )
    summary, trace = generator.generate(_facts())

    assert summary.summary_source == "llm"
    assert summary.short_text.startswith("Retail Rocket")
    assert trace.parse_status == "success"


def test_summary_generator_falls_back_on_invalid_llm_output(monkeypatch) -> None:
    class FakeResponses:
        @staticmethod
        def create(**_: object):
            return types.SimpleNamespace(
                output_text='{"short_text":"Another product is better.","structured_reasoning":["Made something up"],"alternative_notes":[]}'
            )

    class FakeClient:
        def __init__(self, **_: object) -> None:
            self.responses = FakeResponses()

    fake_module = types.SimpleNamespace(OpenAI=FakeClient)
    monkeypatch.setattr(importlib, "import_module", lambda _: fake_module)

    generator = SummaryGenerator(
        api_key="test-key", model="gpt-test", prompt_version="test"
    )
    summary, trace = generator.generate(_facts())

    assert summary.summary_source == "fallback_template"
    assert "omitted the selected product name" in (trace.error or "")
