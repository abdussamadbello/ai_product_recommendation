from __future__ import annotations

import types

import kargo_reco.llm as llm_module
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
    generator = SummaryGenerator(
        api_key=None, base_url=None, model="gpt-test", prompt_version="test"
    )

    summary, trace = generator.generate(_facts())

    assert summary.summary_source == "fallback_template"
    assert trace.parse_status == "fallback"


def test_summary_generator_uses_llm_when_output_is_valid(monkeypatch) -> None:
    class FakeClient:
        def __init__(self, **_: object) -> None:
            pass

        def invoke(self, _: object):
            return types.SimpleNamespace(
                content='{"short_text":"Retail Rocket is the best fit for this request.","structured_reasoning":["Matched client vertical","Fit within budget"],"alternative_notes":["Retail Spotlight ranked lower."]}'
            )

    monkeypatch.setattr(llm_module, "ChatOpenAI", FakeClient)

    generator = SummaryGenerator(
        api_key="test-key", base_url=None, model="gpt-test", prompt_version="test"
    )
    summary, trace = generator.generate(_facts())

    assert summary.summary_source == "llm"
    assert summary.short_text.startswith("Retail Rocket")
    assert trace.parse_status == "success"


def test_summary_generator_falls_back_on_invalid_llm_output(monkeypatch) -> None:
    class FakeClient:
        def __init__(self, **_: object) -> None:
            pass

        def invoke(self, _: object):
            return types.SimpleNamespace(
                content='{"short_text":"Another product is better.","structured_reasoning":["Made something up"],"alternative_notes":[]}'
            )

    monkeypatch.setattr(llm_module, "ChatOpenAI", FakeClient)

    generator = SummaryGenerator(
        api_key="test-key", base_url=None, model="gpt-test", prompt_version="test"
    )
    summary, trace = generator.generate(_facts())

    assert summary.summary_source == "fallback_template"
    assert "omitted the selected product name" in (trace.error or "")


def test_summary_generator_normalizes_dict_alternative_notes(monkeypatch) -> None:
    class FakeClient:
        def __init__(self, **_: object) -> None:
            pass

        def invoke(self, _: object):
            return types.SimpleNamespace(
                content='{"short_text":"Retail Rocket is the best fit for this request.","structured_reasoning":["Matched client vertical",{"reason":"Fit within budget"}],"alternative_notes":[{"creative_name":"Orbit","reason":"Lower CTR than the selected option."},{"note":"Zenith ranked below Retail Rocket."}]}'
            )

    monkeypatch.setattr(llm_module, "ChatOpenAI", FakeClient)

    generator = SummaryGenerator(
        api_key="test-key", base_url=None, model="gpt-test", prompt_version="test"
    )
    summary, trace = generator.generate(_facts())

    assert summary.summary_source == "llm"
    assert summary.structured_reasoning == [
        "Matched client vertical",
        "Fit within budget",
    ]
    assert summary.alternative_notes == [
        "Orbit: Lower CTR than the selected option.",
        "Zenith ranked below Retail Rocket.",
    ]
    assert trace.parse_status == "success"


def test_summary_generator_supports_openai_compatible_local_endpoint(monkeypatch) -> None:
    captured_kwargs: dict[str, object] = {}

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            captured_kwargs.update(kwargs)

        def invoke(self, _: object):
            return types.SimpleNamespace(
                content='{"short_text":"Retail Rocket is the best fit for this request.","structured_reasoning":["Matched client vertical","Fit within budget"],"alternative_notes":[]}'
            )

    monkeypatch.setattr(llm_module, "ChatOpenAI", FakeClient)

    generator = SummaryGenerator(
        api_key=None,
        base_url="http://localhost:11434/v1",
        model="llama3.2",
        prompt_version="test",
    )
    summary, trace = generator.generate(_facts())

    assert summary.summary_source == "llm"
    assert trace.parse_status == "success"
    assert captured_kwargs["base_url"] == "http://localhost:11434/v1"
    assert captured_kwargs["api_key"] == "ollama"
    assert captured_kwargs["model"] == "llama3.2"
