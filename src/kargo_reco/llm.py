from __future__ import annotations

import importlib
import json
import time
from dataclasses import dataclass
from typing import Any

from kargo_reco.reasoning import build_fallback_summary
from kargo_reco.schemas import LLMTrace, SummaryBlock


SYSTEM_PROMPT = """You summarize a deterministic recommendation result.
Return strict JSON with keys: short_text, structured_reasoning, alternative_notes.
Do not invent products, metrics, or constraints. Keep the explanation concise."""


@dataclass
class SummaryGenerator:
    api_key: str | None
    model: str
    prompt_version: str
    timeout_s: float = 20.0

    def _fallback(self, facts: dict[str, Any], error: str | None = None) -> tuple[SummaryBlock, LLMTrace]:
        short_text, reasoning, alternatives = build_fallback_summary(facts)
        summary = SummaryBlock(
            short_text=short_text,
            structured_reasoning=reasoning,
            alternative_notes=alternatives,
            summary_source="fallback_template",
        )
        trace = LLMTrace(
            prompt_version=self.prompt_version,
            model=None,
            input_payload=facts,
            output_text=short_text,
            latency_ms=0,
            parse_status="fallback",
            error=error,
            summary_source="fallback_template",
        )
        return summary, trace

    def generate(self, facts: dict[str, Any]) -> tuple[SummaryBlock, LLMTrace]:
        if not self.api_key:
            return self._fallback(facts, error="OPENAI_API_KEY not configured")

        try:
            openai_module = importlib.import_module("openai")
        except ImportError:
            return self._fallback(facts, error="openai package not installed")

        try:
            client = openai_module.OpenAI(api_key=self.api_key, timeout=self.timeout_s)
            started = time.perf_counter()
            response = client.responses.create(
                model=self.model,
                input=[
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": SYSTEM_PROMPT}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": json.dumps(facts, indent=2, sort_keys=True),
                            }
                        ],
                    },
                ],
            )
            latency_ms = int((time.perf_counter() - started) * 1000)
            output_text = getattr(response, "output_text", "") or ""
            parsed = json.loads(output_text)
            self._validate_parsed_output(parsed, facts)
            summary = SummaryBlock(
                short_text=parsed["short_text"],
                structured_reasoning=list(parsed["structured_reasoning"]),
                alternative_notes=parsed.get("alternative_notes"),
                summary_source="llm",
            )
            trace = LLMTrace(
                prompt_version=self.prompt_version,
                model=self.model,
                input_payload=facts,
                output_text=output_text,
                latency_ms=latency_ms,
                parse_status="success",
                error=None,
                summary_source="llm",
            )
            return summary, trace
        except Exception as exc:  # pragma: no cover - exercised through tests with stubs
            return self._fallback(facts, error=str(exc))

    @staticmethod
    def _validate_parsed_output(parsed: dict[str, Any], facts: dict[str, Any]) -> None:
        if not isinstance(parsed.get("short_text"), str):
            raise ValueError("short_text must be a string")
        if not isinstance(parsed.get("structured_reasoning"), list):
            raise ValueError("structured_reasoning must be a list")
        if parsed.get("alternative_notes") is not None and not isinstance(
            parsed.get("alternative_notes"), list
        ):
            raise ValueError("alternative_notes must be a list or null")

        short_text = parsed["short_text"].lower()
        if facts["decision_status"] == "success":
            selected_name = str(facts["selected_product"]["creative_name"]).lower()
            if selected_name not in short_text:
                raise ValueError("model summary omitted the selected product name")
        else:
            if "no eligible product" not in short_text and "no product" not in short_text:
                raise ValueError("model summary contradicts the no-match result")
