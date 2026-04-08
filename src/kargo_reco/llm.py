from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, field_validator

from kargo_reco.reasoning import build_fallback_summary
from kargo_reco.schemas import LLMTrace, SummaryBlock


class LLMSummaryPayload(BaseModel):
    short_text: str
    structured_reasoning: list[str]
    alternative_notes: list[str] | None = None

    @field_validator("structured_reasoning", "alternative_notes", mode="before")
    @classmethod
    def normalize_string_list(cls, value: Any) -> Any:
        if value is None:
            return None
        if not isinstance(value, list):
            return value

        normalized_items: list[str] = []
        for item in value:
            if isinstance(item, str):
                normalized_items.append(item)
                continue
            if isinstance(item, dict):
                creative_name = item.get("creative_name")
                reason = item.get("reason")
                if isinstance(creative_name, str) and isinstance(reason, str):
                    normalized_items.append(f"{creative_name}: {reason}")
                    continue
                if isinstance(item.get("note"), str):
                    normalized_items.append(item["note"])
                    continue
                if isinstance(item.get("reason"), str):
                    normalized_items.append(item["reason"])
                    continue
                normalized_items.append(json.dumps(item, sort_keys=True))
                continue
            normalized_items.append(str(item))
        return normalized_items


PROMPT_TEMPLATE = """You summarize a deterministic recommendation result.
Use only the facts provided.
Do not invent products, metrics, or constraints.
Mention the selected product by name when the decision status is success.
If the decision status is no_match, clearly say that no eligible product was found.
Each item in structured_reasoning and alternative_notes must be a plain string.

{format_instructions}

Facts:
{facts_json}
"""


@dataclass
class SummaryGenerator:
    api_key: str | None
    base_url: str | None
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
        effective_api_key = self.api_key or ("ollama" if self.base_url else None)
        if not effective_api_key:
            return self._fallback(
                facts,
                error="Neither OPENAI_API_KEY nor OPENAI_BASE_URL is configured",
            )

        try:
            parser = PydanticOutputParser(pydantic_object=LLMSummaryPayload)
            prompt = PromptTemplate(
                template=PROMPT_TEMPLATE,
                input_variables=["facts_json"],
                partial_variables={
                    "format_instructions": parser.get_format_instructions()
                },
            )
            llm = ChatOpenAI(
                model=self.model,
                api_key=effective_api_key,
                base_url=self.base_url,
                temperature=0,
                timeout=self.timeout_s,
            )
            prompt_value = prompt.invoke(
                {"facts_json": json.dumps(facts, indent=2, sort_keys=True)}
            )
            started = time.perf_counter()
            message = llm.invoke(prompt_value)
            latency_ms = int((time.perf_counter() - started) * 1000)
            output_text = self._coerce_message_text(getattr(message, "content", ""))
            parsed = parser.parse(output_text)
            self._validate_parsed_output(parsed, facts)
            summary = SummaryBlock(
                short_text=parsed.short_text,
                structured_reasoning=list(parsed.structured_reasoning),
                alternative_notes=parsed.alternative_notes,
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
    def _coerce_message_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if isinstance(item, str):
                    chunks.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    chunks.append(item["text"])
            return "\n".join(chunks)
        return str(content)

    @staticmethod
    def _validate_parsed_output(parsed: LLMSummaryPayload, facts: dict[str, Any]) -> None:
        short_text = parsed.short_text.lower()
        if facts["decision_status"] == "success":
            selected_name = str(facts["selected_product"]["creative_name"]).lower()
            if selected_name not in short_text:
                raise ValueError("model summary omitted the selected product name")
        else:
            if "no eligible product" not in short_text and "no product" not in short_text:
                raise ValueError("model summary contradicts the no-match result")
