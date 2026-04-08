from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


NormalizedKPI = Literal["click_through_rate", "in_view_rate"]
SummarySource = Literal["llm", "fallback_template"]
ResponseStatus = Literal["success", "no_match", "error"]


class RecommendationRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    client_name: str
    kpi: NormalizedKPI
    client_vertical: str
    budget: float = Field(validation_alias=AliasChoices("budget", "minimum_budget"))

    @field_validator("client_name", "client_vertical")
    @classmethod
    def validate_non_empty_text(cls, value: str) -> str:
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("budget", mode="before")
    @classmethod
    def normalize_budget(cls, value: Any) -> float:
        try:
            budget = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("budget must be numeric") from exc
        if budget <= 0:
            raise ValueError("budget must be greater than zero")
        return budget

    @field_validator("kpi", mode="before")
    @classmethod
    def normalize_kpi(cls, value: Any) -> NormalizedKPI:
        if not isinstance(value, str):
            raise ValueError("kpi must be a string")
        normalized = value.strip().lower().replace(" ", "_")
        aliases = {
            "click_through_rate": "click_through_rate",
            "ctr": "click_through_rate",
            "in_view_rate": "in_view_rate",
            "ivr": "in_view_rate",
        }
        if normalized not in aliases:
            raise ValueError("unsupported kpi")
        return aliases[normalized]


class SelectionReason(BaseModel):
    matched_vertical: bool
    within_budget: bool
    optimized_kpi: NormalizedKPI


class RecommendationItem(BaseModel):
    creative_name: str
    vertical: str
    minimum_budget: float
    click_through_rate: float
    in_view_rate: float
    rank: int
    selection_reason: SelectionReason


class NearestAlternative(BaseModel):
    creative_name: str
    vertical: str
    minimum_budget: float
    click_through_rate: float
    in_view_rate: float
    block_reason: str
    budget_shortfall: float | None = None


class SummaryBlock(BaseModel):
    short_text: str
    structured_reasoning: list[str]
    alternative_notes: list[str] | None = None
    summary_source: SummarySource


class ResponseMeta(BaseModel):
    decision_mode: str = "deterministic_selection_llm_explanation"
    status: ResponseStatus
    request_id: str
    summary_source: SummarySource
    trace_path: str | None = None


class RecommendationResponse(BaseModel):
    request: RecommendationRequest
    recommendations: list[RecommendationItem]
    summary: SummaryBlock
    meta: ResponseMeta
    nearest_alternative: NearestAlternative | None = None


class BenchmarkMetadata(BaseModel):
    file: str
    file_hash: str
    row_count: int
    schema_valid: bool
    loaded_at: str


class DecisionTrace(BaseModel):
    pre_filter_count: int = 0
    post_vertical_count: int = 0
    post_budget_count: int = 0
    ranking_table: list[dict[str, Any]] = Field(default_factory=list)
    selected_product: dict[str, Any] | None = None
    excluded_alternatives: list[dict[str, Any]] = Field(default_factory=list)
    nearest_alternative: dict[str, Any] | None = None
    tie_break_notes: list[str] = Field(default_factory=list)
    no_match_reason: str | None = None


class LLMTrace(BaseModel):
    prompt_version: str
    model: str | None = None
    input_payload: dict[str, Any] = Field(default_factory=dict)
    output_text: str | None = None
    latency_ms: int | None = None
    parse_status: str
    error: str | None = None
    summary_source: SummarySource


class TraceArtifact(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    source: str = "api"
    raw_request: dict[str, Any] = Field(default_factory=dict)
    normalized_request: dict[str, Any] = Field(default_factory=dict)
    benchmark_metadata: BenchmarkMetadata | None = None
    decision_trace: DecisionTrace = Field(default_factory=DecisionTrace)
    llm_trace: LLMTrace | None = None
    final_response: dict[str, Any] = Field(default_factory=dict)
    step_latencies_ms: dict[str, int] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
