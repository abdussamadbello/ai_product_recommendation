from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


NormalizedKPI = Literal["click_through_rate", "in_view_rate"]
ResponseStatus = Literal["success", "no_match", "guardrail_fallback"]
SummarySource = Literal["llm", "guardrail_fallback"]


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


class RecommendationItem(BaseModel):
    creative_name: str
    vertical: str
    minimum_budget: float
    click_through_rate: float
    in_view_rate: float
    rank: int


class AgentStep(BaseModel):
    step_number: int
    tool_name: str
    tool_input: dict[str, Any] = Field(default_factory=dict)
    tool_output: dict[str, Any] = Field(default_factory=dict)
    agent_reasoning: str | None = None
    latency_ms: int = 0


class ResponseMeta(BaseModel):
    status: ResponseStatus
    request_id: str
    model: str
    total_tokens: int | None = None
    agent_steps: int = 0
    latency_ms: int = 0
    source: SummarySource
    trace_path: str | None = None


class RecommendationResponse(BaseModel):
    request: RecommendationRequest
    recommendations: list[RecommendationItem]
    summary: str
    agent_trace: list[AgentStep] = Field(default_factory=list)
    meta: ResponseMeta


class BenchmarkMetadata(BaseModel):
    file: str
    file_hash: str
    row_count: int
    schema_valid: bool
    loaded_at: str


class TraceArtifact(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    source: str = "api"
    raw_request: dict[str, Any] = Field(default_factory=dict)
    normalized_request: dict[str, Any] = Field(default_factory=dict)
    benchmark_metadata: BenchmarkMetadata | None = None
    agent_trace: list[AgentStep] = Field(default_factory=list)
    guardrail_violations: list[str] = Field(default_factory=list)
    final_response: dict[str, Any] = Field(default_factory=dict)
    step_latencies_ms: dict[str, int] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
