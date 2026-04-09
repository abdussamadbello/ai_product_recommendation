# Agentic Product Recommendation Engine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the Kargo product recommendation engine so an LLM agent drives product selection through tool calls, replacing the current deterministic pipeline.

**Architecture:** LangGraph `create_react_agent` with 6 fine-grained tools (filter, sort, inspect, bundle, finalize) operates inside a guardrailed workflow. Pre-validation (Pydantic) ensures clean input; post-validation catches hallucinations and constraint violations, falling back to deterministic logic when needed.

**Tech Stack:** LangGraph (create_react_agent), LangChain (ChatOpenAI), Pydantic v2, Pandas, Gradio, pytest

**Spec:** `docs/superpowers/specs/2026-04-09-agentic-recommendation-engine-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/kargo_reco/schemas.py` | REWRITE | Request, RecommendationItem, AgentStep, ResponseMeta, RecommendationResponse, TraceArtifact |
| `src/kargo_reco/config.py` | MODIFY | Bump timeout default to 30s |
| `src/kargo_reco/benchmark_loader.py` | UNCHANGED | BenchmarkRepository, CSV loading |
| `src/kargo_reco/tools.py` | CREATE | 6 agent tool functions |
| `src/kargo_reco/agent.py` | CREATE | create_react_agent setup + system prompt |
| `src/kargo_reco/guardrails.py` | CREATE | Post-validation + deterministic fallback |
| `src/kargo_reco/workflow.py` | REWRITE | 5-node orchestration |
| `src/kargo_reco/trace.py` | CREATE | AgentStep extraction from message history |
| `src/kargo_reco/ui.py` | REWRITE | Gradio app with agent reasoning steps |
| `src/kargo_reco/reasoning.py` | DELETE | Replaced by agent reasoning |
| `src/kargo_reco/llm.py` | DELETE | Replaced by agent.py |
| `src/kargo_reco/recommender.py` | DELETE | Logic absorbed into guardrails.py |
| `app.py` | UNCHANGED | Entry point |
| `tests/conftest.py` | REWRITE | New fixtures for agent-based workflow |
| `tests/test_tools.py` | CREATE | Tool unit tests |
| `tests/test_guardrails.py` | CREATE | Post-validation tests |
| `tests/test_agent.py` | CREATE | Agent integration tests with mocked LLM |
| `tests/test_schemas.py` | REWRITE | Updated schema tests |
| `tests/test_workflow.py` | REWRITE | End-to-end workflow tests |
| `tests/test_ui.py` | REWRITE | UI tests |
| `tests/test_benchmark_loader.py` | UNCHANGED | Existing tests still valid |
| `tests/test_llm.py` | DELETE | Replaced by test_agent.py |
| `tests/test_recommender.py` | DELETE | Logic tested via test_guardrails.py |

---

### Task 1: Rewrite schemas.py

**Files:**
- Rewrite: `src/kargo_reco/schemas.py`
- Test: `tests/test_schemas.py`

- [ ] **Step 1: Write failing tests for new schemas**

Create `tests/test_schemas.py` with tests for all new/changed models:

```python
from __future__ import annotations

import pytest
from pydantic import ValidationError

from kargo_reco.schemas import (
    AgentStep,
    RecommendationItem,
    RecommendationRequest,
    RecommendationResponse,
    ResponseMeta,
    TraceArtifact,
)


def test_recommendation_request_normalizes_ctr_alias() -> None:
    req = RecommendationRequest(
        client_name="Acme", kpi="CTR", client_vertical="Retail", budget=25000
    )
    assert req.kpi == "click_through_rate"


def test_recommendation_request_normalizes_ivr_alias() -> None:
    req = RecommendationRequest(
        client_name="Acme", kpi="IVR", client_vertical="Retail", budget=25000
    )
    assert req.kpi == "in_view_rate"


def test_recommendation_request_rejects_empty_client_name() -> None:
    with pytest.raises(ValidationError):
        RecommendationRequest(
            client_name="", kpi="click_through_rate", client_vertical="Retail", budget=25000
        )


def test_recommendation_request_rejects_zero_budget() -> None:
    with pytest.raises(ValidationError):
        RecommendationRequest(
            client_name="Acme", kpi="click_through_rate", client_vertical="Retail", budget=0
        )


def test_recommendation_request_coerces_string_budget() -> None:
    req = RecommendationRequest(
        client_name="Acme", kpi="click_through_rate", client_vertical="Retail", budget="25000"
    )
    assert req.budget == 25000.0


def test_agent_step_stores_tool_call_data() -> None:
    step = AgentStep(
        step_number=1,
        tool_name="filter_by_vertical",
        tool_input={"vertical": "Retail"},
        tool_output={"products": [{"creative_name": "Alpha"}]},
        agent_reasoning="I need to find Retail products first.",
        latency_ms=42,
    )
    assert step.tool_name == "filter_by_vertical"
    assert step.agent_reasoning is not None


def test_agent_step_allows_none_reasoning() -> None:
    step = AgentStep(
        step_number=1,
        tool_name="filter_by_vertical",
        tool_input={},
        tool_output={},
        agent_reasoning=None,
        latency_ms=10,
    )
    assert step.agent_reasoning is None


def test_recommendation_item_stores_product_data() -> None:
    item = RecommendationItem(
        creative_name="Alpha",
        vertical="Retail",
        minimum_budget=20000,
        click_through_rate=0.42,
        in_view_rate=0.79,
        rank=1,
    )
    assert item.creative_name == "Alpha"
    assert item.rank == 1


def test_response_meta_accepts_all_status_values() -> None:
    for status in ("success", "no_match", "guardrail_fallback"):
        meta = ResponseMeta(
            status=status,
            request_id="abc-123",
            model="gpt-4.1-mini",
            total_tokens=None,
            agent_steps=3,
            latency_ms=500,
            source="llm",
        )
        assert meta.status == status


def test_recommendation_response_assembles_correctly() -> None:
    response = RecommendationResponse(
        request=RecommendationRequest(
            client_name="Acme", kpi="click_through_rate", client_vertical="Retail", budget=25000
        ),
        recommendations=[
            RecommendationItem(
                creative_name="Alpha",
                vertical="Retail",
                minimum_budget=20000,
                click_through_rate=0.42,
                in_view_rate=0.79,
                rank=1,
            )
        ],
        summary="Alpha is the best pick for Retail CTR.",
        agent_trace=[],
        meta=ResponseMeta(
            status="success",
            request_id="abc-123",
            model="gpt-4.1-mini",
            total_tokens=100,
            agent_steps=4,
            latency_ms=1200,
            source="llm",
        ),
    )
    assert len(response.recommendations) == 1
    assert response.summary.startswith("Alpha")


def test_trace_artifact_generates_request_id() -> None:
    trace = TraceArtifact(source="test", raw_request={"client_name": "Acme"})
    assert trace.request_id is not None
    assert len(trace.request_id) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_schemas.py -v`
Expected: FAIL — old schemas don't have `AgentStep`, new `ResponseMeta` fields, etc.

- [ ] **Step 3: Rewrite schemas.py**

Replace `src/kargo_reco/schemas.py` with:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_schemas.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/kargo_reco/schemas.py tests/test_schemas.py
git commit -m "feat: rewrite schemas for agentic architecture

Replace SummaryBlock/SelectionReason/NearestAlternative with AgentStep,
simplified RecommendationResponse, and new ResponseMeta fields for agent
observability."
```

---

### Task 2: Update config.py

**Files:**
- Modify: `src/kargo_reco/config.py`

- [ ] **Step 1: Update config.py with new timeout default**

Edit `src/kargo_reco/config.py` — change line 28:

```python
    openai_timeout_s: float = float(os.getenv("OPENAI_TIMEOUT_S", "30"))
```

Also update the description in pyproject.toml line 4:

```python
description = "Agentic AI product recommendation engine for Kargo."
```

- [ ] **Step 2: Commit**

```bash
git add src/kargo_reco/config.py pyproject.toml
git commit -m "chore: bump LLM timeout to 30s for agent tool-call loops"
```

---

### Task 3: Create tools.py

**Files:**
- Create: `src/kargo_reco/tools.py`
- Test: `tests/test_tools.py`

- [ ] **Step 1: Write failing tests for all 6 tools**

Create `tests/test_tools.py`:

```python
from __future__ import annotations

import pandas as pd
import pytest

from kargo_reco.tools import (
    check_budget_remaining,
    filter_by_budget,
    filter_by_vertical,
    finalize_recommendation,
    get_product_details,
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


# --- filter_by_vertical ---

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


# --- filter_by_budget ---

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


# --- sort_by_kpi ---

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


# --- get_product_details ---

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


# --- check_budget_remaining ---

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


# --- finalize_recommendation ---

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tools.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kargo_reco.tools'`

- [ ] **Step 3: Implement tools.py**

Create `src/kargo_reco/tools.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tools.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/kargo_reco/tools.py tests/test_tools.py
git commit -m "feat: add 6 agent tool functions for product exploration

filter_by_vertical, filter_by_budget, sort_by_kpi, get_product_details,
check_budget_remaining, finalize_recommendation — all pure functions
operating on DataFrames, testable without LLM."
```

---

### Task 4: Create agent.py

**Files:**
- Create: `src/kargo_reco/agent.py`
- Test: `tests/test_agent.py`

- [ ] **Step 1: Write failing test for agent construction**

Create `tests/test_agent.py`:

```python
from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from kargo_reco.agent import build_agent, build_tools, SYSTEM_PROMPT


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"creative_name": "Alpha", "click_through_rate": 0.42, "in_view_rate": 0.70, "vertical": "Retail", "minimum_budget": 20000, "vertical_normalized": "retail"},
            {"creative_name": "Beta", "click_through_rate": 0.38, "in_view_rate": 0.82, "vertical": "Retail", "minimum_budget": 18000, "vertical_normalized": "retail"},
        ]
    )


def test_build_tools_returns_six_tools(sample_df: pd.DataFrame) -> None:
    tools = build_tools(sample_df, vertical="Retail")
    assert len(tools) == 6
    tool_names = {t.name for t in tools}
    assert tool_names == {
        "filter_by_vertical",
        "filter_by_budget",
        "sort_by_kpi",
        "get_product_details",
        "check_budget_remaining",
        "finalize_recommendation",
    }


def test_system_prompt_mentions_strategy() -> None:
    assert "vertical" in SYSTEM_PROMPT.lower()
    assert "budget" in SYSTEM_PROMPT.lower()
    assert "kpi" in SYSTEM_PROMPT.lower()


def test_build_agent_returns_compiled_graph(sample_df: pd.DataFrame) -> None:
    agent = build_agent(
        model_name="gpt-4.1-mini",
        api_key="test-key",
        base_url=None,
        timeout_s=10,
        df=sample_df,
        vertical="Retail",
    )
    # create_react_agent returns a CompiledStateGraph
    assert hasattr(agent, "invoke")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_agent.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kargo_reco.agent'`

- [ ] **Step 3: Implement agent.py**

Create `src/kargo_reco/agent.py`:

```python
from __future__ import annotations

from typing import Any

import pandas as pd
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from kargo_reco import tools as tool_fns


SYSTEM_PROMPT = """You are a media strategist assistant for Kargo. Given a client request, use the available tools to explore the product catalog, find the best product(s) for the client's vertical and KPI, and stay within budget.

You MUST call finalize_recommendation as your last tool call to commit your selection.

Suggested strategy:
1. Use filter_by_vertical to find products in the client's vertical.
2. Use filter_by_budget to narrow to affordable products.
3. Use sort_by_kpi to rank by the client's target KPI.
4. Use get_product_details to inspect the top candidate if needed.
5. Use check_budget_remaining to see if a second product can be added.
6. If budget allows, pick the next-best product that fits.
7. Call finalize_recommendation with your selected product(s) and reasoning.

When reasoning, explain trade-offs: why you picked this product over alternatives, how budget utilization factored in, and whether bundling adds value.
"""


def build_tools(df: pd.DataFrame, vertical: str) -> list[StructuredTool]:
    """Create the 6 LangChain tools that the agent can call, bound to a DataFrame."""

    def _filter_by_vertical(vertical: str) -> list[dict[str, Any]]:
        """Find all products available in a given vertical."""
        return tool_fns.filter_by_vertical(df, vertical=vertical)

    def _filter_by_budget(budget: float, products: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Filter a list of products to only those within the given budget."""
        return tool_fns.filter_by_budget(budget=budget, products=products)

    def _sort_by_kpi(kpi: str, products: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
        """Sort products by a KPI (click_through_rate or in_view_rate) descending. Returns top N."""
        return tool_fns.sort_by_kpi(kpi=kpi, products=products, limit=limit)

    def _get_product_details(product_name: str, vertical: str) -> dict[str, Any] | None:
        """Get full details for a specific product by name and vertical."""
        return tool_fns.get_product_details(df, product_name=product_name, vertical=vertical)

    def _check_budget_remaining(budget: float, selected: list[str]) -> dict[str, Any]:
        """Check how much budget remains after selecting products, and what else is affordable."""
        return tool_fns.check_budget_remaining(df, budget=budget, selected=selected, vertical=vertical)

    def _finalize_recommendation(products: list[str], reasoning: str) -> dict[str, Any]:
        """Commit your final product selection with reasoning. This MUST be your last tool call."""
        return tool_fns.finalize_recommendation(df, products=products, reasoning=reasoning, vertical=vertical)

    return [
        StructuredTool.from_function(_filter_by_vertical, name="filter_by_vertical"),
        StructuredTool.from_function(_filter_by_budget, name="filter_by_budget"),
        StructuredTool.from_function(_sort_by_kpi, name="sort_by_kpi"),
        StructuredTool.from_function(_get_product_details, name="get_product_details"),
        StructuredTool.from_function(_check_budget_remaining, name="check_budget_remaining"),
        StructuredTool.from_function(_finalize_recommendation, name="finalize_recommendation"),
    ]


def build_agent(
    *,
    model_name: str,
    api_key: str | None,
    base_url: str | None,
    timeout_s: float,
    df: pd.DataFrame,
    vertical: str,
):
    """Construct a LangGraph ReAct agent with the 6 product tools."""
    effective_api_key = api_key or ("ollama" if base_url else "not-set")
    llm = ChatOpenAI(
        model=model_name,
        api_key=effective_api_key,
        base_url=base_url,
        temperature=0,
        timeout=timeout_s,
    )
    tools = build_tools(df, vertical)
    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=SYSTEM_PROMPT,
        name="product_recommendation_agent",
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_agent.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/kargo_reco/agent.py tests/test_agent.py
git commit -m "feat: add ReAct agent with 6 product exploration tools

build_agent constructs a LangGraph create_react_agent with system prompt
guiding the strategy. build_tools binds pure tool functions to a DataFrame."
```

---

### Task 5: Create trace.py

**Files:**
- Create: `src/kargo_reco/trace.py`
- No separate test file — tested as part of workflow integration tests

- [ ] **Step 1: Implement trace.py**

Create `src/kargo_reco/trace.py`:

```python
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from kargo_reco.schemas import AgentStep, TraceArtifact


class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if hasattr(record, "event_payload"):
            payload["event_payload"] = record.event_payload
        return json.dumps(payload, default=str)


def get_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("kargo_reco")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = JsonLineFormatter()

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_dir / "kargo_reco.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def extract_agent_steps(messages: list[BaseMessage]) -> list[AgentStep]:
    """Extract AgentStep records from the agent's message history.

    The message history alternates: AIMessage (with tool_calls) → ToolMessage (result).
    We pair them up into AgentStep objects.
    """
    steps: list[AgentStep] = []
    step_number = 0
    pending_reasoning: str | None = None
    pending_tool_calls: list[dict[str, Any]] = []

    for msg in messages:
        if isinstance(msg, AIMessage):
            # Capture the agent's reasoning text (content before tool calls)
            pending_reasoning = msg.content if isinstance(msg.content, str) and msg.content.strip() else None
            # Capture tool calls from this AI message
            pending_tool_calls = list(msg.tool_calls) if msg.tool_calls else []

        elif isinstance(msg, ToolMessage) and pending_tool_calls:
            # Match this tool result to the corresponding tool call
            tool_call = None
            for tc in pending_tool_calls:
                if tc.get("id") == msg.tool_call_id:
                    tool_call = tc
                    break

            if tool_call is None and pending_tool_calls:
                tool_call = pending_tool_calls[0]

            if tool_call:
                step_number += 1
                # Parse tool output
                try:
                    tool_output = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                except (json.JSONDecodeError, TypeError):
                    tool_output = {"raw": str(msg.content)}

                steps.append(AgentStep(
                    step_number=step_number,
                    tool_name=tool_call.get("name", "unknown"),
                    tool_input=tool_call.get("args", {}),
                    tool_output=tool_output if isinstance(tool_output, dict) else {"result": tool_output},
                    agent_reasoning=pending_reasoning,
                    latency_ms=0,
                ))
                pending_reasoning = None
                pending_tool_calls = [tc for tc in pending_tool_calls if tc.get("id") != msg.tool_call_id]

    return steps


@dataclass
class TraceManager:
    traces_dir: Path
    logger: logging.Logger

    def build_path(self, request_id: str) -> Path:
        dated_dir = self.traces_dir / datetime.now(timezone.utc).strftime("%Y-%m-%d")
        dated_dir.mkdir(parents=True, exist_ok=True)
        return dated_dir / f"{request_id}.json"

    def write(self, trace: TraceArtifact) -> Path:
        path = self.build_path(trace.request_id)
        path.write_text(trace.model_dump_json(indent=2), encoding="utf-8")
        self.logger.info(
            "trace_written",
            extra={
                "event_payload": {
                    "request_id": trace.request_id,
                    "path": str(path),
                    "status": trace.final_response.get("meta", {}).get("status"),
                }
            },
        )
        return path

    def log_event(self, message: str, payload: dict[str, Any]) -> None:
        self.logger.info(message, extra={"event_payload": payload})
```

- [ ] **Step 2: Commit**

```bash
git add src/kargo_reco/trace.py
git commit -m "feat: add trace module with AgentStep extraction from message history

Replaces tracing.py. Extracts tool call/result pairs from LangGraph agent
messages into AgentStep records for the UI reasoning panel."
```

---

### Task 6: Create guardrails.py

**Files:**
- Create: `src/kargo_reco/guardrails.py`
- Test: `tests/test_guardrails.py`

- [ ] **Step 1: Write failing tests for guardrails**

Create `tests/test_guardrails.py`:

```python
from __future__ import annotations

import pandas as pd
import pytest

from kargo_reco.guardrails import validate_agent_output, deterministic_fallback
from kargo_reco.schemas import RecommendationRequest


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"creative_name": "Alpha", "click_through_rate": 0.42, "in_view_rate": 0.70, "vertical": "Retail", "minimum_budget": 20000, "vertical_normalized": "retail"},
            {"creative_name": "Beta", "click_through_rate": 0.38, "in_view_rate": 0.82, "vertical": "Retail", "minimum_budget": 18000, "vertical_normalized": "retail"},
            {"creative_name": "Delta", "click_through_rate": 0.60, "in_view_rate": 0.90, "vertical": "Finance", "minimum_budget": 50000, "vertical_normalized": "finance"},
        ]
    )


@pytest.fixture
def retail_request() -> RecommendationRequest:
    return RecommendationRequest(
        client_name="Acme", kpi="click_through_rate", client_vertical="Retail", budget=25000
    )


# --- validate_agent_output ---

def test_validate_passes_correct_output(sample_df: pd.DataFrame, retail_request: RecommendationRequest) -> None:
    agent_output = {
        "recommendations": [
            {"creative_name": "Alpha", "vertical": "Retail", "minimum_budget": 20000, "click_through_rate": 0.42, "in_view_rate": 0.70, "rank": 1},
        ],
        "reasoning": "Alpha is the best CTR pick.",
    }
    violations = validate_agent_output(agent_output, sample_df, retail_request)
    assert violations == []


def test_validate_catches_hallucinated_product(sample_df: pd.DataFrame, retail_request: RecommendationRequest) -> None:
    agent_output = {
        "recommendations": [
            {"creative_name": "FakeProduct", "vertical": "Retail", "minimum_budget": 20000, "click_through_rate": 0.99, "in_view_rate": 0.99, "rank": 1},
        ],
        "reasoning": "FakeProduct is great.",
    }
    violations = validate_agent_output(agent_output, sample_df, retail_request)
    assert any("not found" in v for v in violations)


def test_validate_catches_vertical_mismatch(sample_df: pd.DataFrame, retail_request: RecommendationRequest) -> None:
    agent_output = {
        "recommendations": [
            {"creative_name": "Delta", "vertical": "Finance", "minimum_budget": 50000, "click_through_rate": 0.60, "in_view_rate": 0.90, "rank": 1},
        ],
        "reasoning": "Delta has the best CTR.",
    }
    violations = validate_agent_output(agent_output, sample_df, retail_request)
    assert any("vertical" in v.lower() for v in violations)


def test_validate_catches_budget_overrun(sample_df: pd.DataFrame, retail_request: RecommendationRequest) -> None:
    agent_output = {
        "recommendations": [
            {"creative_name": "Alpha", "vertical": "Retail", "minimum_budget": 20000, "click_through_rate": 0.42, "in_view_rate": 0.70, "rank": 1},
            {"creative_name": "Beta", "vertical": "Retail", "minimum_budget": 18000, "click_through_rate": 0.38, "in_view_rate": 0.82, "rank": 2},
        ],
        "reasoning": "Bundle both.",
    }
    # Budget is 25000, Alpha (20000) + Beta (18000) = 38000 > 25000
    violations = validate_agent_output(agent_output, sample_df, retail_request)
    assert any("budget" in v.lower() for v in violations)


def test_validate_passes_bundled_within_budget(sample_df: pd.DataFrame) -> None:
    request = RecommendationRequest(
        client_name="Acme", kpi="click_through_rate", client_vertical="Retail", budget=50000
    )
    agent_output = {
        "recommendations": [
            {"creative_name": "Alpha", "vertical": "Retail", "minimum_budget": 20000, "click_through_rate": 0.42, "in_view_rate": 0.70, "rank": 1},
            {"creative_name": "Beta", "vertical": "Retail", "minimum_budget": 18000, "click_through_rate": 0.38, "in_view_rate": 0.82, "rank": 2},
        ],
        "reasoning": "Both fit within budget.",
    }
    violations = validate_agent_output(agent_output, sample_df, request)
    assert violations == []


# --- deterministic_fallback ---

def test_deterministic_fallback_returns_best_by_kpi(sample_df: pd.DataFrame, retail_request: RecommendationRequest) -> None:
    result = deterministic_fallback(sample_df, retail_request)
    assert result["status"] == "success"
    assert len(result["recommendations"]) == 1
    assert result["recommendations"][0]["creative_name"] == "Alpha"  # highest CTR in Retail


def test_deterministic_fallback_no_match(sample_df: pd.DataFrame) -> None:
    request = RecommendationRequest(
        client_name="Acme", kpi="click_through_rate", client_vertical="Travel", budget=100000
    )
    result = deterministic_fallback(sample_df, request)
    assert result["status"] == "no_match"
    assert result["recommendations"] == []


def test_deterministic_fallback_budget_too_low(sample_df: pd.DataFrame) -> None:
    request = RecommendationRequest(
        client_name="Acme", kpi="click_through_rate", client_vertical="Retail", budget=1000
    )
    result = deterministic_fallback(sample_df, request)
    assert result["status"] == "no_match"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_guardrails.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kargo_reco.guardrails'`

- [ ] **Step 3: Implement guardrails.py**

Create `src/kargo_reco/guardrails.py`:

```python
from __future__ import annotations

from typing import Any

import pandas as pd

from kargo_reco.schemas import RecommendationRequest


def validate_agent_output(
    agent_output: dict[str, Any],
    df: pd.DataFrame,
    request: RecommendationRequest,
) -> list[str]:
    """Validate the agent's recommendation against hard constraints.

    Returns a list of violation descriptions. Empty list means valid.
    """
    violations: list[str] = []
    recommendations = agent_output.get("recommendations", [])

    if not recommendations:
        return violations

    requested_vertical = request.client_vertical.strip().lower()
    total_budget_used = 0.0

    for rec in recommendations:
        name = rec.get("creative_name", "")
        vertical = rec.get("vertical", "").strip().lower()

        # Check product exists in CSV
        matches = df[
            (df["creative_name"] == name) & (df["vertical_normalized"] == requested_vertical)
        ]
        if matches.empty:
            violations.append(f"Product '{name}' not found in {request.client_vertical} vertical")
            continue

        # Check vertical matches request
        if vertical != requested_vertical:
            violations.append(
                f"Product '{name}' vertical '{rec.get('vertical')}' does not match requested '{request.client_vertical}'"
            )

        total_budget_used += float(matches.iloc[0]["minimum_budget"])

    # Check total budget
    if total_budget_used > request.budget:
        violations.append(
            f"Total minimum budget ${total_budget_used:,.0f} exceeds client budget ${request.budget:,.0f}"
        )

    return violations


def deterministic_fallback(
    df: pd.DataFrame,
    request: RecommendationRequest,
) -> dict[str, Any]:
    """Produce a correct recommendation using deterministic logic.

    Used when the agent's output fails post-validation.
    """
    requested_vertical = request.client_vertical.strip().lower()

    # Filter by vertical
    vertical_matches = df[df["vertical_normalized"] == requested_vertical].copy()
    if vertical_matches.empty:
        return {
            "status": "no_match",
            "recommendations": [],
            "reasoning": f"No products found for vertical '{request.client_vertical}'.",
        }

    # Filter by budget
    affordable = vertical_matches[vertical_matches["minimum_budget"] <= request.budget].copy()
    if affordable.empty:
        return {
            "status": "no_match",
            "recommendations": [],
            "reasoning": (
                f"No products in '{request.client_vertical}' fit within "
                f"budget ${request.budget:,.0f}."
            ),
        }

    # Sort by requested KPI descending, budget ascending as tiebreak
    sorted_df = affordable.sort_values(
        by=[request.kpi, "minimum_budget"],
        ascending=[False, True],
    ).reset_index(drop=True)

    best = sorted_df.iloc[0]
    return {
        "status": "success",
        "recommendations": [
            {
                "creative_name": best["creative_name"],
                "vertical": best["vertical"],
                "minimum_budget": float(best["minimum_budget"]),
                "click_through_rate": float(best["click_through_rate"]),
                "in_view_rate": float(best["in_view_rate"]),
                "rank": 1,
            }
        ],
        "reasoning": (
            f"{best['creative_name']} was selected as the deterministic fallback: "
            f"highest {request.kpi} in {request.client_vertical} within budget."
        ),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_guardrails.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/kargo_reco/guardrails.py tests/test_guardrails.py
git commit -m "feat: add post-validation guardrails and deterministic fallback

validate_agent_output checks for hallucinations, vertical mismatches,
and budget overruns. deterministic_fallback provides a safety net when
the agent's output fails validation."
```

---

### Task 7: Rewrite workflow.py

**Files:**
- Rewrite: `src/kargo_reco/workflow.py`
- Rewrite: `tests/conftest.py`
- Rewrite: `tests/test_workflow.py`

- [ ] **Step 1: Write failing workflow integration tests**

Rewrite `tests/conftest.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from kargo_reco.benchmark_loader import BenchmarkRepository
from kargo_reco.trace import TraceManager, get_logger
from kargo_reco.workflow import WorkflowRunner


@pytest.fixture
def benchmark_csv(tmp_path: Path) -> Path:
    path = tmp_path / "benchmarks.csv"
    path.write_text(
        "\n".join(
            [
                "creative_name,click_through_rate,in_view_rate,vertical,minimum_budget",
                "Retail Rocket,0.42,0.79,Retail,20000",
                "Retail Spotlight,0.42,0.75,Retail,18000",
                "Retail Video+,0.39,0.82,Retail,25000",
                "Auto Motion,0.31,0.88,Automotive,30000",
                "Auto Highview,0.28,0.91,Automotive,22000",
                "Finance Focus,0.25,0.86,Finance,15000",
                "Finance Premium,0.29,0.89,Finance,28000",
            ]
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def workflow_runner(tmp_path: Path, benchmark_csv: Path) -> WorkflowRunner:
    traces_dir = tmp_path / "traces"
    logs_dir = tmp_path / "logs"
    uploads_dir = tmp_path / "uploads"
    trace_manager = TraceManager(traces_dir, get_logger(logs_dir))
    repository = BenchmarkRepository(benchmark_csv)
    return WorkflowRunner(
        repository=repository,
        trace_manager=trace_manager,
        uploads_dir=uploads_dir,
        model_name="gpt-4.1-mini",
        api_key=None,
        base_url=None,
        timeout_s=10,
    )
```

Rewrite `tests/test_workflow.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from langchain_core.messages import AIMessage, ToolMessage

from kargo_reco.schemas import RecommendationRequest


def _mock_agent_response(tool_calls_sequence: list[tuple[str, dict, str]]) -> list:
    """Build a mock message history for the agent.

    Each tuple is (tool_name, tool_args, tool_result_json).
    """
    messages = []
    for tool_name, tool_args, tool_result in tool_calls_sequence:
        call_id = f"call_{tool_name}"
        messages.append(AIMessage(
            content=f"I'll use {tool_name}.",
            tool_calls=[{"id": call_id, "name": tool_name, "args": tool_args}],
        ))
        messages.append(ToolMessage(
            content=tool_result,
            tool_call_id=call_id,
        ))
    # Final AI message with no tool calls (agent done)
    messages.append(AIMessage(content="Done."))
    return messages


def test_workflow_success_with_mocked_agent(workflow_runner) -> None:
    mock_messages = _mock_agent_response([
        ("filter_by_vertical", {"vertical": "Retail"}, json.dumps([
            {"creative_name": "Retail Rocket", "click_through_rate": 0.42, "in_view_rate": 0.79, "vertical": "Retail", "minimum_budget": 20000},
            {"creative_name": "Retail Spotlight", "click_through_rate": 0.42, "in_view_rate": 0.75, "vertical": "Retail", "minimum_budget": 18000},
        ])),
        ("filter_by_budget", {"budget": 25000, "products": []}, json.dumps([
            {"creative_name": "Retail Rocket", "click_through_rate": 0.42, "in_view_rate": 0.79, "vertical": "Retail", "minimum_budget": 20000},
            {"creative_name": "Retail Spotlight", "click_through_rate": 0.42, "in_view_rate": 0.75, "vertical": "Retail", "minimum_budget": 18000},
        ])),
        ("sort_by_kpi", {"kpi": "click_through_rate", "products": [], "limit": 3}, json.dumps([
            {"creative_name": "Retail Rocket", "click_through_rate": 0.42, "in_view_rate": 0.79, "vertical": "Retail", "minimum_budget": 20000},
        ])),
        ("finalize_recommendation", {"products": ["Retail Rocket"], "reasoning": "Retail Rocket has the highest CTR at 42% within the $25k budget."}, json.dumps({
            "recommendations": [{"creative_name": "Retail Rocket", "vertical": "Retail", "minimum_budget": 20000, "click_through_rate": 0.42, "in_view_rate": 0.79, "rank": 1}],
            "reasoning": "Retail Rocket has the highest CTR at 42% within the $25k budget.",
        })),
    ])

    with patch.object(workflow_runner, "_run_agent", return_value=mock_messages):
        response = workflow_runner.run_from_payload(
            {"client_name": "Acme", "kpi": "CLICK_THROUGH_RATE", "client_vertical": "Retail", "budget": 25000},
            source="test",
        )

    assert response.meta.status == "success"
    assert response.recommendations[0].creative_name == "Retail Rocket"
    assert response.meta.source == "llm"
    assert len(response.agent_trace) > 0
    assert response.meta.trace_path is not None


def test_workflow_guardrail_fallback_on_hallucination(workflow_runner) -> None:
    mock_messages = _mock_agent_response([
        ("finalize_recommendation", {"products": ["FakeProduct"], "reasoning": "Best pick."}, json.dumps({
            "recommendations": [{"creative_name": "FakeProduct", "vertical": "Retail", "minimum_budget": 10000, "click_through_rate": 0.99, "in_view_rate": 0.99, "rank": 1}],
            "reasoning": "Best pick.",
        })),
    ])

    with patch.object(workflow_runner, "_run_agent", return_value=mock_messages):
        response = workflow_runner.run_from_payload(
            {"client_name": "Acme", "kpi": "CLICK_THROUGH_RATE", "client_vertical": "Retail", "budget": 25000},
            source="test",
        )

    assert response.meta.status in ("success", "guardrail_fallback")
    assert response.meta.source == "guardrail_fallback"
    # Fallback should pick a real product
    assert response.recommendations[0].creative_name in ("Retail Rocket", "Retail Spotlight")


def test_workflow_no_match_vertical(workflow_runner) -> None:
    mock_messages = _mock_agent_response([
        ("filter_by_vertical", {"vertical": "Travel"}, json.dumps([])),
        ("finalize_recommendation", {"products": [], "reasoning": "No Travel products found."}, json.dumps({
            "recommendations": [],
            "reasoning": "No Travel products found.",
        })),
    ])

    with patch.object(workflow_runner, "_run_agent", return_value=mock_messages):
        response = workflow_runner.run_from_payload(
            {"client_name": "Explorer", "kpi": "in_view_rate", "client_vertical": "Travel", "budget": 50000},
            source="test",
        )

    assert response.meta.status == "no_match"


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
            ]
        ),
        encoding="utf-8",
    )

    metadata = workflow_runner.upload_benchmarks(str(uploaded_csv))
    assert metadata["row_count"] == 1
    assert "uploads" in metadata["file"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_workflow.py -v`
Expected: FAIL — old `WorkflowRunner` doesn't accept new constructor args

- [ ] **Step 3: Rewrite workflow.py**

Replace `src/kargo_reco/workflow.py` with:

```python
from __future__ import annotations

import shutil
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.messages import BaseMessage

from kargo_reco.agent import build_agent
from kargo_reco.benchmark_loader import BenchmarkRepository
from kargo_reco.config import get_settings
from kargo_reco.guardrails import deterministic_fallback, validate_agent_output
from kargo_reco.schemas import (
    AgentStep,
    RecommendationItem,
    RecommendationRequest,
    RecommendationResponse,
    ResponseMeta,
    TraceArtifact,
)
from kargo_reco.trace import TraceManager, extract_agent_steps, get_logger


class WorkflowRunner:
    def __init__(
        self,
        repository: BenchmarkRepository,
        trace_manager: TraceManager,
        uploads_dir: Path,
        model_name: str,
        api_key: str | None,
        base_url: str | None,
        timeout_s: float,
    ) -> None:
        self.repository = repository
        self.trace_manager = trace_manager
        self.uploads_dir = uploads_dir
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.timeout_s = timeout_s

    def _run_agent(
        self, df: pd.DataFrame, request: RecommendationRequest
    ) -> list[BaseMessage]:
        """Run the ReAct agent and return its message history."""
        agent = build_agent(
            model_name=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            timeout_s=self.timeout_s,
            df=df,
            vertical=request.client_vertical,
        )
        user_message = (
            f"Client: {request.client_name}\n"
            f"Vertical: {request.client_vertical}\n"
            f"KPI to optimize: {request.kpi}\n"
            f"Budget: ${request.budget:,.0f}\n\n"
            f"Find the best product(s) for this client."
        )
        result = agent.invoke({"messages": [("user", user_message)]})
        return result["messages"]

    def _extract_finalize_output(self, messages: list[BaseMessage]) -> dict[str, Any] | None:
        """Find the last finalize_recommendation tool result in the message history."""
        from langchain_core.messages import ToolMessage
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                import json
                try:
                    parsed = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                    if isinstance(parsed, dict) and "recommendations" in parsed:
                        return parsed
                except (json.JSONDecodeError, TypeError):
                    continue
        return None

    def run_from_payload(
        self, payload: dict[str, Any], *, source: str = "api"
    ) -> RecommendationResponse:
        trace = TraceArtifact(source=source, raw_request=payload)
        self.trace_manager.log_event(
            "request_received",
            {"request_id": trace.request_id, "raw_request": payload, "source": source},
        )

        started = time.perf_counter()

        try:
            # Pre-validation
            request = RecommendationRequest.model_validate(payload)
            trace.normalized_request = request.model_dump()

            # Load benchmarks
            df, metadata = self.repository.get_snapshot()
            trace.benchmark_metadata = metadata

            # Run agent
            agent_start = time.perf_counter()
            try:
                messages = self._run_agent(df, request)
            except Exception as agent_err:
                trace.errors.append(f"Agent failed: {agent_err}")
                messages = []
            agent_latency = int((time.perf_counter() - agent_start) * 1000)
            trace.step_latencies_ms["run_agent"] = agent_latency

            # Extract agent trace
            agent_steps = extract_agent_steps(messages)
            trace.agent_trace = agent_steps

            # Extract agent output
            agent_output = self._extract_finalize_output(messages)

            # Post-validation
            if agent_output and agent_output.get("recommendations"):
                violations = validate_agent_output(agent_output, df, request)
                trace.guardrail_violations = violations

                if not violations:
                    # Agent output is valid
                    response = self._build_response(
                        request=request,
                        agent_output=agent_output,
                        agent_steps=agent_steps,
                        source="llm",
                        status="success",
                        trace=trace,
                        latency_ms=int((time.perf_counter() - started) * 1000),
                    )
                else:
                    # Guardrail triggered — use fallback
                    fallback = deterministic_fallback(df, request)
                    response = self._build_response(
                        request=request,
                        agent_output=fallback,
                        agent_steps=agent_steps,
                        source="guardrail_fallback",
                        status=fallback["status"],
                        trace=trace,
                        latency_ms=int((time.perf_counter() - started) * 1000),
                    )
            elif agent_output and not agent_output.get("recommendations"):
                # Agent found no match
                response = self._build_response(
                    request=request,
                    agent_output=agent_output,
                    agent_steps=agent_steps,
                    source="llm",
                    status="no_match",
                    trace=trace,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                )
            else:
                # Agent didn't produce valid output — fallback
                fallback = deterministic_fallback(df, request)
                trace.guardrail_violations.append("Agent did not call finalize_recommendation")
                response = self._build_response(
                    request=request,
                    agent_output=fallback,
                    agent_steps=agent_steps,
                    source="guardrail_fallback",
                    status=fallback["status"],
                    trace=trace,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                )

            trace.final_response = response.model_dump()
            trace_path = str(self.trace_manager.write(trace))
            response.meta.trace_path = trace_path
            return response

        except Exception as exc:
            trace.errors.append(str(exc))
            trace.final_response = {"error": str(exc)}
            self.trace_manager.write(trace)
            raise RuntimeError(f"Recommendation workflow failed. Trace: {trace.request_id}") from exc

    def _build_response(
        self,
        *,
        request: RecommendationRequest,
        agent_output: dict[str, Any],
        agent_steps: list[AgentStep],
        source: str,
        status: str,
        trace: TraceArtifact,
        latency_ms: int,
    ) -> RecommendationResponse:
        recommendations = [
            RecommendationItem(**rec) for rec in agent_output.get("recommendations", [])
        ]
        return RecommendationResponse(
            request=request,
            recommendations=recommendations,
            summary=agent_output.get("reasoning", "No reasoning provided."),
            agent_trace=agent_steps,
            meta=ResponseMeta(
                status=status,
                request_id=trace.request_id,
                model=self.model_name,
                total_tokens=None,
                agent_steps=len(agent_steps),
                latency_ms=latency_ms,
                source=source,
            ),
        )

    def reload_benchmarks(self) -> dict[str, Any]:
        metadata = self.repository.reload()
        return metadata.model_dump()

    def upload_benchmarks(self, uploaded_path: str) -> dict[str, Any]:
        source_path = Path(uploaded_path)
        if not source_path.exists():
            raise FileNotFoundError(f"uploaded benchmark file not found: {source_path}")

        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        sanitized_name = source_path.name.replace(" ", "_")
        target_path = self.uploads_dir / (
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{sanitized_name}"
        )
        shutil.copy2(source_path, target_path)
        metadata = self.repository.activate_path(target_path)
        self.trace_manager.log_event(
            "benchmark_uploaded",
            {"file": metadata.file, "row_count": metadata.row_count},
        )
        return metadata.model_dump()

    def run_recommendation(
        self,
        request: RecommendationRequest,
        *,
        raw_request: dict[str, Any] | None = None,
        source: str = "api",
    ) -> RecommendationResponse:
        payload = raw_request or request.model_dump()
        return self.run_from_payload(payload, source=source)


@lru_cache(maxsize=1)
def get_default_runner() -> WorkflowRunner:
    settings = get_settings()
    trace_manager = TraceManager(settings.traces_dir, get_logger(settings.logs_dir))
    repository = BenchmarkRepository(settings.benchmark_csv_path)
    return WorkflowRunner(
        repository=repository,
        trace_manager=trace_manager,
        uploads_dir=settings.uploads_dir,
        model_name=settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        timeout_s=settings.openai_timeout_s,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_workflow.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/kargo_reco/workflow.py tests/conftest.py tests/test_workflow.py
git commit -m "feat: rewrite workflow with agent-driven recommendation loop

WorkflowRunner now runs a ReAct agent, extracts its tool call history,
validates output via guardrails, and falls back to deterministic logic
when the agent hallucinates or violates constraints."
```

---

### Task 8: Rewrite ui.py

**Files:**
- Rewrite: `src/kargo_reco/ui.py`
- Rewrite: `tests/test_ui.py`

- [ ] **Step 1: Write UI tests**

Rewrite `tests/test_ui.py`:

```python
from __future__ import annotations

from unittest.mock import MagicMock, patch

from kargo_reco.ui import (
    _format_currency,
    _format_rate,
    _render_recommendation_cards,
    _render_summary,
    _render_agent_steps,
    _build_request_preview,
)
from kargo_reco.schemas import (
    AgentStep,
    RecommendationItem,
    RecommendationRequest,
    RecommendationResponse,
    ResponseMeta,
)


def _sample_response() -> RecommendationResponse:
    return RecommendationResponse(
        request=RecommendationRequest(
            client_name="Acme", kpi="click_through_rate", client_vertical="Retail", budget=25000
        ),
        recommendations=[
            RecommendationItem(
                creative_name="Alpha", vertical="Retail", minimum_budget=20000,
                click_through_rate=0.42, in_view_rate=0.79, rank=1,
            ),
        ],
        summary="Alpha is the best pick for Retail CTR.",
        agent_trace=[
            AgentStep(step_number=1, tool_name="filter_by_vertical", tool_input={"vertical": "Retail"}, tool_output={"products": []}, agent_reasoning="Start by filtering.", latency_ms=50),
            AgentStep(step_number=2, tool_name="finalize_recommendation", tool_input={"products": ["Alpha"]}, tool_output={}, agent_reasoning=None, latency_ms=30),
        ],
        meta=ResponseMeta(
            status="success", request_id="abc-123", model="gpt-4.1-mini",
            total_tokens=100, agent_steps=2, latency_ms=500, source="llm",
        ),
    )


def test_format_currency() -> None:
    assert _format_currency(25000) == "$25,000"
    assert _format_currency(None) == "N/A"


def test_format_rate() -> None:
    assert _format_rate(0.42) == "42%"
    assert _format_rate(None) == "N/A"


def test_render_recommendation_cards_success() -> None:
    html = _render_recommendation_cards(_sample_response())
    assert "Alpha" in html
    assert "Retail" in html


def test_render_summary() -> None:
    html = _render_summary(_sample_response())
    assert "Alpha is the best pick" in html


def test_render_agent_steps() -> None:
    html = _render_agent_steps(_sample_response())
    assert "filter_by_vertical" in html
    assert "finalize_recommendation" in html
    assert "Start by filtering" in html


def test_build_request_preview() -> None:
    preview = _build_request_preview("Acme", "click_through_rate", "Retail", 25000)
    assert preview["client_name"] == "Acme"
    assert preview["budget"] == 25000
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ui.py -v`
Expected: FAIL — old UI doesn't have `_render_agent_steps`, `_render_recommendation_cards`

- [ ] **Step 3: Rewrite ui.py**

Replace `src/kargo_reco/ui.py` with:

```python
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

from kargo_reco.config import get_settings
from kargo_reco.schemas import RecommendationResponse
from kargo_reco.workflow import WorkflowRunner, get_default_runner


APP_CSS = """
.result-card, .summary-card, .steps-card {
  background: rgba(255, 255, 255, 0.90);
  border: 1px solid rgba(15, 23, 42, 0.08);
  border-radius: 22px;
  box-shadow: 0 18px 48px rgba(15, 23, 42, 0.08);
  padding: 22px 24px;
  margin-bottom: 16px;
}
.eyebrow {
  display: inline-block;
  font-size: 0.75rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #0369a1;
  margin-bottom: 8px;
}
.result-title { margin: 0 0 12px; font-size: 1.9rem; line-height: 1.1; }
.status-chip {
  display: inline-block; margin-bottom: 12px; padding: 6px 10px;
  border-radius: 999px; font-size: 0.8rem; font-weight: 600;
  background: #dcfce7; color: #166534;
}
.status-chip.no-match { background: #fef3c7; color: #92400e; }
.status-chip.fallback { background: #fee2e2; color: #991b1b; }
.metric-grid {
  display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px; margin-top: 18px;
}
.metric {
  padding: 14px 16px; border-radius: 16px;
  background: #f8fafc; border: 1px solid rgba(15, 23, 42, 0.06);
}
.metric-label { display: block; font-size: 0.8rem; color: #64748b; margin-bottom: 4px; }
.metric-value { font-size: 1.1rem; font-weight: 600; color: #0f172a; }
.budget-bar-container {
  margin-top: 16px; background: #f1f5f9; border-radius: 8px;
  height: 28px; position: relative; overflow: hidden;
}
.budget-bar-fill {
  height: 100%; border-radius: 8px; background: #0369a1;
  display: flex; align-items: center; padding-left: 10px;
  color: white; font-size: 0.8rem; font-weight: 600;
}
.step-card {
  border: 1px solid #e2e8f0; border-radius: 12px; padding: 14px;
  margin-bottom: 10px; background: #f8fafc;
}
.step-header { font-weight: 600; color: #0f172a; margin-bottom: 6px; }
.step-thought { color: #475569; font-style: italic; margin-bottom: 8px; }
.step-detail { font-size: 0.85rem; color: #64748b; }
.step-detail code { background: #e2e8f0; padding: 2px 6px; border-radius: 4px; font-size: 0.8rem; }
.summary-card h3 { margin: 0 0 10px; font-size: 1.1rem; }
.summary-card p { margin: 0 0 16px; line-height: 1.55; }
"""


def _format_currency(value: float | int | None) -> str:
    if value is None:
        return "N/A"
    return f"${float(value):,.0f}"


def _format_rate(value: float | int | None) -> str:
    if value is None:
        return "N/A"
    return f"{float(value) * 100:.0f}%"


def _humanize_token(text: str) -> str:
    normalized = text.replace("_", " ").strip()
    replacements = {
        "click through rate": "Click-through rate",
        "in view rate": "In-view rate",
    }
    lowered = normalized.lower()
    return replacements.get(lowered, normalized[:1].upper() + normalized[1:])


def _status_chip(status: str) -> str:
    cls = ""
    label = status.replace("_", " ").title()
    if status == "no_match":
        cls = " no-match"
        label = "No eligible match"
    elif status == "guardrail_fallback":
        cls = " fallback"
        label = "Guardrail fallback"
    else:
        label = "Eligible and selected"
    return f'<div class="status-chip{cls}">{label}</div>'


def _render_recommendation_cards(response: RecommendationResponse) -> str:
    if not response.recommendations:
        return f"""
<div class="result-card">
  <div class="eyebrow">Recommendation Status</div>
  {_status_chip(response.meta.status)}
  <h2 class="result-title">No eligible product found</h2>
  <p>{response.summary}</p>
</div>"""

    budget = response.request.budget
    total_used = sum(r.minimum_budget for r in response.recommendations)
    pct = min(total_used / budget * 100, 100) if budget > 0 else 0

    cards = []
    for item in response.recommendations:
        cards.append(f"""
<div class="result-card">
  <div class="eyebrow">Recommended Product #{item.rank}</div>
  {_status_chip(response.meta.status)}
  <h2 class="result-title">{item.creative_name}</h2>
  <div class="metric-grid">
    <div class="metric"><span class="metric-label">Vertical</span><span class="metric-value">{item.vertical}</span></div>
    <div class="metric"><span class="metric-label">Minimum budget</span><span class="metric-value">{_format_currency(item.minimum_budget)}</span></div>
    <div class="metric"><span class="metric-label">Click-through rate</span><span class="metric-value">{_format_rate(item.click_through_rate)}</span></div>
    <div class="metric"><span class="metric-label">In-view rate</span><span class="metric-value">{_format_rate(item.in_view_rate)}</span></div>
  </div>
</div>""")

    budget_bar = f"""
<div class="budget-bar-container">
  <div class="budget-bar-fill" style="width: {pct:.0f}%">{_format_currency(total_used)} / {_format_currency(budget)} ({pct:.0f}%)</div>
</div>"""

    return "\n".join(cards) + budget_bar


def _render_summary(response: RecommendationResponse) -> str:
    return f"""
<div class="summary-card">
  <div class="eyebrow">Agent Summary</div>
  <p>{response.summary}</p>
  <p style="font-size: 0.8rem; color: #64748b;">Source: {response.meta.source} | Model: {response.meta.model} | Steps: {response.meta.agent_steps}</p>
</div>"""


def _render_agent_steps(response: RecommendationResponse) -> str:
    if not response.agent_trace:
        return '<div class="steps-card"><p>No agent steps recorded.</p></div>'

    steps_html = []
    for step in response.agent_trace:
        thought = f'<div class="step-thought">"{step.agent_reasoning}"</div>' if step.agent_reasoning else ""
        input_str = json.dumps(step.tool_input, indent=2) if step.tool_input else "{}"
        output_preview = json.dumps(step.tool_output, indent=2) if step.tool_output else "{}"
        if len(output_preview) > 500:
            output_preview = output_preview[:500] + "..."

        steps_html.append(f"""
<div class="step-card">
  <div class="step-header">Step {step.step_number}: {step.tool_name}</div>
  {thought}
  <div class="step-detail"><strong>Input:</strong> <code>{input_str}</code></div>
  <div class="step-detail"><strong>Output:</strong> <code>{output_preview}</code></div>
</div>""")

    return f'<div class="steps-card"><div class="eyebrow">Agent Reasoning Steps</div>{"".join(steps_html)}</div>'


def _build_request_preview(
    client_name: str, kpi: str, client_vertical: str, budget: float | int | None,
) -> dict[str, Any]:
    return {
        "client_name": client_name,
        "kpi": kpi,
        "client_vertical": client_vertical,
        "budget": budget,
    }


def _handle_payload(
    runner: WorkflowRunner, payload: dict[str, Any], source: str
) -> tuple[str, str, str, dict[str, Any], dict[str, Any], str | None]:
    response = runner.run_from_payload(payload, source=source)
    trace_payload = {}
    if response.meta.trace_path:
        trace_path = Path(response.meta.trace_path)
        if trace_path.exists():
            trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
    return (
        _render_recommendation_cards(response),
        _render_summary(response),
        _render_agent_steps(response),
        response.model_dump(),
        trace_payload,
        response.meta.trace_path,
    )


def _load_sample_requests(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("client_requests.json must contain a list of request objects")
    return [item for item in payload if isinstance(item, dict)]


def _sample_request_label(sample: dict[str, Any]) -> str:
    return (
        f"{sample.get('client_name', 'Unknown')} | "
        f"{sample.get('client_vertical', 'Unknown')} | "
        f"{sample.get('kpi', 'unknown')} | "
        f"{sample.get('budget', sample.get('minimum_budget', 'n/a'))}"
    )


def _apply_sample_request(
    selected_label: str | None,
    labels_to_samples: dict[str, dict[str, Any]],
) -> tuple[str, str, str, float | int | None, dict[str, Any], str]:
    sample = labels_to_samples.get(selected_label or "")
    if not sample:
        preview = _build_request_preview("", "click_through_rate", "", None)
        return "", "click_through_rate", "", None, preview, json.dumps(preview, indent=2)

    kpi = sample.get("kpi", "click_through_rate")
    if isinstance(kpi, str):
        kpi = kpi.strip().lower().replace(" ", "_")
    budget = sample.get("budget", sample.get("minimum_budget"))
    preview = _build_request_preview(
        str(sample.get("client_name", "")), kpi,
        str(sample.get("client_vertical", "")), budget,
    )
    return preview["client_name"], preview["kpi"], preview["client_vertical"], preview["budget"], preview, json.dumps(preview, indent=2)


def build_app(runner: WorkflowRunner | None = None) -> gr.Blocks:
    workflow_runner = runner or get_default_runner()
    settings = get_settings()
    sample_requests = _load_sample_requests(settings.client_requests_path)
    sample_labels = [_sample_request_label(s) for s in sample_requests]
    labels_to_samples = {_sample_request_label(s): s for s in sample_requests}
    default_sample = sample_requests[0] if sample_requests else {
        "client_name": "Acme Shoes", "kpi": "click_through_rate", "client_vertical": "Retail", "budget": 25000,
    }
    default_preview = _build_request_preview(
        str(default_sample.get("client_name", "")),
        str(default_sample.get("kpi", "click_through_rate")).strip().lower().replace(" ", "_"),
        str(default_sample.get("client_vertical", "")),
        default_sample.get("budget", default_sample.get("minimum_budget")),
    )

    with gr.Blocks(title="Kargo Product Recommendation Engine") as app:
        app.css = APP_CSS
        gr.Markdown("# Kargo Product Recommendation Engine")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Request Builder")
                sample_request = gr.Dropdown(
                    label="Sample Client Requests", choices=sample_labels,
                    value=sample_labels[0] if sample_labels else None, allow_custom_value=False,
                )
                with gr.Accordion("Benchmark Management", open=False):
                    benchmark_upload = gr.File(label="Upload Benchmark CSV", file_types=[".csv"], type="filepath")
                    with gr.Row():
                        reload_button = gr.Button("Reload", variant="secondary")
                        upload_button = gr.Button("Use Upload", variant="secondary")
                    reload_status = gr.JSON(label="Benchmark Status")
                with gr.Tab("Form Input"):
                    client_name = gr.Textbox(label="Client Name", value=default_preview["client_name"])
                    kpi = gr.Dropdown(label="KPI", choices=["click_through_rate", "in_view_rate"], value=default_preview["kpi"])
                    client_vertical = gr.Textbox(label="Client Vertical", value=default_preview["client_vertical"])
                    budget = gr.Number(label="Budget", value=default_preview["budget"])
                    request_preview = gr.JSON(label="Request JSON Preview", value=default_preview)
                    submit_form = gr.Button("Submit Request", variant="primary")
                with gr.Tab("JSON Input"):
                    json_input = gr.Code(label="Request JSON", language="json", value=json.dumps(default_preview, indent=2))
                    submit_json = gr.Button("Submit JSON", variant="primary")

            with gr.Column(scale=1):
                recommendation_card = gr.HTML(label="Recommendation")
                summary_block = gr.HTML(label="Summary")
                agent_steps_block = gr.HTML(label="Agent Steps")
                response_json = gr.JSON(label="Response JSON")
                trace_download = gr.File(label="Download Trace JSON")

        with gr.Accordion("Debug Panel", open=False):
            trace_json = gr.JSON(label="Trace Artifact")

        def on_reload() -> dict[str, Any]:
            return workflow_runner.reload_benchmarks()

        def on_upload(uploaded_file_path: str | None) -> dict[str, Any]:
            if not uploaded_file_path:
                raise gr.Error("Select a benchmark CSV before uploading.")
            return workflow_runner.upload_benchmarks(uploaded_file_path)

        def on_form_submit(cn: str, k: str, cv: str, b: float):
            payload = {"client_name": cn, "kpi": k, "client_vertical": cv, "budget": b}
            return _handle_payload(workflow_runner, payload, "ui_form")

        def on_json_submit(json_value: str):
            payload = json.loads(json_value)
            return _handle_payload(workflow_runner, payload, "ui_json")

        def on_field_change(cn: str, k: str, cv: str, b: float | int | None):
            preview = _build_request_preview(cn, k, cv, b)
            return preview, json.dumps(preview, indent=2)

        reload_button.click(on_reload, outputs=reload_status)
        upload_button.click(on_upload, inputs=[benchmark_upload], outputs=reload_status)
        sample_request.change(
            lambda label: _apply_sample_request(label, labels_to_samples),
            inputs=[sample_request],
            outputs=[client_name, kpi, client_vertical, budget, request_preview, json_input],
        )
        for field in [client_name, kpi, client_vertical, budget]:
            field.change(on_field_change, inputs=[client_name, kpi, client_vertical, budget], outputs=[request_preview, json_input])

        outputs = [recommendation_card, summary_block, agent_steps_block, response_json, trace_json, trace_download]
        submit_form.click(on_form_submit, inputs=[client_name, kpi, client_vertical, budget], outputs=outputs)
        submit_json.click(on_json_submit, inputs=[json_input], outputs=outputs)

    return app
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ui.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/kargo_reco/ui.py tests/test_ui.py
git commit -m "feat: rewrite UI with agent reasoning steps panel

Shows recommendation cards with budget utilization bar, agent summary,
expandable step-by-step reasoning trace, and debug panel. Supports
form input, JSON input, sample requests, and benchmark management."
```

---

### Task 9: Delete old files and clean up

**Files:**
- Delete: `src/kargo_reco/reasoning.py`
- Delete: `src/kargo_reco/llm.py`
- Delete: `src/kargo_reco/recommender.py`
- Delete: `src/kargo_reco/tracing.py`
- Delete: `tests/test_llm.py`
- Delete: `tests/test_recommender.py`

- [ ] **Step 1: Delete old source files**

```bash
git rm src/kargo_reco/reasoning.py src/kargo_reco/llm.py src/kargo_reco/recommender.py src/kargo_reco/tracing.py
```

- [ ] **Step 2: Delete old test files**

```bash
git rm tests/test_llm.py tests/test_recommender.py
```

- [ ] **Step 3: Run full test suite to verify nothing is broken**

Run: `pytest tests/ -v`
Expected: All PASS — no remaining imports of deleted modules

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: remove old deterministic pipeline modules

Delete reasoning.py, llm.py, recommender.py, tracing.py and their tests.
Logic absorbed into agent.py, guardrails.py, trace.py."
```

---

### Task 10: End-to-end verification

**Files:** None — verification only

- [ ] **Step 1: Run the full test suite**

```bash
pytest tests/ -v
```

Expected: All tests pass across test_schemas.py, test_tools.py, test_guardrails.py, test_agent.py, test_workflow.py, test_ui.py, test_benchmark_loader.py.

- [ ] **Step 2: Run the app locally to verify the UI launches**

```bash
python app.py
```

Expected: Gradio launches without errors. Visit the local URL and verify:
- Sample dropdown loads client requests
- Form submission triggers the workflow (will use fallback without API key)
- Agent reasoning steps panel appears (empty if no agent ran)
- Debug panel shows trace JSON

- [ ] **Step 3: Test with a real LLM (if API key available)**

Set `OPENAI_API_KEY` in `.env` and run:

```bash
python app.py
```

Submit a request and verify:
- Agent reasoning steps show multiple tool calls
- Recommendation card shows a real product
- Budget utilization bar renders
- Summary contains agent-generated reasoning
- Status chip shows "Eligible and selected"

- [ ] **Step 4: Final commit with any fixes**

If any fixes were needed, commit them:

```bash
git add -A
git commit -m "fix: address end-to-end verification issues"
```
