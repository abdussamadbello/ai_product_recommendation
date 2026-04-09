# Agentic Product Recommendation Engine — Design Spec

**Date:** 2026-04-09
**Status:** Approved
**Scope:** Ground-up rebuild of the Kargo product recommendation engine from a deterministic pipeline to an LLM-agent-driven system.

---

## 1. Problem Statement

The current implementation uses a deterministic pipeline (sort + filter) with an LLM that only narrates pre-computed decisions. For an AI engineering role, the system must demonstrate genuine agentic behavior: the LLM should reason about product selection through tool use, not just summarize a `sort_values()` result.

## 2. Architecture Overview

```
Request
  │
  ├── Pre-validation (Pydantic)
  │
  ├── Load benchmarks (CSV → DataFrame, cached)
  │
  ├── LLM Agent (LangGraph create_react_agent, ReAct loop)
  │     ├── tool: filter_by_vertical
  │     ├── tool: filter_by_budget
  │     ├── tool: sort_by_kpi
  │     ├── tool: get_product_details
  │     ├── tool: check_budget_remaining
  │     └── tool: finalize_recommendation
  │
  ├── Post-validation guardrail
  │     ├── pass → agent's selection
  │     └── fail → deterministic fallback + log violation
  │
  └── JSON Response + Trace
```

**Approach: Guardrailed Agent.** The LLM agent has full autonomy over product exploration and selection within the ReAct loop. Deterministic guardrails validate input before and output after the agent runs. If the agent hallucinates or violates constraints, a deterministic fallback produces a correct result.

## 3. Agent Core & Tools

### 3.1 Framework

LangGraph `create_react_agent` with a ReAct loop. The agent receives a system prompt suggesting a strategy (filter → sort → bundle → finalize) but is free to deviate.

### 3.2 Tool Definitions

| Tool | Input | Output | Purpose |
|------|-------|--------|---------|
| `filter_by_vertical` | `vertical: str` | List of product dicts in that vertical | Discover available products |
| `filter_by_budget` | `budget: float, products: list[dict]` | Products where minimum_budget <= budget | Narrow by affordability |
| `sort_by_kpi` | `kpi: str, products: list[dict], limit: int` | Top N products ranked by KPI descending | Identify best performers (single KPI) |
| `score_products` | `products: list[dict], weights: dict[str, float], limit: int` | Products ranked by weighted score | Agent generates its own ranking formula |
| `get_product_details` | `product_name: str, vertical: str` | Full product record | Inspect individual candidates |
| `check_budget_remaining` | `budget: float, selected: list[str]` | `{ remaining: float, can_add_more: bool, affordable: list[dict] }` | Enable multi-product bundling |
| `finalize_recommendation` | `products: list[str], reasoning: str` | Structured recommendation dict | Agent commits to its selection |

### 3.3 System Prompt

The system prompt instructs the agent to act as a media strategist assistant. It suggests a strategy:

1. Filter products by the client's vertical
2. Filter by budget
3. Use `score_products` to create a weighted ranking formula reflecting the client's priorities (e.g., weight CTR at 0.8 and IVR at 0.2), or use `sort_by_kpi` for a simple single-KPI sort
4. Check if budget allows adding more products
5. If yes, find the next best product that fits the remaining budget
6. Finalize with selection and explain reasoning — including what weights were chosen and why

The prompt suggests but does not force this order. The agent may call tools in any sequence. The key differentiator is `score_products`: the agent generates its own ranking strategy rather than relying on a hardcoded sort.

### 3.4 Data Layer

Tools read from a `BenchmarkRepository` (existing code, unchanged) that loads and caches the CSV as a DataFrame. Tools operate on the DataFrame but return plain dicts/lists so the LLM sees clean data.

### 3.5 Selection Behavior

- **Single best product** by default: filter vertical → filter budget → sort by KPI → select top
- **Multi-product bundling** when budget allows: after selecting the top product, agent uses `check_budget_remaining` to see if another product fits. If yes, it can add the next-best product and repeat.
- The agent decides whether to bundle — this is where agentic reasoning adds value over a deterministic sort.

## 4. Guardrails & Validation

### 4.1 Pre-validation (before agent runs)

- Pydantic validates `RecommendationRequest` (client_name, kpi, client_vertical, budget)
- KPI normalization: "CTR" → "click_through_rate", "IVR" → "in_view_rate"
- Budget > 0, non-empty strings
- Benchmark CSV loaded and schema-validated
- If pre-validation fails: return error immediately, no agent invocation

### 4.2 Post-validation (after agent finishes)

| Check | Catches |
|-------|---------|
| Every recommended product exists in the CSV | Hallucinated product names |
| Every product matches the requested vertical | Vertical constraint violation |
| Sum of minimum_budget across selected products <= client budget | Budget overrun |
| Top selection has highest or near-highest KPI among eligible | Suboptimal pick |

### 4.3 Fallback behavior

If post-validation fails:
- Log the violation to the trace artifact
- Fall back to deterministic single-best selection (existing `compute_recommendation` logic preserved as safety net)
- Mark `source: "guardrail_fallback"` in response metadata
- The UI displays a status chip indicating the fallback was triggered

## 5. Workflow Orchestration

5 nodes in a LangGraph StateGraph:

```
START → validate_input → load_benchmarks → run_agent → post_validate → build_response → END
```

- `validate_input`: Pydantic parsing, returns `RecommendationRequest`
- `load_benchmarks`: Load CSV via `BenchmarkRepository`, returns DataFrame + metadata
- `run_agent`: Invoke `create_react_agent` ReAct loop, returns agent message history
- `post_validate`: Check agent output against constraints, fallback if needed
- `build_response`: Assemble `RecommendationResponse` + trace artifact, write trace JSON

The agent node (`run_agent`) contains the ReAct loop internally — LangGraph manages the tool-call cycle.

## 6. Schemas & Data Model

### 6.1 Request (unchanged)

```python
RecommendationRequest:
    client_name: str
    kpi: "click_through_rate" | "in_view_rate"
    client_vertical: str
    budget: float
```

### 6.2 Agent Step Trace (new)

```python
AgentStep:
    step_number: int
    tool_name: str
    tool_input: dict
    tool_output: dict
    agent_reasoning: str | None
    latency_ms: int
```

### 6.3 Response (revised)

```python
RecommendationItem:
    creative_name: str
    vertical: str
    minimum_budget: float
    click_through_rate: float
    in_view_rate: float
    rank: int

RecommendationResponse:
    request: RecommendationRequest
    recommendations: list[RecommendationItem]
    summary: str
    agent_trace: list[AgentStep]
    meta: ResponseMeta

ResponseMeta:
    status: "success" | "no_match" | "guardrail_fallback"
    request_id: str
    model: str
    total_tokens: int | None
    agent_steps: int
    latency_ms: int
    source: "llm" | "guardrail_fallback"
```

### 6.4 Changes from current schemas

- `SummaryBlock` → plain `summary: str` (agent writes free-form)
- `AgentStep` is new (captures tool-use trace)
- `SelectionReason` removed (agent's reasoning is the reason)
- `NearestAlternative` removed (agent discusses alternatives in summary)
- `ResponseMeta` gains `agent_steps`, `total_tokens`, `model`
- `status` gains `"guardrail_fallback"` as third state
- `BenchmarkMetadata` unchanged
- `TraceArtifact` extended with `agent_trace: list[AgentStep]`

## 7. UI Design

Gradio web interface. Two-column layout.

### 7.1 Left column: Request Builder

- Sample request dropdown (loads from `client_requests.json`)
- Benchmark management accordion (upload CSV, reload, status display)
- Form input tab (client_name, kpi dropdown, client_vertical, budget, submit button)
- JSON input tab (code editor, submit button)

### 7.2 Right column: Results

- **Recommendation card(s):** One card per selected product with metrics grid. Multiple cards if bundled.
- **Budget utilization bar:** Progress bar showing used / total budget. Makes bundling visible.
- **Agent summary:** Plain text block with the agent's free-form reasoning.
- **Agent reasoning steps:** Expandable accordion, one entry per `AgentStep`. Each expands to show: agent thought, tool input, tool output.
- **Response JSON:** Collapsible full response.
- **Status chip:** Green "success" / amber "no match" / red "guardrail fallback"

### 7.3 Debug panel (collapsed accordion below main content)

- Full trace JSON
- Ranking table (if available from agent's sort_by_kpi calls)

## 8. File Structure

```
src/kargo_reco/
├── config.py              # Settings (mostly unchanged)
├── schemas.py             # Redesigned: Request, Response, AgentStep, Meta, Trace
├── benchmark_loader.py    # BenchmarkRepository (unchanged)
├── tools.py               # 6 agent tool functions (NEW)
├── agent.py               # create_react_agent setup + system prompt (NEW)
├── guardrails.py          # Post-validation + deterministic fallback (NEW)
├── workflow.py            # 5-node orchestration (REWRITTEN)
├── trace.py               # AgentStep extraction from message history (REWRITTEN)
├── ui.py                  # Gradio app (REWRITTEN)

app.py                     # Entry point (unchanged)
```

Deleted: `reasoning.py`, `llm.py`
Absorbed: `recommender.py` logic moved into `guardrails.py` as fallback

## 9. LLM Configuration

Configurable via environment variables (existing pattern):

- `OPENAI_API_KEY`: API key for OpenAI (or omit for Ollama)
- `OPENAI_BASE_URL`: Override for Ollama or other OpenAI-compatible endpoints
- `OPENAI_MODEL`: Model name (default: `gpt-4.1-mini`)
- `OPENAI_TIMEOUT_S`: Request timeout (default: 30s, increased from 20s for agent loops)

When `OPENAI_BASE_URL` is set and no API key is provided, use `"ollama"` as the API key (existing pattern). Document in README the trade-offs of local models vs. cloud models for agent quality.

## 10. Testing Strategy

### Layer 1: Tool unit tests (`test_tools.py`)

- Each tool tested against a fixture DataFrame
- Verify correctness of filtering, sorting, budget math
- Edge cases: unknown vertical, zero budget, empty product list

### Layer 2: Guardrail tests (`test_guardrails.py`)

- Valid agent output passes through
- Hallucinated product name caught, fallback triggered
- Vertical mismatch caught
- Budget overrun caught
- Fallback produces correct deterministic result

### Layer 3: Agent integration tests (`test_agent.py`)

- Mock LLM with scripted tool-call responses (no real API in CI)
- Verify agent calls tools in reasonable order
- Verify agent output passes post-validation
- Optional: golden path tests with real LLM (marked slow, skipped in CI)

### Principle

Deterministic parts tested deterministically. Agentic parts tested by verifying the guardrail catches failures. LLM reasoning quality is non-deterministic and not unit-tested — the system produces correct output regardless via the safety net.
