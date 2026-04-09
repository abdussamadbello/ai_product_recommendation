# Kargo Product Recommendation Engine — Slide Deck

## Slide 1: Title

**Kargo Product Recommendation Engine**
An Agentic AI Approach to Media Product Selection

---

## Slide 2: Problem Framing

**The Task:**
Media strategists manually match client requirements (vertical, KPI, budget) to Kargo's product catalog. This is time-consuming and subjective.

**The Challenge:**
- 50 products across 5 verticals with varying CTR, IVR, and budget thresholds
- Clients optimize for different KPIs (click-through rate vs. in-view rate)
- Budget constraints require trade-off reasoning, not just sorting
- Multi-product bundling adds combinatorial complexity

**What We Built:**
A recommendation engine where an LLM agent explores the catalog through tool calls, generates its own ranking strategy, and explains its reasoning — with deterministic guardrails as a safety net.

---

## Slide 3: Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Gradio Web UI                            │
│  ┌──────────┐  ┌──────────────────────────────────────────┐ │
│  │ Request   │  │ Results                                  │ │
│  │ Builder   │  │ • Recommendation Cards + Budget Bar      │ │
│  │           │  │ • Agent Summary (structured)             │ │
│  │ Form/JSON │  │ • Agent Reasoning Steps (expandable)     │ │
│  │ Inputs    │  │ • Response JSON + Debug Trace            │ │
│  └──────────┘  └──────────────────────────────────────────┘ │
└────────────────────────────┬────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  WorkflowRunner  │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
    ┌─────────▼──┐  ┌───────▼───────┐  ┌──▼──────────┐
    │ Pre-validate│  │  ReAct Agent  │  │Post-validate│
    │  (Pydantic) │  │  (LangGraph)  │  │ (Guardrails)│
    └─────────────┘  └───────┬───────┘  └──────┬──────┘
                             │                 │
                    ┌────────▼────────┐        │ fail
                    │   5 Tools       │   ┌────▼──────┐
                    │ • get_eligible  │   │Deterministic│
                    │ • score_products│   │ Fallback    │
                    │ • check_budget  │   └────────────┘
                    │ • get_details   │
                    │ • finalize      │
                    └─────────────────┘
```

---

## Slide 4: Data Flow — End to End

```
1. USER INPUT
   { client_name, kpi, client_vertical, budget }
          │
2. PRE-VALIDATION (Pydantic)
   • Normalize KPI: "CTR" → "click_through_rate"
   • Validate budget > 0, non-empty strings
   • Accept both "budget" and "minimum_budget" field names
          │
3. LOAD BENCHMARKS
   • product_benchmarks.csv → Pandas DataFrame (cached)
   • 50 products × 5 columns × 5 verticals
   • Schema validation + string/numeric normalization
          │
4. AGENT EXECUTION (ReAct Loop)
   • LLM receives client request as natural language
   • Calls tools iteratively: explore → rank → bundle → finalize
   • Generates weighted scoring formula for KPI optimization
   • Produces structured reasoning explaining trade-offs
          │
5. POST-VALIDATION (Guardrails)
   • Every product exists in CSV? (catch hallucinations)
   • Every product matches requested vertical?
   • Total budget ≤ client budget?
   • Pass → agent's output | Fail → deterministic fallback
          │
6. RESPONSE
   { request, recommendations[], summary, agent_trace[], meta }
          │
7. UI RENDERING
   • Product cards with metrics
   • Budget utilization bar
   • Expandable reasoning steps
   • Full JSON + trace artifact
```

---

## Slide 5: The Agent — ReAct Loop in Detail

**Framework:** LangGraph `create_react_agent`

**How ReAct Works:**
```
LLM thinks → picks a tool → tool executes → LLM observes result → thinks again → ...
```

**Typical Execution (4 tool calls, ~15 seconds):**

| Step | Agent Thought | Tool Call | Result |
|------|--------------|-----------|--------|
| 1 | "I need to find Retail products within $25k" | `get_eligible_products(vertical="Retail", budget=25000)` | 7 products found |
| 2 | "I'll weight CTR at 85% since that's the client's KPI" | `score_products(weights={"ctr": 0.85, "ivr": 0.15}, budget=25000)` | Ranked list with scores |
| 3 | "Aurora Story is top. Budget leaves $7k — anything else fit?" | `check_budget_remaining(budget=25000, selected=["Aurora Story"])` | No affordable additions |
| 4 | "Aurora Story is the best pick. Finalizing." | `finalize_recommendation(products=["Aurora Story"], reasoning="...")` | Structured output |

**Key Insight:** The agent generates the scoring formula (Step 2). A deterministic system would hardcode `sort_values(by="ctr")`. The agent reasons about *how much* to weight each metric based on the client context.

---

## Slide 6: Tool Design

### 5 Tools — Each Independently Callable

| Tool | Input | Output | Design Choice |
|------|-------|--------|--------------|
| `get_eligible_products` | vertical, budget | Filtered product list + counts | Combines vertical + budget filtering in one call to reduce round-trips |
| `score_products` | weights, budget | Scored + ranked products | Agent generates its own formula — the core differentiator |
| `check_budget_remaining` | budget, selected[] | Remaining $, affordable products | Enables informed bundling decisions |
| `get_product_details` | product_name, vertical | Single product record | Optional deep inspection |
| `finalize_recommendation` | products[], reasoning | Structured output | Forces the agent to commit and explain |

### Why This Design?

**Problem:** LLMs make one API call per tool use. More tools = more latency.

**Solution:** Consolidated tools that do more per call.
- `get_eligible_products` replaces separate `filter_by_vertical` + `filter_by_budget` (2 calls → 1)
- `score_products` accepts optional `budget` param to auto-filter (eliminates a separate filter step)

**Result:** 4 tool calls typical vs. 9 in the original design (~15s vs. ~40s)

### The `score_products` Tool — Agent-Generated Strategy

```
Deterministic approach:    df.sort_values(by="click_through_rate")
Agentic approach:          agent decides: {"click_through_rate": 0.85, "in_view_rate": 0.15}
```

The agent can reason: *"Finance clients care about viewability for compliance, so I'll weight IVR higher even though CTR was requested."* This is strategy generation — not possible with a hardcoded sort.

---

## Slide 7: Guardrails — The Trust Sandwich

```
    ┌─────────────────────────┐
    │   PRE-VALIDATION        │   Deterministic
    │   Pydantic schemas      │   (before agent)
    └───────────┬─────────────┘
                │
    ┌───────────▼─────────────┐
    │   AGENT EXECUTION       │   Agentic
    │   Full autonomy         │   (LLM decides)
    │   within the ReAct loop │
    └───────────┬─────────────┘
                │
    ┌───────────▼─────────────┐
    │   POST-VALIDATION       │   Deterministic
    │   Guardrail checks      │   (after agent)
    └───────────┬─────────────┘
                │
         ┌──────┴──────┐
         │             │
    ┌────▼───┐   ┌─────▼──────┐
    │  PASS  │   │   FAIL     │
    │ Agent's│   │ Fallback to│
    │ output │   │ deterministic│
    └────────┘   └────────────┘
```

### What the Guardrail Checks

| Check | Why It Exists | Example Failure |
|-------|--------------|-----------------|
| Product exists in CSV | LLMs hallucinate names | Agent recommends "Super Banner Pro" (doesn't exist) |
| Vertical matches request | LLMs ignore constraints | Agent picks a Finance product for a Retail client |
| Total budget ≤ limit | LLMs can't do math reliably | Agent bundles $70k + $60k on an $80k budget |

### Fallback Behavior

When the guardrail catches a violation:
1. Log the violation to the trace artifact (debuggable)
2. Run deterministic selection: highest KPI → within vertical → within budget
3. Mark response as `guardrail_fallback` so the UI shows it
4. Agent's reasoning steps are still visible (you can see what went wrong)

---

## Slide 8: The Scoring Algorithm

### Deterministic Approach (What We Replaced)

```python
# Simple sort — no reasoning, no trade-offs
df.sort_values(by=kpi, ascending=False).iloc[0]
```

### Agentic Approach (What We Built)

```python
# Agent generates weights based on client context
weights = {"click_through_rate": 0.85, "in_view_rate": 0.15}

# Weighted score for each product
score = sum(product[metric] * weight for metric, weight in weights.items())

# Products ranked by composite score
scored_products = sorted(products, key=lambda p: p["score"], reverse=True)
```

### Multi-Product Bundling Algorithm

```
1. Rank all eligible products by weighted score
2. Select #1 product
3. remaining_budget = client_budget - product_1.minimum_budget
4. For each remaining product (in score order):
     if product.minimum_budget <= remaining_budget:
       add to bundle
       remaining_budget -= product.minimum_budget
5. Return bundle
```

The agent decides when to stop bundling. It might reason: *"I have $7k remaining but the cheapest product is $8k. No additional product fits."* Or: *"Budget allows one more product, but its KPI is significantly lower — not worth adding."*

---

## Slide 9: UI Design

### Layout

| Left Column | Right Column |
|------------|-------------|
| Sample request dropdown | Recommendation cards (1 per product) |
| Benchmark management (upload/reload) | Budget utilization bar |
| Form input tab (4 fields) | Agent summary (structured paragraphs) |
| JSON input tab (raw editor) | Agent reasoning steps (expandable) |
| | Response JSON |
| | Debug panel (full trace) |

### Key UI Elements

1. **Recommendation Cards** — One card per selected product showing all metrics
2. **Budget Utilization Bar** — Visual progress bar: `$used / $total (XX%)`
3. **Agent Reasoning Steps** — Each tool call expandable to show:
   - Agent's thought before the call
   - Tool input parameters
   - Tool output (truncated if large)
4. **Status Chip** — Green (success) / Amber (no match) / Red (guardrail fallback)

---

## Slide 10: Design Choices and Trade-offs

### Choice 1: LangGraph ReAct vs. Linear Pipeline

| | ReAct Agent | Linear Pipeline |
|---|------------|----------------|
| **Flexibility** | Agent decides tool order | Fixed step sequence |
| **Reasoning** | LLM explains trade-offs | Template-based summary |
| **Latency** | ~15s (4 LLM round-trips) | ~3s (1 LLM call for summary) |
| **Correctness** | Needs guardrails | Deterministic |
| **Interview signal** | "I built an agent" | "I wrote a script" |

**Why we chose ReAct:** The product selection problem has genuine decision points (weighting KPIs, bundling vs. single product, budget allocation). An agent can reason about these trade-offs. A linear pipeline just sorts.

### Choice 2: Fine-Grained Tools vs. Single "Recommend" Tool

We started with 7 small tools and consolidated to 5 after seeing 9-step traces (40s latency).

**Trade-off:** More granular tools = richer reasoning trace but more latency. We found the sweet spot at 5 tools / ~4 steps.

### Choice 3: Guardrails as Safety Net

**Alternative:** Let the agent handle everything, trust its output.
**Why we didn't:** LLMs hallucinate product names and can't reliably do budget arithmetic. The guardrail costs ~0ms (pure Python) and catches real failures.

### Choice 4: score_products — Agent-Generated Formulas

**Alternative:** Just `sort_by_kpi` on a single column.
**Why we didn't:** The spec says "optimizing the requested KPI" — but a media strategist also considers secondary metrics. Letting the agent set weights demonstrates genuine reasoning.

---

## Slide 11: Agentic vs. Programmatic — Pros and Cons

### Pros of Agentic Approach

| Advantage | Example |
|-----------|---------|
| **Adaptive strategy** | Agent weights KPIs differently based on vertical context |
| **Natural language reasoning** | "I chose Aurora because..." vs. `selected_by=sort_order[0]` |
| **Handles ambiguity** | Agent can reason about bundling trade-offs |
| **Extensible** | Add new tools (e.g., "check_competitor_pricing") without rewriting logic |
| **Transparent** | Every decision is traceable via tool call history |

### Cons of Agentic Approach

| Disadvantage | Mitigation |
|-------------|-----------|
| **Latency** (~15s vs. ~1s) | Consolidated tools, could add caching |
| **Non-deterministic** | Guardrail fallback ensures correct output |
| **Cost** ($0.01-0.05 per request) | Use cheaper models (gpt-5.1-mini), limit steps |
| **Hallucination risk** | Post-validation catches all constraint violations |
| **Debugging complexity** | Full trace artifact with every tool call logged |

### When to Use Each

| Use Agentic When | Use Programmatic When |
|-----------------|---------------------|
| Trade-offs matter | Rules are exhaustive |
| Reasoning needs to be explained | Speed is critical |
| Requirements may evolve | Correctness must be 100% |
| Users need transparency | Output is binary (yes/no) |

---

## Slide 12: Testing Strategy

### Three Layers

```
┌─────────────────────────────────┐
│  Layer 3: Agent Integration     │  Mock LLM, verify full pipeline
│  (test_workflow.py — 5 tests)   │  Tests: success, hallucination
│                                 │  fallback, no-match, upload
├─────────────────────────────────┤
│  Layer 2: Guardrail Tests       │  No LLM needed
│  (test_guardrails.py — 8 tests) │  Tests: valid output, hallucination
│                                 │  vertical mismatch, budget overrun
├─────────────────────────────────┤
│  Layer 1: Tool Unit Tests       │  No LLM needed
│  (test_tools.py — 23 tests)     │  Tests: filter, sort, score,
│                                 │  budget math, edge cases
└─────────────────────────────────┘
```

**Principle:** Deterministic parts tested deterministically. Agentic parts tested by verifying the safety net catches failures. 59 tests total, all run without an API key.

---

## Slide 13: Potential Improvements

### Short-Term
- **Streaming** — Stream agent steps to the UI in real-time instead of waiting for completion
- **Caching** — Cache agent results for identical requests (same vertical + KPI + budget range)
- **Parallel tool calls** — LangGraph supports parallel tool execution for independent calls

### Medium-Term
- **Multi-turn conversation** — Let the user refine: "What if I increase the budget to $50k?"
- **Historical performance data** — Feed actual campaign results back to improve recommendations
- **A/B testing** — Compare agent recommendations vs. deterministic for quality measurement

### Long-Term
- **Fine-tuned model** — Train on historical media planner decisions for domain-specific reasoning
- **Multi-agent** — Specialist agents for different verticals or KPI types
- **Real-time data** — Connect to live campaign performance APIs instead of static CSV

---

## Slide 14: Live Demo

### Demo Flow

1. **Sample request** — Select "Acme Shoes" (Retail, CTR, $25k)
   - Show: product card, reasoning steps, budget bar
2. **Large budget** — Enter $250k Retail IVR request
   - Show: multi-product bundle, budget utilization
3. **No match** — Enter "Travel" with $1k budget
   - Show: no-match status, agent reasoning about why
4. **JSON input** — Paste raw JSON request
   - Show: response JSON structure
5. **Guardrail demo** — (if time allows) Show trace where agent hallucinated and fallback activated

---

## Slide 15: Summary

**What we built:**
- A genuinely agentic recommendation engine where the LLM drives product selection through tool calls
- Not a "deterministic pipeline with an LLM narrator" — the agent reasons about trade-offs and generates scoring strategies

**Key design decisions:**
- Guardrailed agent: full autonomy inside the loop, validated output outside
- Agent-generated scoring formulas via `score_products` — the LLM creates strategy, not just follows rules
- Consolidated 5-tool design balancing reasoning depth vs. latency

**Tech stack:**
- LangGraph (create_react_agent) + LangChain (ChatOpenAI) + Gradio + Pydantic + Pandas
- 59 tests, no API key needed for CI
- Configurable: OpenAI API or local Ollama

---

## Appendix A: Module Map

```
src/kargo_reco/
│
├── agent.py          ← ReAct agent + system prompt
│   Uses: tools.py, langgraph, langchain_openai
│
├── tools.py          ← 5 pure tool functions
│   Uses: pandas (DataFrame operations only)
│
├── guardrails.py     ← Post-validation + fallback
│   Uses: schemas.py, pandas
│
├── workflow.py       ← Orchestrator
│   Uses: agent.py, guardrails.py, trace.py, schemas.py
│
├── schemas.py        ← All Pydantic models
│   Uses: pydantic
│
├── trace.py          ← AgentStep extraction + logging
│   Uses: langchain_core.messages, schemas.py
│
├── benchmark_loader.py ← CSV loading + validation
│   Uses: pandas
│
├── config.py         ← Environment settings
│   Uses: dotenv
│
└── ui.py             ← Gradio interface
    Uses: workflow.py, config.py, schemas.py
```

## Appendix B: Sample Trace Artifact

```json
{
  "request_id": "5a3ecaff-d0d9-48cd-a6f3-fc68c8c0e108",
  "timestamp": "2026-04-09T14:23:01.442Z",
  "source": "ui_form",
  "normalized_request": {
    "client_name": "Acme Shoes",
    "kpi": "click_through_rate",
    "client_vertical": "Retail",
    "budget": 25000
  },
  "agent_trace": [
    {
      "step_number": 1,
      "tool_name": "get_eligible_products",
      "tool_input": { "vertical": "Retail", "budget": 25000 },
      "tool_output": { "total_in_vertical": 10, "within_budget": 7, "products": ["..."] },
      "agent_reasoning": "I need to find all Retail products within the $25k budget."
    },
    {
      "step_number": 2,
      "tool_name": "score_products",
      "tool_input": { "weights": { "click_through_rate": 0.85, "in_view_rate": 0.15 }, "budget": 25000 },
      "tool_output": [{ "creative_name": "Aurora Story", "score": 0.4685 }, "..."],
      "agent_reasoning": "CTR is the primary KPI. I'll weight it at 85%."
    },
    {
      "step_number": 3,
      "tool_name": "check_budget_remaining",
      "tool_input": { "budget": 25000, "selected": ["Aurora Story"] },
      "tool_output": { "remaining": 7000, "can_add_more": false },
      "agent_reasoning": "Aurora Story costs $18k. Checking if I can add another product."
    },
    {
      "step_number": 4,
      "tool_name": "finalize_recommendation",
      "tool_input": { "products": ["Aurora Story"], "reasoning": "..." },
      "agent_reasoning": null
    }
  ],
  "guardrail_violations": [],
  "step_latencies_ms": { "run_agent": 14523 },
  "meta": { "status": "success", "source": "llm", "model": "gpt-5.1-mini" }
}
```
