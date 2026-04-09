# Kargo Product Recommendation Engine

An agentic AI product recommendation engine that uses a LangGraph ReAct agent to explore a product catalog, reason about trade-offs, and build optimal product bundles for media strategists.

## Architecture

```
Request JSON
     │
     ├─ Pre-validation (Pydantic)
     │
     ├─ Load Benchmarks (CSV → DataFrame)
     │
     ├─ LLM Agent (LangGraph ReAct loop)
     │    ├─ get_eligible_products  → vertical + budget filtering
     │    ├─ score_products         → weighted KPI ranking
     │    ├─ check_budget_remaining → bundling decisions
     │    ├─ get_product_details    → inspect candidates
     │    └─ finalize_recommendation → commit selection + reasoning
     │
     ├─ Post-validation Guardrail
     │    ├─ pass → agent's selection
     │    └─ fail → deterministic fallback
     │
     └─ JSON Response + Trace Artifact
```

The agent drives all product selection decisions through tool calls. A post-validation guardrail catches hallucinations, vertical mismatches, and budget overruns — falling back to deterministic logic when needed.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- An LLM API key (OpenAI) **or** a local Ollama instance

## Quick Start

### 1. Install dependencies

```bash
uv sync --extra dev
```

### 2. Configure LLM

Create a `.env` file in the project root:

**Option A: OpenAI API**
```bash
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-5.1-mini
```

**Option B: Ollama (local LLM)**
```bash
# Start Ollama first: ollama serve
# Pull a model: ollama pull llama3.2
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_MODEL=llama3.2
```

When `OPENAI_BASE_URL` is set, the app uses that endpoint and does not require an OpenAI API key.

### 3. Run the app

```bash
uv run python app.py
```

Or using the package entry point:

```bash
uv run kargo-reco
```

The Gradio UI opens at `http://localhost:7860`.

### 4. Dev reload

For auto-reload on file changes:

```bash
uv run gradio app.py
```

## Usage

1. **Select a sample request** from the dropdown, or fill in the form manually:
   - Client Name, KPI (click_through_rate or in_view_rate), Client Vertical, Budget
2. **Click "Submit Request"** — the agent explores the product catalog and returns recommendations
3. **View results:**
   - **Recommendation cards** — selected products with metrics and budget utilization bar
   - **Agent summary** — structured reasoning explaining the selection
   - **Agent reasoning steps** — expandable trace of every tool call the agent made
   - **Response JSON** — full API response
   - **Debug panel** — complete trace artifact

## How the Agent Works

The ReAct agent has 5 tools and follows this strategy:

| Step | Tool | What it does |
|------|------|-------------|
| 1 | `get_eligible_products` | Finds all products matching the vertical within budget |
| 2 | `score_products` | Applies a weighted formula (e.g., 85% CTR + 15% IVR) to rank products |
| 3 | `check_budget_remaining` | Checks if budget allows adding another product |
| 4 | `get_product_details` | (Optional) Inspects a specific candidate |
| 5 | `finalize_recommendation` | Commits the selection with detailed reasoning |

The agent generates its own scoring formula via `score_products` — this is the key differentiator from a deterministic sort. It weights KPIs based on the client context rather than using a hardcoded `sort_values()`.

## Guardrails

After the agent finishes, a post-validation step checks:

| Check | Catches |
|-------|---------|
| Product exists in CSV | Hallucinated product names |
| Vertical matches request | Wrong-vertical selections |
| Total budget ≤ client budget | Overspending |

If any check fails, the system falls back to deterministic selection (highest KPI within vertical and budget) and marks the response as `guardrail_fallback`.

## Configuration

All settings are configurable via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | OpenAI API key |
| `OPENAI_BASE_URL` | — | Override for Ollama or compatible endpoint |
| `OPENAI_MODEL` | `gpt-5.1-mini` | Model name |
| `OPENAI_TIMEOUT_S` | `30` | LLM request timeout (seconds) |
| `BENCHMARK_CSV_PATH` | `data/product_benchmarks.csv` | Path to benchmark data |
| `CLIENT_REQUESTS_PATH` | `data/client_requests.json` | Path to sample requests |
| `TRACE_DIR` | `traces/` | Where trace artifacts are written |

## Project Structure

```
src/kargo_reco/
├── agent.py            # ReAct agent construction + system prompt
├── tools.py            # 5 tool functions (pure DataFrame operations)
├── guardrails.py       # Post-validation + deterministic fallback
├── workflow.py         # Orchestration: validate → agent → guardrail → respond
├── schemas.py          # Pydantic models (request, response, agent step, trace)
├── trace.py            # AgentStep extraction from LangGraph messages
├── benchmark_loader.py # CSV loading + validation + caching
├── config.py           # Environment-based settings
├── ui.py               # Gradio web interface
└── main.py             # Entry point

data/
├── product_benchmarks.csv  # 50 products × 5 verticals
└── client_requests.json    # 5 sample requests

tests/
├── test_tools.py           # Tool unit tests (23 tests)
├── test_guardrails.py      # Validation + fallback tests (8 tests)
├── test_agent.py           # Agent construction tests (3 tests)
├── test_schemas.py         # Schema validation tests (11 tests)
├── test_workflow.py        # Integration tests with mocked LLM (5 tests)
├── test_ui.py              # UI rendering tests (6 tests)
└── test_benchmark_loader.py # CSV loading tests (3 tests)
```

## Testing

```bash
uv run pytest              # run all 59 tests
uv run pytest -v           # verbose output
uv run pytest tests/test_tools.py  # run specific test file
```

Tests use mocked LLM responses — no API key needed for CI.

## Sample Requests

| Client | Vertical | KPI | Budget |
|--------|----------|-----|--------|
| Acme Shoes | Retail | CTR | $25,000 |
| Stellar Bank | Finance | IVR | $80,000 |
| Wanderlust Air | Travel | CTR | $60,000 |
| Burger Bazaar | QSR | CTR | $15,000 |
| CineVerse | Entertainment | IVR | $40,000 |
