# Kargo Product Recommendation Engine

V1 prototype for deterministic benchmark-driven recommendation with AI-assisted explanation, trace artifacts, and a Gradio UI.

## Requirements

- Python 3.12+
- `uv`
- Optional: `OPENAI_API_KEY` for live explanation generation

## Install

```bash
uv sync --extra dev
```

## Run

```bash
uv run kargo-reco
```

The app uses `data/product_benchmarks.csv` by default. Override it with:

```bash
BENCHMARK_CSV_PATH=/path/to/file.csv uv run kargo-reco
```

## Test

```bash
uv run pytest
```

## Notes

- Recommendation selection is deterministic.
- AI is used only to generate the summary and falls back to a template if no model is configured or the call fails.
- Per-request trace artifacts are written under `traces/`.
