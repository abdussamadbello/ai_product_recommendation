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

## Dev Reload

For local development with auto-reload on file changes:

```bash
uv run gradio app.py
```

This uses Gradio's reload mode and watches the app for source changes.

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

## Local LLMs

The summary step can also target an OpenAI-compatible local endpoint such as Ollama.

Example configuration:

```bash
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_MODEL=llama3.2
OPENAI_API_KEY=
uv run kargo-reco
```

When `OPENAI_BASE_URL` is set, the app will use that endpoint and does not require a real OpenAI API key.
