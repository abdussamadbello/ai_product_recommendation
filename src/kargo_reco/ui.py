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
.result-card, .summary-card {
  background: rgba(255, 255, 255, 0.90);
  border: 1px solid rgba(15, 23, 42, 0.08);
  border-radius: 22px;
  box-shadow: 0 18px 48px rgba(15, 23, 42, 0.08);
}
.result-card, .summary-card {
  padding: 22px 24px;
}
.eyebrow {
  display: inline-block;
  font-size: 0.75rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #0369a1;
  margin-bottom: 8px;
}
.result-title {
  margin: 0 0 12px;
  font-size: 1.9rem;
  line-height: 1.1;
}
.status-chip {
  display: inline-block;
  margin-bottom: 12px;
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 0.8rem;
  font-weight: 600;
  background: #dcfce7;
  color: #166534;
}
.status-chip.no-match {
  background: #fef3c7;
  color: #92400e;
}
.metric-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
  margin-top: 18px;
}
.metric {
  padding: 14px 16px;
  border-radius: 16px;
  background: #f8fafc;
  border: 1px solid rgba(15, 23, 42, 0.06);
}
.metric-label {
  display: block;
  font-size: 0.8rem;
  color: #64748b;
  margin-bottom: 4px;
}
.metric-value {
  font-size: 1.1rem;
  font-weight: 600;
  color: #0f172a;
}
.summary-card h3 {
  margin: 0 0 10px;
  font-size: 1.1rem;
}
.summary-card p {
  margin: 0 0 16px;
  line-height: 1.55;
}
.summary-card ul {
  margin: 8px 0 0;
  padding-left: 20px;
}
.summary-card li {
  margin: 0 0 8px;
}
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
        "budget shortfall": "Budget shortfall",
        "vertical mismatch": "Vertical mismatch",
    }
    lowered = normalized.lower()
    if lowered in replacements:
        return replacements[lowered]
    return normalized[:1].upper() + normalized[1:]


def _clean_sentence(text: str) -> str:
    sentence = re.sub(r"\bminimum_budget\b", "minimum budget", text)
    sentence = re.sub(r"\bclick_through_rate\b", "click-through rate", sentence)
    sentence = re.sub(r"\bin_view_rate\b", "in-view rate", sentence)
    sentence = re.sub(r"\bpre_filter_count\b", "pre-filter count", sentence)
    sentence = re.sub(r"\bpost_vertical_count\b", "post-vertical count", sentence)
    sentence = re.sub(r"\bpost_budget_count\b", "post-budget count", sentence)
    sentence = sentence.replace("  ", " ").strip()
    return sentence[:1].upper() + sentence[1:] if sentence else sentence


def _render_list(items: list[str] | None) -> str:
    if not items:
        return ""
    rendered = "".join(f"<li>{_clean_sentence(item)}</li>" for item in items)
    return f"<ul>{rendered}</ul>"


def _render_recommendation_card(response: RecommendationResponse) -> str:
    if response.meta.status == "success":
        item = response.recommendations[0]
        return f"""
<div class="result-card">
  <div class="eyebrow">Recommended Product</div>
  <div class="status-chip">Eligible and selected</div>
  <h2 class="result-title">{item.creative_name}</h2>
  <div class="metric-grid">
    <div class="metric">
      <span class="metric-label">Vertical</span>
      <span class="metric-value">{item.vertical}</span>
    </div>
    <div class="metric">
      <span class="metric-label">Minimum budget</span>
      <span class="metric-value">{_format_currency(item.minimum_budget)}</span>
    </div>
    <div class="metric">
      <span class="metric-label">Click-through rate</span>
      <span class="metric-value">{_format_rate(item.click_through_rate)}</span>
    </div>
    <div class="metric">
      <span class="metric-label">In-view rate</span>
      <span class="metric-value">{_format_rate(item.in_view_rate)}</span>
    </div>
  </div>
</div>
"""
    if response.nearest_alternative:
        alt = response.nearest_alternative
        return f"""
<div class="result-card">
  <div class="eyebrow">Recommendation Status</div>
  <div class="status-chip no-match">No eligible match</div>
  <h2 class="result-title">{alt.creative_name}</h2>
  <p>This was the closest alternative, but it could not be selected.</p>
  <div class="metric-grid">
    <div class="metric">
      <span class="metric-label">Vertical</span>
      <span class="metric-value">{alt.vertical}</span>
    </div>
    <div class="metric">
      <span class="metric-label">Block reason</span>
      <span class="metric-value">{_humanize_token(alt.block_reason)}</span>
    </div>
    <div class="metric">
      <span class="metric-label">Minimum budget</span>
      <span class="metric-value">{_format_currency(alt.minimum_budget)}</span>
    </div>
    <div class="metric">
      <span class="metric-label">Budget shortfall</span>
      <span class="metric-value">{_format_currency(alt.budget_shortfall or 0)}</span>
    </div>
  </div>
</div>
"""
    return """
<div class="result-card">
  <div class="eyebrow">Recommendation Status</div>
  <div class="status-chip no-match">No eligible match</div>
  <h2 class="result-title">No alternative available</h2>
  <p>The benchmark data did not contain a usable fallback product for this request.</p>
</div>
"""


def _render_summary(response: RecommendationResponse) -> str:
    reasoning = _render_list(response.summary.structured_reasoning)
    alternatives = _render_list(response.summary.alternative_notes)
    alternatives_section = ""
    if alternatives:
        alternatives_section = f"<h3>Other considerations</h3>{alternatives}"
    return f"""
<div class="summary-card">
  <div class="eyebrow">Recommendation Summary</div>
  <p>{_clean_sentence(response.summary.short_text)}</p>
  <h3>Why this result</h3>
  {reasoning}
  {alternatives_section}
</div>
"""


def _trace_summary(response: RecommendationResponse) -> dict[str, Any]:
    return {
        "request_id": response.meta.request_id,
        "status": response.meta.status,
        "summary_source": response.meta.summary_source,
        "trace_path": response.meta.trace_path,
    }


def _load_trace_file(trace_path: str | None) -> dict[str, Any]:
    if not trace_path:
        return {}
    return json.loads(Path(trace_path).read_text(encoding="utf-8"))


def _ranking_dataframe(trace_payload: dict[str, Any]) -> pd.DataFrame:
    ranking = trace_payload.get("decision_trace", {}).get("ranking_table", [])
    if not ranking:
        return pd.DataFrame(columns=["creative_name", "rank"])
    return pd.DataFrame(ranking)


def _handle_payload(
    runner: WorkflowRunner, payload: dict[str, Any], source: str
) -> tuple[str, str, dict[str, Any], dict[str, Any], pd.DataFrame, str | None]:
    response = runner.run_from_payload(payload, source=source)
    trace_payload = _load_trace_file(response.meta.trace_path)
    return (
        _render_recommendation_card(response),
        _render_summary(response),
        response.model_dump(),
        trace_payload,
        _ranking_dataframe(trace_payload),
        response.meta.trace_path,
    )


def _build_request_preview(
    client_name: str,
    kpi: str,
    client_vertical: str,
    budget: float | int | None,
) -> dict[str, Any]:
    return {
        "client_name": client_name,
        "kpi": kpi,
        "client_vertical": client_vertical,
        "budget": budget,
    }


def _build_request_outputs(
    client_name: str,
    kpi: str,
    client_vertical: str,
    budget: float | int | None,
) -> tuple[dict[str, Any], str]:
    preview = _build_request_preview(client_name, kpi, client_vertical, budget)
    return preview, json.dumps(preview, indent=2)


def _normalize_ui_kpi(kpi: str) -> str:
    return kpi.strip().lower().replace(" ", "_")


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


def _sample_request_choices(samples: list[dict[str, Any]]) -> list[str]:
    return [_sample_request_label(sample) for sample in samples]


def _apply_sample_request(
    selected_label: str | None,
    labels_to_samples: dict[str, dict[str, Any]],
) -> tuple[str, str, str, float | int | None, dict[str, Any], str]:
    sample = labels_to_samples.get(selected_label or "")
    if not sample:
        preview = _build_request_preview("", "click_through_rate", "", None)
        return "", "click_through_rate", "", None, preview, json.dumps(preview, indent=2)

    budget = sample.get("budget", sample.get("minimum_budget"))
    preview = _build_request_preview(
        str(sample.get("client_name", "")),
        _normalize_ui_kpi(str(sample.get("kpi", "click_through_rate"))),
        str(sample.get("client_vertical", "")),
        budget,
    )
    return (
        preview["client_name"],
        preview["kpi"],
        preview["client_vertical"],
        preview["budget"],
        preview,
        json.dumps(preview, indent=2),
    )


def build_app(runner: WorkflowRunner | None = None) -> gr.Blocks:
    workflow_runner = runner or get_default_runner()
    settings = get_settings()
    sample_requests = _load_sample_requests(settings.client_requests_path)
    sample_labels = _sample_request_choices(sample_requests)
    labels_to_samples = {
        _sample_request_label(sample): sample for sample in sample_requests
    }
    default_sample = sample_requests[0] if sample_requests else {
        "client_name": "Acme Shoes",
        "kpi": "click_through_rate",
        "client_vertical": "Retail",
        "budget": 25000,
    }
    default_preview = _build_request_preview(
        str(default_sample.get("client_name", "")),
        _normalize_ui_kpi(str(default_sample.get("kpi", "click_through_rate"))),
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
                    label="Sample Client Requests",
                    choices=sample_labels,
                    value=sample_labels[0] if sample_labels else None,
                    allow_custom_value=False,
                    info="Load a sample request, then edit any field before submitting.",
                )
                with gr.Accordion("Benchmark Management", open=False, elem_classes=["compact-section"]):
                    benchmark_upload = gr.File(
                        label="Upload Benchmark CSV",
                        file_types=[".csv"],
                        type="filepath",
                    )
                    with gr.Row():
                        reload_button = gr.Button("Reload", variant="secondary")
                        upload_button = gr.Button("Use Upload", variant="secondary")
                    reload_status = gr.JSON(label="Benchmark Status")
                with gr.Tab("Form Input"):
                    client_name = gr.Textbox(
                        label="Client Name", value=default_preview["client_name"]
                    )
                    kpi = gr.Dropdown(
                        label="KPI",
                        choices=["click_through_rate", "in_view_rate"],
                        value=default_preview["kpi"],
                    )
                    client_vertical = gr.Textbox(
                        label="Client Vertical", value=default_preview["client_vertical"]
                    )
                    budget = gr.Number(label="Budget", value=default_preview["budget"])
                    request_preview = gr.JSON(
                        label="Request JSON Preview",
                        value=default_preview,
                    )
                    submit_form = gr.Button("Submit Request", variant="primary")
                with gr.Tab("JSON Input"):
                    json_input = gr.Code(
                        label="Request JSON",
                        language="json",
                        value=json.dumps(default_preview, indent=2),
                    )
                    submit_json = gr.Button("Submit JSON", variant="primary")
            with gr.Column(scale=1):
                recommendation_card = gr.Markdown(label="Recommendation")
                summary_block = gr.Markdown(label="Summary")
                response_json = gr.JSON(label="Response JSON")
                trace_download = gr.File(label="Download Trace JSON")

        with gr.Accordion("Expanded Debug Panel", open=False):
            trace_json = gr.JSON(label="Trace Artifact")
            ranking_table = gr.Dataframe(label="Ranking Table", row_count=5)

        def on_reload() -> dict[str, Any]:
            return workflow_runner.reload_benchmarks()

        def on_upload(uploaded_file_path: str | None) -> dict[str, Any]:
            if not uploaded_file_path:
                raise gr.Error("Select a benchmark CSV before uploading.")
            return workflow_runner.upload_benchmarks(uploaded_file_path)

        def on_form_submit(
            client_name_value: str,
            kpi_value: str,
            client_vertical_value: str,
            budget_value: float,
        ):
            payload = {
                "client_name": client_name_value,
                "kpi": kpi_value,
                "client_vertical": client_vertical_value,
                "budget": budget_value,
            }
            return _handle_payload(workflow_runner, payload, "ui_form")

        def on_json_submit(json_value: str):
            payload = json.loads(json_value)
            return _handle_payload(workflow_runner, payload, "ui_json")

        reload_button.click(on_reload, outputs=reload_status)
        upload_button.click(on_upload, inputs=[benchmark_upload], outputs=reload_status)
        sample_request.change(
            lambda label: _apply_sample_request(label, labels_to_samples),
            inputs=[sample_request],
            outputs=[client_name, kpi, client_vertical, budget, request_preview, json_input],
        )
        client_name.change(
            _build_request_outputs,
            inputs=[client_name, kpi, client_vertical, budget],
            outputs=[request_preview, json_input],
        )
        kpi.change(
            _build_request_outputs,
            inputs=[client_name, kpi, client_vertical, budget],
            outputs=[request_preview, json_input],
        )
        client_vertical.change(
            _build_request_outputs,
            inputs=[client_name, kpi, client_vertical, budget],
            outputs=[request_preview, json_input],
        )
        budget.change(
            _build_request_outputs,
            inputs=[client_name, kpi, client_vertical, budget],
            outputs=[request_preview, json_input],
        )
        submit_form.click(
            on_form_submit,
            inputs=[client_name, kpi, client_vertical, budget],
            outputs=[
                recommendation_card,
                summary_block,
                response_json,
                trace_json,
                ranking_table,
                trace_download,
            ],
        )
        submit_json.click(
            on_json_submit,
            inputs=[json_input],
            outputs=[
                recommendation_card,
                summary_block,
                response_json,
                trace_json,
                ranking_table,
                trace_download,
            ],
        )

    return app
