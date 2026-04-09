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
