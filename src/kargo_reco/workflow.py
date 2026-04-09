from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.messages import BaseMessage, ToolMessage

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
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
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
