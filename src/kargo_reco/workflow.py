from __future__ import annotations

import shutil
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from kargo_reco.benchmark_loader import BenchmarkRepository
from kargo_reco.config import get_settings
from kargo_reco.llm import SummaryGenerator
from kargo_reco.reasoning import build_reasoning_facts
from kargo_reco.recommender import RecommendationComputation, compute_recommendation
from kargo_reco.schemas import (
    RecommendationRequest,
    RecommendationResponse,
    ResponseMeta,
    SummaryBlock,
    TraceArtifact,
)
from kargo_reco.tracing import TraceManager, get_logger


class WorkflowState(TypedDict, total=False):
    source: str
    raw_request: dict[str, Any]
    request: RecommendationRequest
    benchmark_df: pd.DataFrame
    benchmark_metadata: dict[str, Any]
    computation: RecommendationComputation
    reasoning_facts: dict[str, Any]
    summary: SummaryBlock
    llm_trace: dict[str, Any]
    trace: TraceArtifact
    response: RecommendationResponse
    trace_path: str
    next_step: str


class WorkflowRunner:
    def __init__(
        self,
        repository: BenchmarkRepository,
        summary_generator: SummaryGenerator,
        trace_manager: TraceManager,
        uploads_dir: Path,
    ) -> None:
        self.repository = repository
        self.summary_generator = summary_generator
        self.trace_manager = trace_manager
        self.uploads_dir = uploads_dir
        self.graph = self._build_graph()

    def _timed_node(self, state: WorkflowState, name: str, fn):
        started = time.perf_counter()
        updates = fn(state)
        trace = state["trace"]
        trace.step_latencies_ms[name] = int((time.perf_counter() - started) * 1000)
        return updates

    def _build_graph(self):
        builder = StateGraph(WorkflowState)
        builder.add_node("validate_input", self._validate_input)
        builder.add_node("normalize_request", self._normalize_request)
        builder.add_node("load_benchmarks", self._load_benchmarks)
        builder.add_node("filter_candidates", self._filter_candidates)
        builder.add_node("rank_candidates", self._rank_candidates)
        builder.add_node("build_reasoning_facts", self._build_reasoning)
        builder.add_node("generate_llm_summary", self._generate_llm_summary)
        builder.add_node("generate_fallback_summary", self._generate_fallback_summary)
        builder.add_node("persist_trace", self._persist_trace)
        builder.add_node("return_response", self._return_response)

        builder.add_edge(START, "validate_input")
        builder.add_edge("validate_input", "normalize_request")
        builder.add_edge("normalize_request", "load_benchmarks")
        builder.add_edge("load_benchmarks", "filter_candidates")
        builder.add_edge("filter_candidates", "rank_candidates")
        builder.add_edge("rank_candidates", "build_reasoning_facts")
        builder.add_edge("build_reasoning_facts", "generate_llm_summary")
        builder.add_conditional_edges(
            "generate_llm_summary",
            lambda state: state["next_step"],
            {
                "persist_trace": "persist_trace",
                "generate_fallback_summary": "generate_fallback_summary",
            },
        )
        builder.add_edge("generate_fallback_summary", "persist_trace")
        builder.add_edge("persist_trace", "return_response")
        builder.add_edge("return_response", END)
        return builder.compile()

    def _validate_input(self, state: WorkflowState):
        return self._timed_node(
            state,
            "validate_input",
            lambda inner_state: {
                "raw_request": inner_state["raw_request"],
            },
        )

    def _normalize_request(self, state: WorkflowState):
        def normalize(_: WorkflowState):
            request = RecommendationRequest.model_validate(state["raw_request"])
            state["trace"].normalized_request = request.model_dump()
            return {"request": request}

        return self._timed_node(state, "normalize_request", normalize)

    def _load_benchmarks(self, state: WorkflowState):
        def load(_: WorkflowState):
            frame, metadata = self.repository.get_snapshot()
            state["trace"].benchmark_metadata = metadata
            self.trace_manager.log_event(
                "benchmark_loaded",
                {
                    "request_id": state["trace"].request_id,
                    "row_count": metadata.row_count,
                    "file": metadata.file,
                },
            )
            return {"benchmark_df": frame, "benchmark_metadata": metadata.model_dump()}

        return self._timed_node(state, "load_benchmarks", load)

    def _filter_candidates(self, state: WorkflowState):
        return self._timed_node(state, "filter_candidates", lambda _: {})

    def _rank_candidates(self, state: WorkflowState):
        def rank(_: WorkflowState):
            computation = compute_recommendation(state["benchmark_df"], state["request"])
            ranking_table = (
                computation.ranking_table.to_dict(orient="records")
                if not computation.ranking_table.empty
                else []
            )
            state["trace"].decision_trace.pre_filter_count = computation.pre_filter_count
            state["trace"].decision_trace.post_vertical_count = computation.post_vertical_count
            state["trace"].decision_trace.post_budget_count = computation.post_budget_count
            state["trace"].decision_trace.ranking_table = ranking_table
            state["trace"].decision_trace.tie_break_notes = computation.tie_break_notes
            state["trace"].decision_trace.no_match_reason = computation.no_match_reason
            if computation.selected:
                state["trace"].decision_trace.selected_product = computation.selected.model_dump()
            if computation.nearest_alternative:
                state["trace"].decision_trace.nearest_alternative = (
                    computation.nearest_alternative.model_dump()
                )
            return {"computation": computation}

        return self._timed_node(state, "rank_candidates", rank)

    def _build_reasoning(self, state: WorkflowState):
        def build(_: WorkflowState):
            facts = build_reasoning_facts(state["request"], state["computation"])
            state["trace"].decision_trace.excluded_alternatives = facts["eligible_alternatives"]
            return {"reasoning_facts": facts}

        return self._timed_node(state, "build_reasoning_facts", build)

    def _generate_llm_summary(self, state: WorkflowState):
        def generate(_: WorkflowState):
            summary, llm_trace = self.summary_generator.generate(state["reasoning_facts"])
            state["trace"].llm_trace = llm_trace
            next_step = "persist_trace"
            if summary.summary_source == "fallback_template":
                next_step = "generate_fallback_summary"
            return {
                "summary": summary,
                "llm_trace": llm_trace.model_dump(),
                "next_step": next_step,
            }

        return self._timed_node(state, "generate_llm_summary", generate)

    def _generate_fallback_summary(self, state: WorkflowState):
        def fallback(_: WorkflowState):
            if state["summary"].summary_source == "fallback_template":
                return {"summary": state["summary"]}
            return {}

        return self._timed_node(state, "generate_fallback_summary", fallback)

    def _persist_trace(self, state: WorkflowState):
        def persist(_: WorkflowState):
            trace_path = str(self.trace_manager.build_path(state["trace"].request_id))
            response = self._assemble_response(state, trace_path=trace_path)
            state["trace"].final_response = response.model_dump()
            self.trace_manager.write(state["trace"])
            return {"trace_path": trace_path}

        return self._timed_node(state, "persist_trace", persist)

    def _return_response(self, state: WorkflowState):
        def finalize(_: WorkflowState):
            response = self._assemble_response(state, trace_path=state["trace_path"])
            state["trace"].final_response = response.model_dump()
            return {"response": response}

        return self._timed_node(state, "return_response", finalize)

    def _assemble_response(
        self, state: WorkflowState, trace_path: str | None
    ) -> RecommendationResponse:
        computation = state["computation"]
        summary = state["summary"]
        status = "success" if computation.selected else "no_match"
        recommendations = [computation.selected] if computation.selected else []
        return RecommendationResponse(
            request=state["request"],
            recommendations=recommendations,
            summary=summary,
            meta=ResponseMeta(
                status=status,
                request_id=state["trace"].request_id,
                summary_source=summary.summary_source,
                trace_path=trace_path,
            ),
            nearest_alternative=computation.nearest_alternative,
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

    def run_from_payload(
        self, payload: dict[str, Any], *, source: str = "api"
    ) -> RecommendationResponse:
        trace = TraceArtifact(source=source, raw_request=payload)
        self.trace_manager.log_event(
            "request_received",
            {"request_id": trace.request_id, "raw_request": payload, "source": source},
        )
        try:
            result = self.graph.invoke(
                {"source": source, "raw_request": payload, "trace": trace}
            )
            return result["response"]
        except Exception as exc:
            trace.errors.append(str(exc))
            trace.step_latencies_ms.setdefault("error", 0)
            trace.final_response = {"error": str(exc)}
            path = self.trace_manager.write(trace)
            raise RuntimeError(f"Recommendation workflow failed. Trace: {path}") from exc

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
    summary_generator = SummaryGenerator(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        prompt_version=settings.prompt_version,
        timeout_s=settings.openai_timeout_s,
    )
    return WorkflowRunner(
        repository,
        summary_generator,
        trace_manager,
        uploads_dir=settings.uploads_dir,
    )


def run_recommendation(
    request: RecommendationRequest,
    *,
    raw_request: dict[str, Any] | None = None,
    source: str = "api",
) -> RecommendationResponse:
    return get_default_runner().run_recommendation(
        request, raw_request=raw_request, source=source
    )
