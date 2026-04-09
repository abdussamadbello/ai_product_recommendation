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

    The message history alternates: AIMessage (with tool_calls) -> ToolMessage (result).
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
