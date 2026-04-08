from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kargo_reco.schemas import TraceArtifact


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
