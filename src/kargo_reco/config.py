from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    client_requests_path: Path = Path(
        os.getenv("CLIENT_REQUESTS_PATH", ROOT_DIR / "data" / "client_requests.json")
    )
    benchmark_csv_path: Path = Path(
        os.getenv("BENCHMARK_CSV_PATH", ROOT_DIR / "data" / "product_benchmarks.csv")
    )
    uploads_dir: Path = Path(os.getenv("UPLOADS_DIR", ROOT_DIR / "uploads"))
    traces_dir: Path = Path(os.getenv("TRACE_DIR", ROOT_DIR / "traces"))
    logs_dir: Path = Path(os.getenv("LOG_DIR", ROOT_DIR / "logs"))
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    openai_timeout_s: float = float(os.getenv("OPENAI_TIMEOUT_S", "20"))
    prompt_version: str = os.getenv("PROMPT_VERSION", "v1")


def get_settings() -> Settings:
    return Settings()
