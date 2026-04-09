from __future__ import annotations

from pathlib import Path

import pytest

from kargo_reco.benchmark_loader import BenchmarkRepository
from kargo_reco.trace import TraceManager, get_logger
from kargo_reco.workflow import WorkflowRunner


@pytest.fixture
def benchmark_csv(tmp_path: Path) -> Path:
    path = tmp_path / "benchmarks.csv"
    path.write_text(
        "\n".join(
            [
                "creative_name,click_through_rate,in_view_rate,vertical,minimum_budget",
                "Retail Rocket,0.42,0.79,Retail,20000",
                "Retail Spotlight,0.42,0.75,Retail,18000",
                "Retail Video+,0.39,0.82,Retail,25000",
                "Auto Motion,0.31,0.88,Automotive,30000",
                "Auto Highview,0.28,0.91,Automotive,22000",
                "Finance Focus,0.25,0.86,Finance,15000",
                "Finance Premium,0.29,0.89,Finance,28000",
            ]
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def workflow_runner(tmp_path: Path, benchmark_csv: Path) -> WorkflowRunner:
    traces_dir = tmp_path / "traces"
    logs_dir = tmp_path / "logs"
    uploads_dir = tmp_path / "uploads"
    trace_manager = TraceManager(traces_dir, get_logger(logs_dir))
    repository = BenchmarkRepository(benchmark_csv)
    return WorkflowRunner(
        repository=repository,
        trace_manager=trace_manager,
        uploads_dir=uploads_dir,
        model_name="gpt-4.1-mini",
        api_key=None,
        base_url=None,
        timeout_s=10,
    )
