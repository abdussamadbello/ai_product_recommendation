from __future__ import annotations

from pathlib import Path

import pytest

from kargo_reco.benchmark_loader import BenchmarkRepository, load_benchmark_dataframe


def test_load_benchmark_dataframe_validates_and_normalizes(benchmark_csv: Path) -> None:
    frame, metadata = load_benchmark_dataframe(benchmark_csv)

    assert metadata.schema_valid is True
    assert metadata.row_count == 7
    assert "vertical_normalized" in frame.columns
    assert frame["minimum_budget"].dtype.kind in {"i", "f"}


def test_load_benchmark_dataframe_rejects_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "broken.csv"
    path.write_text("creative_name,vertical\nA,Retail\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        load_benchmark_dataframe(path)


def test_benchmark_repository_reload_returns_metadata(benchmark_csv: Path) -> None:
    repository = BenchmarkRepository(benchmark_csv)
    metadata = repository.reload()
    snapshot, snapshot_metadata = repository.get_snapshot()

    assert metadata.file == str(benchmark_csv)
    assert snapshot_metadata.file_hash == metadata.file_hash
    assert len(snapshot) == 7
