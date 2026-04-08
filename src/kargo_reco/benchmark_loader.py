from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from kargo_reco.schemas import BenchmarkMetadata


REQUIRED_COLUMNS = [
    "creative_name",
    "click_through_rate",
    "in_view_rate",
    "vertical",
    "minimum_budget",
]
NUMERIC_COLUMNS = ["click_through_rate", "in_view_rate", "minimum_budget"]
STRING_COLUMNS = ["creative_name", "vertical"]


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize_strings(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in STRING_COLUMNS:
        if result[column].isna().any():
            raise ValueError(f"column {column} contains null values")
        result[column] = result[column].astype(str).str.strip()
        if (result[column] == "").any():
            raise ValueError(f"column {column} contains empty values")
    result["vertical_normalized"] = result["vertical"].str.lower()
    return result


def _normalize_numerics(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in NUMERIC_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        if result[column].isna().any():
            raise ValueError(f"column {column} contains non-numeric values")
    return result


def load_benchmark_dataframe(path: Path) -> tuple[pd.DataFrame, BenchmarkMetadata]:
    if not path.exists():
        raise FileNotFoundError(f"benchmark file not found: {path}")

    frame = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"benchmark file missing required columns: {sorted(missing)}")

    frame = frame[REQUIRED_COLUMNS].copy()
    frame = _normalize_strings(frame)
    frame = _normalize_numerics(frame)

    metadata = BenchmarkMetadata(
        file=str(path),
        file_hash=_file_hash(path),
        row_count=len(frame),
        schema_valid=True,
        loaded_at=datetime.now(timezone.utc).isoformat(),
    )
    return frame, metadata


@dataclass
class BenchmarkRepository:
    path: Path
    _frame: pd.DataFrame | None = None
    _metadata: BenchmarkMetadata | None = None

    def reload(self) -> BenchmarkMetadata:
        frame, metadata = load_benchmark_dataframe(self.path)
        self._frame = frame
        self._metadata = metadata
        return metadata

    def activate_path(self, path: Path) -> BenchmarkMetadata:
        frame, metadata = load_benchmark_dataframe(path)
        self.path = path
        self._frame = frame
        self._metadata = metadata
        return metadata

    def get_snapshot(self) -> tuple[pd.DataFrame, BenchmarkMetadata]:
        if self._frame is None or self._metadata is None:
            self.reload()
        assert self._frame is not None
        assert self._metadata is not None
        return self._frame.copy(), self._metadata
