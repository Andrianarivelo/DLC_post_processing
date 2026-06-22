"""External frame-time loading and alignment helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FrameTimeData:
    path: str
    times: np.ndarray
    frames: Optional[np.ndarray] = None
    time_column: str = ""
    frame_column: str = ""

    @property
    def summary(self) -> str:
        if self.times.size == 0:
            return "0 timestamps"
        frame_text = f", frames {int(self.frames[0])}-{int(self.frames[-1])}" if self.frames is not None and self.frames.size else ""
        return f"{self.times.size} timestamps{frame_text}, {self.times[0]:.3f}-{self.times[-1]:.3f} s"


_TIME_EXT = {".csv", ".txt", ".tsv"}
_TIME_NAME_TOKENS = (
    "frame_time",
    "frame-times",
    "frame_times",
    "aligned_frame",
    "timestamps",
    "timestamp",
    "time_s",
)
_PREFERRED_TIME_COLS = (
    "time_s",
    "time_sec",
    "timestamp_s",
    "timestamp",
    "timestamps",
    "time",
    "frame_time_s",
    "frame_time",
)
_PREFERRED_FRAME_COLS = (
    "frame",
    "frame_idx",
    "frame_index",
    "frame_number",
)


def is_frame_time_file(path: str | Path) -> bool:
    p = Path(path)
    if p.suffix.lower() not in _TIME_EXT:
        return False
    name = p.name.lower()
    return any(token in name for token in _TIME_NAME_TOKENS)


def load_frame_times(path: str | Path) -> FrameTimeData:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Time file not found: {p}")
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return _load_csv_frame_times(p)
    if suffix in {".txt", ".tsv"}:
        return _load_text_frame_times(p)
    raise ValueError(f"Unsupported time file type: {suffix}")


def align_times_to_dfs(time_data, animal_dfs: dict) -> Optional[np.ndarray]:
    """Return timestamps aligned to active tracking rows.

    If tracking rows preserve source-video frame numbers, those are used to
    select matching timestamps from the full-video frame-time table.
    """
    if time_data is None:
        return None
    if isinstance(time_data, FrameTimeData):
        times = np.asarray(time_data.times, dtype=np.float64).reshape(-1)
        time_frames = None if time_data.frames is None else np.asarray(time_data.frames, dtype=np.int64).reshape(-1)
    else:
        times = np.asarray(time_data, dtype=np.float64).reshape(-1)
        time_frames = None

    if times.size == 0:
        return None
    if not animal_dfs:
        return times

    first_df = next(iter(animal_dfs.values()), None)
    if first_df is None:
        return times
    n_rows = len(first_df)
    frame_numbers = getattr(first_df, "attrs", {}).get("frame_numbers")

    if frame_numbers is not None:
        frames = np.asarray(frame_numbers, dtype=np.int64).reshape(-1)
        if len(frames) == n_rows and len(frames) > 0:
            if time_frames is not None and len(time_frames) == len(times):
                series = pd.Series(times, index=time_frames)
                aligned = series.reindex(frames).to_numpy(dtype=np.float64)
                if np.isfinite(aligned).any():
                    return aligned
            max_frame = int(np.nanmax(frames))
            min_frame = int(np.nanmin(frames))
            if min_frame >= 0 and max_frame < len(times):
                return times[frames]

    if len(times) >= n_rows:
        return times[:n_rows]
    out = np.full(n_rows, np.nan, dtype=np.float64)
    out[: len(times)] = times
    return out


def _load_csv_frame_times(path: Path) -> FrameTimeData:
    # TimeSy-style files may append '# method,...' metadata rows. comment="#"
    # keeps the data table numeric instead of reading footer labels as values.
    df = pd.read_csv(path, comment="#")
    if df.empty:
        raise ValueError("No frame-time rows found.")

    time_col, time_values = _select_time_column(df)
    frame_col, frame_values = _select_frame_column(df, len(time_values))

    valid = np.isfinite(time_values)
    if frame_values is not None:
        valid &= np.isfinite(frame_values)
    time_values = time_values[valid]
    frames = None
    if frame_values is not None:
        frames = np.rint(frame_values[valid]).astype(np.int64)

    if time_values.size < 2:
        raise ValueError("Time file needs at least 2 numeric timestamps.")
    return FrameTimeData(
        path=str(path),
        times=time_values.astype(np.float64),
        frames=frames,
        time_column=time_col,
        frame_column=frame_col,
    )


def _load_text_frame_times(path: Path) -> FrameTimeData:
    rows: list[list[float]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.replace(",", " ").replace(";", " ").split()
            nums: list[float] = []
            for part in parts:
                try:
                    nums.append(float(part))
                except ValueError:
                    continue
            if nums:
                rows.append(nums)
    if not rows:
        raise ValueError("No numeric frame-time rows found.")

    max_cols = max(len(row) for row in rows)
    arr = np.full((len(rows), max_cols), np.nan, dtype=np.float64)
    for i, row in enumerate(rows):
        arr[i, : len(row)] = row

    if arr.shape[1] >= 2 and np.isfinite(arr[:, 1]).sum() >= 2:
        valid = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])
        frames = np.rint(arr[valid, 0]).astype(np.int64)
        times = arr[valid, 1]
    else:
        frames = None
        times = arr[:, 0]
        times = times[np.isfinite(times)]
    if times.size < 2:
        raise ValueError("Time file needs at least 2 numeric timestamps.")
    return FrameTimeData(path=str(path), times=times, frames=frames)


def _select_time_column(df: pd.DataFrame) -> tuple[str, np.ndarray]:
    lower_to_col = {str(col).strip().lower(): col for col in df.columns}
    best_name = ""
    best_values: Optional[np.ndarray] = None

    for name in _PREFERRED_TIME_COLS:
        col = lower_to_col.get(name)
        if col is None:
            continue
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
        if np.isfinite(values).sum() >= 2:
            return str(col), values

    best_count = -1
    for col in df.columns:
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
        count = int(np.isfinite(values).sum())
        if count > best_count:
            best_count = count
            best_name = str(col)
            best_values = values

    if best_values is None or best_count < 2:
        raise ValueError("No numeric timestamp column found.")
    return best_name, best_values


def _select_frame_column(df: pd.DataFrame, n_times: int) -> tuple[str, Optional[np.ndarray]]:
    lower_to_col = {str(col).strip().lower(): col for col in df.columns}
    for name in _PREFERRED_FRAME_COLS:
        col = lower_to_col.get(name)
        if col is None:
            continue
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
        if np.isfinite(values).sum() >= min(2, n_times):
            return str(col), values
    return "", None
