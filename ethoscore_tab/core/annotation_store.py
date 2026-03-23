"""In-memory behavior annotation store.

Stores per-behavior (start_frame, stop_frame) intervals.
Produces 1-D integer label arrays suitable for ML training.

Supports save/load as CSV (compatible with common ethoscore annotation format).
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class AnnotationStore:
    """Manage behavior labels as a collection of (start, stop) intervals."""

    def __init__(self) -> None:
        # {behavior_name: [(start, stop), …]}
        self._intervals: dict[str, list[tuple[int, int]]] = {}
        self._n_frames: int = 0
        self._pending_starts: dict[str, int] = {}  # behavior → open start

    # ── Setup ─────────────────────────────────────────────────────────────────

    def set_n_frames(self, n: int) -> None:
        self._n_frames = n

    def behaviors(self) -> list[str]:
        return list(self._intervals.keys())

    def add_behavior(self, name: str) -> None:
        if name not in self._intervals:
            self._intervals[name] = []
            logger.debug("Added behavior: %s", name)

    def remove_behavior(self, name: str) -> None:
        self._intervals.pop(name, None)
        self._pending_starts.pop(name, None)

    # ── Tagging ───────────────────────────────────────────────────────────────

    def tag_start(self, behavior: str, frame: int) -> None:
        self._pending_starts[behavior] = frame

    def tag_stop(self, behavior: str, frame: int) -> None:
        start = self._pending_starts.pop(behavior, None)
        if start is None:
            logger.warning("tag_stop called without prior tag_start for %r", behavior)
            return
        if start > frame:
            start, frame = frame, start  # allow reversed marking
        self.add_interval(behavior, start, frame)

    def add_interval(self, behavior: str, start: int, stop: int) -> None:
        if behavior not in self._intervals:
            self._intervals[behavior] = []
        self._intervals[behavior].append((start, stop))
        self._intervals[behavior].sort()
        logger.debug("Annotated %s: frames %d–%d", behavior, start, stop)

    def remove_last_interval(self, behavior: str) -> None:
        if self._intervals.get(behavior):
            removed = self._intervals[behavior].pop()
            logger.debug("Removed last interval for %s: %s", behavior, removed)

    def clear_behavior(self, behavior: str) -> None:
        self._intervals[behavior] = []

    def intervals(self, behavior: str) -> list[tuple[int, int]]:
        return list(self._intervals.get(behavior, []))

    # ── Label arrays ──────────────────────────────────────────────────────────

    def get_binary_array(self, behavior: str) -> np.ndarray:
        """Return a (N_frames,) boolean array for one behavior."""
        n = self._n_frames
        if n == 0:
            return np.array([], dtype=bool)
        arr = np.zeros(n, dtype=bool)
        for start, stop in self._intervals.get(behavior, []):
            s = max(0, start)
            e = min(n - 1, stop)
            arr[s:e+1] = True
        return arr

    def get_label_array(self) -> np.ndarray:
        """Return (N_frames,) integer array with behavior index (0 = background).

        If multiple behaviors overlap, last one wins.
        """
        n = self._n_frames
        arr = np.zeros(n, dtype=np.int32)
        for i, name in enumerate(self.behaviors(), start=1):
            mask = self.get_binary_array(name)
            arr[mask] = i
        return arr

    def get_multilabel_matrix(self) -> tuple[np.ndarray, list[str]]:
        """Return (N_frames, N_behaviors) binary matrix + behavior names."""
        names = self.behaviors()
        if not names or self._n_frames == 0:
            return np.zeros((max(self._n_frames, 0), 0), dtype=np.float32), []
        mat = np.stack([self.get_binary_array(b).astype(np.float32) for b in names], axis=1)
        return mat, names

    def labeled_frame_count(self) -> int:
        if self._n_frames == 0:
            return 0
        return int(self.get_label_array().astype(bool).sum())

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_csv(self, path: str | Path) -> None:
        path = Path(path)
        rows = []
        for behavior, intervals in self._intervals.items():
            for start, stop in intervals:
                rows.append({"behavior": behavior, "start": start, "stop": stop})
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["behavior", "start", "stop"])
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Annotations saved → %s", path)

    def load_csv(self, path: str | Path) -> None:
        path = Path(path)
        self._intervals.clear()
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                behavior = row["behavior"]
                start    = int(row["start"])
                stop     = int(row["stop"])
                if behavior not in self._intervals:
                    self._intervals[behavior] = []
                self._intervals[behavior].append((start, stop))
        for name in self._intervals:
            self._intervals[name].sort()
        logger.info("Annotations loaded ← %s (%d behaviors)", path, len(self._intervals))
