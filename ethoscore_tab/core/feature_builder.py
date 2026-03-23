"""Build a 2-D feature matrix (Frames × Features) from DLC data + kinematics.

Combines:
  - Raw DLC keypoint coordinates (x, y) per animal/bodypart
  - Kinematic columns appended by kinematics.py
  - Social behaviour boolean arrays from social_behaviors.py
  - Optional timestamp column

Output: normalised NumPy array + matching feature name list.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """Assemble and normalise a feature matrix from various sources."""

    def __init__(self) -> None:
        self._animal_dfs: dict[str, pd.DataFrame] = {}
        self._behavior_arrays: dict[str, np.ndarray] = {}
        self._selected_columns: list[str] = []
        self._feature_names: list[str] = []
        self._matrix: Optional[np.ndarray] = None
        self._n_frames: int = 0

    # ── Source data setters ───────────────────────────────────────────────────

    def set_animal_dfs(self, dfs: dict[str, pd.DataFrame]) -> None:
        self._animal_dfs = dfs
        if dfs:
            self._n_frames = max(len(df) for df in dfs.values())

    def set_behavior_arrays(self, arrays: dict[str, np.ndarray]) -> None:
        self._behavior_arrays = arrays

    # ── Column enumeration ────────────────────────────────────────────────────

    def available_columns(self) -> list[str]:
        """Return all available column names from all sources."""
        cols: list[str] = []
        for aid, df in self._animal_dfs.items():
            for col in df.columns:
                cols.append(f"{aid}/{col}")
        for name in self._behavior_arrays:
            cols.append(f"behavior/{name}")
        return cols

    def set_selected_columns(self, columns: list[str]) -> None:
        self._selected_columns = columns

    # ── Build matrix ──────────────────────────────────────────────────────────

    def build(
        self,
        selected: Optional[list[str]] = None,
        normalize: bool = True,
        impute: bool = True,
    ) -> tuple[np.ndarray, list[str]]:
        """Build the feature matrix.

        Parameters
        ----------
        selected  : list of column keys from available_columns(); None = all
        normalize : z-score normalise each column
        impute    : forward-fill then zero-fill NaN values

        Returns
        -------
        (matrix, feature_names) where matrix has shape (N_frames, N_features)
        """
        cols = selected if selected is not None else self.available_columns()
        if not cols:
            raise ValueError("No columns selected for feature matrix.")

        n = self._n_frames
        if n == 0:
            raise ValueError("No data loaded.")

        arrays: list[np.ndarray] = []
        names:  list[str] = []

        for key in cols:
            arr = self._resolve_column(key, n)
            if arr is not None:
                arrays.append(arr)
                names.append(key)

        if not arrays:
            raise ValueError("None of the selected columns could be resolved.")

        matrix = np.column_stack(arrays).astype(np.float64)

        if impute:
            matrix = _impute(matrix)

        if normalize:
            matrix = _zscore(matrix)

        self._matrix = matrix
        self._feature_names = names
        logger.info("Feature matrix built: shape %s", matrix.shape)
        return matrix, names

    def matrix(self) -> Optional[np.ndarray]:
        return self._matrix

    def feature_names(self) -> list[str]:
        return self._feature_names

    # ── Internal ──────────────────────────────────────────────────────────────

    def _resolve_column(self, key: str, n: int) -> Optional[np.ndarray]:
        if "/" not in key:
            return None
        prefix, col = key.split("/", 1)

        if prefix == "behavior":
            arr = self._behavior_arrays.get(col)
            if arr is None:
                return None
            return _pad(arr.astype(np.float64), n)

        # Animal column
        df = self._animal_dfs.get(prefix)
        if df is None:
            return None
        if col not in df.columns:
            return None
        arr = df[col].to_numpy(dtype=np.float64)
        return _pad(arr, n)


# ── Utilities ─────────────────────────────────────────────────────────────────

def _pad(arr: np.ndarray, n: int) -> np.ndarray:
    if len(arr) >= n:
        return arr[:n]
    return np.concatenate([arr, np.full(n - len(arr), np.nan)])


def _impute(matrix: np.ndarray) -> np.ndarray:
    """Forward-fill then zero-fill NaN values column-wise."""
    out = matrix.copy()
    for c in range(out.shape[1]):
        col = out[:, c]
        nans = np.isnan(col)
        if not nans.any():
            continue
        # Forward fill
        valid = np.where(~nans)[0]
        if valid.size == 0:
            col[:] = 0.0
            continue
        # Fill before first valid
        col[:valid[0]] = col[valid[0]]
        for i in range(len(valid) - 1):
            col[valid[i]:valid[i+1]] = col[valid[i]]
        col[valid[-1]:] = col[valid[-1]]
        out[:, c] = col
    out = np.nan_to_num(out, nan=0.0)
    return out


def _zscore(matrix: np.ndarray) -> np.ndarray:
    out = matrix.copy()
    std = np.std(out, axis=0)
    std[std < 1e-8] = 1.0
    out = (out - np.mean(out, axis=0)) / std
    return out
