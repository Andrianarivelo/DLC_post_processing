"""ROI (Region-of-Interest) analysis for DLC tracking data.

Polygonal ROIs are defined in pixel coordinates.
For each frame, determines whether a keypoint falls inside each ROI.
Computes time-in-zone statistics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from dlc_processor.core.dlc_loader import get_bodyparts

logger = logging.getLogger(__name__)


@dataclass
class ROI:
    name: str
    polygon: list[tuple[float, float]]  # [(x, y), …] pixel coords

    def as_contour(self) -> np.ndarray:
        pts = np.array([[int(x), int(y)] for x, y in self.polygon], dtype=np.int32)
        return pts.reshape((-1, 1, 2))


@dataclass
class ZoneResult:
    roi_name: str
    animal_id: str
    keypoint: str
    occupancy: np.ndarray       # bool array (N_frames,)
    frames_in: int = 0
    frames_total: int = 0
    time_in_s: float = 0.0
    pct_in: float = 0.0


class ROIAnalyzer:
    """Compute per-frame occupancy for a set of polygon ROIs."""

    def __init__(self, fps: float = 25.0):
        self.fps  = fps
        self.rois: list[ROI] = []

    def add_roi(self, name: str, polygon: list[tuple[float, float]]) -> None:
        self.rois.append(ROI(name=name, polygon=polygon))
        logger.debug("ROI added: %s (%d vertices)", name, len(polygon))

    def remove_roi(self, name: str) -> None:
        self.rois = [r for r in self.rois if r.name != name]

    def clear(self) -> None:
        self.rois.clear()

    def analyze(
        self,
        df: pd.DataFrame,
        animal_id: str = "animal",
        keypoint: str = "center",
    ) -> list[ZoneResult]:
        """Compute per-ROI occupancy for one keypoint.

        Parameters
        ----------
        df          : flat per-animal DataFrame
        animal_id   : label for logging/results
        keypoint    : bodypart name to use (e.g. "center")

        Returns
        -------
        List of ZoneResult — one per ROI.
        """
        bps = get_bodyparts(df)
        kp = _resolve_keypoint(keypoint, bps)
        if kp is None:
            logger.warning("Keypoint %r not found in bodyparts: %s", keypoint, bps)
            return []

        x = df.get(f"{kp}_x", pd.Series(dtype=float)).to_numpy(np.float64)
        y = df.get(f"{kp}_y", pd.Series(dtype=float)).to_numpy(np.float64)
        n = len(x)

        results: list[ZoneResult] = []
        for roi in self.rois:
            if len(roi.polygon) < 3:
                continue
            contour = roi.as_contour()
            occupancy = np.zeros(n, dtype=bool)
            for i in range(n):
                if np.isnan(x[i]) or np.isnan(y[i]):
                    continue
                # cv2.pointPolygonTest returns +ve if inside, 0 on boundary, -ve outside
                val = cv2.pointPolygonTest(contour, (float(x[i]), float(y[i])), False)
                occupancy[i] = val >= 0

            frames_in = int(occupancy.sum())
            results.append(ZoneResult(
                roi_name=roi.name,
                animal_id=animal_id,
                keypoint=kp,
                occupancy=occupancy,
                frames_in=frames_in,
                frames_total=n,
                time_in_s=frames_in / max(self.fps, 1e-6),
                pct_in=100.0 * frames_in / max(n, 1),
            ))

        return results

    def analyze_all_animals(
        self,
        animal_dfs: dict[str, pd.DataFrame],
        keypoint: str = "center",
    ) -> list[ZoneResult]:
        all_results: list[ZoneResult] = []
        for animal_id, df in animal_dfs.items():
            all_results.extend(self.analyze(df, animal_id, keypoint))
        return all_results


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_keypoint(name: str, bps: list[str]) -> Optional[str]:
    """Case-insensitive match of keypoint name against available bodyparts."""
    nl = name.lower()
    for bp in bps:
        if bp.lower() == nl:
            return bp
    # Fallback candidates
    for cand in ["center", "Centre", "body_centre", "mid"]:
        if cand in bps:
            return cand
    return bps[0] if bps else None
