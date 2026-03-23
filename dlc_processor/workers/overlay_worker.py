"""QThread worker that renders DLC keypoint overlays onto video frames.

Each rendered frame is emitted as a QImage signal for display in the UI.
Behaviour subtitles are drawn as a banner at the bottom of the frame.

Emits
-----
  frame_ready(QImage, int)   — rendered frame + absolute frame index
  finished()
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from dlc_processor.core.dlc_loader import get_bodyparts

logger = logging.getLogger(__name__)

# Pastel colour palette (BGR) for up to 8 animals
_ANIMAL_COLORS_BGR = [
    (255, 120, 50),    # blue
    (50, 200, 255),    # orange
    (80, 240, 80),     # green
    (80, 80, 255),     # red
    (240, 100, 240),   # pink
    (240, 240, 80),    # cyan
    (150, 80, 240),    # purple
    (80, 200, 200),    # yellow-green
]

# Keypoint circle radius
_KP_RADIUS = 5
_KP_THICKNESS = -1

# Skeleton connections — default 7-bodypart mouse layout
_SKELETON_EDGES = [
    ("nose", "left_ear"),
    ("nose", "right_ear"),
    ("left_ear", "neck"),
    ("right_ear", "neck"),
    ("neck", "left_hip"),
    ("neck", "right_hip"),
    ("left_hip", "tail"),
    ("right_hip", "tail"),
]

# Common name aliases — maps canonical names to alternatives
_BP_ALIASES: dict[str, list[str]] = {
    "nose": ["Nose", "snout", "Snout"],
    "left_ear": ["Left_ear", "left_Ear", "ear_left", "Ear_left", "lear"],
    "right_ear": ["Right_ear", "right_Ear", "ear_right", "Ear_right", "rear"],
    "neck": ["Neck", "spine1", "Spine_1", "spine_1", "nape"],
    "left_hip": ["Left_hip", "left_Hip", "hip_left", "Hip_left", "lhip"],
    "right_hip": ["Right_hip", "right_Hip", "hip_right", "Hip_right", "rhip"],
    "tail": ["Tail", "tailbase", "Tailbase", "tail_base", "Tail_base", "tail_tip"],
    "body_centre": ["body_center", "center", "Center", "Centre", "centroid", "spine2", "Spine_2"],
}


def resolve_skeleton_edges(
    edges: list[tuple[str, str]],
    available_bps: list[str],
) -> list[tuple[str, str]]:
    """Resolve skeleton edge names to actual bodypart names found in the data.

    Uses exact match first, then case-insensitive match, then alias lookup.
    Returns only edges where both bodyparts are found.
    """
    # Build lookup: canonical -> actual name in data
    bp_set = set(available_bps)
    bp_lower = {bp.lower(): bp for bp in available_bps}

    def _resolve(name: str) -> Optional[str]:
        # Exact match
        if name in bp_set:
            return name
        # Case-insensitive match
        actual = bp_lower.get(name.lower())
        if actual:
            return actual
        # Alias lookup
        for canonical, aliases in _BP_ALIASES.items():
            if name == canonical or name in aliases:
                # Try to find canonical or any alias in data
                if canonical in bp_set:
                    return canonical
                cand = bp_lower.get(canonical.lower())
                if cand:
                    return cand
                for alias in aliases:
                    if alias in bp_set:
                        return alias
                    cand = bp_lower.get(alias.lower())
                    if cand:
                        return cand
        return None

    resolved = []
    for bp1, bp2 in edges:
        r1 = _resolve(bp1)
        r2 = _resolve(bp2)
        if r1 is not None and r2 is not None:
            resolved.append((r1, r2))
    return resolved


class OverlayWorker(QThread):
    frame_ready = Signal(object, int)   # QImage, frame_idx
    finished    = Signal()

    def __init__(
        self,
        video_path: str,
        animal_dfs: dict[str, "pd.DataFrame"],
        behavior_arrays: Optional[dict[str, "np.ndarray"]] = None,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        draw_skeleton: bool = True,
        draw_labels: bool = True,
        draw_behaviors: bool = True,
        skeleton_edges: Optional[list[tuple[str, str]]] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.video_path      = video_path
        self.animal_dfs      = animal_dfs
        self.behavior_arrays = behavior_arrays or {}
        self.start_frame     = start_frame
        self.end_frame       = end_frame
        self.draw_skeleton   = draw_skeleton
        self.draw_labels     = draw_labels
        self.draw_behaviors  = draw_behaviors
        self.skeleton_edges  = skeleton_edges
        self.fill_body       = False
        self._abort          = False
        self._seek_frame: Optional[int] = None    # for single-frame seek

    def abort(self) -> None:
        self._abort = True

    def seek(self, frame_idx: int) -> None:
        """Render a single frame (called from UI thread)."""
        self._seek_frame = frame_idx

    def run(self) -> None:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logger.error("Cannot open video: %s", self.video_path)
            self.finished.emit()
            return

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        end   = min(self.end_frame or total, total)
        animals = list(self.animal_dfs.keys())

        cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

        for fi in range(self.start_frame, end):
            if self._abort:
                break

            # Support single-frame seek request
            if self._seek_frame is not None:
                target = self._seek_frame
                self._seek_frame = None
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                ret, frame = cap.read()
                if ret:
                    rendered = self._render(frame, target, animals)
                    self.frame_ready.emit(_to_qimage(rendered), target)
                continue

            ret, frame = cap.read()
            if not ret:
                break

            rendered = self._render(frame, fi, animals)
            self.frame_ready.emit(_to_qimage(rendered), fi)

        cap.release()
        self.finished.emit()

    # ── rendering ─────────────────────────────────────────────────────────────

    def _render(self, frame: np.ndarray, fi: int, animals: list[str]) -> np.ndarray:
        out = frame.copy()
        fh, fw = out.shape[:2]

        raw_edges = self.skeleton_edges if self.skeleton_edges else _SKELETON_EDGES

        # Resolve skeleton edge names against actual bodypart names
        if animals and animals[0] in self.animal_dfs:
            all_bps = get_bodyparts(self.animal_dfs[animals[0]])
            edges = resolve_skeleton_edges(raw_edges, all_bps)
        else:
            edges = raw_edges

        for ai, animal_id in enumerate(animals):
            df = self.animal_dfs[animal_id]
            if fi >= len(df):
                continue
            color = _ANIMAL_COLORS_BGR[ai % len(_ANIMAL_COLORS_BGR)]
            row   = df.iloc[fi]
            bps   = get_bodyparts(df)

            kp_coords: dict[str, tuple[int, int]] = {}
            for bp in bps:
                x = row.get(f"{bp}_x", np.nan)
                y = row.get(f"{bp}_y", np.nan)
                if np.isnan(x) or np.isnan(y):
                    continue
                ix, iy = int(round(x)), int(round(y))
                kp_coords[bp] = (ix, iy)

            # Body fill (convex hull) — drawn first so circles/lines appear on top
            if getattr(self, 'fill_body', False) and len(kp_coords) >= 3:
                pts = np.array(list(kp_coords.values()), dtype=np.int32)
                hull = cv2.convexHull(pts)
                fill_color = tuple(min(255, c + 80) for c in color)
                overlay = out.copy()
                cv2.fillConvexPoly(overlay, hull, fill_color)
                cv2.addWeighted(overlay, 0.3, out, 0.7, 0, out)

            # Draw keypoint circles
            for bp, (ix, iy) in kp_coords.items():
                cv2.circle(out, (ix, iy), _KP_RADIUS, color, _KP_THICKNESS)

            if self.draw_skeleton:
                for bp1, bp2 in edges:
                    if bp1 in kp_coords and bp2 in kp_coords:
                        cv2.line(out, kp_coords[bp1], kp_coords[bp2], color, 2)

            if self.draw_labels and kp_coords:
                cx = int(np.mean([v[0] for v in kp_coords.values()]))
                cy = int(np.mean([v[1] for v in kp_coords.values()]))
                label = animal_id

                # Mobility state indicator
                is_immobile_val = row.get("is_immobile", None)
                if is_immobile_val is not None and not (
                    isinstance(is_immobile_val, float) and np.isnan(is_immobile_val)
                ):
                    immobile = bool(is_immobile_val)
                    dot_color = (80, 80, 255) if immobile else (80, 240, 80)  # red / green BGR
                    dot_x = cx + len(animal_id) * 7 + 4
                    dot_y = cy - 16
                    cv2.circle(out, (dot_x, dot_y), 5, dot_color, -1)

                cv2.putText(
                    out, label,
                    (cx - 20, cy - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
                )

        # Behaviour subtitle banner (boolean behaviours only, skip continuous)
        draw_behaviors = getattr(self, "draw_behaviors", True)
        if draw_behaviors:
            active_behaviors = [
                _pretty_behavior_name(name) for name, arr in self.behavior_arrays.items()
                if arr.dtype == bool and fi < len(arr) and arr[fi]
            ]
            if active_behaviors:
                banner_h = 32
                banner   = np.zeros((banner_h, fw, 3), dtype=np.uint8)
                text     = "  |  ".join(active_behaviors)
                cv2.putText(
                    banner, text, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA,
                )
                out[-banner_h:] = cv2.addWeighted(out[-banner_h:], 0.35, banner, 0.65, 0)

        return out


def _to_qimage(frame_bgr: np.ndarray) -> QImage:
    """Convert an OpenCV BGR frame to a QImage (RGB888), ensuring the buffer stays alive."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    img = QImage(rgb.data, w, h, w * ch, QImage.Format.Format_RGB888)
    return img.copy()   # copy so the numpy buffer can be freed safely


def _pretty_behavior_name(name: str) -> str:
    pretty = {
        "nose2nose": "nose\u2194nose",
        "sidebyside": "side-by-side",
        "sidereside": "side\u2194rear",
        "a_nose2anogenital_b": "A\u2192B anogenital",
        "b_nose2anogenital_a": "B\u2192A anogenital",
        "a_nose2body_b": "A\u2192B body",
        "b_nose2body_a": "B\u2192A body",
        "a_following_b": "A follows B",
        "b_following_a": "B follows A",
        "a_oriented_toward_b": "A oriented\u2192B",
        "b_oriented_toward_a": "B oriented\u2192A",
        "passive_anogenital": "passive anogenital",
        "passive_investigation": "passive investigation",
        "passive_being_followed": "passive followed",
        "rearing": "rearing",
    }
    return pretty.get(name, name.replace("_", " "))
