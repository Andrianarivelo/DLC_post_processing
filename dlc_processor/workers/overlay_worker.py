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

# Subtitle pill colours (BGR) — matches Gantt palette order
_SUBTITLE_COLORS_BGR = [
    (168, 139, 243),   # pink-red
    (250, 180, 137),   # blue
    (161, 227, 166),   # green
    (135, 179, 250),   # peach/orange
    (247, 166, 203),   # mauve
    (175, 226, 249),   # yellow
    (213, 226, 148),   # teal
    (231, 194, 245),   # pink
    (236, 199, 116),   # sapphire
]

# Keypoint circle radius
_KP_RADIUS = 5
_KP_THICKNESS = -1

# Skeleton connections — default mouse layout.
# resolve_skeleton_edges() filters out edges whose bodyparts are absent,
# so this list intentionally includes optional bodyparts (fhip, body_center).
# Duplicate / self edges are stripped automatically during resolution.
_SKELETON_EDGES = [
    # Head
    ("nose", "left_ear"),
    ("nose", "right_ear"),
    ("left_ear", "neck"),
    ("right_ear", "neck"),
    # Forebody (fhip = forehip / shoulder — optional)
    ("neck", "left_fhip"),
    ("neck", "right_fhip"),
    # Centre spine
    ("neck", "body_center"),
    ("body_center", "left_hip"),
    ("body_center", "right_hip"),
    # Fallback: direct neck-to-hip when no body_center/fhip
    ("neck", "left_hip"),
    ("neck", "right_hip"),
    # Forehip to hip
    ("left_fhip", "left_hip"),
    ("right_fhip", "right_hip"),
    # Hindquarters
    ("left_hip", "tail"),
    ("right_hip", "tail"),
]

# Common name aliases — maps canonical names to alternatives found in DLC configs.
_BP_ALIASES: dict[str, list[str]] = {
    "nose": ["Nose", "snout", "Snout", "nose_tip", "Nose_tip"],
    "left_ear": ["Left_ear", "left_Ear", "ear_left", "Ear_left", "lear", "leftear", "l_ear"],
    "right_ear": ["Right_ear", "right_Ear", "ear_right", "Ear_right", "rear", "rightear", "r_ear"],
    "neck": ["Neck", "spine1", "Spine_1", "spine_1", "nape", "Nape"],
    "left_fhip": ["Left_fhip", "left_forehip", "lfhip", "left_shoulder", "Left_shoulder"],
    "right_fhip": ["Right_fhip", "right_forehip", "rfhip", "right_shoulder", "Right_shoulder"],
    "left_hip": ["Left_hip", "left_Hip", "hip_left", "Hip_left", "lhip", "lefthip"],
    "right_hip": ["Right_hip", "right_Hip", "hip_right", "Hip_right", "rhip", "righthip"],
    "tail": ["Tail", "tailbase", "Tailbase", "tail_base", "Tail_base", "tail_tip", "tailtip"],
    "body_center": [
        "body_centre", "center", "Center", "Centre", "centroid",
        "spine2", "Spine_2", "spine_2", "mid_body", "midbody", "body_centre",
    ],
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
    seen: set[tuple[str, str]] = set()
    for bp1, bp2 in edges:
        r1 = _resolve(bp1)
        r2 = _resolve(bp2)
        if r1 is not None and r2 is not None and r1 != r2:
            key = (min(r1, r2), max(r1, r2))  # order-independent
            if key not in seen:
                seen.add(key)
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
        self.frame_index_mode = "video"
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
            row_idx = _resolve_data_row(df, fi, getattr(self, "frame_index_mode", "video"))
            if row_idx is None:
                continue
            color = _ANIMAL_COLORS_BGR[ai % len(_ANIMAL_COLORS_BGR)]
            row   = df.iloc[row_idx]
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

        # Behaviour subtitle pills — each active behaviour gets a colored badge
        draw_behaviors = getattr(self, "draw_behaviors", True)
        if draw_behaviors:
            behavior_idx = fi
            if animals and animals[0] in self.animal_dfs:
                resolved_idx = _resolve_data_row(
                    self.animal_dfs[animals[0]],
                    fi,
                    getattr(self, "frame_index_mode", "video"),
                )
                if resolved_idx is None:
                    behavior_idx = -1
                else:
                    behavior_idx = resolved_idx
            all_bool_names = [
                name for name, arr in self.behavior_arrays.items()
                if arr.dtype == bool
            ]
            active = [
                (name, all_bool_names.index(name))
                for name in all_bool_names
                if 0 <= behavior_idx < len(self.behavior_arrays[name])
                and self.behavior_arrays[name][behavior_idx]
            ]
            if active:
                pill_h = 34
                pad_y = 6
                font_scale = 0.7
                thickness = 2
                margin_x = fw // 2   # centre pills horizontally
                y_bottom = fh - 10

                for name, color_idx in reversed(active):
                    label = _pretty_behavior_name(name)
                    bgr = _SUBTITLE_COLORS_BGR[color_idx % len(_SUBTITLE_COLORS_BGR)]
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    pill_w = tw + 20
                    x1 = margin_x - pill_w // 2
                    y1 = y_bottom - pill_h
                    x2 = x1 + pill_w
                    y2 = y_bottom

                    # Semi-transparent filled rectangle
                    overlay = out.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), bgr, -1)
                    cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)

                    # White text centred in pill
                    tx = x1 + (pill_w - tw) // 2
                    ty = y2 - (pill_h - th) // 2
                    cv2.putText(out, label, (tx, ty),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                (255, 255, 255), thickness, cv2.LINE_AA)
                    y_bottom = y1 - pad_y

        return out


def _to_qimage(frame_bgr: np.ndarray) -> QImage:
    """Convert an OpenCV BGR frame to a QImage (RGB888), ensuring the buffer stays alive."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    img = QImage(rgb.data, w, h, w * ch, QImage.Format.Format_RGB888)
    return img.copy()   # copy so the numpy buffer can be freed safely


def _resolve_data_row(df, frame_idx: int, mode: str) -> Optional[int]:
    if mode == "row":
        row_idx = int(frame_idx)
        return row_idx if 0 <= row_idx < len(df) else None

    frames = getattr(df, "attrs", {}).get("frame_numbers")
    if frames is None:
        row_idx = int(frame_idx)
        return row_idx if 0 <= row_idx < len(df) else None

    arr = np.asarray(frames, dtype=np.int64).reshape(-1)
    if len(arr) != len(df) or len(arr) == 0:
        row_idx = int(frame_idx)
        return row_idx if 0 <= row_idx < len(df) else None

    target = int(frame_idx)
    if len(arr) > 1 and np.all(np.diff(arr) == 1):
        row_idx = target - int(arr[0])
        return row_idx if 0 <= row_idx < len(arr) else None

    pos = int(np.searchsorted(arr, target))
    if 0 <= pos < len(arr) and int(arr[pos]) == target:
        return pos
    return None


def _pretty_behavior_name(name: str) -> str:
    if "__" in name:
        animal, metric = name.split("__", 1)
        if metric in {"immobile", "is_immobile"}:
            return f"{animal} immobile"
        if metric in {"mobile", "is_mobile"}:
            return f"{animal} mobile"
    pretty = {
        "nose2nose": "nose-to-nose",
        "mask_contact": "mask contact",
        "fighting": "fighting",
        "attacks": "attacks",
        "sidebyside": "side-by-side",
        "sidereside": "side-reverse",
        "a_nose2anogenital_b": "A to B anogenital",
        "b_nose2anogenital_a": "B to A anogenital",
        "a_nose2body_b": "A to B body",
        "b_nose2body_a": "B to A body",
        "a_following_b": "A follows B",
        "b_following_a": "B follows A",
        "a_chasing_b": "A chases B",
        "b_chasing_a": "B chases A",
        "a_approaches_b": "A approaches B",
        "b_approaches_a": "B approaches A",
        "a_withdraws_from_b": "A withdraws from B",
        "b_withdraws_from_a": "B withdraws from A",
        "a_escapes_b": "A escapes B",
        "b_escapes_a": "B escapes A",
        "a_withdrawal_after_contact_b": "A withdraws after contact B",
        "b_withdrawal_after_contact_a": "B withdraws after contact A",
        "a_oriented_toward_b": "A oriented to B",
        "b_oriented_toward_a": "B oriented to A",
        "passive_anogenital": "passive anogenital",
        "passive_investigation": "passive investigation",
        "passive_being_followed": "passive followed",
        "passive_being_chased": "passive chased",
        "passive_withdrawal": "passive withdrawal",
        "rearing": "rearing",
    }
    return pretty.get(name, name.replace("_", " "))
