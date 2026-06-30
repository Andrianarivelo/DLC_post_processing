"""QThread worker that renders DLC keypoint overlays onto video frames.

Rendering is delegated to :mod:`dlc_processor.core.overlay_renderer` (pure
drawing). This worker's job is the *data* side: resolving the data row for a
given video frame, extracting keypoints, mapping detected behaviours to the
animals they describe, and computing a per-animal confidence score so the
overlay can show broadcast-style badges such as ``M1: nose-to-nose 0.59``.

The same :class:`OverlayWorker._render` method powers both the live video
preview and the offline export worker, so improving it upgrades both at once.

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
from dlc_processor.core import overlay_renderer as _orr

logger = logging.getLogger(__name__)

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
    bp_set = set(available_bps)
    bp_lower = {bp.lower(): bp for bp in available_bps}

    def _resolve(name: str) -> Optional[str]:
        if name in bp_set:
            return name
        actual = bp_lower.get(name.lower())
        if actual:
            return actual
        for canonical, aliases in _BP_ALIASES.items():
            if name == canonical or name in aliases:
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
        """Initialise the worker.

        Args:
            video_path: Absolute path to the source video file.
            animal_dfs: Mapping of animal ID to its DLC keypoint DataFrame.
                Each DataFrame is expected to have columns of the form
                ``<bodypart>_x``, ``<bodypart>_y``, and
                ``<bodypart>_likelihood``.
            behavior_arrays: Optional mapping of behaviour key to a boolean
                numpy array aligned to the same row index as the DataFrames.
                Only rows where the value is ``True`` produce a badge.
            start_frame: First video frame to render (inclusive).
            end_frame: Last video frame to render (exclusive). Defaults to the
                total frame count reported by OpenCV.
            draw_skeleton: Whether to draw limb edges between keypoints.
            draw_labels: Whether to draw the small animal-name tag near each
                body centroid.
            draw_behaviors: Whether to draw behaviour badges.
            skeleton_edges: Custom edge list overriding the built-in mouse
                skeleton. Use this when the DLC project has non-standard
                bodypart names not covered by ``_BP_ALIASES``.
            parent: Optional Qt parent object.
        """
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
        # Premium rendering defaults (read via getattr in _render, so callers
        # that build the worker with __new__ inherit these too).
        self.draw_keypoints  = True
        self.fill_body       = False
        self.outline         = True
        self.line_thickness  = 2
        self.kp_radius       = 4
        self.font_scale      = 0.0          # 0 -> auto from frame width
        self.badge_mode      = "per_animal"  # "per_animal" | "banner"
        self.show_scores     = True
        self.max_badges_per_animal = 4
        self.animal_label_mode = "id"       # "id" | "mouse" | "none"
        self.frame_index_mode = "video"
        self._abort          = False
        self._seek_frame: Optional[int] = None    # for single-frame seek

    def abort(self) -> None:
        """Signal the run loop to stop after the current frame.

        Thread-safe: sets a plain Python bool that the run loop checks at
        the top of each iteration. The loop exits cleanly without terminating
        the OS thread forcefully.
        """
        self._abort = True

    def seek(self, frame_idx: int) -> None:
        """Render a single frame (called from UI thread)."""
        self._seek_frame = frame_idx

    def run(self) -> None:
        """Main QThread entry point: decode frames and emit rendered QImages.

        Opens the video at ``video_path``, seeks to ``start_frame``, and
        iterates forward frame by frame until ``end_frame`` or an abort
        signal. A pending seek request (set via :meth:`seek`) is serviced on
        the next iteration so a UI scrub does not block the main thread.
        Emits :attr:`frame_ready` for every rendered frame and
        :attr:`finished` when done (even on error).
        """
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
        """Paint the premium overlay for frame *fi* onto a copy of *frame*."""
        out = frame.copy()
        mode = getattr(self, "frame_index_mode", "video")

        # Resolve skeleton edge names against actual bodypart names.
        raw_edges = self.skeleton_edges if self.skeleton_edges else _SKELETON_EDGES
        if animals and animals[0] in self.animal_dfs:
            all_bps = get_bodyparts(self.animal_dfs[animals[0]])
            edges = resolve_skeleton_edges(raw_edges, all_bps)
        else:
            edges = list(raw_edges)

        # Use getattr with defaults throughout so that workers constructed
        # via __new__ (e.g. in tests) still receive sensible values even
        # when __init__ was never called.
        style = _orr.OverlayStyle(
            draw_skeleton=getattr(self, "draw_skeleton", True),
            draw_keypoints=getattr(self, "draw_keypoints", True),
            draw_labels=getattr(self, "draw_labels", True),
            draw_behaviors=getattr(self, "draw_behaviors", True),
            fill_body=getattr(self, "fill_body", False),
            outline=getattr(self, "outline", True),
            line_thickness=int(getattr(self, "line_thickness", 2)),
            kp_radius=int(getattr(self, "kp_radius", 4)),
            badge_mode=getattr(self, "badge_mode", "per_animal"),
            max_badges_per_animal=int(getattr(self, "max_badges_per_animal", 4)),
            font_scale=float(getattr(self, "font_scale", 0.0)),
        )

        show_scores = bool(getattr(self, "show_scores", True))
        label_mode = getattr(self, "animal_label_mode", "id")

        behavior_idx = self._behavior_row_index(fi, animals, mode)
        badges = (
            self._behavior_badges(animals, behavior_idx, show_scores)
            if style.draw_behaviors else {}
        )

        draws: list[_orr.AnimalDraw] = []
        for ai, animal_id in enumerate(animals):
            df = self.animal_dfs.get(animal_id)
            if df is None:
                continue
            row_idx = _resolve_data_row(df, fi, mode)
            if row_idx is None:
                continue
            row = df.iloc[row_idx]
            kp_coords: dict[str, tuple[int, int]] = {}
            for bp in get_bodyparts(df):
                try:
                    x = float(row.get(f"{bp}_x", np.nan))
                    y = float(row.get(f"{bp}_y", np.nan))
                except (TypeError, ValueError):
                    continue
                if np.isfinite(x) and np.isfinite(y):
                    kp_coords[bp] = (int(round(x)), int(round(y)))

            if label_mode == "mouse":
                label = f"M{ai + 1}"
            elif label_mode == "none":
                label = ""
            else:
                label = str(animal_id)

            draws.append(_orr.AnimalDraw(
                keypoints=kp_coords,
                color=_orr.animal_color(ai),
                label=label,
                badges=badges.get(ai, []),
            ))

        try:
            _orr.render_overlay(out, draws, edges, style)
        except Exception:
            logger.warning("Overlay rendering failed for frame %d", fi, exc_info=True)
            return frame
        return out

    # ── behaviour -> animal attribution ───────────────────────────────────────

    def _behavior_row_index(self, fi: int, animals: list[str], mode: str) -> int:
        """Resolve the behaviour-array row index for video frame *fi*."""
        if animals and animals[0] in self.animal_dfs:
            idx = _resolve_data_row(self.animal_dfs[animals[0]], fi, mode)
            return -1 if idx is None else int(idx)
        return int(fi)

    def _behavior_badges(
        self, animals: list[str], behavior_idx: int, show_scores: bool,
    ) -> dict[int, list[str]]:
        """Return ``{animal_index: [badge_text, ...]}`` for the active behaviours."""
        result: dict[int, list[str]] = {}
        if behavior_idx < 0 or not self.behavior_arrays:
            return result

        # Only boolean-dtype arrays are treated as behaviour masks. Float or
        # integer probability arrays (e.g. raw LISBET scores) are intentionally
        # skipped here; callers that want probability-thresholded badges should
        # convert to bool before passing via behavior_arrays.
        active = [
            name for name, arr in self.behavior_arrays.items()
            if getattr(arr, "dtype", None) == bool
            and 0 <= behavior_idx < len(arr) and bool(arr[behavior_idx])
        ]
        if not active:
            return result

        scores: dict[int, Optional[float]] = {}
        if show_scores:
            for ai, aid in enumerate(animals):
                scores[ai] = self._animal_likelihood(aid, behavior_idx)

        for name in active:
            pretty = _pretty_behavior_name(name)
            for ai in self._animals_for_behavior(name, animals):
                text = f"M{ai + 1}: {pretty}"
                score = scores.get(ai) if show_scores else None
                if score is not None:
                    text = f"{text} {score:.2f}"
                result.setdefault(ai, []).append(text)
        return result

    def _animals_for_behavior(self, name: str, animals: list[str]) -> list[int]:
        """Map a behaviour key to the animal indices it describes.

        - ``"<animal_id>__immobile"`` -> that animal only.
        - ``"a_..._b"`` -> the first animal (A); ``"b_..._a"`` -> the second (B).
        - everything else is treated as a mutual/pairwise behaviour (e.g.
          ``nose2nose``) and shown on both animals of the pair.
        """
        n = len(animals)
        if "__" in name:
            prefix = name.split("__", 1)[0]
            for ai, aid in enumerate(animals):
                if str(aid) == prefix:
                    return [ai]
            # Prefix found but no animal ID matched (e.g. the animal was
            # renamed after the behaviour arrays were created). Fall back to
            # both animals of the pair rather than silently dropping the badge.
            return list(range(min(n, 2)))
        low = name.lower()
        if low.startswith("a_"):
            return [0] if n >= 1 else []
        if low.startswith("b_"):
            return [1] if n >= 2 else [0]
        return list(range(min(n, 2)))

    def _animal_likelihood(self, animal_id: str, row_idx: int) -> Optional[float]:
        """Mean keypoint likelihood for an animal at *row_idx*, in [0, 1]."""
        df = self.animal_dfs.get(animal_id)
        if df is None or row_idx < 0 or row_idx >= len(df):
            return None
        row = df.iloc[row_idx]
        vals: list[float] = []
        for col in df.columns:
            if isinstance(col, str) and col.endswith("_likelihood"):
                try:
                    v = float(row[col])
                except (TypeError, ValueError, KeyError):
                    continue
                if np.isfinite(v):
                    vals.append(v)
        if not vals:
            return None
        return float(np.clip(np.mean(vals), 0.0, 1.0))


def _to_qimage(frame_bgr: np.ndarray) -> QImage:
    """Convert an OpenCV BGR frame to a QImage (RGB888), ensuring the buffer stays alive."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    img = QImage(rgb.data, w, h, w * ch, QImage.Format.Format_RGB888)
    return img.copy()   # copy so the numpy buffer can be freed safely


def _resolve_data_row(df, frame_idx: int, mode: str) -> Optional[int]:
    """Translate a video frame index to the corresponding DataFrame row index.

    Two indexing modes are supported:

    - ``"row"``: treat *frame_idx* as a direct 0-based row number. Used when
      the CSV was not subsampled and every row corresponds to one video frame
      in order.
    - ``"video"`` (default): look up the exact frame number stored in
      ``df.attrs["frame_numbers"]``. This handles subsampled or
      non-contiguous exports where the CSV rows do not align 1-to-1 with
      video frames.

    Optimisation: if the stored frame-number array is a contiguous run (all
    differences equal 1) a simple subtraction replaces a binary search.

    Returns the integer row index, or ``None`` if *frame_idx* is out of range
    or cannot be matched.
    """
    if mode == "row":
        row_idx = int(frame_idx)
        return row_idx if 0 <= row_idx < len(df) else None

    frames = getattr(df, "attrs", {}).get("frame_numbers")
    if frames is None:
        # No frame-number metadata: assume rows are contiguous from frame 0.
        row_idx = int(frame_idx)
        return row_idx if 0 <= row_idx < len(df) else None

    arr = np.asarray(frames, dtype=np.int64).reshape(-1)
    if len(arr) != len(df) or len(arr) == 0:
        # Metadata length mismatch: fall back to direct row indexing.
        row_idx = int(frame_idx)
        return row_idx if 0 <= row_idx < len(df) else None

    target = int(frame_idx)
    if len(arr) > 1 and np.all(np.diff(arr) == 1):
        # Fast path: contiguous frame numbers, no search needed.
        row_idx = target - int(arr[0])
        return row_idx if 0 <= row_idx < len(arr) else None

    # General path: binary search for the exact frame number.
    pos = int(np.searchsorted(arr, target))
    if 0 <= pos < len(arr) and int(arr[pos]) == target:
        return pos
    return None


def _pretty_behavior_name(name: str) -> str:
    """Convert a raw behaviour key to a human-readable display string.

    Handles two cases before consulting the lookup table:
    - ``"<animal>__immobile"`` / ``"<animal>__mobile"``: expand the
      ``__``-separated form to ``"<animal> immobile"`` etc.
    - Everything else: look up in the ``pretty`` dict, then fall back to
      replacing underscores with spaces so arbitrary keys still look tidy.
    """
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
