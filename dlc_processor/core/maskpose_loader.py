"""Load the LISBET mask+pose JSONL(.gz) export format.

A ``mask_pose`` export folder holds, per recording, a common base name
``<base> = <stem>_preprocessed`` and these companions::

    keypoint/<base>_keypoints.jsonl.gz            -> per-frame pose
    mask/<base>_mask.jsonl.gz                      -> per-frame segmentation
    _source_maskpose_combined/<base>_maskpose.jsonl.gz  -> both in one file
    overlays/<base>_overlay.mp4                    -> rendered overlay video

Every JSONL line is one frame::

    {"f": <int>, "animals": [
        {"id": <int>,
         "keypoints": [[x, y, conf], ...],          # pose files / combined
         "mask": [[x, y], ...],                      # mask files / combined
         "orientation_deg": <float>,
         "mask_conf": <float>}]}

Keypoints become per-animal DeepLabCut-style DataFrames and masks become a
:class:`~dlc_processor.core.mask_loader.CocoMaskStore` (polygon segmentation),
so the rest of the post-processing pipeline (cleaning, kinematics, social,
overlays, identity refinement, export) consumes them unchanged.
"""

from __future__ import annotations

import gzip
import logging
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# LISBET ``mask_tracking`` skeleton, in the exporter's keypoint order. The
# JSONL keypoint arrays are dumped in exactly this order (it matches the
# ``bodyparts`` row of the sibling DeepLabCut ``*_tracking.csv`` export).
MASKPOSE_BODYPARTS: tuple[str, ...] = (
    "nose",
    "left_ear",
    "right_ear",
    "neck",
    "body",
    "left_hip",
    "right_hip",
    "tail",
)

_KEYPOINT_SUFFIXES = ("_keypoints.jsonl.gz", "_keypoints.jsonl")
_MASK_SUFFIXES = ("_mask.jsonl.gz", "_mask.jsonl")
_COMBINED_SUFFIXES = ("_maskpose.jsonl.gz", "_maskpose.jsonl")
_JSONL_SUFFIXES = (".jsonl.gz", ".jsonl")


# ── Detection ─────────────────────────────────────────────────────────────────

def _name(path: str | Path) -> str:
    return Path(path).name.lower()


def is_maskpose_jsonl(path: str | Path) -> bool:
    """True for any ``*.jsonl`` / ``*.jsonl.gz`` file."""
    return _name(path).endswith(_JSONL_SUFFIXES)


def is_maskpose_keypoint_file(path: str | Path) -> bool:
    """True for keypoint or combined files (anything carrying pose data)."""
    name = _name(path)
    if name.endswith(_KEYPOINT_SUFFIXES) or name.endswith(_COMBINED_SUFFIXES):
        return True
    if name.endswith(_JSONL_SUFFIXES) and not name.endswith(_MASK_SUFFIXES):
        return _sniff_keys(path).get("keypoints", False)
    return False


def is_maskpose_mask_file(path: str | Path) -> bool:
    """True for mask or combined files (anything carrying segmentation data)."""
    name = _name(path)
    if name.endswith(_MASK_SUFFIXES) or name.endswith(_COMBINED_SUFFIXES):
        return True
    if name.endswith(_JSONL_SUFFIXES) and not name.endswith(_KEYPOINT_SUFFIXES):
        return _sniff_keys(path).get("mask", False)
    return False


def is_maskpose_combined_file(path: str | Path) -> bool:
    name = _name(path)
    if name.endswith(_COMBINED_SUFFIXES):
        return True
    if name.endswith(_JSONL_SUFFIXES):
        keys = _sniff_keys(path)
        return bool(keys.get("keypoints") and keys.get("mask"))
    return False


def _sniff_keys(path: str | Path) -> dict[str, bool]:
    """Peek at the first frame carrying animals to see which fields exist."""
    try:
        for obj in _iter_frames(path):
            animals = obj.get("animals") or []
            if not animals:
                continue
            first = animals[0] if isinstance(animals[0], dict) else {}
            return {
                "keypoints": "keypoints" in first,
                "mask": "mask" in first,
            }
    except Exception:
        logger.debug("Could not sniff maskpose JSONL: %s", path, exc_info=True)
    return {"keypoints": False, "mask": False}


# ── Low-level reading ─────────────────────────────────────────────────────────

def _iter_frames(path: str | Path) -> Iterator[dict]:
    """Yield one decoded JSON object per non-empty JSONL line."""
    import json

    path = Path(path)
    opener = gzip.open if str(path).lower().endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except ValueError:
                logger.debug("Skipping malformed JSONL line in %s", path)
                continue


# ── Keypoints -> DataFrames ───────────────────────────────────────────────────

def load_maskpose_keypoints(
    path: str | Path,
    bodyparts: tuple[str, ...] = MASKPOSE_BODYPARTS,
) -> dict[str, pd.DataFrame]:
    """Load a keypoint (or combined) JSONL file as per-animal DataFrames.

    Returns ``{animal_id: DataFrame}`` with flat ``<bp>_x``/``<bp>_y``/
    ``<bp>_likelihood`` columns, a frame-indexed ``RangeIndex``, and the same
    trimming + metadata behaviour as :func:`dlc_loader.load_dlc_file`.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Keypoint file not found: {path}")

    n_bp = len(bodyparts)
    max_frame = -1
    per_animal: dict[int, list[tuple[int, list]]] = {}

    for obj in _iter_frames(path):
        frame = int(obj.get("f", 0))
        if frame > max_frame:
            max_frame = frame
        for animal in obj.get("animals", []) or []:
            if not isinstance(animal, dict):
                continue
            kps = animal.get("keypoints")
            if not kps:
                continue
            per_animal.setdefault(int(animal.get("id", 0)), []).append((frame, kps))

    n_frames = max_frame + 1 if max_frame >= 0 else 0
    if not per_animal or n_frames <= 0:
        raise ValueError(f"No keypoint data found in {path.name}")

    result: dict[str, pd.DataFrame] = {}
    for animal_id in sorted(per_animal):
        coords = np.full((n_frames, n_bp, 3), np.nan, dtype=np.float64)
        for frame, kps in per_animal[animal_id]:
            for i, kp in enumerate(kps[:n_bp]):
                if not kp:
                    continue
                coords[frame, i, 0] = float(kp[0])
                coords[frame, i, 1] = float(kp[1])
                coords[frame, i, 2] = float(kp[2]) if len(kp) > 2 else 1.0

        data: dict[str, np.ndarray] = {}
        for i, bp in enumerate(bodyparts):
            data[f"{bp}_x"] = coords[:, i, 0]
            data[f"{bp}_y"] = coords[:, i, 1]
            data[f"{bp}_likelihood"] = coords[:, i, 2]

        df = pd.DataFrame(data, index=pd.RangeIndex(n_frames))
        df.attrs["frame_numbers"] = np.arange(n_frames, dtype=np.int64)
        result[f"mouse{animal_id}"] = df

    from .dlc_loader import _finalize_loaded_dfs

    return _finalize_loaded_dfs(result, path)


def detect_keypoint_animals(path: str | Path) -> list[str]:
    """Return animal IDs (``mouse1`` ...) without building full DataFrames.

    Scans the whole file because a second animal can first appear thousands of
    frames in (an early-exit would undercount). Parsing IDs only is cheap
    (well under a second even for a 70k-frame recording).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Keypoint file not found: {path}")

    ids: set[int] = set()
    for obj in _iter_frames(path):
        for animal in obj.get("animals") or []:
            if isinstance(animal, dict) and "keypoints" in animal:
                ids.add(int(animal.get("id", 0)))
    return [f"mouse{aid}" for aid in sorted(ids)]


# ── Masks -> CocoMaskStore ────────────────────────────────────────────────────

def probe_video_size(video_path: str | Path) -> Optional[tuple[int, int]]:
    """Return ``(height, width)`` of a video, or ``None`` if unreadable."""
    if not video_path:
        return None
    try:
        import cv2
    except Exception:
        return None
    try:
        cap = cv2.VideoCapture(str(video_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
    except Exception:
        logger.debug("Could not probe video size: %s", video_path, exc_info=True)
        return None
    if width > 0 and height > 0:
        return (height, width)
    return None


def build_maskpose_mask_store(
    path: str | Path,
    frame_size: Optional[tuple[int, int]] = None,
    video_path: Optional[str | Path] = None,
):
    """Build a :class:`CocoMaskStore` from a mask (or combined) JSONL file.

    Polygons are stored as COCO polygon segmentation and rasterised lazily, so
    the masks need a frame ``(height, width)``. It is taken from ``frame_size``
    when given, else probed from ``video_path`` (the paired recording), else
    inferred from the largest polygon extent as a last resort.
    """
    from .mask_loader import CocoMaskStore

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mask file not found: {path}")

    if frame_size is None and video_path:
        frame_size = probe_video_size(video_path)

    frames_data: list[tuple[int, list[tuple[int, list, float]]]] = []
    max_x = 0.0
    max_y = 0.0
    for obj in _iter_frames(path):
        frame = int(obj.get("f", 0))
        annotations: list[tuple[int, list, float]] = []
        for animal in obj.get("animals", []) or []:
            if not isinstance(animal, dict):
                continue
            polygon = animal.get("mask")
            if not polygon or len(polygon) < 3:
                continue
            track_id = int(animal.get("id", 0))
            score = float(animal.get("mask_conf", animal.get("score", 1.0)) or 0.0)
            annotations.append((track_id, polygon, score))
            for point in polygon:
                if point[0] > max_x:
                    max_x = float(point[0])
                if point[1] > max_y:
                    max_y = float(point[1])
        if annotations:
            frames_data.append((frame, annotations))

    if not frames_data:
        raise ValueError(f"No mask polygons found in {path.name}")

    if frame_size is not None and frame_size[0] > 0 and frame_size[1] > 0:
        height, width = int(frame_size[0]), int(frame_size[1])
    else:
        height, width = int(max_y) + 1, int(max_x) + 1
        logger.info(
            "No frame size for %s; rasterising masks on inferred %dx%d canvas",
            path.name, width, height,
        )

    images: list[dict] = []
    annotations_payload: list[dict] = []
    ann_id = 1
    for frame, annotations in frames_data:
        images.append({
            "id": frame,
            "frame_index": frame,
            "height": height,
            "width": width,
        })
        for track_id, polygon, score in annotations:
            xs = [float(p[0]) for p in polygon]
            ys = [float(p[1]) for p in polygon]
            x0, y0 = min(xs), min(ys)
            flat = [float(c) for point in polygon for c in point[:2]]
            annotations_payload.append({
                "id": ann_id,
                "image_id": frame,
                "track_id": track_id,
                "category_id": track_id,
                "bbox": [x0, y0, max(xs) - x0, max(ys) - y0],
                "score": score,
                "segmentation": [flat],
            })
            ann_id += 1

    payload = {
        "info": {
            "description": "mask+pose JSONL import",
            "source_path": str(path),
            "height": height,
            "width": width,
        },
        "images": images,
        "annotations": annotations_payload,
    }
    return CocoMaskStore.from_payload(payload, path=path, keep_payload=False)
