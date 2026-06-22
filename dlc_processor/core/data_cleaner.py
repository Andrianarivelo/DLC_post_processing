"""DLC data cleaning pipeline.

Stage 1 — Confidence filtering  : set x/y to NaN where likelihood < threshold.
Stage 2 — Gap interpolation     : spline-fill gaps up to max_gap_frames long;
                                   longer gaps stay NaN.
Stage 3 — Smoothing             : Savitzky-Golay filter on x and y columns.
Stage 4 — Anatomy fix           : correct impossible geometry (nose behind neck, etc.)

Spatial calibration is handled downstream in analysis/export layers so raw
tracking coordinates remain in pixel space for overlays and ROI geometry.
"""

from __future__ import annotations

import logging
import os
import itertools
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from dlc_processor.core.dlc_loader import get_bodyparts

logger = logging.getLogger(__name__)

_NOSE_CANDIDATES = ["nose", "Nose", "snout", "Snout"]
_NECK_CANDIDATES = ["neck", "Neck", "spine1", "Spine1", "spine_1", "Spine_1", "nape"]
_TAIL_CANDIDATES = ["tailbase", "Tailbase", "tail_base", "TailBase", "tail", "Tail"]
_CENTER_CANDIDATES = [
    "body_center", "body_centre", "center", "Center", "centre", "Centre",
    "centroid", "mid", "spine2", "Spine2", "spine_2", "Spine_2",
]
_LEFT_EAR_CANDIDATES = ["left_ear", "ear_left", "lear", "Left_ear", "Left_Ear"]
_RIGHT_EAR_CANDIDATES = ["right_ear", "ear_right", "rear", "Right_ear", "Right_Ear"]
_LEFT_HIP_CANDIDATES = ["left_hip", "hip_left", "lhip", "Left_hip", "Left_Hip"]
_RIGHT_HIP_CANDIDATES = ["right_hip", "hip_right", "rhip", "Right_hip", "Right_Hip"]


@dataclass
class _GeometryModel:
    axis_scale: float
    expected_pair_signs: dict[tuple[str, str], int]


# ── Public API ────────────────────────────────────────────────────────────────

def clean_animal_df(
    df: pd.DataFrame,
    conf_threshold: float = 0.6,
    max_gap_frames: int = 15,
    sg_window: int = 11,
    sg_polyorder: int = 3,
    apply_conf: bool = True,
    apply_interp: bool = True,
    apply_smooth: bool = True,
    apply_impossible: bool = True,
    px_per_cm: float = 0.0,
    start_frame: int = 0,
    end_frame: int = 0,
) -> pd.DataFrame:
    """Apply the full cleaning pipeline to a single-animal flat DataFrame.

    Parameters
    ----------
    df              : flat per-animal DataFrame (<bp>_x / _y / _likelihood)
    conf_threshold  : likelihood below this sets x/y to NaN
    max_gap_frames  : gaps longer than this are left as NaN
    sg_window       : Savitzky-Golay window length (must be odd, >= polyorder+2)
    sg_polyorder    : Savitzky-Golay polynomial order
    apply_*         : toggle individual stages
    px_per_cm       : deprecated; calibration is no longer applied by cleaning
    start_frame     : if > 0 with end_frame, clean only this sub-range
    end_frame       : end of sub-range (exclusive)

    Returns
    -------
    Cleaned copy of df.
    """
    out = df.copy()
    bodyparts = get_bodyparts(out)

    if px_per_cm > 0:
        logger.warning(
            "Ignoring px_per_cm during cleaning; calibration now propagates "
            "through analysis/export without modifying raw coordinates."
        )

    use_range = end_frame > start_frame
    if use_range:
        # Extract sub-range, clean it, put it back
        target = out.iloc[start_frame:end_frame].copy()
        saved_idx = target.index.copy()
        target = target.reset_index(drop=True)
    else:
        target = out

    t_bps = get_bodyparts(target)

    if apply_conf:
        target = _apply_confidence_filter(target, t_bps, conf_threshold)
        logger.debug("Confidence filter done (threshold=%.2f)", conf_threshold)

    if apply_interp:
        target = _apply_interpolation(target, t_bps, max_gap_frames)
        logger.debug("Interpolation done (max_gap=%d frames)", max_gap_frames)

    if apply_smooth:
        # Ensure window is odd and at least polyorder+2
        w = sg_window if sg_window % 2 == 1 else sg_window + 1
        w = max(w, sg_polyorder + 2 if (sg_polyorder + 2) % 2 == 1 else sg_polyorder + 3)
        target = _apply_smoothing(target, t_bps, w, sg_polyorder)
        logger.debug("SG smoothing done (window=%d, order=%d)", w, sg_polyorder)

    if apply_impossible:
        target = _apply_impossible_geometry_fix(target, t_bps, max_gap_frames=max_gap_frames)
        logger.debug("Impossible-geometry repair done")

    if use_range:
        target.index = saved_idx
        for col in target.columns:
            out.iloc[start_frame:end_frame, out.columns.get_loc(col)] = target[col].values
        logger.debug("Cleaned frames %d–%d only", start_frame, end_frame)
        return out

    return target


def clean_all_animals(
    animal_dfs: dict[str, pd.DataFrame],
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """Apply cleaning to all animals. Kwargs forwarded to :func:`clean_animal_df`.

    If ``computed_bodyparts`` is in kwargs, derived bodyparts are added first.
    """
    computed_defs = kwargs.pop("computed_bodyparts", None)
    max_workers = kwargs.pop("max_workers", None)

    def _clean_one(item: tuple[str, pd.DataFrame]) -> tuple[str, pd.DataFrame]:
        aid, df = item
        out = df
        if computed_defs:
            out = add_computed_bodyparts(out, computed_defs)
        return aid, clean_animal_df(out, **kwargs)

    items = list(animal_dfs.items())
    if len(items) <= 1:
        return dict(_clean_one(item) for item in items)

    workers = _resolve_worker_count(max_workers, len(items))
    if workers <= 1:
        return dict(_clean_one(item) for item in items)

    result: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for aid, cleaned in executor.map(_clean_one, items):
            result[aid] = cleaned
    return result


def fix_impossible_all_animals(
    animal_dfs: dict[str, pd.DataFrame],
    max_gap_frames: int = 15,
    start_frame: int = 0,
    end_frame: int = 0,
    computed_bodyparts: Optional[list[dict]] = None,
) -> dict[str, pd.DataFrame]:
    """Run only the impossible-geometry repair stage on all animals."""
    return clean_all_animals(
        animal_dfs,
        computed_bodyparts=computed_bodyparts,
        apply_conf=False,
        apply_interp=False,
        apply_smooth=False,
        apply_impossible=True,
        max_gap_frames=max_gap_frames,
        start_frame=start_frame,
        end_frame=end_frame,
    )


def fix_identities_from_masks(
    animal_dfs: dict[str, pd.DataFrame],
    mask_store: Any,
    animal_to_track_id: Optional[dict[str, int]] = None,
    *,
    exact_masks: bool = True,
    max_workers: Optional[int] = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    """Use mask track IDs as ground truth and repair swapped keypoint rows.

    The fast path scores how many keypoints from each current identity fall
    inside each mask bbox. When mask bboxes overlap, exact mask pixels are used
    for that frame so contact frames are still assigned by the real masks.
    """
    if not animal_dfs or mask_store is None:
        return animal_dfs, {"frames_checked": 0, "frames_corrected": 0}

    animals = list(animal_dfs.keys())
    if len(animals) < 2:
        return animal_dfs, {"frames_checked": 0, "frames_corrected": 0}

    first_df = next(iter(animal_dfs.values()))
    n_rows = min(len(df) for df in animal_dfs.values())
    if n_rows <= 0:
        return animal_dfs, {"frames_checked": 0, "frames_corrected": 0}

    track_map = _coerce_animal_track_map(animals, animal_to_track_id, mask_store)
    if len(set(track_map.values())) < 2:
        return animal_dfs, {"frames_checked": 0, "frames_corrected": 0}

    frames = _source_frames(first_df, n_rows)
    centers = np.stack([_animal_centers(df, n_rows) for df in animal_dfs.values()], axis=0)
    point_series = [_animal_point_series(df, n_rows) for df in animal_dfs.values()]
    target_tracks = np.asarray([track_map[aid] for aid in animals], dtype=np.int64)
    store_frames = getattr(mask_store, "frames", {}) or {}
    workers = _resolve_worker_count(max_workers, n_rows)
    chunks = _row_chunks(n_rows, workers)

    def _scan_chunk(chunk: tuple[int, int]) -> tuple[list[tuple[int, np.ndarray]], int, int]:
        start, stop = chunk
        hits: list[tuple[int, np.ndarray]] = []
        checked = 0
        exact_checked = 0
        for row_idx in range(start, stop):
            annotations = _mask_annotations_for_tracks(store_frames, int(frames[row_idx]), target_tracks)
            if annotations is None:
                continue
            checked += 1
            assignment, used_exact = _best_identity_assignment_from_masks(
                point_series,
                row_idx,
                centers[:, row_idx, :],
                annotations,
                exact_masks=exact_masks,
            )
            if assignment is None:
                continue
            exact_checked += int(used_exact)
            if not np.array_equal(assignment, np.arange(len(animals), dtype=np.int64)):
                hits.append((row_idx, assignment))
        return hits, checked, exact_checked

    assignments: list[tuple[int, np.ndarray]] = []
    frames_checked = 0
    exact_frames_checked = 0
    if workers > 1 and len(chunks) > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for chunk_hits, chunk_checked, chunk_exact in executor.map(_scan_chunk, chunks):
                assignments.extend(chunk_hits)
                frames_checked += int(chunk_checked)
                exact_frames_checked += int(chunk_exact)
    else:
        for chunk in chunks:
            chunk_hits, chunk_checked, chunk_exact = _scan_chunk(chunk)
            assignments.extend(chunk_hits)
            frames_checked += int(chunk_checked)
            exact_frames_checked += int(chunk_exact)

    if not assignments:
        return _copy_animal_dfs(animal_dfs), {
            "frames_checked": int(frames_checked),
            "frames_corrected": 0,
            "exact_mask_frames_checked": int(exact_frames_checked),
        }

    fixed = _copy_animal_dfs(animal_dfs)
    original = animal_dfs
    row_source_for_target = np.tile(np.arange(len(animals), dtype=np.int64)[:, None], (1, n_rows))
    for row_idx, current_to_target in assignments:
        for source_idx, target_idx in enumerate(current_to_target):
            row_source_for_target[int(target_idx), int(row_idx)] = int(source_idx)

    for target_idx, target_animal in enumerate(animals):
        target_df = fixed[target_animal]
        for source_idx, source_animal in enumerate(animals):
            rows = np.flatnonzero(row_source_for_target[target_idx] == source_idx)
            if rows.size == 0 or source_idx == target_idx:
                continue
            cols = _shared_bodypart_columns(target_df, original[source_animal])
            if not cols:
                continue
            target_locs = [target_df.columns.get_loc(col) for col in cols]
            source_locs = [original[source_animal].columns.get_loc(col) for col in cols]
            target_df.iloc[rows, target_locs] = original[source_animal].iloc[rows, source_locs].to_numpy()

    return fixed, {
        "frames_checked": int(frames_checked),
        "frames_corrected": int(len({row for row, _assignment in assignments})),
        "exact_mask_frames_checked": int(exact_frames_checked),
    }


def _resolve_worker_count(max_workers: Optional[int], work_items: int) -> int:
    if work_items <= 1:
        return 1
    if max_workers is None:
        count = min(work_items, max(1, (os.cpu_count() or 2) - 1))
    else:
        count = int(max_workers)
    return max(1, min(int(work_items), count))


def _row_chunks(n_rows: int, workers: int) -> list[tuple[int, int]]:
    if n_rows <= 0:
        return []
    workers = max(1, int(workers))
    chunk_size = max(512, int(np.ceil(n_rows / workers)))
    return [(start, min(n_rows, start + chunk_size)) for start in range(0, n_rows, chunk_size)]


def _copy_animal_dfs(animal_dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    copied: dict[str, pd.DataFrame] = {}
    for aid, df in animal_dfs.items():
        out = df.copy()
        out.attrs.update(getattr(df, "attrs", {}))
        copied[aid] = out
    return copied


def _source_frames(df: pd.DataFrame, n_rows: int) -> np.ndarray:
    frames = getattr(df, "attrs", {}).get("frame_numbers")
    if frames is not None:
        arr = np.asarray(frames, dtype=np.int64).reshape(-1)
        if len(arr) >= n_rows:
            return arr[:n_rows].copy()
    return np.arange(n_rows, dtype=np.int64)


def _coerce_animal_track_map(
    animals: list[str],
    animal_to_track_id: Optional[dict[str, int]],
    mask_store: Any,
) -> dict[str, int]:
    available = _available_track_ids(mask_store)
    ordered = sorted(available)
    if _prefer_track_order_for_animals(animals, ordered):
        return {
            aid: int(ordered[idx])
            for idx, aid in enumerate(animals)
            if idx < len(ordered)
        }

    out: dict[str, int] = {}
    if animal_to_track_id:
        used: set[int] = set()
        for aid in animals:
            try:
                track_id = int(animal_to_track_id[aid])
            except Exception:
                continue
            if track_id in used:
                continue
            if available and track_id not in available:
                continue
            out[aid] = track_id
            used.add(track_id)
    missing = [aid for aid in animals if aid not in out]
    for idx, aid in enumerate(missing):
        parsed = _numeric_suffix(aid)
        if parsed is not None and (not available or (parsed in available and parsed not in out.values())):
            out[aid] = int(parsed)
        elif parsed == 0 and 1 in available and 1 not in out.values():
            out[aid] = 1
        else:
            for track_id in ordered:
                if track_id not in out.values():
                    out[aid] = int(track_id)
                    break
            if aid in out:
                continue
            out[aid] = len(out) + 1
    if len(out) < len(animals) and ordered:
        return {
            aid: int(ordered[idx])
            for idx, aid in enumerate(animals)
            if idx < len(ordered)
        }
    return out


def _prefer_track_order_for_animals(animals: list[str], ordered_tracks: list[int]) -> bool:
    if len(ordered_tracks) < len(animals) or not animals:
        return False
    suffixes = [_numeric_suffix(aid) for aid in animals]
    if any(value is None for value in suffixes):
        return False
    numeric = [int(value) for value in suffixes if value is not None]
    if sorted(numeric) != list(range(1, len(numeric) + 1)):
        return False
    available = set(int(track) for track in ordered_tracks)
    if set(numeric).issubset(available):
        return False
    return True


def _available_track_ids(mask_store: Any) -> set[int]:
    frames = getattr(mask_store, "frames", None)
    if not isinstance(frames, dict):
        return set()
    ids: set[int] = set()
    for annotations in frames.values():
        for ann in annotations or []:
            try:
                ids.add(int(getattr(ann, "track_id")))
            except Exception:
                continue
    return ids


def _numeric_suffix(text: object) -> Optional[int]:
    import re

    found = re.findall(r"\d+", str(text or ""))
    if not found:
        return None
    return int(found[-1])


def _animal_centers(df: pd.DataFrame, n_rows: int) -> np.ndarray:
    for candidates in (_CENTER_CANDIDATES, _NECK_CANDIDATES):
        point = _first_bodypart_xy(df, candidates, n_rows)
        if point is not None:
            return point

    nose = _first_bodypart_xy(df, _NOSE_CANDIDATES, n_rows)
    tail = _first_bodypart_xy(df, _TAIL_CANDIDATES, n_rows)
    if nose is not None and tail is not None:
        return (nose + tail) / 2.0

    points = []
    for bp in get_bodyparts(df):
        point = _bodypart_xy(df, bp, n_rows)
        if point is not None:
            points.append(point)
    if not points:
        return np.full((n_rows, 2), np.nan, dtype=np.float64)
    stacked = np.stack(points, axis=0)
    with np.errstate(invalid="ignore"):
        return np.nanmean(stacked, axis=0)


def _animal_point_series(df: pd.DataFrame, n_rows: int) -> list[tuple[np.ndarray, Optional[np.ndarray]]]:
    series: list[tuple[np.ndarray, Optional[np.ndarray]]] = []
    for bp in get_bodyparts(df):
        xy = _bodypart_xy(df, bp, n_rows)
        if xy is None:
            continue
        likelihood = None
        lik_col = f"{bp}_likelihood"
        if lik_col in df.columns:
            likelihood = _pad_float(df[lik_col].to_numpy(dtype=np.float64), n_rows)
        series.append((xy, likelihood))
    return series


def _points_for_row(
    series: list[tuple[np.ndarray, Optional[np.ndarray]]],
    row_idx: int,
) -> np.ndarray:
    points: list[np.ndarray] = []
    for xy, likelihood in series:
        if row_idx < 0 or row_idx >= len(xy):
            continue
        point = xy[row_idx]
        if not np.isfinite(point).all():
            continue
        if likelihood is not None and row_idx < len(likelihood):
            lik = float(likelihood[row_idx])
            if np.isfinite(lik) and lik < 0.05:
                continue
        points.append(point.astype(np.float64, copy=False))
    if not points:
        return np.empty((0, 2), dtype=np.float64)
    return np.vstack(points).astype(np.float64, copy=False)


def _first_bodypart_xy(
    df: pd.DataFrame,
    candidates: list[str],
    n_rows: int,
) -> Optional[np.ndarray]:
    lower = {bp.lower(): bp for bp in get_bodyparts(df)}
    for cand in candidates:
        bp = lower.get(str(cand).lower())
        if bp is None:
            continue
        point = _bodypart_xy(df, bp, n_rows)
        if point is not None and np.isfinite(point).any():
            return point
    return None


def _bodypart_xy(df: pd.DataFrame, bp: str, n_rows: int) -> Optional[np.ndarray]:
    x_col = f"{bp}_x"
    y_col = f"{bp}_y"
    if x_col not in df.columns or y_col not in df.columns:
        return None
    x = _pad_float(df[x_col].to_numpy(dtype=np.float64), n_rows)
    y = _pad_float(df[y_col].to_numpy(dtype=np.float64), n_rows)
    return np.column_stack([x, y])


def _pad_float(values: np.ndarray, n_rows: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(arr) >= n_rows:
        return arr[:n_rows].copy()
    out = np.full(n_rows, np.nan, dtype=np.float64)
    out[: len(arr)] = arr
    return out


def _mask_centers_for_tracks(
    store_frames: dict,
    frame_idx: int,
    target_tracks: np.ndarray,
) -> Optional[np.ndarray]:
    annotations = store_frames.get(int(frame_idx), []) or []
    if not annotations:
        return None
    out = np.full((len(target_tracks), 2), np.nan, dtype=np.float64)
    track_to_pos = {int(track): idx for idx, track in enumerate(target_tracks)}
    for ann in annotations:
        try:
            track_id = int(getattr(ann, "track_id"))
        except Exception:
            continue
        pos = track_to_pos.get(track_id)
        if pos is None:
            continue
        center = _bbox_center(getattr(ann, "bbox", None))
        if center is not None:
            out[pos] = center
    if np.isfinite(out).all(axis=1).sum() < 2:
        return None
    return out


def _mask_annotations_for_tracks(
    store_frames: dict,
    frame_idx: int,
    target_tracks: np.ndarray,
) -> Optional[list[Any]]:
    annotations = store_frames.get(int(frame_idx), []) or []
    if not annotations:
        return None
    out: list[Any] = [None] * len(target_tracks)
    track_to_pos = {int(track): idx for idx, track in enumerate(target_tracks)}
    for ann in annotations:
        try:
            track_id = int(getattr(ann, "track_id"))
        except Exception:
            continue
        pos = track_to_pos.get(track_id)
        if pos is not None:
            out[pos] = ann
    if sum(ann is not None for ann in out) < 2:
        return None
    return out


def _bbox_center(bbox: object) -> Optional[np.ndarray]:
    if bbox is None:
        return None
    try:
        x, y, w, h = [float(v) for v in list(bbox)[:4]]
    except Exception:
        return None
    if not np.isfinite([x, y, w, h]).all() or w <= 0 or h <= 0:
        return None
    return np.asarray([x + w / 2.0, y + h / 2.0], dtype=np.float64)


def _best_identity_assignment(
    animal_centers: np.ndarray,
    mask_centers: np.ndarray,
) -> Optional[np.ndarray]:
    n = int(min(len(animal_centers), len(mask_centers)))
    if n < 2:
        return None
    costs = np.full((n, n), np.inf, dtype=np.float64)
    for i in range(n):
        if not np.isfinite(animal_centers[i]).all():
            continue
        for j in range(n):
            if np.isfinite(mask_centers[j]).all():
                costs[i, j] = float(np.linalg.norm(animal_centers[i] - mask_centers[j]))
    if not np.isfinite(costs).any():
        return None
    if n > 7:
        assignment = np.full(n, -1, dtype=np.int64)
        used_targets: set[int] = set()
        candidates = [
            (float(costs[i, j]), i, j)
            for i in range(n)
            for j in range(n)
            if np.isfinite(costs[i, j])
        ]
        for _cost, source_idx, target_idx in sorted(candidates, key=lambda item: item[0]):
            if assignment[source_idx] >= 0 or target_idx in used_targets:
                continue
            assignment[source_idx] = target_idx
            used_targets.add(target_idx)
        if np.any(assignment < 0):
            return None
        return assignment
    best_perm = None
    best_cost = np.inf
    for perm in itertools.permutations(range(n)):
        c = float(sum(costs[i, perm[i]] for i in range(n)))
        if c < best_cost:
            best_cost = c
            best_perm = perm
    if best_perm is None or not np.isfinite(best_cost):
        return None
    return np.asarray(best_perm, dtype=np.int64)


def _best_identity_assignment_from_masks(
    point_series: list[list[tuple[np.ndarray, Optional[np.ndarray]]]],
    row_idx: int,
    animal_centers: np.ndarray,
    annotations: list[Any],
    *,
    exact_masks: bool,
) -> tuple[Optional[np.ndarray], bool]:
    n = int(min(len(point_series), len(annotations)))
    if n < 2:
        return None, False

    use_exact = bool(exact_masks and _annotations_have_overlapping_bboxes(annotations[:n]))
    decoded_masks = [_decode_annotation_mask(ann) if use_exact else None for ann in annotations[:n]]
    if use_exact and not all(mask is not None for mask in decoded_masks):
        use_exact = False
        decoded_masks = [None] * n
    costs = np.full((n, n), np.inf, dtype=np.float64)
    for source_idx in range(n):
        points = _points_for_row(point_series[source_idx], row_idx)
        if points.size == 0:
            continue
        center = _robust_points_center(points, animal_centers[source_idx])
        for target_idx in range(n):
            ann = annotations[target_idx]
            if ann is None:
                continue
            bbox = getattr(ann, "bbox", None)
            costs[source_idx, target_idx] = _mask_identity_cost(
                points,
                center,
                bbox,
                decoded_masks[target_idx],
            )
    assignment = _best_assignment_from_costs(costs)
    return assignment, bool(use_exact and any(mask is not None for mask in decoded_masks))


def _best_assignment_from_costs(costs: np.ndarray) -> Optional[np.ndarray]:
    n = int(min(costs.shape)) if costs.ndim == 2 else 0
    if n < 2 or not np.isfinite(costs).any():
        return None
    if n > 7:
        assignment = np.full(n, -1, dtype=np.int64)
        used_targets: set[int] = set()
        candidates = [
            (float(costs[i, j]), i, j)
            for i in range(n)
            for j in range(n)
            if np.isfinite(costs[i, j])
        ]
        for _cost, source_idx, target_idx in sorted(candidates, key=lambda item: item[0]):
            if assignment[source_idx] >= 0 or target_idx in used_targets:
                continue
            assignment[source_idx] = target_idx
            used_targets.add(target_idx)
        if np.any(assignment < 0):
            return None
        return assignment

    best_perm = None
    best_cost = np.inf
    for perm in itertools.permutations(range(n)):
        cost = float(sum(costs[i, perm[i]] for i in range(n)))
        if cost < best_cost:
            best_cost = cost
            best_perm = perm
    if best_perm is None or not np.isfinite(best_cost):
        return None
    return np.asarray(best_perm, dtype=np.int64)


def _mask_identity_cost(
    points: np.ndarray,
    center: np.ndarray,
    bbox: object,
    mask: Optional[np.ndarray],
) -> float:
    bbox_values = _bbox_values(bbox)
    if bbox_values is None:
        return np.inf
    x, y, w, h = bbox_values
    bbox_center = np.asarray([x + w / 2.0, y + h / 2.0], dtype=np.float64)
    diag = max(float(np.hypot(w, h)), 1.0)
    center_dist = float(np.linalg.norm(center - bbox_center)) if np.isfinite(center).all() else diag
    bbox_hits = _count_points_in_bbox(points, bbox_values, margin=2.0)
    bbox_fraction = float(bbox_hits / max(len(points), 1))

    if mask is not None:
        mask_hits = _count_points_in_mask(points, mask, radius=2)
        mask_fraction = float(mask_hits / max(len(points), 1))
        return (
            center_dist / diag
            - 1000.0 * float(mask_hits)
            - 250.0 * mask_fraction
            - 5.0 * float(bbox_hits)
            - 2.0 * bbox_fraction
        )

    return (
        center_dist / diag
        - 50.0 * float(bbox_hits)
        - 20.0 * bbox_fraction
    )


def _bbox_values(bbox: object) -> Optional[tuple[float, float, float, float]]:
    if bbox is None:
        return None
    try:
        x, y, w, h = [float(v) for v in list(bbox)[:4]]
    except Exception:
        return None
    if not np.isfinite([x, y, w, h]).all() or w <= 0 or h <= 0:
        return None
    return x, y, w, h


def _robust_points_center(points: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    if points.size:
        with np.errstate(invalid="ignore"):
            center = np.nanmedian(points, axis=0)
        if np.isfinite(center).all():
            return np.asarray(center, dtype=np.float64)
    return np.asarray(fallback, dtype=np.float64)


def _count_points_in_bbox(
    points: np.ndarray,
    bbox: tuple[float, float, float, float],
    *,
    margin: float = 0.0,
) -> int:
    if points.size == 0:
        return 0
    x, y, w, h = bbox
    finite = np.isfinite(points).all(axis=1)
    inside = (
        finite
        & (points[:, 0] >= x - margin)
        & (points[:, 0] <= x + w + margin)
        & (points[:, 1] >= y - margin)
        & (points[:, 1] <= y + h + margin)
    )
    return int(np.count_nonzero(inside))


def _count_points_in_mask(points: np.ndarray, mask: np.ndarray, *, radius: int = 0) -> int:
    if points.size == 0 or mask is None:
        return 0
    mask_arr = np.asarray(mask, dtype=bool)
    if mask_arr.ndim != 2 or not mask_arr.any():
        return 0
    h, w = mask_arr.shape
    hits = 0
    r = max(0, int(radius))
    for x_raw, y_raw in points:
        if not np.isfinite([x_raw, y_raw]).all():
            continue
        x = int(round(float(x_raw)))
        y = int(round(float(y_raw)))
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        if r <= 0:
            hits += int(bool(mask_arr[y, x]))
            continue
        x0 = max(0, x - r)
        x1 = min(w, x + r + 1)
        y0 = max(0, y - r)
        y1 = min(h, y + r + 1)
        hits += int(bool(mask_arr[y0:y1, x0:x1].any()))
    return int(hits)


def _annotations_have_overlapping_bboxes(annotations: list[Any]) -> bool:
    bboxes = [_bbox_values(getattr(ann, "bbox", None)) for ann in annotations if ann is not None]
    for idx, bbox_a in enumerate(bboxes):
        if bbox_a is None:
            continue
        for bbox_b in bboxes[idx + 1:]:
            if bbox_b is None:
                continue
            if _bbox_intersection_area(bbox_a, bbox_b) > 0.0:
                return True
    return False


def _bbox_intersection_area(
    bbox_a: tuple[float, float, float, float],
    bbox_b: tuple[float, float, float, float],
) -> float:
    ax, ay, aw, ah = bbox_a
    bx, by, bw, bh = bbox_b
    ix0 = max(ax, bx)
    iy0 = max(ay, by)
    ix1 = min(ax + aw, bx + bw)
    iy1 = min(ay + ah, by + bh)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return float((ix1 - ix0) * (iy1 - iy0))


def _decode_annotation_mask(annotation: Any) -> Optional[np.ndarray]:
    if annotation is None:
        return None
    segmentation = getattr(annotation, "segmentation", None)
    size = getattr(annotation, "size", None)
    if segmentation is None or size is None:
        return None
    try:
        from dlc_processor.core.mask_loader import decode_coco_segmentation

        return decode_coco_segmentation(segmentation, size)
    except Exception:
        logger.debug("Could not decode mask annotation during identity repair", exc_info=True)
        return None


def _shared_bodypart_columns(target_df: pd.DataFrame, source_df: pd.DataFrame) -> list[str]:
    target_bps = get_bodyparts(target_df)
    source_bps = set(get_bodyparts(source_df))
    cols: list[str] = []
    for bp in target_bps:
        if bp not in source_bps:
            continue
        for suffix in ("_x", "_y", "_likelihood"):
            col = f"{bp}{suffix}"
            if col in target_df.columns and col in source_df.columns:
                cols.append(col)
    return cols


# ── Stage 0: calibration ─────────────────────────────────────────────────────

def _apply_calibration(
    df: pd.DataFrame,
    bodyparts: list[str],
    px_per_cm: float,
) -> pd.DataFrame:
    """Divide all x/y coordinate columns by *px_per_cm* to convert to cm."""
    out = df.copy()
    for bp in bodyparts:
        for coord in ("x", "y"):
            col = f"{bp}_{coord}"
            if col in out.columns:
                out[col] = out[col] / px_per_cm
    logger.debug("Converted %d bodyparts from px to cm", len(bodyparts))
    return out


# ── Stage 1: confidence filter ────────────────────────────────────────────────

def _apply_confidence_filter(
    df: pd.DataFrame,
    bodyparts: list[str],
    threshold: float,
) -> pd.DataFrame:
    out = df.copy()
    for bp in bodyparts:
        lik_col = f"{bp}_likelihood"
        x_col   = f"{bp}_x"
        y_col   = f"{bp}_y"
        if lik_col not in out.columns:
            continue
        low_conf = out[lik_col] < threshold
        n_low = int(low_conf.sum())
        if n_low:
            out.loc[low_conf, [x_col, y_col]] = np.nan
            logger.debug("  %s: %d frames below conf threshold", bp, n_low)
    return out


# ── Stage 2: gap interpolation ────────────────────────────────────────────────

def _apply_interpolation(
    df: pd.DataFrame,
    bodyparts: list[str],
    max_gap: int,
) -> pd.DataFrame:
    out = df.copy()
    for bp in bodyparts:
        for coord in ("x", "y"):
            col = f"{bp}_{coord}"
            if col not in out.columns:
                continue
            series = out[col].to_numpy(dtype=np.float64)
            filled = _interpolate_with_gap_limit(series, max_gap)
            out[col] = filled
    return out


def _interpolate_with_gap_limit(arr: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill NaN runs of length <= max_gap using linear interpolation.

    Gaps longer than max_gap are left untouched.
    """
    n = len(arr)
    out = arr.copy()
    valid = ~np.isnan(arr)

    if valid.sum() < 2:
        return out

    # Identify NaN runs
    nan_mask = ~valid
    # Pad for diff
    padded = np.concatenate(([False], nan_mask, [False]))
    run_starts = np.where(np.diff(padded.astype(int)) == 1)[0]
    run_ends   = np.where(np.diff(padded.astype(int)) == -1)[0]

    for s, e in zip(run_starts, run_ends):
        gap_len = e - s
        if gap_len > max_gap:
            continue  # leave long gaps as NaN
        # Need valid neighbours on both sides
        left  = s - 1
        right = e
        if left < 0 or right >= n:
            continue
        if np.isnan(arr[left]) or np.isnan(arr[right]):
            continue
        # Linear interpolation across this gap
        xs = np.array([left, right], dtype=np.float64)
        ys = np.array([arr[left], arr[right]], dtype=np.float64)
        fn = interp1d(xs, ys, kind="linear")
        indices = np.arange(s, e, dtype=np.float64)
        out[s:e] = fn(indices)

    return out


# ── Stage 3: Savitzky-Golay smoothing ────────────────────────────────────────

def _apply_smoothing(
    df: pd.DataFrame,
    bodyparts: list[str],
    window: int,
    polyorder: int,
) -> pd.DataFrame:
    out = df.copy()
    for bp in bodyparts:
        for coord in ("x", "y"):
            col = f"{bp}_{coord}"
            if col not in out.columns:
                continue
            arr = out[col].to_numpy(dtype=np.float64)
            # Only smooth segments without NaN
            smoothed = _smooth_with_nans(arr, window, polyorder)
            out[col] = smoothed
    return out


def add_computed_bodyparts(
    df: pd.DataFrame,
    definitions: list[dict],
) -> pd.DataFrame:
    """Add computed bodyparts to a flat per-animal DataFrame.

    Each definition is a dict:
        {"name": "body_center", "sources": ["neck", "left_hip", "right_hip"], "operation": "mean"}

    Supported operations:
        - "mean": average of source bodypart coordinates
        - "midpoint": midpoint between exactly 2 source bodyparts
        - "weighted_mean": weighted average (definition must include "weights" list)

    For each computed bodypart, creates <name>_x, <name>_y, <name>_likelihood columns.
    Likelihood is the minimum of all source likelihoods.
    """
    out = df.copy()
    for defn in definitions:
        name = defn["name"]
        sources = defn["sources"]
        operation = defn.get("operation", "mean")
        weights = defn.get("weights", None)

        # Check all sources exist
        existing_bps = get_bodyparts(out)
        missing = [s for s in sources if s not in existing_bps]
        if missing:
            logger.warning("Skipping computed bp %r: missing sources %s", name, missing)
            continue

        # Gather source coordinates
        xs = np.column_stack([out[f"{s}_x"].to_numpy(dtype=np.float64) for s in sources])
        ys = np.column_stack([out[f"{s}_y"].to_numpy(dtype=np.float64) for s in sources])
        liks = np.column_stack([
            out[f"{s}_likelihood"].to_numpy(dtype=np.float64)
            if f"{s}_likelihood" in out.columns
            else np.ones(len(out))
            for s in sources
        ])

        if operation == "midpoint" and len(sources) == 2:
            out[f"{name}_x"] = (xs[:, 0] + xs[:, 1]) / 2.0
            out[f"{name}_y"] = (ys[:, 0] + ys[:, 1]) / 2.0
        elif operation == "weighted_mean" and weights:
            w = np.array(weights, dtype=np.float64)
            w /= w.sum()
            out[f"{name}_x"] = np.nansum(xs * w, axis=1)
            out[f"{name}_y"] = np.nansum(ys * w, axis=1)
        else:  # default: mean
            out[f"{name}_x"] = np.nanmean(xs, axis=1)
            out[f"{name}_y"] = np.nanmean(ys, axis=1)

        # Likelihood = min of all sources (conservative)
        out[f"{name}_likelihood"] = np.nanmin(liks, axis=1)

        logger.info("Added computed bodypart %r from %s (%s)", name, sources, operation)

    return out


def save_cleaned_h5(
    animal_dfs: dict[str, pd.DataFrame],
    output_path: str,
    scorer: str = "DLCProcessor",
) -> str:
    """Save cleaned DataFrames as DLC-compatible H5.

    Returns the actual written path. If PyTables is unavailable, falls back to
    DLC-compatible CSV next to the requested H5 path so project persistence
    still works in lightweight environments.
    """
    frames = []
    for animal_id, df in animal_dfs.items():
        frame_numbers = getattr(df, "attrs", {}).get("frame_numbers")
        index = (
            pd.Index(np.asarray(frame_numbers, dtype=np.int64), name="frame")
            if frame_numbers is not None and len(frame_numbers) == len(df)
            else pd.RangeIndex(len(df), name="frame")
        )
        bps = get_bodyparts(df)
        for bp in bps:
            for coord in ("x", "y", "likelihood"):
                col_name = f"{bp}_{coord}"
                if col_name in df.columns:
                    vals = df[col_name].to_numpy(dtype=np.float64)
                else:
                    vals = np.full(len(df), np.nan)
                mi = pd.MultiIndex.from_tuples(
                    [(scorer, animal_id, bp, coord)],
                    names=["scorer", "individuals", "bodyparts", "coords"],
                )
                frames.append(pd.DataFrame(vals, index=index, columns=mi))
    combined = pd.concat(frames, axis=1)
    try:
        combined.to_hdf(output_path, key="df_with_missing", mode="w")
        logger.info("Saved cleaned H5 -> %s (%d animals)", output_path, len(animal_dfs))
        return str(output_path)
    except ImportError as exc:
        fallback_path = str(Path(output_path).with_suffix(".csv"))
        combined.to_csv(fallback_path, index=True)
        logger.warning(
            "PyTables unavailable; saved cleaned tracking as CSV fallback -> %s (%d animals): %s",
            fallback_path,
            len(animal_dfs),
            exc,
        )
        return fallback_path


def save_cleaned_csv(
    animal_dfs: dict[str, pd.DataFrame],
    output_path: str,
    scorer: str = "DLCProcessor",
) -> str:
    """Save cleaned DataFrames back as DLC-compatible CSV with MultiIndex columns."""
    frames = []
    for animal_id, df in animal_dfs.items():
        frame_numbers = getattr(df, "attrs", {}).get("frame_numbers")
        index = (
            pd.Index(np.asarray(frame_numbers, dtype=np.int64), name="frame")
            if frame_numbers is not None and len(frame_numbers) == len(df)
            else pd.RangeIndex(len(df), name="frame")
        )
        bps = get_bodyparts(df)
        for bp in bps:
            for coord in ("x", "y", "likelihood"):
                col_name = f"{bp}_{coord}"
                vals = (
                    df[col_name].to_numpy(dtype=np.float64)
                    if col_name in df.columns
                    else np.full(len(df), np.nan)
                )
                mi = pd.MultiIndex.from_tuples(
                    [(scorer, animal_id, bp, coord)],
                    names=["scorer", "individuals", "bodyparts", "coords"],
                )
                frames.append(pd.DataFrame(vals, index=index, columns=mi))
    combined = pd.concat(frames, axis=1)
    combined.to_csv(output_path, index=True)
    logger.info("Saved cleaned CSV -> %s (%d animals)", output_path, len(animal_dfs))
    return str(output_path)


def _smooth_with_nans(arr: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """Apply SG filter to non-NaN segments separately."""
    out = arr.copy()
    if len(arr) < window:
        return out

    nan_mask = np.isnan(arr)
    if nan_mask.all():
        return out

    # Find contiguous valid segments
    valid = ~nan_mask
    padded = np.concatenate(([False], valid, [False]))
    seg_starts = np.where(np.diff(padded.astype(int)) == 1)[0]
    seg_ends   = np.where(np.diff(padded.astype(int)) == -1)[0]

    for s, e in zip(seg_starts, seg_ends):
        seg = arr[s:e]
        if len(seg) >= window:
            out[s:e] = savgol_filter(seg, window, polyorder)

    return out


# ── Stage 4: impossible geometry repair ──────────────────────────────────────

def _apply_impossible_geometry_fix(
    df: pd.DataFrame,
    bodyparts: list[str],
    max_gap_frames: int,
) -> pd.DataFrame:
    """Repair anatomically impossible placements and infer occluded keypoints.

    Strategy:
      1. Detect impossible positions (nose behind neck, bilateral points on the
         same side of the body axis).
      2. Invalidate those points and interpolate from neighbouring valid frames.
      3. Fill remaining occlusions with within-frame geometric rules.
    """
    if not bodyparts:
        return df

    out = df.copy()
    coords = {
        bp: np.column_stack(
            [
                out[f"{bp}_x"].to_numpy(dtype=np.float64),
                out[f"{bp}_y"].to_numpy(dtype=np.float64),
            ]
        )
        for bp in bodyparts
        if f"{bp}_x" in out.columns and f"{bp}_y" in out.columns
    }
    if not coords:
        return out

    likelihoods: dict[str, Optional[np.ndarray]] = {}
    for bp in bodyparts:
        lik_col = f"{bp}_likelihood"
        if lik_col in out.columns:
            likelihoods[bp] = out[lik_col].to_numpy(dtype=np.float64).copy()
        else:
            likelihoods[bp] = None

    resolved = _resolve_special_bodyparts(bodyparts)
    bilateral_pairs = _discover_bilateral_pairs(bodyparts, resolved)
    model = _estimate_geometry_model(coords, likelihoods, resolved, bilateral_pairs)

    invalid_masks = _detect_impossible_points(
        coords,
        likelihoods,
        resolved,
        bilateral_pairs,
        model,
    )

    repaired_frames = 0
    for bp, mask in invalid_masks.items():
        if not mask.any():
            continue
        repaired_frames += int(mask.sum())
        coords[bp][mask] = np.nan
        lik = likelihoods.get(bp)
        if lik is not None:
            lik[mask] = 0.0

    # Rebuild from the last good frames where geometry was still plausible.
    for bp, pts in coords.items():
        filled_pts, filled_lik, _ = _interpolate_point_series(
            pts,
            max_gap=max_gap_frames,
            likelihood=likelihoods.get(bp),
        )
        coords[bp] = filled_pts
        if filled_lik is not None:
            likelihoods[bp] = filled_lik

    inferred = _infer_missing_bodyparts(
        coords,
        likelihoods,
        resolved,
        bilateral_pairs,
        model,
    )
    if inferred:
        logger.info("Inferred %d occluded/impossible bodypart positions", inferred)
    if repaired_frames:
        logger.info("Invalidated %d anatomically impossible placements", repaired_frames)

    for bp, pts in coords.items():
        out[f"{bp}_x"] = pts[:, 0]
        out[f"{bp}_y"] = pts[:, 1]
        lik = likelihoods.get(bp)
        if lik is not None and f"{bp}_likelihood" in out.columns:
            out[f"{bp}_likelihood"] = lik
    return out


def _resolve_special_bodyparts(bodyparts: list[str]) -> dict[str, Optional[str]]:
    return {
        "nose": _pick_bp(bodyparts, _NOSE_CANDIDATES),
        "neck": _pick_bp(bodyparts, _NECK_CANDIDATES),
        "tail": _pick_bp(bodyparts, _TAIL_CANDIDATES),
        "center": _pick_bp(bodyparts, _CENTER_CANDIDATES),
        "left_ear": _pick_bp(bodyparts, _LEFT_EAR_CANDIDATES),
        "right_ear": _pick_bp(bodyparts, _RIGHT_EAR_CANDIDATES),
        "left_hip": _pick_bp(bodyparts, _LEFT_HIP_CANDIDATES),
        "right_hip": _pick_bp(bodyparts, _RIGHT_HIP_CANDIDATES),
    }


def _discover_bilateral_pairs(
    bodyparts: list[str],
    resolved: dict[str, Optional[str]],
) -> list[tuple[str, str, str]]:
    lower = {bp.lower(): bp for bp in bodyparts}
    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[str, str, str]] = []

    def _add(left_bp: Optional[str], right_bp: Optional[str], label: str) -> None:
        if not left_bp or not right_bp:
            return
        key = (left_bp, right_bp)
        if key in seen:
            return
        seen.add(key)
        pairs.append((left_bp, right_bp, label))

    _add(resolved.get("left_ear"), resolved.get("right_ear"), "ear")
    _add(resolved.get("left_hip"), resolved.get("right_hip"), "hip")

    for bp in bodyparts:
        bp_low = bp.lower()
        counterpart_low = None
        label = None
        if bp_low.startswith("left_"):
            counterpart_low = "right_" + bp_low[5:]
            label = bp_low[5:]
        elif bp_low.endswith("_left"):
            counterpart_low = bp_low[:-5] + "_right"
            label = bp_low[:-5]
        elif bp_low.startswith("left"):
            counterpart_low = "right" + bp_low[4:]
            label = bp_low[4:].lstrip("_")
        if not counterpart_low:
            continue
        counterpart = lower.get(counterpart_low)
        if counterpart:
            _add(bp, counterpart, label or bp)
    return pairs


def _estimate_geometry_model(
    coords: dict[str, np.ndarray],
    likelihoods: dict[str, Optional[np.ndarray]],
    resolved: dict[str, Optional[str]],
    bilateral_pairs: list[tuple[str, str, str]],
) -> _GeometryModel:
    axis_lengths: list[float] = []
    pair_signs: dict[tuple[str, str], list[int]] = {}
    n_frames = len(next(iter(coords.values())))

    for i in range(n_frames):
        landmarks = _frame_landmarks(coords, i, resolved)
        axis = _select_body_axis(landmarks, allow_nose=False)
        if axis is None:
            axis = _select_body_axis(landmarks, allow_nose=True)
        if axis is None:
            continue
        origin, unit, axis_len = axis
        axis_lengths.append(axis_len)
        side_tol = max(2.0, 0.02 * axis_len)

        for left_bp, right_bp, _label in bilateral_pairs:
            left_pt = _frame_point(coords, left_bp, i)
            right_pt = _frame_point(coords, right_bp, i)
            if left_pt is None or right_pt is None:
                continue
            dl = _signed_distance_to_axis(left_pt, origin, unit)
            dr = _signed_distance_to_axis(right_pt, origin, unit)
            if not np.isfinite(dl) or not np.isfinite(dr):
                continue
            if abs(dl) <= side_tol or abs(dr) <= side_tol:
                continue
            if np.sign(dl) == -np.sign(dr):
                pair_signs.setdefault((left_bp, right_bp), []).append(int(np.sign(dl)))

    axis_scale = float(np.nanmedian(axis_lengths)) if axis_lengths else 40.0
    expected_pair_signs = {}
    for pair, signs in pair_signs.items():
        if not signs:
            continue
        expected_pair_signs[pair] = 1 if float(np.nanmedian(signs)) >= 0 else -1
    return _GeometryModel(axis_scale=axis_scale, expected_pair_signs=expected_pair_signs)


def _detect_impossible_points(
    coords: dict[str, np.ndarray],
    likelihoods: dict[str, Optional[np.ndarray]],
    resolved: dict[str, Optional[str]],
    bilateral_pairs: list[tuple[str, str, str]],
    model: _GeometryModel,
) -> dict[str, np.ndarray]:
    n_frames = len(next(iter(coords.values())))
    invalid = {bp: np.zeros(n_frames, dtype=bool) for bp in coords}

    for i in range(n_frames):
        landmarks = _frame_landmarks(coords, i, resolved)
        axis = _select_body_axis(landmarks, allow_nose=False)
        if axis is None:
            axis = _select_body_axis(landmarks, allow_nose=True)

        if axis is not None:
            origin, unit, axis_len = axis
            nose_tol = max(2.0, 0.05 * max(axis_len, model.axis_scale))
            nose_bp = resolved.get("nose")
            head_ref = _first_available_point(landmarks.get("neck"), landmarks.get("ears_mid"))
            if nose_bp and landmarks.get("nose") is not None and head_ref is not None:
                nose_proj = _project_to_axis(landmarks["nose"], origin, unit)
                head_proj = _project_to_axis(head_ref, origin, unit)
                if nose_proj < head_proj - nose_tol:
                    invalid[nose_bp][i] = True

            side_tol = max(2.0, 0.02 * max(axis_len, model.axis_scale))
            for left_bp, right_bp, _label in bilateral_pairs:
                if invalid[left_bp][i] or invalid[right_bp][i]:
                    continue
                left_pt = _frame_point(coords, left_bp, i)
                right_pt = _frame_point(coords, right_bp, i)
                if left_pt is None or right_pt is None:
                    continue
                dl = _signed_distance_to_axis(left_pt, origin, unit)
                dr = _signed_distance_to_axis(right_pt, origin, unit)
                if abs(dl) <= side_tol or abs(dr) <= side_tol:
                    continue
                if np.sign(dl) != np.sign(dr):
                    continue

                expected = model.expected_pair_signs.get((left_bp, right_bp))
                if expected in (-1, 1):
                    if np.sign(dl) == expected and np.sign(dr) != -expected:
                        invalid[right_bp][i] = True
                        continue
                    if np.sign(dr) == -expected and np.sign(dl) != expected:
                        invalid[left_bp][i] = True
                        continue

                left_lik = _frame_likelihood(likelihoods, left_bp, i)
                right_lik = _frame_likelihood(likelihoods, right_bp, i)
                if np.isfinite(left_lik) and np.isfinite(right_lik) and abs(left_lik - right_lik) > 0.05:
                    invalid[left_bp if left_lik < right_lik else right_bp][i] = True
                    continue

                # When confidence gives no answer, discard the point closer to the
                # symmetry axis because it is usually the collapsed misprediction.
                if abs(dl) < abs(dr):
                    invalid[left_bp][i] = True
                elif abs(dr) < abs(dl):
                    invalid[right_bp][i] = True
                else:
                    invalid[left_bp][i] = True
                    invalid[right_bp][i] = True

    return invalid


def _infer_missing_bodyparts(
    coords: dict[str, np.ndarray],
    likelihoods: dict[str, Optional[np.ndarray]],
    resolved: dict[str, Optional[str]],
    bilateral_pairs: list[tuple[str, str, str]],
    model: _GeometryModel,
) -> int:
    inferred = 0
    n_frames = len(next(iter(coords.values())))

    for _pass in range(2):
        changed = False
        for i in range(n_frames):
            landmarks = _frame_landmarks(coords, i, resolved)

            neck_bp = resolved.get("neck")
            if neck_bp and _frame_point(coords, neck_bp, i) is None and landmarks.get("ears_mid") is not None:
                added = _set_point(coords, likelihoods, neck_bp, i, landmarks["ears_mid"], _source_likelihood(
                    likelihoods, i, resolved.get("left_ear"), resolved.get("right_ear")
                ))
                inferred += int(added)
                changed = changed or added
                if added:
                    landmarks = _frame_landmarks(coords, i, resolved)

            tail_bp = resolved.get("tail")
            if tail_bp and _frame_point(coords, tail_bp, i) is None and landmarks.get("hips_mid") is not None:
                added = _set_point(coords, likelihoods, tail_bp, i, landmarks["hips_mid"], _source_likelihood(
                    likelihoods, i, resolved.get("left_hip"), resolved.get("right_hip")
                ))
                inferred += int(added)
                changed = changed or added
                if added:
                    landmarks = _frame_landmarks(coords, i, resolved)

            center_bp = resolved.get("center")
            if center_bp and _frame_point(coords, center_bp, i) is None:
                center_pt, center_lik = _infer_center_point(landmarks)
                if center_pt is not None:
                    added = _set_point(coords, likelihoods, center_bp, i, center_pt, center_lik)
                    inferred += int(added)
                    changed = changed or added
                    if added:
                        landmarks = _frame_landmarks(coords, i, resolved)

            nose_bp = resolved.get("nose")
            if nose_bp and _frame_point(coords, nose_bp, i) is None:
                nose_pt = _infer_nose_from_ears(landmarks)
                if nose_pt is not None:
                    nose_lik = _source_likelihood(
                        likelihoods,
                        i,
                        resolved.get("left_ear"),
                        resolved.get("right_ear"),
                        resolved.get("neck"),
                    )
                    added = _set_point(coords, likelihoods, nose_bp, i, nose_pt, nose_lik)
                    inferred += int(added)
                    changed = changed or added
                    if added:
                        landmarks = _frame_landmarks(coords, i, resolved)

            axis = _select_body_axis(landmarks, allow_nose=True)
            if axis is None:
                continue
            origin, unit, _axis_len = axis

            for left_bp, right_bp, _label in bilateral_pairs:
                left_pt = _frame_point(coords, left_bp, i)
                right_pt = _frame_point(coords, right_bp, i)
                expected = model.expected_pair_signs.get((left_bp, right_bp))
                if left_pt is None and right_pt is not None:
                    mirrored = _reflect_point_across_axis(right_pt, origin, unit)
                    if expected in (-1, 1):
                        side = np.sign(_signed_distance_to_axis(mirrored, origin, unit))
                        if side not in (0, expected):
                            mirrored = _reflect_point_across_axis(right_pt, origin, unit)
                    added = _set_point(coords, likelihoods, left_bp, i, mirrored, _source_likelihood(
                        likelihoods, i, right_bp
                    ))
                    inferred += int(added)
                    changed = changed or added
                elif right_pt is None and left_pt is not None:
                    mirrored = _reflect_point_across_axis(left_pt, origin, unit)
                    added = _set_point(coords, likelihoods, right_bp, i, mirrored, _source_likelihood(
                        likelihoods, i, left_bp
                    ))
                    inferred += int(added)
                    changed = changed or added

        if not changed:
            break

    return inferred


def _infer_center_point(
    landmarks: dict[str, Optional[np.ndarray]],
) -> tuple[Optional[np.ndarray], float]:
    if landmarks.get("neck") is not None and landmarks.get("left_hip") is not None and landmarks.get("right_hip") is not None:
        centre = np.nanmean(
            np.stack([landmarks["neck"], landmarks["left_hip"], landmarks["right_hip"]]),
            axis=0,
        )
        return centre, 0.8
    if landmarks.get("neck") is not None and landmarks.get("hips_mid") is not None:
        return (landmarks["neck"] + landmarks["hips_mid"]) / 2.0, 0.75
    if landmarks.get("neck") is not None and landmarks.get("tail") is not None:
        return (landmarks["neck"] + landmarks["tail"]) / 2.0, 0.75
    if landmarks.get("hips_mid") is not None:
        return landmarks["hips_mid"], 0.7
    return None, np.nan


def _infer_nose_from_ears(landmarks: dict[str, Optional[np.ndarray]]) -> Optional[np.ndarray]:
    left_ear = landmarks.get("left_ear")
    right_ear = landmarks.get("right_ear")
    if left_ear is None or right_ear is None:
        return None

    ear_vec = right_ear - left_ear
    ear_dist = float(np.hypot(ear_vec[0], ear_vec[1]))
    if ear_dist <= 1e-6:
        return None

    ear_mid = (left_ear + right_ear) / 2.0
    perp = np.array([-ear_vec[1], ear_vec[0]], dtype=np.float64) / ear_dist
    height = (np.sqrt(3.0) / 2.0) * ear_dist
    cand_a = ear_mid + perp * height
    cand_b = ear_mid - perp * height

    rear_anchor = _first_available_point(
        landmarks.get("center"),
        landmarks.get("hips_mid"),
        landmarks.get("tail"),
        landmarks.get("neck"),
    )
    if rear_anchor is None:
        return None

    forward = ear_mid - rear_anchor
    if float(np.hypot(forward[0], forward[1])) <= 1e-6:
        return None

    score_a = float(np.dot(cand_a - ear_mid, forward))
    score_b = float(np.dot(cand_b - ear_mid, forward))
    return cand_a if score_a >= score_b else cand_b


def _frame_landmarks(
    coords: dict[str, np.ndarray],
    frame_idx: int,
    resolved: dict[str, Optional[str]],
) -> dict[str, Optional[np.ndarray]]:
    left_ear = _frame_point(coords, resolved.get("left_ear"), frame_idx)
    right_ear = _frame_point(coords, resolved.get("right_ear"), frame_idx)
    left_hip = _frame_point(coords, resolved.get("left_hip"), frame_idx)
    right_hip = _frame_point(coords, resolved.get("right_hip"), frame_idx)
    ears_mid = _midpoint(left_ear, right_ear)
    hips_mid = _midpoint(left_hip, right_hip)

    neck = _frame_point(coords, resolved.get("neck"), frame_idx)
    if neck is None:
        neck = ears_mid

    tail = _frame_point(coords, resolved.get("tail"), frame_idx)
    if tail is None:
        tail = hips_mid

    center = _frame_point(coords, resolved.get("center"), frame_idx)
    if center is None:
        if neck is not None and hips_mid is not None:
            center = (neck + hips_mid) / 2.0
        elif neck is not None and tail is not None:
            center = (neck + tail) / 2.0
        elif hips_mid is not None:
            center = hips_mid

    return {
        "nose": _frame_point(coords, resolved.get("nose"), frame_idx),
        "neck": neck,
        "tail": tail,
        "center": center,
        "left_ear": left_ear,
        "right_ear": right_ear,
        "left_hip": left_hip,
        "right_hip": right_hip,
        "ears_mid": ears_mid,
        "hips_mid": hips_mid,
    }


def _select_body_axis(
    landmarks: dict[str, Optional[np.ndarray]],
    allow_nose: bool,
) -> Optional[tuple[np.ndarray, np.ndarray, float]]:
    candidates = [
        (landmarks.get("tail"), landmarks.get("neck")),
        (landmarks.get("tail"), landmarks.get("center")),
        (landmarks.get("center"), landmarks.get("neck")),
        (landmarks.get("tail"), landmarks.get("ears_mid")),
        (landmarks.get("center"), landmarks.get("ears_mid")),
    ]
    if allow_nose:
        candidates.extend(
            [
                (landmarks.get("tail"), landmarks.get("nose")),
                (landmarks.get("center"), landmarks.get("nose")),
                (landmarks.get("neck"), landmarks.get("nose")),
            ]
        )

    for rear, front in candidates:
        if rear is None or front is None:
            continue
        vec = front - rear
        norm = float(np.hypot(vec[0], vec[1]))
        if norm > 1e-6:
            return rear, vec / norm, norm
    return None


def _interpolate_point_series(
    points: np.ndarray,
    max_gap: int,
    likelihood: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    valid = np.isfinite(points).all(axis=1)
    out = points.copy()
    lik_out = None if likelihood is None else likelihood.copy()
    fill_mask = np.zeros(len(points), dtype=bool)

    if valid.sum() < 2:
        return out, lik_out, fill_mask

    padded = np.concatenate(([False], ~valid, [False]))
    run_starts = np.where(np.diff(padded.astype(int)) == 1)[0]
    run_ends = np.where(np.diff(padded.astype(int)) == -1)[0]

    for start, end in zip(run_starts, run_ends):
        gap_len = end - start
        if gap_len > max_gap:
            continue
        left = start - 1
        right = end
        if left < 0 or right >= len(points):
            continue
        if not valid[left] or not valid[right]:
            continue

        fill_mask[start:end] = True
        idx = np.arange(start, end, dtype=np.float64)
        for dim in range(2):
            out[start:end, dim] = np.interp(
                idx,
                [left, right],
                [out[left, dim], out[right, dim]],
            )

        if lik_out is not None:
            neigh = [lik_out[left], lik_out[right]]
            interp_lik = float(np.nanmean(neigh)) if np.isfinite(np.nanmean(neigh)) else 0.75
            lik_out[start:end] = np.clip(interp_lik, 0.0, 1.0)

    return out, lik_out, fill_mask


def _pick_bp(bodyparts: list[str], candidates: list[str]) -> Optional[str]:
    by_lower = {bp.lower(): bp for bp in bodyparts}
    for cand in candidates:
        if cand in bodyparts:
            return cand
        match = by_lower.get(cand.lower())
        if match:
            return match
    return None


def _frame_point(
    coords: dict[str, np.ndarray],
    bp: Optional[str],
    frame_idx: int,
) -> Optional[np.ndarray]:
    if not bp or bp not in coords:
        return None
    point = coords[bp][frame_idx]
    if not np.isfinite(point).all():
        return None
    return point.copy()


def _frame_likelihood(
    likelihoods: dict[str, Optional[np.ndarray]],
    bp: str,
    frame_idx: int,
) -> float:
    lik = likelihoods.get(bp)
    if lik is None or frame_idx >= len(lik):
        return np.nan
    return float(lik[frame_idx])


def _source_likelihood(
    likelihoods: dict[str, Optional[np.ndarray]],
    frame_idx: int,
    *bodyparts: Optional[str],
) -> float:
    vals = []
    for bp in bodyparts:
        if not bp:
            continue
        lik = likelihoods.get(bp)
        if lik is None or frame_idx >= len(lik):
            continue
        vals.append(float(lik[frame_idx]))
    if vals:
        val = float(np.nanmin(vals))
        if np.isfinite(val):
            return float(np.clip(val, 0.5, 1.0))
    return 0.75


def _set_point(
    coords: dict[str, np.ndarray],
    likelihoods: dict[str, Optional[np.ndarray]],
    bp: str,
    frame_idx: int,
    point: np.ndarray,
    likelihood: float,
) -> bool:
    if bp not in coords or point is None or not np.isfinite(point).all():
        return False
    if np.isfinite(coords[bp][frame_idx]).all():
        return False
    coords[bp][frame_idx] = point
    lik = likelihoods.get(bp)
    if lik is not None:
        lik[frame_idx] = float(np.clip(likelihood, 0.0, 1.0)) if np.isfinite(likelihood) else 0.75
    return True


def _midpoint(p1: Optional[np.ndarray], p2: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if p1 is None or p2 is None:
        return None
    return (p1 + p2) / 2.0


def _first_available_point(*points: Optional[np.ndarray]) -> Optional[np.ndarray]:
    for point in points:
        if point is not None:
            return point
    return None


def _project_to_axis(point: np.ndarray, origin: np.ndarray, unit: np.ndarray) -> float:
    return float(np.dot(point - origin, unit))


def _signed_distance_to_axis(point: np.ndarray, origin: np.ndarray, unit: np.ndarray) -> float:
    rel = point - origin
    return float(unit[0] * rel[1] - unit[1] * rel[0])


def _reflect_point_across_axis(
    point: np.ndarray,
    origin: np.ndarray,
    unit: np.ndarray,
) -> np.ndarray:
    rel = point - origin
    parallel = unit * np.dot(rel, unit)
    perp = rel - parallel
    return origin + parallel - perp
