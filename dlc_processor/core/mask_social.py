"""Mask-derived social contact helpers."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd


_CENTRE_ALIASES = ("body_center", "body_centre", "center", "centre", "centroid", "neck")
_NOSE_ALIASES = ("nose", "snout")
_TAIL_ALIASES = ("tailbase", "tail_base", "tail")


def pair_mask_contact(
    mask_store: Any,
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    *,
    max_edge_gap_px: int = 2,
    max_edge_gap_percent: float = 0.0,
    candidate_frames: Optional[np.ndarray] = None,
    exact_masks: bool = True,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> np.ndarray:
    """Return frames where assigned masks for two animals touch or nearly touch.

    Masks are assigned to animals by nearest mask bounding-box centre to the
    animal body centre. This makes the contact signal independent of arbitrary
    model track IDs while keeping keypoints responsible for contact type.
    """
    n = max(len(df_a), len(df_b))
    out = np.zeros(n, dtype=bool)
    if mask_store is None or n <= 0:
        return out

    frames = _frame_numbers(df_a, n)
    centers_a = _animal_center(df_a, n)
    centers_b = _animal_center(df_b, n)
    candidates = _coerce_candidate_frames(candidate_frames, n)

    store_frames = getattr(mask_store, "frames", None)
    if isinstance(store_frames, dict):
        return _pair_mask_contact_from_annotations(
            mask_store,
            store_frames,
            frames,
            centers_a,
            centers_b,
            out,
            candidates,
            max_edge_gap_px=max_edge_gap_px,
            max_edge_gap_percent=max_edge_gap_percent,
            exact_masks=exact_masks,
            progress_callback=progress_callback,
        )

    row_indices = np.flatnonzero(candidates)
    total = int(len(row_indices))
    _report_progress(progress_callback, 0, f"Preparing mask contact ({total}/{n} candidate frames)")
    if total == 0:
        _report_progress(progress_callback, 100, "Mask contact complete")
        return out
    update_every = max(total // 100, 1)
    for done, row_idx in enumerate(row_indices, start=1):
        frame_idx = frames[int(row_idx)]
        masks = mask_store.masks_for_frame(int(frame_idx))
        if len(masks) < 2:
            if done == total or done % update_every == 0:
                _report_progress(progress_callback, int(100 * done / total), f"Mask contact {done}/{total}")
            continue
        idx_a, idx_b = _assign_two_masks(masks, centers_a[row_idx], centers_b[row_idx])
        if idx_a is None or idx_b is None or idx_a == idx_b:
            if done == total or done % update_every == 0:
                _report_progress(progress_callback, int(100 * done / total), f"Mask contact {done}/{total}")
            continue
        gap = _contact_gap_px(masks[idx_a][2], masks[idx_b][2], max_edge_gap_px, max_edge_gap_percent)
        if not exact_masks:
            out[row_idx] = not _bbox_separated_by_more_than(masks[idx_a][2], masks[idx_b][2], gap)
            if done == total or done % update_every == 0:
                _report_progress(progress_callback, int(100 * done / total), f"Mask contact {done}/{total}")
            continue
        out[row_idx] = _masks_touch(
            masks[idx_a][1],
            masks[idx_b][1],
            max_edge_gap_px=gap,
            bbox_a=masks[idx_a][2],
            bbox_b=masks[idx_b][2],
        )
        if done == total or done % update_every == 0:
            _report_progress(progress_callback, int(100 * done / total), f"Mask contact {done}/{total}")
    return out


def _pair_mask_contact_from_annotations(
    mask_store: Any,
    store_frames: dict,
    frames: np.ndarray,
    centers_a: np.ndarray,
    centers_b: np.ndarray,
    out: np.ndarray,
    candidate_frames: np.ndarray,
    *,
    max_edge_gap_px: int,
    max_edge_gap_percent: float,
    exact_masks: bool,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> np.ndarray:
    if not store_frames:
        return out

    available = np.fromiter((int(k) for k in store_frames.keys()), dtype=np.int64)
    row_indices = np.flatnonzero(np.isin(frames, available) & candidate_frames)
    total = int(len(row_indices))
    _report_progress(progress_callback, 0, f"Preparing mask contact ({total}/{len(frames)} candidate frames)")
    if total == 0:
        _report_progress(progress_callback, 100, "Mask contact complete")
        return out
    update_every = max(total // 100, 1)

    for done, row_idx in enumerate(row_indices, start=1):
        frame_idx = int(frames[int(row_idx)])
        annotations = list(store_frames.get(frame_idx, []) or [])
        if len(annotations) < 2:
            if done == total or done % update_every == 0:
                _report_progress(progress_callback, int(100 * done / total), f"Mask contact {done}/{total}")
            continue

        idx_a, idx_b = _assign_two_annotations(annotations, centers_a[row_idx], centers_b[row_idx])
        if idx_a is None or idx_b is None or idx_a == idx_b:
            if done == total or done % update_every == 0:
                _report_progress(progress_callback, int(100 * done / total), f"Mask contact {done}/{total}")
            continue

        bbox_a = _annotation_bbox(annotations[idx_a])
        bbox_b = _annotation_bbox(annotations[idx_b])
        if bbox_a is None or bbox_b is None:
            if done == total or done % update_every == 0:
                _report_progress(progress_callback, int(100 * done / total), f"Mask contact {done}/{total}")
            continue

        gap = _contact_gap_px(bbox_a, bbox_b, max_edge_gap_px, max_edge_gap_percent)
        if _bbox_separated_by_more_than(bbox_a, bbox_b, gap):
            if done == total or done % update_every == 0:
                _report_progress(progress_callback, int(100 * done / total), f"Mask contact {done}/{total}")
            continue
        if not exact_masks:
            out[row_idx] = True
            if done == total or done % update_every == 0:
                _report_progress(progress_callback, int(100 * done / total), f"Mask contact {done}/{total}")
            continue

        mask_a = _annotation_mask(mask_store, frame_idx, idx_a, annotations[idx_a])
        mask_b = _annotation_mask(mask_store, frame_idx, idx_b, annotations[idx_b])
        if mask_a is None or mask_b is None:
            if done == total or done % update_every == 0:
                _report_progress(progress_callback, int(100 * done / total), f"Mask contact {done}/{total}")
            continue
        score_a = float(getattr(annotations[idx_a], "score", 1.0) or 0.0)
        score_b = float(getattr(annotations[idx_b], "score", 1.0) or 0.0)
        if score_a <= 0.0 or score_b <= 0.0:
            if done == total or done % update_every == 0:
                _report_progress(progress_callback, int(100 * done / total), f"Mask contact {done}/{total}")
            continue
        out[row_idx] = _masks_touch(
            mask_a,
            mask_b,
            max_edge_gap_px=gap,
            bbox_a=bbox_a,
            bbox_b=bbox_b,
        )
        if done == total or done % update_every == 0:
            _report_progress(progress_callback, int(100 * done / total), f"Mask contact {done}/{total}")
    return out


def _coerce_candidate_frames(candidate_frames: Optional[np.ndarray], n: int) -> np.ndarray:
    if candidate_frames is None:
        return np.ones(n, dtype=bool)
    arr = np.asarray(candidate_frames, dtype=bool).reshape(-1)
    if len(arr) >= n:
        return arr[:n].copy()
    out = np.zeros(n, dtype=bool)
    out[: len(arr)] = arr
    return out


def _report_progress(
    callback: Optional[Callable[[int, str], None]],
    pct: int,
    message: str,
) -> None:
    if callback is not None:
        callback(max(0, min(100, int(pct))), message)


def _frame_numbers(df: pd.DataFrame, n: int) -> np.ndarray:
    frames = getattr(df, "attrs", {}).get("frame_numbers")
    if frames is not None and len(frames) >= n:
        return np.asarray(frames, dtype=np.int64)[:n]
    return np.arange(n, dtype=np.int64)


def _animal_center(df: pd.DataFrame, n: int) -> np.ndarray:
    x, y = _point_from_aliases(df, _CENTRE_ALIASES, n)
    if np.isfinite(x).any() and np.isfinite(y).any():
        return np.column_stack([x, y])

    nx, ny = _point_from_aliases(df, _NOSE_ALIASES, n)
    tx, ty = _point_from_aliases(df, _TAIL_ALIASES, n)
    if np.isfinite(nx).any() and np.isfinite(tx).any():
        return np.column_stack([(nx + tx) / 2.0, (ny + ty) / 2.0])

    xs = []
    ys = []
    for col in df.columns:
        if str(col).endswith("_x"):
            base = str(col)[:-2]
            y_col = f"{base}_y"
            if y_col in df.columns:
                xs.append(_pad_numeric(df[col], n))
                ys.append(_pad_numeric(df[y_col], n))
    if xs:
        return np.column_stack([np.nanmean(np.vstack(xs), axis=0), np.nanmean(np.vstack(ys), axis=0)])
    return np.full((n, 2), np.nan, dtype=np.float64)


def _point_from_aliases(df: pd.DataFrame, aliases: tuple[str, ...], n: int) -> tuple[np.ndarray, np.ndarray]:
    lower_to_bp = {
        str(col)[:-2].lower(): str(col)[:-2]
        for col in df.columns
        if str(col).endswith("_x")
    }
    for alias in aliases:
        bp = lower_to_bp.get(alias.lower())
        if bp and f"{bp}_y" in df.columns:
            return _pad_numeric(df[f"{bp}_x"], n), _pad_numeric(df[f"{bp}_y"], n)
    return np.full(n, np.nan), np.full(n, np.nan)


def _pad_numeric(series: pd.Series, n: int) -> np.ndarray:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    if len(arr) >= n:
        return arr[:n]
    out = np.full(n, np.nan, dtype=np.float64)
    out[: len(arr)] = arr
    return out


def _assign_two_masks(
    masks: list[tuple[int, np.ndarray, tuple[float, float, float, float], float]],
    center_a: np.ndarray,
    center_b: np.ndarray,
) -> tuple[int | None, int | None]:
    if not np.isfinite(center_a).all() or not np.isfinite(center_b).all():
        return 0, 1 if len(masks) > 1 else None
    centers = np.asarray([_bbox_center(item[2]) for item in masks], dtype=np.float64)
    best: tuple[float, int | None, int | None] = (np.inf, None, None)
    for i in range(len(masks)):
        for j in range(len(masks)):
            if i == j:
                continue
            score = float(np.linalg.norm(centers[i] - center_a) + np.linalg.norm(centers[j] - center_b))
            if score < best[0]:
                best = (score, i, j)
    return best[1], best[2]


def _assign_two_annotations(
    annotations: list[Any],
    center_a: np.ndarray,
    center_b: np.ndarray,
) -> tuple[int | None, int | None]:
    bboxes = [_annotation_bbox(ann) for ann in annotations]
    if not np.isfinite(center_a).all() or not np.isfinite(center_b).all():
        valid = [idx for idx, bbox in enumerate(bboxes) if bbox is not None]
        if len(valid) < 2:
            return None, None
        return valid[0], valid[1]

    centers = np.asarray([
        _bbox_center(bbox) if bbox is not None else (np.nan, np.nan)
        for bbox in bboxes
    ], dtype=np.float64)
    best: tuple[float, int | None, int | None] = (np.inf, None, None)
    for i in range(len(annotations)):
        if not np.isfinite(centers[i]).all():
            continue
        for j in range(len(annotations)):
            if i == j or not np.isfinite(centers[j]).all():
                continue
            score = float(np.linalg.norm(centers[i] - center_a) + np.linalg.norm(centers[j] - center_b))
            if score < best[0]:
                best = (score, i, j)
    return best[1], best[2]


def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x, y, w, h = bbox
    return float(x) + float(w) / 2.0, float(y) + float(h) / 2.0


def _contact_gap_px(
    bbox_a: tuple[float, float, float, float],
    bbox_b: tuple[float, float, float, float],
    max_edge_gap_px: int,
    max_edge_gap_percent: float,
) -> int:
    base = int(max(0, max_edge_gap_px))
    pct = float(max(0.0, max_edge_gap_percent))
    if pct <= 0.0:
        return base
    edges_a = _bbox_edges(bbox_a)
    edges_b = _bbox_edges(bbox_b)
    if edges_a is None or edges_b is None:
        return base
    ax0, ay0, ax1, ay1 = edges_a
    bx0, by0, bx1, by1 = edges_b
    diag_a = float(np.hypot(ax1 - ax0, ay1 - ay0))
    diag_b = float(np.hypot(bx1 - bx0, by1 - by0))
    if not np.isfinite(diag_a + diag_b) or diag_a <= 0 or diag_b <= 0:
        return base
    pct_gap = int(round(((diag_a + diag_b) * 0.5) * pct / 100.0))
    return max(base, pct_gap)


def _annotation_bbox(annotation: Any) -> tuple[float, float, float, float] | None:
    bbox = getattr(annotation, "bbox", None)
    if bbox is None and isinstance(annotation, dict):
        bbox = annotation.get("bbox")
    if bbox is None:
        return None
    try:
        values = tuple(float(v) for v in list(bbox)[:4])
    except Exception:
        return None
    return values if len(values) == 4 else None


def _annotation_mask(
    mask_store: Any,
    frame_idx: int,
    annotation_idx: int,
    annotation: Any,
) -> np.ndarray | None:
    decoder = getattr(mask_store, "mask_for_annotation", None)
    if callable(decoder):
        return decoder(frame_idx, annotation_idx)

    segmentation = getattr(annotation, "segmentation", None)
    size = getattr(annotation, "size", None)
    if isinstance(annotation, dict):
        segmentation = annotation.get("segmentation", segmentation)
        size = annotation.get("size", size)
    if segmentation is None or size is None:
        return None
    try:
        from dlc_processor.core.mask_loader import decode_coco_segmentation

        return decode_coco_segmentation(segmentation, size)
    except Exception:
        return None


def _masks_touch(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    *,
    max_edge_gap_px: int,
    bbox_a: tuple[float, float, float, float] | None = None,
    bbox_b: tuple[float, float, float, float] | None = None,
) -> bool:
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    if a.shape != b.shape:
        return False

    gap = int(max(0, max_edge_gap_px))
    if bbox_a is not None and bbox_b is not None:
        if _bbox_separated_by_more_than(bbox_a, bbox_b, gap):
            return False
        bounds = _expanded_union_bounds(a.shape, bbox_a, bbox_b, gap)
        if bounds is not None:
            y0, y1, x0, x1 = bounds
            a = a[y0:y1, x0:x1]
            b = b[y0:y1, x0:x1]

    if a.size == 0 or b.size == 0 or not a.any() or not b.any():
        return False
    if np.any(a & b):
        return True
    if gap <= 0:
        return False
    import cv2

    kernel = _dilation_kernel(gap)
    dilated = cv2.dilate(a.astype(np.uint8), kernel, iterations=1).astype(bool)
    return bool(np.any(dilated & b))


def _bbox_separated_by_more_than(
    bbox_a: tuple[float, float, float, float],
    bbox_b: tuple[float, float, float, float],
    gap: int,
) -> bool:
    edges_a = _bbox_edges(bbox_a)
    edges_b = _bbox_edges(bbox_b)
    if edges_a is None or edges_b is None:
        return False
    ax0, ay0, ax1, ay1 = edges_a
    bx0, by0, bx1, by1 = edges_b
    sep_x = max(bx0 - ax1, ax0 - bx1, 0.0)
    sep_y = max(by0 - ay1, ay0 - by1, 0.0)
    return sep_x > gap or sep_y > gap


def _expanded_union_bounds(
    shape: tuple[int, ...],
    bbox_a: tuple[float, float, float, float],
    bbox_b: tuple[float, float, float, float],
    gap: int,
) -> tuple[int, int, int, int] | None:
    edges_a = _bbox_edges(bbox_a)
    edges_b = _bbox_edges(bbox_b)
    if edges_a is None or edges_b is None or len(shape) < 2:
        return None

    height, width = int(shape[0]), int(shape[1])
    ax0, ay0, ax1, ay1 = edges_a
    bx0, by0, bx1, by1 = edges_b
    x0 = max(0, int(np.floor(min(ax0, bx0))) - gap)
    y0 = max(0, int(np.floor(min(ay0, by0))) - gap)
    x1 = min(width, int(np.ceil(max(ax1, bx1))) + gap)
    y1 = min(height, int(np.ceil(max(ay1, by1))) + gap)
    if x1 <= x0 or y1 <= y0:
        return None
    return y0, y1, x0, x1


def _bbox_edges(bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float] | None:
    try:
        x, y, w, h = (float(v) for v in bbox[:4])
    except Exception:
        return None
    if not all(np.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
        return None
    return x, y, x + w, y + h


@lru_cache(maxsize=16)
def _dilation_kernel(gap: int) -> np.ndarray:
    import cv2

    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap + 1, 2 * gap + 1))
