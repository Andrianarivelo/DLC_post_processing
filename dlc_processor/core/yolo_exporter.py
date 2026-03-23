"""Export annotated DLC frames in YOLO instance-detection format.

For each selected frame:
  1. Compute a padded bounding box from all valid keypoints of each animal.
  2. Normalise to [0, 1] relative to frame dimensions.
  3. Write a YOLO .txt label (class cx cy w h) alongside the JPG image.

Directory layout produced:
  <output_dir>/
    images/train/   *.jpg
    images/val/     *.jpg
    labels/train/   *.txt
    labels/val/     *.txt
    dataset.yaml
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from dlc_processor.core.dlc_loader import get_bodyparts

logger = logging.getLogger(__name__)


def export_yolo(
    video_path: str,
    animal_dfs: dict[str, pd.DataFrame],
    output_dir: str,
    padding_px: int = 20,
    val_frac: float = 0.20,
    frame_indices: Optional[list[int]] = None,
    class_map: Optional[dict[str, int]] = None,
) -> dict:
    """Generate a YOLO detection dataset from DLC tracking data + video.

    Parameters
    ----------
    video_path    : Source video file
    animal_dfs    : {animal_id: flat_df} — cleaned tracking data
    output_dir    : Root directory for dataset output
    padding_px    : Bounding-box padding in pixels
    val_frac      : Fraction of frames assigned to val split
    frame_indices : Which frames to export; None = all
    class_map     : {animal_id: class_index}; auto-assigned if None

    Returns
    -------
    dict with keys: n_train, n_val, output_dir, yaml_path
    """
    out_dir = Path(output_dir)
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    if class_map is None:
        class_map = {aid: i for i, aid in enumerate(sorted(animal_dfs.keys()))}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if frame_indices is None:
        # Export every frame that has at least one valid keypoint
        frame_indices = list(range(total_frames))

    # Shuffle and split
    indices = list(frame_indices)
    random.shuffle(indices)
    n_val = max(1, int(len(indices) * val_frac))
    val_set = set(indices[:n_val])

    n_train = n_val_written = 0

    for fi in sorted(indices):
        if fi < 0 or fi >= total_frames:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue

        labels: list[str] = []
        for animal_id, df in animal_dfs.items():
            if fi >= len(df):
                continue
            bbox = _compute_bbox(df, fi, fw, fh, padding_px)
            if bbox is None:
                continue
            cx, cy, bw, bh = bbox
            cls = class_map.get(animal_id, 0)
            labels.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if not labels:
            continue

        split = "val" if fi in val_set else "train"
        stem  = f"frame_{fi:07d}"
        img_path = out_dir / "images" / split / f"{stem}.jpg"
        lbl_path = out_dir / "labels" / split / f"{stem}.txt"

        cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        lbl_path.write_text("\n".join(labels))

        if split == "val":
            n_val_written += 1
        else:
            n_train += 1

    cap.release()

    # Write dataset.yaml
    yaml_path = out_dir / "dataset.yaml"
    class_names = [aid for aid, _ in sorted(class_map.items(), key=lambda kv: kv[1])]
    yaml_path.write_text(
        f"path: {out_dir.resolve()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"nc: {len(class_names)}\n"
        f"names: {class_names}\n"
    )

    logger.info(
        "YOLO export complete — train=%d  val=%d  → %s",
        n_train, n_val_written, out_dir,
    )
    return {
        "n_train": n_train,
        "n_val": n_val_written,
        "output_dir": str(out_dir),
        "yaml_path": str(yaml_path),
    }


def _compute_bbox(
    df: pd.DataFrame,
    frame_idx: int,
    fw: int,
    fh: int,
    padding_px: int,
) -> Optional[tuple[float, float, float, float]]:
    """Return (cx, cy, w, h) normalised to [0,1], or None if no valid keypoints."""
    bps = get_bodyparts(df)
    row = df.iloc[frame_idx]
    xs, ys = [], []
    for bp in bps:
        x = row.get(f"{bp}_x", np.nan)
        y = row.get(f"{bp}_y", np.nan)
        if not (np.isnan(x) or np.isnan(y)):
            xs.append(float(x))
            ys.append(float(y))

    if not xs:
        return None

    x1 = max(0, min(xs) - padding_px)
    y1 = max(0, min(ys) - padding_px)
    x2 = min(fw - 1, max(xs) + padding_px)
    y2 = min(fh - 1, max(ys) + padding_px)

    cx = (x1 + x2) / 2 / fw
    cy = (y1 + y2) / 2 / fh
    bw = (x2 - x1) / fw
    bh = (y2 - y1) / fh

    if bw <= 0 or bh <= 0:
        return None
    return cx, cy, bw, bh
