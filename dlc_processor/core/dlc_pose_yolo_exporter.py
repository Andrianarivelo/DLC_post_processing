"""Export a multianimal DeepLabCut project as a YOLO Pose dataset.

Reads:
  <project>/config.yaml             — individuals, multianimalbodyparts, skeleton
  <project>/labeled-data/<video>/CollectedData_<scorer>.h5
  <project>/labeled-data/<video>/img*.png

Writes a YOLO pose dataset that ultralytics can train directly:

  <output>/
    images/train/   *.jpg
    images/val/     *.jpg
    labels/train/   *.txt   (one row per visible animal:
                              "<class> cx cy w h px1 py1 v1 ... pxK pyK vK")
    labels/val/     *.txt
    dataset.yaml

The dataset.yaml includes:
  - kpt_shape: [K, 3]            # K keypoints, (x, y, visibility)
  - flip_idx: [...]              # auto-detected from left_/right_ pairings
  - skeleton: [[i, j], ...]      # zero-based bodypart-index pairs
  - names:    {0: 'mouse'}       # all individuals collapse to a single class
                                 # (identity is reserved for the tracker;
                                 #  pose models train per-instance, not per-id)

A separate ``names`` mode (one class per individual) is available via
``one_class_per_individual=True`` for projects where the user wants identity
classes baked into the model.
"""

from __future__ import annotations

import logging
import random
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# ── Project loading ──────────────────────────────────────────────────────────


def load_dlc_project_config(project_dir: str | Path) -> dict:
    """Read ``config.yaml`` from a DLC project root."""
    project_root = Path(project_dir)
    config_path = project_root / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"DLC config.yaml not found at {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    cfg["__project_root__"] = str(project_root.resolve())
    return cfg


def project_individuals(cfg: dict) -> list[str]:
    return [str(name) for name in (cfg.get("individuals") or []) if name]


def project_bodyparts(cfg: dict) -> list[str]:
    bps = cfg.get("multianimalbodyparts") or cfg.get("bodyparts") or []
    if isinstance(bps, str) and bps == "MULTI!":
        return []
    return [str(bp) for bp in bps if bp]


def project_skeleton(cfg: dict) -> list[tuple[str, str]]:
    edges_raw = cfg.get("skeleton") or []
    edges: list[tuple[str, str]] = []
    for edge in edges_raw:
        if isinstance(edge, (list, tuple)) and len(edge) == 2:
            a, b = edge
            edges.append((str(a), str(b)))
    return edges


# ── Helpers ──────────────────────────────────────────────────────────────────


def _flip_index(bodyparts: list[str]) -> list[int]:
    """Auto-detect a YOLO ``flip_idx`` from left_/right_ prefixes.

    YOLO's flip augmentation needs an index list of the same length as the
    keypoint count. Index ``i`` should map to the keypoint that is its mirror
    image. Bodyparts without a mirror map to themselves.
    """
    name_to_idx = {bp: i for i, bp in enumerate(bodyparts)}
    flip_idx = list(range(len(bodyparts)))
    for i, bp in enumerate(bodyparts):
        lower = bp.lower()
        mirror: Optional[str] = None
        if lower.startswith("left_"):
            mirror = "right_" + bp[5:]
        elif lower.startswith("right_"):
            mirror = "left_" + bp[6:]
        elif lower.startswith("l_"):
            mirror = "r_" + bp[2:]
        elif lower.startswith("r_"):
            mirror = "l_" + bp[2:]
        if mirror is not None and mirror in name_to_idx:
            flip_idx[i] = name_to_idx[mirror]
    return flip_idx


def _skeleton_to_indices(
    edges: list[tuple[str, str]],
    bodyparts: list[str],
) -> list[list[int]]:
    """Map (name, name) edges to ultralytics ``[[i+1, j+1], ...]`` index pairs.

    Ultralytics uses **1-based** indexing for the skeleton field in dataset.yaml.
    """
    name_to_idx = {bp: i + 1 for i, bp in enumerate(bodyparts)}
    out: list[list[int]] = []
    for a, b in edges:
        if a in name_to_idx and b in name_to_idx:
            out.append([name_to_idx[a], name_to_idx[b]])
    return out


def _bbox_from_keypoints(
    points: list[tuple[float, float]],
    fw: int,
    fh: int,
    padding_frac: float,
) -> Optional[tuple[float, float, float, float]]:
    """Return a normalized (cx, cy, w, h) bbox enclosing the visible keypoints."""
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1 = min(xs)
    y1 = min(ys)
    x2 = max(xs)
    y2 = max(ys)
    pad_x = max(8.0, (x2 - x1) * padding_frac)
    pad_y = max(8.0, (y2 - y1) * padding_frac)
    x1 = max(0.0, x1 - pad_x)
    y1 = max(0.0, y1 - pad_y)
    x2 = min(float(fw - 1), x2 + pad_x)
    y2 = min(float(fh - 1), y2 + pad_y)
    if x2 <= x1 or y2 <= y1:
        return None
    cx = (x1 + x2) / 2.0 / float(fw)
    cy = (y1 + y2) / 2.0 / float(fh)
    bw = (x2 - x1) / float(fw)
    bh = (y2 - y1) / float(fh)
    return cx, cy, bw, bh


def _row_to_pose_label(
    row: pd.Series,
    bodyparts: list[str],
    individual: str,
    fw: int,
    fh: int,
    padding_frac: float,
    class_index: int,
) -> Optional[str]:
    """Build one YOLO pose label row from a DLC labeled-data row."""
    visible_points: list[tuple[float, float]] = []
    kp_tokens: list[str] = []
    for bp in bodyparts:
        x = row.get((individual, bp, "x"))
        y = row.get((individual, bp, "y"))
        if x is None or y is None or not np.isfinite(x) or not np.isfinite(y):
            kp_tokens.extend(("0.000000", "0.000000", "0"))
            continue
        # Visibility: training labels have no likelihood, so present=2.
        visible_points.append((float(x), float(y)))
        kp_tokens.extend(
            (
                f"{float(x) / float(fw):.6f}",
                f"{float(y) / float(fh):.6f}",
                "2",
            )
        )
    bbox = _bbox_from_keypoints(visible_points, fw, fh, padding_frac)
    if bbox is None:
        return None
    cx, cy, bw, bh = bbox
    return (
        f"{class_index} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} " + " ".join(kp_tokens)
    )


# ── Main exporter ────────────────────────────────────────────────────────────


def export_dlc_to_yolo_pose(
    project_dir: str | Path,
    output_dir: str | Path,
    *,
    skeleton_edges: Optional[list[tuple[str, str]]] = None,
    bodyparts_override: Optional[list[str]] = None,
    one_class_per_individual: bool = False,
    val_frac: float = 0.20,
    padding_frac: float = 0.20,
    seed: int = 1234,
) -> dict:
    """Convert a multianimal DLC project into a YOLO Pose dataset.

    Parameters
    ----------
    project_dir
        Root of the DLC project (the folder that contains ``config.yaml``).
    output_dir
        Destination folder for the YOLO dataset.
    skeleton_edges
        Optional skeleton (list of name pairs). When omitted, the project's
        ``config.yaml`` skeleton is used. Use the ``SkeletonEditorDialog`` to
        let the user edit and pass the result here.
    bodyparts_override
        Optional explicit bodypart ordering. When omitted, the project's
        ``multianimalbodyparts`` order is used.
    one_class_per_individual
        If True, each individual gets its own class id (used for projects
        where identity should be baked into the model). Default False —
        all animals share class 0 ("mouse"), and identity is recovered by
        the runtime tracker.
    val_frac
        Fraction of labeled images reserved for validation.
    padding_frac
        Bounding-box padding as a fraction of the keypoint span.
    seed
        Deterministic train/val split seed.

    Returns
    -------
    dict with: n_train, n_val, n_skipped, individuals, bodyparts,
    skeleton, output_dir, yaml_path
    """
    project_root = Path(project_dir)
    cfg = load_dlc_project_config(project_root)
    individuals = project_individuals(cfg)
    if not individuals:
        raise ValueError(
            f"DLC project at {project_root} has no 'individuals' key (not multianimal?)"
        )
    bodyparts = list(bodyparts_override) if bodyparts_override else project_bodyparts(cfg)
    if not bodyparts:
        raise ValueError(
            f"DLC project at {project_root} has no multianimalbodyparts."
        )
    edges = list(skeleton_edges) if skeleton_edges is not None else project_skeleton(cfg)

    out_root = Path(output_dir)
    for split in ("train", "val"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    if one_class_per_individual:
        class_map = {name: i for i, name in enumerate(individuals)}
        class_names = list(individuals)
    else:
        class_map = {name: 0 for name in individuals}
        class_names = ["mouse"]

    scorer = str(cfg.get("scorer", "")).strip() or None

    labeled_root = project_root / "labeled-data"
    if not labeled_root.exists():
        raise FileNotFoundError(f"labeled-data/ not found under {project_root}")

    rng = random.Random(seed)
    n_train = 0
    n_val = 0
    n_skipped = 0

    video_dirs = sorted(p for p in labeled_root.iterdir() if p.is_dir())
    logger.info(
        "DLC→YOLO pose export starting: project=%s, video_dirs=%d, individuals=%s, bodyparts=%s",
        project_root.name, len(video_dirs), individuals, bodyparts,
    )

    for video_dir in video_dirs:
        # Find the CollectedData_<scorer>.h5 file.
        h5_files = list(video_dir.glob("CollectedData_*.h5"))
        if not h5_files:
            logger.debug("No CollectedData_*.h5 in %s; skipping", video_dir)
            continue
        h5_path = h5_files[0]
        try:
            df = pd.read_hdf(str(h5_path))
        except Exception:
            logger.exception("Could not read %s", h5_path)
            continue

        if not isinstance(df.index, pd.MultiIndex):
            logger.warning("Unexpected index in %s; expected MultiIndex with image filename", h5_path)
            continue

        for index_value in df.index:
            row = df.loc[index_value]
            # Image path: index typically (labeled-data, <video>, img0001.png).
            if isinstance(index_value, tuple):
                img_name = str(index_value[-1])
            else:
                img_name = str(index_value)
            src_image = video_dir / img_name
            if not src_image.exists():
                n_skipped += 1
                continue

            frame = cv2.imread(str(src_image), cv2.IMREAD_COLOR)
            if frame is None:
                n_skipped += 1
                continue
            fh, fw = frame.shape[:2]

            label_lines: list[str] = []
            for individual in individuals:
                line = _row_to_pose_label(
                    row,
                    bodyparts,
                    individual,
                    fw,
                    fh,
                    padding_frac,
                    class_map[individual],
                )
                if line is not None:
                    label_lines.append(line)
            if not label_lines:
                n_skipped += 1
                continue

            split = "val" if rng.random() < float(val_frac) else "train"
            stem = f"{video_dir.name}_{Path(img_name).stem}"
            dst_image = out_root / "images" / split / f"{stem}.jpg"
            dst_label = out_root / "labels" / split / f"{stem}.txt"
            cv2.imwrite(str(dst_image), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            dst_label.write_text("\n".join(label_lines))
            if split == "val":
                n_val += 1
            else:
                n_train += 1

    if n_train == 0 and n_val == 0:
        raise RuntimeError(
            f"DLC→YOLO pose export wrote zero images for {project_root}; "
            "check that labeled-data/ contains CollectedData_*.h5 files with valid annotations."
        )

    flip_idx = _flip_index(bodyparts)
    skeleton_yaml = _skeleton_to_indices(edges, bodyparts)

    yaml_path = out_root / "dataset.yaml"
    yaml_payload = {
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(class_names),
        "names": {i: name for i, name in enumerate(class_names)},
        "kpt_shape": [len(bodyparts), 3],
        "flip_idx": flip_idx,
        "keypoint_names": list(bodyparts),
        "skeleton": skeleton_yaml,
    }
    with yaml_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(yaml_payload, fh, sort_keys=False)

    logger.info(
        "DLC→YOLO pose export complete: train=%d val=%d skipped=%d → %s",
        n_train, n_val, n_skipped, out_root,
    )

    return {
        "n_train": n_train,
        "n_val": n_val,
        "n_skipped": n_skipped,
        "individuals": individuals,
        "bodyparts": bodyparts,
        "skeleton": edges,
        "flip_idx": flip_idx,
        "output_dir": str(out_root.resolve()),
        "yaml_path": str(yaml_path.resolve()),
        "class_names": class_names,
    }
