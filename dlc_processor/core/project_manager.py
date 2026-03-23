"""DLC Processor project persistence and folder scanning."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
_DLC_EXT = {".h5", ".hdf5", ".csv"}


def scan_folder(folder: Path) -> tuple[list[str], list[str]]:
    """Scan *folder* for video and DLC files, sorted alphabetically.

    Returns (video_paths, dlc_paths).
    """
    videos = sorted(
        str(p) for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in _VIDEO_EXT
    )
    dlc_files = sorted(
        str(p) for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in _DLC_EXT
    )
    return videos, dlc_files


def save_project(path: Path, data: dict[str, Any]) -> None:
    """Save a project to a .dlcproj JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "version": 1,
        "video_files": data.get("video_files", []),
        "dlc_files": data.get("dlc_files", []),
        "config_path": data.get("config_path", ""),
        "skeleton_edges": [list(e) for e in data.get("skeleton_edges", [])],
        "calibration_px_per_cm": data.get("calibration_px_per_cm", 0.0),
        "fps": data.get("fps", 25.0),
        "settings": data.get("settings", {}),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    logger.info("Project saved -> %s", path)


def load_project(path: Path) -> dict[str, Any]:
    """Load a project from a .dlcproj JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    # Normalize skeleton edges to tuples
    data["skeleton_edges"] = [
        tuple(e) for e in data.get("skeleton_edges", [])
    ]
    logger.info("Project loaded <- %s", path)
    return data
