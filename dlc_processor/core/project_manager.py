"""DLC Processor project persistence and folder scanning."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
_DLC_EXT = {".h5", ".hdf5", ".csv"}
_MASK_JSON_SUFFIXES = ("_masks_coco.json", "_mask_polygons.json")
_TIME_EXT = {".csv", ".txt", ".tsv"}


def scan_folder(folder: Path) -> tuple[list[str], list[str]]:
    """Scan *folder* for video and DLC files, sorted alphabetically.

    Returns (video_paths, dlc_paths).
    """
    videos, dlc_files, _mask_files, _time_files = scan_folder_with_sidecars(folder)
    return videos, dlc_files


def scan_folder_with_masks(folder: Path) -> tuple[list[str], list[str], list[str]]:
    """Scan *folder* for video, DLC tracking, and mask files.
    Returns (video_paths, dlc_paths, mask_paths).
    """
    videos, dlc_files, mask_files, _time_files = scan_folder_with_sidecars(folder)
    return videos, dlc_files, mask_files


def scan_folder_with_sidecars(folder: Path) -> tuple[list[str], list[str], list[str], list[str]]:
    """Scan *folder* for video, DLC tracking, mask, and time files.

    LISBET/MouseTracker output folders can keep the source video outside the
    output directory. In that case the source path is recovered from
    ``tracking_manifest.json`` or ``*_tracking_meta.json``.
    """
    folder = Path(folder)
    lisbet = discover_lisbet_output(folder)

    videos = sorted(
        str(p) for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in _VIDEO_EXT
    )
    dlc_files = sorted(
        str(p) for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in _DLC_EXT
        and not _looks_like_time_file(p)
        and not _looks_like_metadata_table(p)
    )
    mask_files = sorted(
        str(p) for p in folder.iterdir()
        if p.is_file() and _looks_like_mask_json(p)
    )
    time_files = sorted(
        str(p) for p in folder.iterdir()
        if p.is_file() and _looks_like_time_file(p)
    )
    if lisbet["videos"] or lisbet["dlc_files"] or lisbet["mask_files"] or lisbet["time_files"]:
        videos = _merge_paths(lisbet["videos"], videos)
        dlc_files = _merge_paths(lisbet["dlc_files"], dlc_files)
        mask_files = _merge_paths(lisbet["mask_files"], mask_files)
        time_files = _merge_paths(lisbet["time_files"], time_files)
    return videos, dlc_files, mask_files, time_files


def discover_lisbet_output(path: Path) -> dict[str, list[str]]:
    """Return source videos, tracking CSVs/H5s, and COCO masks from an export."""
    p = Path(path)
    folder = p if p.is_dir() else p.parent
    videos: list[str] = []
    dlc_files: list[str] = []
    mask_files: list[str] = []
    time_files: list[str] = []

    def add_file(target: str | Path, bucket: list[str]) -> None:
        if not target:
            return
        item = Path(target)
        if not item.is_absolute():
            item = folder / item
        if item.exists():
            text = str(item)
            if text not in bucket:
                bucket.append(text)

    manifest = folder / "tracking_manifest.json"
    if manifest.exists():
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            logger.debug("Could not read tracking manifest: %s", manifest, exc_info=True)
            payload = {}
        if isinstance(payload, dict):
            for item in payload.get("csv_outputs", []) or []:
                add_file(item, dlc_files)
            for item in payload.get("coco_outputs", []) or []:
                add_file(item, mask_files)
            for item in payload.get("time_outputs", []) or []:
                add_file(item, time_files)
            config = payload.get("config") or {}
            for item in config.get("input_paths", []) or []:
                add_file(item, videos)

    for meta in sorted(folder.glob("*_tracking_meta.json")):
        try:
            payload = json.loads(meta.read_text(encoding="utf-8"))
        except Exception:
            logger.debug("Could not read tracking metadata: %s", meta, exc_info=True)
            continue
        if isinstance(payload, dict):
            add_file(payload.get("source_video", ""), videos)

    for candidate in sorted(folder.glob("*_tracking.csv")) + sorted(folder.glob("*_tracking.h5")):
        add_file(candidate, dlc_files)
    for candidate in sorted(folder.glob("*.json")):
        if _looks_like_mask_json(candidate):
            add_file(candidate, mask_files)
    for candidate in sorted(folder.iterdir()):
        if candidate.is_file() and _looks_like_time_file(candidate):
            add_file(candidate, time_files)

    if p.is_file():
        if _looks_like_time_file(p):
            add_file(p, time_files)
        elif p.suffix.lower() in _DLC_EXT:
            if not _looks_like_metadata_table(p):
                add_file(p, dlc_files)
        elif _looks_like_mask_json(p):
            add_file(p, mask_files)
        elif p.suffix.lower() in _VIDEO_EXT:
            add_file(p, videos)

    return {
        "videos": sorted(videos),
        "dlc_files": sorted(dlc_files),
        "mask_files": sorted(mask_files),
        "time_files": sorted(time_files),
    }


def _looks_like_mask_json(path: Path) -> bool:
    name = path.name.lower()
    return path.suffix.lower() == ".json" and (
        name.endswith(_MASK_JSON_SUFFIXES)
        or ("mask" in name and "coco" in name)
    )


def _looks_like_time_file(path: Path) -> bool:
    if path.suffix.lower() not in _TIME_EXT:
        return False
    name = path.name.lower()
    return any(
        token in name
        for token in (
            "frame_time",
            "frame-times",
            "frame_times",
            "aligned_frame",
            "timestamp",
            "time_s",
        )
    )


def _looks_like_metadata_table(path: Path) -> bool:
    name = path.name.lower()
    return path.suffix.lower() == ".csv" and (
        name in {"dlc_metadata.csv", "metadata.csv"}
        or name.endswith("_metadata.csv")
        or name.endswith("_metadata_table.csv")
    )


def is_metadata_table_file(path: str | Path) -> bool:
    return _looks_like_metadata_table(Path(path))


def _merge_paths(*groups: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for path in group:
            if path in seen:
                continue
            seen.add(path)
            out.append(path)
    return out


def save_project(path: Path, data: dict[str, Any]) -> None:
    """Save a project to a .dlcproj JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "version": 1,
        "video_files": data.get("video_files", []),
        "dlc_files": data.get("dlc_files", []),
        "dlc_animal_ids_by_path": data.get("dlc_animal_ids_by_path", {}),
        "mask_files": data.get("mask_files", []),
        "time_files": data.get("time_files", []),
        "config_path": data.get("config_path", ""),
        "skeleton_edges": [list(e) for e in data.get("skeleton_edges", [])],
        "calibration_px_per_cm": data.get("calibration_px_per_cm", 0.0),
        "video_calibrations_px_per_cm": data.get("video_calibrations_px_per_cm", {}),
        "edited_dlc_files": data.get("edited_dlc_files", {}),
        "social_behavior_cache": data.get("social_behavior_cache", {}),
        "metadata_rows": data.get("metadata_rows", []),
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
