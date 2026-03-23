"""Persistent user settings for DLC Processor."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SETTINGS_DIR = Path.home() / ".dlc_processor"
_SETTINGS_FILE = _SETTINGS_DIR / "settings.json"

DEFAULT_SETTINGS: dict[str, Any] = {
    # Animal overlay colors (BGR tuples for OpenCV)
    "animal_colors_bgr": [
        [255, 120, 50], [50, 200, 255], [80, 240, 80], [80, 80, 255],
        [240, 100, 240], [240, 240, 80], [150, 80, 240], [80, 200, 200],
    ],
    # Plot appearance
    "plot_bg_color": "#1e1e2e",
    "plot_fg_color": "#cdd6f4",
    "plot_line_width": 1.5,
    "plot_antialias": False,
    "plot_grid_alpha": 0.15,
    # Skeleton
    "skeleton_edges": [],
    # Directories
    "last_dlc_dir": "",
    "last_video_dir": "",
    "last_project_path": "",
    # Recent files/projects (most recent first, max 10)
    "recent_paths": [],
    # Last session files (loaded together via "Load Last Session")
    "last_session_dlc_files": [],
    "last_session_video_files": [],
    # Misc
    "fps": 25.0,
    "calibration_px_per_cm": 0.0,
}

_MAX_RECENT = 10


def add_recent_path(settings: dict[str, Any], path: str) -> None:
    """Push *path* to the front of the recent list, deduplicating."""
    recents: list[str] = settings.get("recent_paths", [])
    path = str(Path(path).resolve())
    if path in recents:
        recents.remove(path)
    recents.insert(0, path)
    settings["recent_paths"] = recents[:_MAX_RECENT]


def load_settings() -> dict[str, Any]:
    """Load settings from disk, falling back to defaults for missing keys."""
    settings = DEFAULT_SETTINGS.copy()
    if _SETTINGS_FILE.exists():
        try:
            with open(_SETTINGS_FILE, "r", encoding="utf-8") as fh:
                saved = json.load(fh)
            settings.update(saved)
        except Exception as exc:
            logger.warning("Could not load settings: %s", exc)
    return settings


def save_settings(settings: dict[str, Any]) -> None:
    """Persist settings to disk."""
    try:
        _SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(_SETTINGS_FILE, "w", encoding="utf-8") as fh:
            json.dump(settings, fh, indent=2)
    except Exception as exc:
        logger.warning("Could not save settings: %s", exc)
