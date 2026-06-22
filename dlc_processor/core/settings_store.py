"""Persistent user settings for DLC Processor."""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SETTINGS_DIR = Path.home() / ".dlc_processor"
_SETTINGS_FILE = _SETTINGS_DIR / "settings.json"
DEFAULT_DLC_CONDA_ENV = "dcl3"
DEFAULT_DLC_EXECUTION_MODE = "subprocess"

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
    "last_session_time_files": [],
    "last_session_mask_files": [],
    # Batch framewise export
    "batch_export_output_mode": "time_file_folder",
    "batch_export_custom_dir": "",
    "batch_summary_group_by": "animal",
    "batch_summary_animal_filter": "",
    "batch_summary_mouse_id_filter": "",
    "batch_summary_plot_style": "box_strip",
    "batch_summary_comparison": "holm_ttest",
    "batch_summary_color_map": "",
    "batch_export_position_maps": True,
    "batch_compute_social": True,
    "batch_fix_identity_from_masks": False,
    "batch_generate_summary": True,
    "batch_analysis_frame_window_enabled": False,
    "batch_analysis_start_frame": 0,
    "batch_analysis_end_frame": 0,
    # Overlay video export
    "video_export_frame_window_enabled": False,
    "video_export_start_frame": 0,
    "video_export_end_frame": 0,
    # Misc
    "fps": 30.0,
    "calibration_px_per_cm": 0.0,
    # Inference panel
    "inference_panel": {
        "video_paths": [],
        "config_path": "",
        "checkpoint_path": "",
        "modelprefix": "",
        "conda_env_name": DEFAULT_DLC_CONDA_ENV,
        "python_exe": "",
        "execution_mode": DEFAULT_DLC_EXECUTION_MODE,
        "engine": "auto",
        "videotype": "",
        "destfolder": "",
        "use_gpu": True,
        "gpu_index": 0,
        "device": "",
        "use_openvino": "",
        "batchsize": 0,
        "save_as_csv": True,
        "overwrite": False,
        "load_results": True,
        "shuffle": 1,
        "trainingsetindex": 0,
        "snapshot_index": -1,
        "detector_snapshot_index": -1,
        "auto_track": False,
        "n_tracks": 0,
        "calibrate": False,
        "identity_only": False,
        "cropping_enabled": False,
        "crop_x1": 0,
        "crop_x2": 0,
        "crop_y1": 0,
        "crop_y2": 0,
        "dynamic_enabled": False,
        "dynamic_threshold": 0.5,
        "dynamic_margin": 10,
        "tfgpuinference": True,
        "allow_growth": False,
        "use_shelve": False,
        "robust_nframes": False,
        "extra_kwargs_json": "",
        "current_section": "inputs",
    },
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
    settings = copy.deepcopy(DEFAULT_SETTINGS)
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
