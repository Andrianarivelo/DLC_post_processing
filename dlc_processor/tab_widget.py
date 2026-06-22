"""DLC Processor tab — right-side vertical activity-bar layout.

Layout
------
  ┌──────────────────────────────────────────────────────┐
  │  menu bar                                            │
  ├───────────────────────────┬────────────┬────┐
  │                           │            │    │
  │   central (stretch)       │ slide      │ AB │  ← vertical bar
  │                           │ panel      │    │
  └───────────────────────────┴────────────┴────┘

Activity buttons
----------------
  load      — Load Files panel
  clean     — Data Cleaning panel
  kinematics — Kinematics panel
  social    — Social Behaviors panel
  export    — Export / Inference panel
  refine    — ID Refinement panel
  plots     — toggle plot panel visibility
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QAction, QColor, QDragEnterEvent, QDragMoveEvent, QDropEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QMenu,
    QMenuBar,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from shared.sidebar_layout import SidebarLayout

from dlc_processor.ui.load_panel       import LoadPanel
from dlc_processor.ui.cleaning_panel   import CleaningPanel
from dlc_processor.ui.kinematics_panel import KinematicsPanel
from dlc_processor.ui.social_panel     import SocialPanel
from dlc_processor.ui.video_panel      import VideoPanel
from dlc_processor.ui.plot_panel       import PlotPanel
from dlc_processor.ui.refinement_panel import RefinementPanel
from dlc_processor.ui.export_panel     import ExportPanel
from dlc_processor.ui.batch_panel      import BatchPanel
from dlc_processor.ui.inference_panel  import InferencePanel
from dlc_processor.ui.roi_panel        import ROIPanel
from dlc_processor.core.settings_store import load_settings, save_settings, add_recent_path
from dlc_processor.core.project_manager import (
    scan_folder_with_sidecars, save_project, load_project,
)
from dlc_processor.core.time_loader import align_times_to_dfs, is_frame_time_file

logger = logging.getLogger(__name__)

_MENU_QSS = """
QMenuBar {
    background: #181825; color: #cdd6f4;
    border-bottom: 1px solid #313244; font-size: 11px;
}
QMenuBar::item:selected { background: #313244; }
QMenu {
    background: #1e1e2e; color: #cdd6f4; border: 1px solid #313244;
}
QMenu::item:selected { background: #313244; }
"""


class DLCProcessorTab(SidebarLayout):
    """Main DLC Processor tab — sidebar-based layout."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(bar_position="right", parent=parent)
        self._animal_dfs: dict = {}
        self._source_animal_dfs: dict = {}
        self._source_behavior_arrays: dict = {}
        self._modified_dlc_cache: dict[str, dict] = {}
        self._social_behavior_cache: dict[str, dict] = {}
        self._fps = 30.0
        self._custom_time: object = None  # Optional np.ndarray of timestamps
        self._mask_store = None
        self._panel_time_override = None
        self._analysis_time_range_enabled = False
        self._analysis_start_s = 0.0
        self._analysis_end_s = 0.0
        self._current_video_path = ""
        self._video_calibrations_px_per_cm: dict[str, float] = {}
        self._project_path = ""
        self._settings = load_settings()
        self.setAcceptDrops(True)
        self._build_panels()
        self._build_menu()
        self._connect_signals()
        self._apply_settings()

    # ── Panel construction ─────────────────────────────────────────────────────

    def _build_panels(self) -> None:
        self.load_panel       = LoadPanel()
        self.cleaning_panel   = CleaningPanel()
        self.kinematics_panel = KinematicsPanel()
        self.social_panel     = SocialPanel()
        self.batch_panel      = BatchPanel()
        self.export_panel     = ExportPanel()
        self.inference_panel  = InferencePanel()
        self.refinement_panel = RefinementPanel()
        self.roi_panel        = ROIPanel()
        self.video_panel      = VideoPanel()
        self.plot_panel       = PlotPanel()

        self.add_activity("load",       "file-plus",  "Load",       "Load Files",        self.load_panel)
        self.add_activity("clean",      "sliders",    "Clean",      "Data Cleaning",     self.cleaning_panel)
        self.add_activity("kinematics", "move",       "Kinematics", "Kinematics",        self.kinematics_panel)
        self.add_activity("social",     "users",      "Social",     "Social Behaviors",  self.social_panel)
        self.add_activity("batch",      "bar-chart-2", "Batch",     "Batch / Metadata",  self.batch_panel)
        self.add_activity("roi",        "crosshair",  "ROI",        "Regions of Interest", self.roi_panel)
        self.add_bar_separator()
        self.add_activity("refine",     "edit-3",     "Refine",     "ID Refinement",     self.refinement_panel)
        self.add_activity("infer",      "cpu",        "Infer",      "DLC Inference",     self.inference_panel)
        self.add_activity("export",     "download",   "Export",     "Export", self.export_panel)

        # Plot toggle
        self.add_bar_separator()
        btn_plots = self._bar.add_button("plots", "bar-chart-2", "Plots")
        btn_plots.pressed.connect(self._toggle_plot_panel)
        btn_plots.setProperty("active", "true")

        # Central: menu bar + splitter
        self._central_splitter = QSplitter(Qt.Orientation.Vertical)
        self._central_splitter.setStyleSheet(
            "QSplitter::handle { background: #313244; height: 3px; }"
        )
        self._central_splitter.addWidget(self.video_panel)
        self._central_splitter.addWidget(self.plot_panel)
        self._central_splitter.setSizes([400, 300])
        self._plot_visible = True

        # Wrapper: menu on top, splitter below
        central_w = QWidget()
        central_lay = QVBoxLayout(central_w)
        central_lay.setContentsMargins(0, 0, 0, 0)
        central_lay.setSpacing(0)
        self._menu_bar = QMenuBar()
        self._menu_bar.setStyleSheet(_MENU_QSS)
        central_lay.addWidget(self._menu_bar)
        central_lay.addWidget(self._central_splitter, 1)

        # Status bar
        self._status_bar = QLabel("Ready")
        self._status_bar.setStyleSheet(
            "color: #a6adc8; background: #181825; font-size: 10px;"
            " padding: 2px 6px; border-top: 1px solid #313244;"
        )
        self._status_bar.setFixedHeight(20)
        central_lay.addWidget(self._status_bar)

        self.set_central(central_w)

    # ── Menu bar ───────────────────────────────────────────────────────────────

    def _build_menu(self) -> None:
        mb = self._menu_bar

        # ── File ──
        file_menu = mb.addMenu("&File")

        act_new = QAction("New Project", self)
        act_new.setShortcut("Ctrl+N")
        act_new.triggered.connect(self._new_project)
        file_menu.addAction(act_new)

        act_open = QAction("Open Project…", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self._open_project)
        file_menu.addAction(act_open)

        self._recent_menu = QMenu("Open Recent", self)
        self._recent_menu.setStyleSheet(_MENU_QSS)
        file_menu.addMenu(self._recent_menu)
        self._rebuild_recent_menu()

        act_save = QAction("Save Project", self)
        act_save.setShortcut("Ctrl+S")
        act_save.triggered.connect(self._save_project)
        file_menu.addAction(act_save)

        act_saveas = QAction("Save Project As…", self)
        act_saveas.setShortcut("Ctrl+Shift+S")
        act_saveas.triggered.connect(self._save_project_as)
        file_menu.addAction(act_saveas)

        file_menu.addSeparator()

        act_last = QAction("Load Last Session", self)
        act_last.setShortcut("Ctrl+L")
        act_last.triggered.connect(self._load_last_session)
        file_menu.addAction(act_last)

        act_folder = QAction("Open Folder…", self)
        act_folder.setShortcut("Ctrl+Shift+O")
        act_folder.triggered.connect(self._open_folder)
        file_menu.addAction(act_folder)

        # ── Settings ──
        settings_menu = mb.addMenu("&Settings")

        act_anim_colors = QAction("Animal Colors…", self)
        act_anim_colors.triggered.connect(self._pick_animal_colors)
        settings_menu.addAction(act_anim_colors)

        act_plot_bg = QAction("Plot Background…", self)
        act_plot_bg.triggered.connect(self._pick_plot_bg)
        settings_menu.addAction(act_plot_bg)

        act_plot_fg = QAction("Plot Foreground…", self)
        act_plot_fg.triggered.connect(self._pick_plot_fg)
        settings_menu.addAction(act_plot_fg)

        settings_menu.addSeparator()

        act_analysis_range = QAction("Analysis Time Range...", self)
        act_analysis_range.triggered.connect(self._edit_analysis_time_range)
        settings_menu.addAction(act_analysis_range)

        # Register shortcuts on the widget so they work when tab is active
        for act in file_menu.actions() + settings_menu.actions():
            self.addAction(act)

    # ── Signal wiring ──────────────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        lp = self.load_panel
        lp.data_loaded.connect(self._on_data_loaded)
        lp.video_loaded.connect(self._on_video_loaded)
        lp.config_loaded.connect(self.export_panel.set_config_path)
        lp.config_loaded.connect(self.inference_panel.set_config_path)
        lp.time_loaded.connect(self._on_time_loaded)
        lp.masks_loaded.connect(self._on_masks_loaded)
        lp.files_changed.connect(self._sync_project_batch_count)
        lp.metadata_csv_loaded.connect(self._on_metadata_csv_loaded)

        self.cleaning_panel.cleaning_requested.connect(self._on_clean)
        self.cleaning_panel.impossible_fix_requested.connect(self._on_fix_impossible)
        self.cleaning_panel.identity_swap_requested.connect(self._on_identity_swap)
        self.cleaning_panel.identity_rename_requested.connect(self._on_identity_rename)
        self.cleaning_panel.mask_identity_fix_requested.connect(self._on_mask_identity_fix)
        self.cleaning_panel.reset_cleaning_requested.connect(self._on_reset_cleaning)
        self.cleaning_panel.calibration_changed.connect(self._on_calibration)
        self.cleaning_panel.track_editor_requested.connect(self._open_track_editor)
        self.cleaning_panel.range_selection_requested.connect(
            self._show_plot_cleaning_region
        )
        self.cleaning_panel.range_changed.connect(
            self.plot_panel.set_cleaning_region
        )
        self.plot_panel.cleaning_region_changed.connect(
            self.cleaning_panel.set_range
        )
        self.video_panel.range_start_flag_requested.connect(self.cleaning_panel.flag_range_start)
        self.video_panel.range_end_flag_requested.connect(self.cleaning_panel.flag_range_end)
        self.video_panel.range_clear_requested.connect(self.cleaning_panel.clear_range)
        self.video_panel.identity_swap_requested.connect(self.cleaning_panel.request_identity_swap)
        self.plot_panel.range_start_flag_requested.connect(self.cleaning_panel.flag_range_start)
        self.plot_panel.range_end_flag_requested.connect(self.cleaning_panel.flag_range_end)
        self.plot_panel.range_clear_requested.connect(self.cleaning_panel.clear_range)
        self.plot_panel.identity_swap_requested.connect(self.cleaning_panel.request_identity_swap)
        self.kinematics_panel.computed.connect(self._on_kinematics)
        self.kinematics_panel.heatmap_requested.connect(self._open_egocentric_heatmap)
        self.social_panel.detected.connect(self._on_social)
        self.social_panel.batch_process_requested.connect(self._on_social_batch_process_requested)
        self.social_panel.mask_contact_calibration_requested.connect(self._open_mask_contact_calibration)
        self.video_panel.frame_changed.connect(self.refinement_panel.set_current_frame)
        self.video_panel.frame_changed.connect(self.plot_panel.set_frame_cursor)
        self.video_panel.frame_changed.connect(self.roi_panel.set_current_frame)
        self.plot_panel.cursor_moved.connect(self._on_plot_cursor)
        self.refinement_panel.data_changed.connect(self._on_data_refined)
        self.export_panel.inference_requested.connect(self._on_inference_done)
        self.inference_panel.inference_finished.connect(self._on_inference_outputs_ready)
        self.export_panel.skeleton_edges_needed.connect(self._provide_skeleton_edges)
        self.export_panel.project_batch_requested.connect(self._on_project_batch_export_requested)
        self.batch_panel.batch_compute_requested.connect(self._on_batch_panel_compute_requested)
        self.roi_panel.rois_changed.connect(self._on_rois_changed)

    # ── Settings persistence ─────────────────────────────────────────────────

    def _apply_settings(self) -> None:
        """Apply loaded settings to panels."""
        s = self._settings
        import pyqtgraph as pg
        pg.setConfigOptions(
            antialias=s.get("plot_antialias", False),
            background=s.get("plot_bg_color", "#1e1e2e"),
            foreground=s.get("plot_fg_color", "#cdd6f4"),
        )
        edges = s.get("skeleton_edges", [])
        if edges:
            self.video_panel._skeleton_edges = [tuple(e) for e in edges]
        self.inference_panel.restore_state(s.get("inference_panel", {}))
        settings_calibrations = {
            str(k): float(v)
            for k, v in (s.get("video_calibrations_px_per_cm", {}) or {}).items()
            if float(v or 0.0) > 0
        }
        self._video_calibrations_px_per_cm.update(settings_calibrations)
        px_cm = float(s.get("calibration_px_per_cm", 0.0) or 0.0)
        if not self._current_video_path and px_cm > 0:
            self.cleaning_panel.set_calibration(px_cm, use_for_analysis=True)
        self._propagate_calibration()
        self._analysis_time_range_enabled = bool(s.get("analysis_time_range_enabled", False))
        self._analysis_start_s = float(s.get("analysis_start_s", 0.0) or 0.0)
        self._analysis_end_s = float(s.get("analysis_end_s", 0.0) or 0.0)
        self.social_panel.set_mask_contact_margin_percent(float(s.get("social_mask_contact_margin_percent", 5.0) or 5.0))
        self.export_panel.restore_batch_export_settings(s)
        self.batch_panel.restore_state(s)

    def _save_current_settings(self) -> None:
        """Persist current settings to disk."""
        save_settings(self._collect_current_settings())

    def _collect_current_settings(self) -> dict:
        """Return a fresh settings snapshot from all panels."""
        self._settings["skeleton_edges"] = [
            list(e) for e in self.video_panel._skeleton_edges
        ]
        self._settings["fps"] = self._fps
        self._settings["inference_panel"] = self.inference_panel.export_state()
        self._settings["analysis_time_range_enabled"] = self._analysis_time_range_enabled
        self._settings["analysis_start_s"] = self._analysis_start_s
        self._settings["analysis_end_s"] = self._analysis_end_s
        self._settings["video_calibrations_px_per_cm"] = dict(self._video_calibrations_px_per_cm)
        self._settings["calibration_px_per_cm"] = self.cleaning_panel.px_per_cm()
        self._settings["social_mask_contact_margin_percent"] = self.social_panel.mask_contact_margin_percent()
        self._settings.update(self.export_panel.batch_export_state())
        self._settings.update(self.batch_panel.export_state())
        if self._project_path:
            self._settings["last_project_path"] = self._project_path
        return dict(self._settings)

    # ── Recent files ─────────────────────────────────────────────────────────

    def _rebuild_recent_menu(self) -> None:
        """Populate the Open Recent submenu from settings."""
        self._recent_menu.clear()
        recents = self._settings.get("recent_paths", [])
        if not recents:
            act = self._recent_menu.addAction("(empty)")
            act.setEnabled(False)
            return
        for path in recents:
            name = Path(path).name
            act = self._recent_menu.addAction(name)
            act.setToolTip(path)
            act.triggered.connect(lambda checked=False, p=path: self._open_recent(p))
        self._recent_menu.addSeparator()
        act_clear = self._recent_menu.addAction("Clear Recent")
        act_clear.triggered.connect(self._clear_recent)

    def _open_recent(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            logger.warning("Recent path no longer exists: %s", path)
            return
        if p.suffix == ".dlcproj":
            self._project_path = path
            data = load_project(p)
            self._restore_project(data)
        elif is_frame_time_file(p):
            self.load_panel.add_time_path(path)
        elif p.suffix.lower() in {".h5", ".hdf5", ".csv"}:
            self.load_panel.add_dlc_path(path)
        elif p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm"}:
            self.load_panel.add_video_path(path)
        elif p.is_dir():
            videos, dlc_files, mask_files, time_files = scan_folder_with_sidecars(p)
            if videos or dlc_files or mask_files or time_files:
                self.load_panel.load_from_folder(videos, dlc_files, mask_files, time_files)

    def _clear_recent(self) -> None:
        self._settings["recent_paths"] = []
        save_settings(self._settings)
        self._rebuild_recent_menu()

    def _add_to_recent(self, path: str) -> None:
        add_recent_path(self._settings, path)
        save_settings(self._settings)
        self._rebuild_recent_menu()

    # ── Project system ────────────────────────────────────────────────────────

    def _new_project(self) -> None:
        self._animal_dfs = {}
        self._source_animal_dfs = {}
        self._source_behavior_arrays = {}
        self._modified_dlc_cache = {}
        self._social_behavior_cache = {}
        self._panel_time_override = None
        self._current_video_path = ""
        self._project_path = ""
        self.video_panel.set_data({})
        self.plot_panel.set_animal_dfs({})
        logger.info("New project created")

    def _open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open DLC Project", "",
            "DLC Project (*.dlcproj);;All Files (*)",
        )
        if not path:
            return
        self._project_path = path
        data = load_project(Path(path))
        self._restore_project(data)
        self._add_to_recent(path)
        logger.info("Project loaded: %s", path)

    def _restore_project(self, data: dict) -> None:
        """Apply a loaded project dict to all panels."""
        video_files = data.get("video_files", [])
        dlc_files = data.get("dlc_files", [])
        mask_files = data.get("mask_files", [])
        time_files = data.get("time_files", [])
        self._modified_dlc_cache = {}
        self._social_behavior_cache = {}
        self._fps = data.get("fps", 25.0)
        self._load_project_tracking_edits(data.get("edited_dlc_files", {}) or {})
        self._load_project_social_cache(data.get("social_behavior_cache", {}) or {})
        project_calibrations = data.get("video_calibrations_px_per_cm", {}) or {}
        if project_calibrations:
            self._video_calibrations_px_per_cm.update(
                {
                    str(k): float(v)
                    for k, v in project_calibrations.items()
                    if float(v or 0.0) > 0
                }
            )
            self._settings["video_calibrations_px_per_cm"] = dict(self._video_calibrations_px_per_cm)
        if video_files or dlc_files or mask_files or time_files:
            self.load_panel.load_from_folder(video_files, dlc_files, mask_files, time_files)
            saved_animals = data.get("dlc_animal_ids_by_path", {}) or {}
            if saved_animals and hasattr(self.load_panel, "_animal_ids_by_path"):
                self.load_panel._animal_ids_by_path.update(
                    {
                        str(path): [str(aid) for aid in (animals or []) if str(aid)]
                        for path, animals in saved_animals.items()
                    }
                )
            self._preload_project_dlc_metadata()
        self.batch_panel.restore_metadata_rows(data.get("metadata_rows", []) or [])

        config_path = data.get("config_path", "")
        if config_path:
            self.load_panel.set_config_path(config_path)

        edges = data.get("skeleton_edges", [])
        if edges:
            self.video_panel._skeleton_edges = [tuple(e) for e in edges]

        project_calibrations = data.get("video_calibrations_px_per_cm", {}) or {}
        if project_calibrations:
            self._video_calibrations_px_per_cm.update(
                {
                    str(k): float(v)
                    for k, v in project_calibrations.items()
                    if float(v or 0.0) > 0
                }
            )
            self._settings["video_calibrations_px_per_cm"] = dict(self._video_calibrations_px_per_cm)
        elif not self._current_video_path:
            px_cm = data.get("calibration_px_per_cm", 0.0)
            if px_cm > 0:
                self.cleaning_panel.set_calibration(px_cm, use_for_analysis=True)
                self._propagate_calibration()

        proj_settings = data.get("settings", {})
        if proj_settings:
            self._settings.update(proj_settings)
            self._apply_settings()
            self._reapply_source_data_if_available()

        if self._current_video_path:
            self._apply_calibration_for_video(self._current_video_path)
        self._try_restore_social_cache_for_active()

    def _preload_project_dlc_metadata(self) -> None:
        """Populate batch metadata animal rows without activating every file."""
        if not hasattr(self.load_panel, "preload_dlc_metadata"):
            return
        from PySide6.QtWidgets import QApplication

        def _progress(done: int, total: int, name: str) -> None:
            if total <= 0:
                return
            if name:
                self._status_bar.setText(f"Reading DLC metadata {done + 1}/{total}: {name}")
            QApplication.processEvents()

        loaded = self.load_panel.preload_dlc_metadata(progress_callback=_progress)
        if loaded:
            self._sync_project_batch_count()
            self._status_bar.setText(f"Loaded metadata for {loaded} DLC recording(s)")

    def _save_project(self) -> None:
        if not self._project_path:
            self._save_project_as()
            return
        self._do_save_project(Path(self._project_path))

    def _save_project_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save DLC Project", "",
            "DLC Project (*.dlcproj)",
        )
        if not path:
            return
        self._project_path = path
        self._do_save_project(Path(path))

    def _do_save_project(self, path: Path) -> None:
        self._ensure_project_dlc_metadata_for_save()
        edited_dlc_files = self._save_project_tracking_edits(path)
        self._remember_current_social_cache()
        social_behavior_cache = self._save_project_social_cache(path)
        current_settings = self._collect_current_settings()
        data = {
            "video_files": getattr(self.load_panel, "_video_paths", []),
            "dlc_files": getattr(self.load_panel, "_dlc_paths", []),
            "mask_files": getattr(self.load_panel, "_mask_paths", []),
            "time_files": getattr(self.load_panel, "_time_paths", []),
            "dlc_animal_ids_by_path": getattr(self.load_panel, "_animal_ids_by_path", {}),
            "config_path": self.load_panel.config_path,
            "skeleton_edges": self.video_panel._skeleton_edges,
            "calibration_px_per_cm": getattr(self.cleaning_panel, "_px_per_cm", 0.0),
            "video_calibrations_px_per_cm": dict(self._video_calibrations_px_per_cm),
            "edited_dlc_files": edited_dlc_files,
            "social_behavior_cache": social_behavior_cache,
            "metadata_rows": self.batch_panel.metadata_rows(),
            "fps": self._fps,
            "settings": {
                k: v for k, v in current_settings.items()
                if k not in ("last_dlc_dir", "last_video_dir", "last_project_path")
            },
        }
        save_project(path, data)
        self._add_to_recent(str(path))
        save_settings(current_settings)
        logger.info("Project saved: %s", path)

    def _ensure_project_dlc_metadata_for_save(self) -> None:
        records = getattr(self.load_panel, "paired_records", lambda: [])()
        needs_metadata = any(
            str(rec.get("dlc_path", "") or "")
            and not rec.get("animal_ids")
            and not rec.get("animal_dfs")
            for rec in records
        )
        if not needs_metadata:
            return
        loaded = self.load_panel.preload_dlc_metadata()
        if loaded:
            self._sync_project_batch_count()

    def _open_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Open Folder with Videos + DLC Files")
        if not folder:
            return
        videos, dlc_files, mask_files, time_files = scan_folder_with_sidecars(Path(folder))
        if videos or dlc_files or mask_files or time_files:
            self.load_panel.load_from_folder(videos, dlc_files, mask_files, time_files)
            self._add_to_recent(folder)
        logger.info(
            "Opened folder: %d videos, %d DLC files, %d mask files, %d time files",
            len(videos), len(dlc_files), len(mask_files), len(time_files),
        )

    def _on_metadata_csv_loaded(self, path: str) -> None:
        try:
            count = self.batch_panel.import_metadata_csv(path)
            self._sync_project_batch_count()
            self._status_bar.setText(f"Imported metadata: {Path(path).name} ({count} rows)")
        except Exception as exc:
            logger.exception("Metadata import failed: %s", path)
            self._status_bar.setText(f"Metadata import failed: {exc}")

    # ── Settings dialogs ──────────────────────────────────────────────────────

    def _pick_animal_colors(self) -> None:
        color = QColorDialog.getColor(QColor("#89b4fa"), self, "Pick Animal Color")
        if color.isValid():
            bgr = [color.blue(), color.green(), color.red()]
            colors = self._settings.get("animal_colors_bgr", [])
            colors.append(bgr)
            self._settings["animal_colors_bgr"] = colors
            self._save_current_settings()

    def _pick_plot_bg(self) -> None:
        cur = QColor(self._settings.get("plot_bg_color", "#1e1e2e"))
        color = QColorDialog.getColor(cur, self, "Plot Background")
        if color.isValid():
            self._settings["plot_bg_color"] = color.name()
            self._apply_settings()
            self._save_current_settings()

    def _pick_plot_fg(self) -> None:
        cur = QColor(self._settings.get("plot_fg_color", "#cdd6f4"))
        color = QColorDialog.getColor(cur, self, "Plot Foreground")
        if color.isValid():
            self._settings["plot_fg_color"] = color.name()
            self._apply_settings()
            self._save_current_settings()

    def _edit_analysis_time_range(self) -> None:
        """Configure an optional time window used by analysis panels."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Analysis Time Range")
        lay = QVBoxLayout(dlg)

        chk_enabled = QCheckBox("Restrict analysis to this time range")
        chk_enabled.setChecked(self._analysis_time_range_enabled)
        lay.addWidget(chk_enabled)

        form = QFormLayout()
        spin_start = QDoubleSpinBox()
        spin_start.setRange(-1_000_000.0, 1_000_000_000.0)
        spin_start.setDecimals(3)
        spin_start.setSingleStep(1.0)
        spin_start.setSuffix(" s")
        spin_start.setValue(float(self._analysis_start_s))
        form.addRow("Start:", spin_start)

        spin_end = QDoubleSpinBox()
        spin_end.setRange(-1_000_000.0, 1_000_000_000.0)
        spin_end.setDecimals(3)
        spin_end.setSingleStep(1.0)
        spin_end.setSuffix(" s")
        spin_end.setValue(float(self._analysis_end_s))
        form.addRow("End:", spin_end)
        lay.addLayout(form)

        hint = QLabel("End must be greater than start. If no time file is loaded, FPS-based time is used.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#a6adc8; font-size:11px;")
        lay.addWidget(hint)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        lay.addWidget(buttons)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        self._analysis_time_range_enabled = bool(chk_enabled.isChecked())
        self._analysis_start_s = float(spin_start.value())
        self._analysis_end_s = float(spin_end.value())
        if self._analysis_time_range_enabled and self._analysis_end_s <= self._analysis_start_s:
            self._analysis_time_range_enabled = False
            self._status_bar.setText("Analysis time range disabled: end must be greater than start")
        self._save_current_settings()
        self._reapply_source_data_if_available()

    # ── Handlers ───────────────────────────────────────────────────────────────

    @Slot(dict, str)
    def _on_data_loaded(self, dfs: dict, path: str) -> None:
        modified = self._get_modified_tracking(path)
        if modified is not None:
            dfs = modified
            if hasattr(self.load_panel, "replace_loaded_data"):
                self.load_panel.replace_loaded_data(path, dfs, activate=False)
        self._source_animal_dfs = dfs
        self._source_behavior_arrays = {}
        self._apply_source_animal_dfs()
        restored_social = self._try_restore_social_cache_for_active()
        # Save current session files for "Load Last Session"
        self._settings["last_session_dlc_files"] = list(self.load_panel._dlc_paths)
        self._settings["last_session_video_files"] = list(self.load_panel._video_paths)
        self._settings["last_session_time_files"] = list(getattr(self.load_panel, "_time_paths", []))
        self._settings["last_session_mask_files"] = list(getattr(self.load_panel, "_mask_paths", []))
        save_settings(self._settings)
        self._sync_project_batch_count()
        n = len(dfs)
        if not restored_social:
            self._status_bar.setText(f"Loaded {n} animal{'s' if n != 1 else ''} from {Path(path).name}")
        logger.info("DLC data loaded: %d animals from %s", len(dfs), path)

    @Slot(str)
    def _on_video_loaded(self, path: str) -> None:
        self._current_video_path = path
        self.video_panel.set_video(path)
        self.export_panel.set_video_path(path)
        self.inference_panel.set_current_video_path(path)
        self.cleaning_panel.set_video_path(path)
        self.roi_panel.set_video_path(path)
        self.roi_panel.set_video_info(
            self.video_panel._frame_w, self.video_panel._frame_h,
        )
        # Update session files
        self._settings["last_session_video_files"] = list(self.load_panel._video_paths)
        self._apply_calibration_for_video(path)
        save_settings(self._settings)
        self._sync_project_batch_count()

    @Slot(object)
    def _on_time_loaded(self, times) -> None:
        """Handle custom timestamps from external file."""
        self._custom_time = times
        self._settings["last_session_time_files"] = list(getattr(self.load_panel, "_time_paths", []))
        save_settings(self._settings)
        self._sync_project_batch_count()
        if self._analysis_time_range_enabled and self._source_animal_dfs:
            self._apply_source_animal_dfs()
        aligned = self._apply_time_to_panels()
        if aligned is not None:
            source_count = len(getattr(times, "times", aligned))
            self._status_bar.setText(f"Loaded frame times: {len(aligned)} aligned timestamp(s)")
            logger.info("Custom time loaded: %d source timestamps, %d aligned", source_count, len(aligned))
        else:
            self._status_bar.setText("Using FPS-based time")
            logger.info("Custom time cleared, using FPS-based time")

    @Slot(object, str)
    def _on_masks_loaded(self, mask_store, path: str) -> None:
        self._mask_store = mask_store
        self.social_panel.set_mask_store(mask_store)
        if mask_store is None:
            self.video_panel.set_masks(None)
            self._settings["last_session_mask_files"] = list(getattr(self.load_panel, "_mask_paths", []))
            save_settings(self._settings)
            self._status_bar.setText("Masks cleared")
            self._sync_project_batch_count()
            return
        self.video_panel.set_masks(mask_store)
        self._settings["last_session_mask_files"] = list(getattr(self.load_panel, "_mask_paths", []))
        save_settings(self._settings)
        rng = mask_store.frame_range
        suffix = f" frames {rng[0]}-{rng[1]}" if rng else ""
        self._status_bar.setText(
            f"Loaded masks: {Path(path).name} ({mask_store.annotation_count} masks{suffix})"
        )
        self._sync_project_batch_count()

    @Slot(dict)
    def _on_clean(self, params: dict) -> None:
        if not self._animal_dfs:
            return
        self._status_bar.setText("Cleaning data...")
        from dlc_processor.core.data_cleaner import clean_all_animals
        cleaned = clean_all_animals(self._animal_dfs, **params)
        self._commit_active_tracking_update(cleaned, status="Cleaning complete")
        self._status_bar.setText("Cleaning complete")

    @Slot(dict)
    def _on_fix_impossible(self, params: dict) -> None:
        if not self._animal_dfs:
            return
        from dlc_processor.core.data_cleaner import fix_impossible_all_animals
        fixed = fix_impossible_all_animals(self._animal_dfs, **params)
        self._commit_active_tracking_update(fixed, status="Impossible-condition fix applied")

    @Slot(dict)
    def _on_identity_swap(self, params: dict) -> None:
        animal_a = str(params.get("animal_a", ""))
        animal_b = str(params.get("animal_b", ""))
        source = self._source_animal_dfs or self._animal_dfs
        if not source or animal_a == animal_b or animal_a not in source or animal_b not in source:
            return

        indices = None
        if "start_frame" in params and "end_frame" in params:
            indices = self._current_rows_to_source_indices(
                int(params.get("start_frame", 0)),
                int(params.get("end_frame", 0)),
            )
            if indices is not None and len(indices) == 0:
                self._status_bar.setText("Identity swap skipped: empty frame range")
                return

        swapped = _swap_identity_data(source, animal_a, animal_b, indices)
        mask_status = ""
        if self._mask_store is not None:
            mask_status = self._swap_active_masks_for_identity(source, animal_a, animal_b, indices)
        active_path = getattr(self.load_panel, "_active_path", "")

        csv_path = ""
        if params.get("write_csv", True) and active_path:
            from dlc_processor.core.data_cleaner import save_cleaned_csv

            src = Path(active_path)
            csv_path = str(src.with_name(f"{src.stem}_identity_swapped.csv"))
            save_cleaned_csv(swapped, csv_path)

        frame_text = "whole recording" if indices is None else f"{len(indices)} frame(s)"
        suffix_parts = []
        if csv_path:
            suffix_parts.append(f"-> {Path(csv_path).name}")
        if mask_status:
            suffix_parts.append(mask_status)
        suffix = f" ({'; '.join(suffix_parts)})" if suffix_parts else ""
        self._commit_active_tracking_update(
            swapped,
            status=f"Swapped {animal_a} <-> {animal_b} over {frame_text}{suffix}",
        )

    def _swap_active_masks_for_identity(self, source_dfs: dict, animal_a: str, animal_b: str, indices) -> str:
        track_a = _mask_track_id_for_animal(animal_a, source_dfs, self._mask_store)
        track_b = _mask_track_id_for_animal(animal_b, source_dfs, self._mask_store)
        if track_a == track_b:
            fallback_a, fallback_b = _mask_track_ids_by_animal_order(animal_a, animal_b, source_dfs, self._mask_store)
            if fallback_a is not None and fallback_b is not None and fallback_a != fallback_b:
                track_a, track_b = fallback_a, fallback_b
        if track_a is None or track_b is None or track_a == track_b:
            return ""

        source_frames = _source_frames_for_row_indices(source_dfs, indices)
        swapper = getattr(self._mask_store, "with_swapped_track_ids", None)
        if swapper is None:
            return ""

        self._mask_store = swapper(track_a, track_b, source_frames)
        if hasattr(self.load_panel, "_mask_store"):
            self.load_panel._mask_store = self._mask_store
        self.social_panel.set_mask_store(self._mask_store)
        self.video_panel.set_masks(self._mask_store)

        frame_text = "all mask frames" if source_frames is None else f"{len(source_frames)} mask frame(s)"
        saved_text = ""
        mask_path = str(getattr(self.load_panel, "_active_mask_path", "") or "")
        if mask_path:
            try:
                cleaned_path = _cleaned_mask_json_path(mask_path)
                saver = getattr(self._mask_store, "save_json", None)
                if saver is not None:
                    saved_path = saver(cleaned_path)
                    saved_text = f" -> {Path(saved_path).name}"
            except Exception:
                logger.exception("Failed to save cleaned mask COCO JSON for %s", mask_path)
                saved_text = " (mask JSON save failed)"
        return f"masks id {track_a}<->{track_b} over {frame_text}{saved_text}"

    @Slot()
    def _on_reset_cleaning(self) -> None:
        active_path = str(getattr(self.load_panel, "_active_path", "") or "")
        if not active_path:
            self._status_bar.setText("Reset skipped: no active DLC file")
            return

        try:
            from dlc_processor.core.dlc_loader import load_dlc_file

            dfs = load_dlc_file(active_path)
        except Exception as exc:
            logger.exception("Failed to reload original DLC file: %s", active_path)
            self._status_bar.setText(f"Reset failed: {exc}")
            return

        self._forget_modified_tracking(active_path)
        self._source_animal_dfs = dfs
        self._source_behavior_arrays = {}
        self._forget_social_cache(active_path)
        if hasattr(self.load_panel, "replace_loaded_data"):
            self.load_panel.replace_loaded_data(active_path, dfs, activate=False)

        mask_path = str(getattr(self.load_panel, "_active_mask_path", "") or "")
        if mask_path:
            try:
                self.load_panel._set_masks(mask_path)
            except Exception as exc:
                logger.exception("Failed to reload original mask file: %s", mask_path)
                self._status_bar.setText(f"Tracking reset; mask reload failed: {exc}")
                self._apply_source_animal_dfs()
                return
        else:
            self._mask_store = None
            self.social_panel.set_mask_store(None)
            self.video_panel.set_masks(None)

        self._apply_source_animal_dfs()
        self._status_bar.setText("Reset to original DLC tracking and masks")

    @Slot(dict)
    def _on_identity_rename(self, params: dict) -> None:
        old_label = str(params.get("old_label", "")).strip()
        new_label = str(params.get("new_label", "")).strip()
        source = self._source_animal_dfs or self._animal_dfs
        if not source or not old_label or not new_label or old_label not in source:
            return
        if new_label in source and new_label != old_label:
            self._status_bar.setText(f"Cannot rename {old_label}: {new_label} already exists")
            return
        if new_label == old_label:
            return

        renamed = _rename_identity_data(source, old_label, new_label)
        active_path = getattr(self.load_panel, "_active_path", "")

        csv_path = ""
        if params.get("write_csv", True) and active_path:
            from dlc_processor.core.data_cleaner import save_cleaned_csv

            src = Path(active_path)
            csv_path = str(src.with_name(f"{src.stem}_renamed.csv"))
            save_cleaned_csv(renamed, csv_path)

        suffix = f" -> {Path(csv_path).name}" if csv_path else ""
        self._commit_active_tracking_update(
            renamed,
            status=f"Renamed {old_label} -> {new_label}{suffix}",
        )

    @Slot(dict)
    def _on_mask_identity_fix(self, params: dict) -> None:
        source = self._source_animal_dfs or self._animal_dfs
        if not source:
            self._status_bar.setText("Mask identity fix skipped: no tracking data loaded")
            return
        if self._mask_store is None:
            self._status_bar.setText("Mask identity fix skipped: no masks loaded")
            return
        if len(source) == 1:
            fixed, updated_store, summary = _normalize_single_animal_mask_identity(source, self._mask_store)
            self._mask_store = updated_store
            if hasattr(self.load_panel, "_mask_store"):
                self.load_panel._mask_store = updated_store
            self.social_panel.set_mask_store(updated_store)
            self.video_panel.set_masks(updated_store)

            active_path = getattr(self.load_panel, "_active_path", "")
            csv_path = ""
            if params.get("write_csv", True) and active_path:
                from dlc_processor.core.data_cleaner import save_cleaned_csv

                src = Path(active_path)
                csv_path = str(src.with_name(f"{src.stem}_mask_identity_cleaned.csv"))
                save_cleaned_csv(fixed, csv_path)

            mask_path = str(getattr(self.load_panel, "_active_mask_path", "") or "")
            mask_json_path = ""
            if bool(summary.get("mask_reassigned", False)) and mask_path:
                try:
                    saver = getattr(updated_store, "save_json", None)
                    if saver is not None:
                        mask_json_path = saver(_cleaned_mask_json_path(mask_path))
                except Exception:
                    logger.exception("Failed to save single-animal cleaned mask COCO JSON for %s", mask_path)

            suffix_parts = []
            if csv_path:
                suffix_parts.append(f"tracking -> {Path(csv_path).name}")
            if mask_json_path:
                suffix_parts.append(f"masks -> {Path(mask_json_path).name}")
            elif bool(summary.get("mask_reassigned", False)):
                suffix_parts.append("mask JSON save failed")
            suffix = f" ({'; '.join(suffix_parts)})" if suffix_parts else ""
            old_label = str(summary.get("old_label", ""))
            old_track = summary.get("old_track_id")
            track_text = f"; mask id {old_track}->1" if old_track not in (None, 1) else ""
            self._commit_active_tracking_update(
                fixed,
                status=f"Single-animal mask identity: {old_label}->mouse1{track_text}{suffix}",
            )
            return

        from dlc_processor.core.data_cleaner import fix_identities_from_masks

        animal_to_track = {}
        for aid in source:
            track_id = _mask_track_id_for_animal(str(aid), source, self._mask_store)
            if track_id is not None:
                animal_to_track[str(aid)] = int(track_id)
        fixed, summary = fix_identities_from_masks(
            source,
            self._mask_store,
            animal_to_track_id=animal_to_track,
        )

        active_path = getattr(self.load_panel, "_active_path", "")
        csv_path = ""
        if params.get("write_csv", True) and active_path:
            from dlc_processor.core.data_cleaner import save_cleaned_csv

            src = Path(active_path)
            csv_path = str(src.with_name(f"{src.stem}_mask_identity_cleaned.csv"))
            save_cleaned_csv(fixed, csv_path)

        corrected = int(summary.get("frames_corrected", 0) or 0)
        checked = int(summary.get("frames_checked", 0) or 0)
        suffix = f" -> {Path(csv_path).name}" if csv_path else ""
        self._commit_active_tracking_update(
            fixed,
            status=f"Mask identity fix: corrected {corrected}/{checked} frame(s){suffix}",
        )

    @Slot(float)
    def _on_calibration(self, px_per_cm: float) -> None:
        raw_px_cm = float(self.cleaning_panel.px_per_cm())
        video_path = self._current_video_path or self.cleaning_panel.video_path()
        if video_path:
            self._current_video_path = video_path
        if video_path and raw_px_cm > 0:
            for key in _video_calibration_keys(video_path):
                self._video_calibrations_px_per_cm[key] = raw_px_cm
            self._settings["video_calibrations_px_per_cm"] = dict(self._video_calibrations_px_per_cm)
        self._settings["calibration_px_per_cm"] = raw_px_cm
        self._propagate_calibration(px_per_cm)
        self._save_current_settings()
        scope = Path(video_path).name if video_path else "global"
        self._status_bar.setText(f"Calibration for {scope}: {px_per_cm:.2f} px/cm")
        logger.info("Calibration updated for %s: %.2f px/cm (active=%.2f)", scope, raw_px_cm, px_per_cm)

    @Slot(dict)
    def _on_kinematics(self, result: dict) -> None:
        self._fps = self.kinematics_panel.fps()
        self.export_panel.set_fps(self._fps)
        self._apply_animal_dfs(result, behavior_arrays={})
        self._status_bar.setText(f"Kinematics computed for {len(result)} animals")

    @Slot(dict)
    def _on_social(self, arrays: dict) -> None:
        merged = _merge_mobility_behavior_arrays(self._animal_dfs, arrays)
        self._source_behavior_arrays = dict(arrays or {})
        self._remember_current_social_cache(visible_arrays=arrays)
        self.video_panel.set_data(self._animal_dfs, merged)
        self.plot_panel.set_behavior_arrays(merged)
        self.export_panel.set_behavior_arrays(merged)
        n_bool = sum(1 for a in merged.values() if hasattr(a, 'dtype') and a.dtype == bool)
        self._status_bar.setText(f"Social detection: {n_bool} behaviours detected")

    @Slot(float)
    def _open_mask_contact_calibration(self, margin_percent: float) -> None:
        if not self._current_video_path:
            self._status_bar.setText("Load a video before calibrating mask contact")
            return
        if self._mask_store is None:
            self._status_bar.setText("Load masks before calibrating mask contact")
            return
        if len(self._animal_dfs) < 2:
            self._status_bar.setText("Load at least two animals before calibrating mask contact")
            return

        aid_a, aid_b = self.social_panel.current_pair()
        if aid_a not in self._animal_dfs or aid_b not in self._animal_dfs or aid_a == aid_b:
            self._status_bar.setText("Choose two different animals before calibrating mask contact")
            return

        from dlc_processor.ui.mask_contact_calibration import MaskContactCalibrationDialog

        dlg = MaskContactCalibrationDialog(
            video_path=self._current_video_path,
            mask_store=self._mask_store,
            animal_dfs=self._animal_dfs,
            aid_a=aid_a,
            aid_b=aid_b,
            current_row=getattr(self.video_panel, "_current_frame", 0),
            margin_percent=margin_percent,
            parent=self,
        )
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.social_panel.set_mask_contact_margin_percent(dlg.margin_percent())
            self._settings["social_mask_contact_margin_percent"] = dlg.margin_percent()
            self._save_current_settings()
            self._status_bar.setText(f"Mask contact margin set to {dlg.margin_percent():.1f}%")

    @Slot(dict)
    def _on_data_refined(self, dfs: dict) -> None:
        self._commit_active_tracking_update(dfs, status="Track edits applied")

    @Slot()
    def _show_plot_cleaning_region(self) -> None:
        """Show cleaning region on the plot, synced with cleaning panel spinboxes."""
        start = self.cleaning_panel._spin_range_start.value()
        end = self.cleaning_panel._spin_range_end.value()
        self.plot_panel.show_cleaning_region(start, end)

    @Slot()
    def _open_egocentric_heatmap(self) -> None:
        """Open the egocentric position heatmap dialog."""
        if not self._animal_dfs or len(self._animal_dfs) < 2:
            logger.warning("Need at least 2 animals for egocentric heatmap.")
            return
        from dlc_processor.ui.egocentric_heatmap import EgocentricHeatmapDialog
        current_frame = self.video_panel._slider.value() if hasattr(self.video_panel, "_slider") else 0
        dlg = EgocentricHeatmapDialog(
            self._animal_dfs,
            fps=self._fps,
            current_frame=current_frame,
            px_per_cm=self.cleaning_panel.active_px_per_cm(),
            parent=self,
        )
        # Wire frame cursor updates for real-time windowed heatmap
        self.video_panel.frame_changed.connect(dlg.set_current_frame)
        dlg.exec()

    @Slot()
    def _open_track_editor(self) -> None:
        """Launch the track editor dialog with current animal data."""
        if not self._animal_dfs:
            logger.warning("No data loaded — cannot open track editor.")
            return
        from dlc_processor.ui.track_editor import TrackEditorDialog
        dlg = TrackEditorDialog(self._animal_dfs, parent=self)
        dlg.data_changed.connect(self._on_track_editor_done)
        dlg.exec()

    @Slot(dict)
    def _on_track_editor_done(self, dfs: dict) -> None:
        """Apply edited data from the track editor."""
        self._commit_active_tracking_update(dfs, status="Track editor changes applied")
        logger.info("Track editor changes committed to active recording")

    @Slot(int)
    def _on_plot_cursor(self, frame_idx: int) -> None:
        """When user clicks on the plot, seek the video to that frame."""
        self.video_panel._slider.setValue(frame_idx)

    @Slot()
    def _provide_skeleton_edges(self) -> None:
        """Pass skeleton edge definitions from overlay_worker to export panel."""
        edges = self.video_panel._skeleton_edges
        if not edges:
            from dlc_processor.workers.overlay_worker import _SKELETON_EDGES
            edges = list(_SKELETON_EDGES)
        self.export_panel.set_skeleton_edges(edges)

    def _toggle_plot_panel(self) -> None:
        """Show / hide the plot panel in the central splitter."""
        self._plot_visible = not self._plot_visible
        self.plot_panel.setVisible(self._plot_visible)
        for btn in self._bar._buttons:
            if btn.objectName() == "plots":
                btn.setProperty("active", "true" if self._plot_visible else "false")
                from shared.icons import icon_qicon
                color = "#cba6f7" if self._plot_visible else "#6c7086"
                btn.setIcon(icon_qicon("bar-chart-2", size=18, color=color))
                btn.style().unpolish(btn)
                btn.style().polish(btn)
                break

    @Slot(str, str)
    def _on_inference_done(self, h5_path: str, _config: str) -> None:
        if h5_path:
            self.load_panel.load_file_programmatic(h5_path)

    @Slot(list, str)
    def _on_inference_outputs_ready(self, h5_paths: list, _config: str) -> None:
        for h5_path in h5_paths:
            if h5_path:
                self.load_panel.load_file_programmatic(str(h5_path))

    def _load_last_session(self) -> None:
        """Reload the DLC and video files from the previous session."""
        dlc_files = self._settings.get("last_session_dlc_files", [])
        video_files = self._settings.get("last_session_video_files", [])
        mask_files = self._settings.get("last_session_mask_files", [])
        time_files = self._settings.get("last_session_time_files", [])
        # Filter to files that still exist
        dlc_files = [p for p in dlc_files if Path(p).exists()]
        video_files = [p for p in video_files if Path(p).exists()]
        mask_files = [p for p in mask_files if Path(p).exists()]
        time_files = [p for p in time_files if Path(p).exists()]
        if not dlc_files and not video_files and not mask_files and not time_files:
            logger.info("No previous session files found.")
            return
        self.load_panel.load_from_folder(video_files, dlc_files, mask_files, time_files)
        logger.info(
            "Last session restored: %d videos, %d DLC files, %d mask files, %d time files",
            len(video_files), len(dlc_files), len(mask_files), len(time_files),
        )

    @Slot(list)
    def _on_rois_changed(self, rois: list) -> None:
        """Handle updated ROI definitions from ROI panel."""
        # Pass ROIs to video panel for overlay rendering
        self.video_panel._rois = rois
        if self.video_panel._video_path:
            self.video_panel._render_frame(self.video_panel._current_frame)

    @Slot(dict)
    def _on_project_batch_export_requested(self, options: dict) -> None:
        """Export framewise metrics+behaviours for every loaded recording row."""
        self._run_project_batch_export(options)

    @Slot(dict)
    def _on_batch_panel_compute_requested(self, options: dict) -> None:
        """Run the visible Batch / Metadata panel workflow."""
        self._run_project_batch_export(options)

    @Slot(dict)
    def _on_social_batch_process_requested(self, options: dict) -> None:
        """Run all loaded recordings from the Social panel using current thresholds."""
        batch_state = self.batch_panel.export_state()
        export_state = self.export_panel.batch_export_state()
        social_options = dict(options or {})
        use_masks = (
            bool(social_options.get("use_masks_for_social", False))
            or bool(batch_state.get("batch_social_use_masks", False))
            or bool(export_state.get("batch_social_use_masks", False))
        )
        run_options = {
            "compute_social": True,
            "use_masks_for_social": use_masks,
            "fix_identity_from_masks": bool(
                batch_state.get("batch_fix_identity_from_masks", False)
                or export_state.get("batch_fix_identity_from_masks", False)
            ),
            "output_mode": batch_state.get("batch_export_output_mode")
            or export_state.get("batch_export_output_mode")
            or "time_file_folder",
            "custom_output_dir": batch_state.get("batch_export_custom_dir")
            or export_state.get("batch_export_custom_dir")
            or "",
            "generate_summary": bool(batch_state.get("batch_generate_summary", True)),
            "export_position_maps": bool(batch_state.get("batch_export_position_maps", True)),
            "group_by": str(batch_state.get("batch_summary_group_by", "animal") or "animal").split(","),
            "animal_filter": str(batch_state.get("batch_summary_animal_filter", "") or ""),
            "mouse_id_filter": str(batch_state.get("batch_summary_mouse_id_filter", "") or ""),
            "plot_style": str(batch_state.get("batch_summary_plot_style", "box_strip") or "box_strip"),
            "comparison_method": str(batch_state.get("batch_summary_comparison", "holm_ttest") or "holm_ttest"),
            "tab10_color_map": batch_state.get("batch_summary_color_map", ""),
            "n_jobs": int(batch_state.get("batch_cpu_jobs", 0) or export_state.get("batch_cpu_jobs", 0) or 0),
            "analysis_frame_window_enabled": bool(batch_state.get("batch_analysis_frame_window_enabled", False)),
            "analysis_start_frame": int(batch_state.get("batch_analysis_start_frame", 0) or 0),
            "analysis_end_frame": int(batch_state.get("batch_analysis_end_frame", 0) or 0),
            "social_params": dict(social_options.get("social_params") or {}),
        }
        self._run_project_batch_export(run_options)

    def _run_project_batch_export(self, options: dict) -> None:
        records = self._project_batch_records()
        if not records:
            self.export_panel.set_project_batch_result("No loaded DLC recordings to export.")
            self.batch_panel.set_result("No loaded DLC recordings to export.")
            self.social_panel.set_batch_process_result("No loaded DLC recordings to export.")
            return
        records = self.batch_panel.attach_metadata_to_records(records)

        self._settings.update(self.export_panel.batch_export_state())
        self._settings.update(self.batch_panel.export_state())
        self._settings["batch_cpu_jobs"] = int(options.get("n_jobs", self._settings.get("batch_cpu_jobs", 0)) or 0)
        self._save_current_settings()

        from PySide6.QtWidgets import QApplication
        from dlc_processor.core.batch_processor import batch_export_recordings

        self.export_panel.set_project_batch_running(True)
        self.batch_panel.set_running(True)
        self.social_panel.set_batch_process_running(True)

        def _progress(pct: int, msg: str) -> None:
            self.export_panel.set_project_batch_progress(pct, msg)
            self.batch_panel.set_progress(pct, msg)
            self.social_panel.set_batch_process_progress(pct, msg)
            QApplication.processEvents()

        analysis_range = None
        analysis_frame_range = None
        batch_frame_window_enabled = bool(options.get("analysis_frame_window_enabled", False))
        batch_start_frame = int(options.get("analysis_start_frame", 0) or 0)
        batch_end_frame = int(options.get("analysis_end_frame", 0) or 0)
        if batch_frame_window_enabled:
            if batch_end_frame <= 0:
                analysis_frame_range = (batch_start_frame, 0)
            elif batch_end_frame >= batch_start_frame:
                analysis_frame_range = (batch_start_frame, batch_end_frame)

        try:
            social_kwargs = self._batch_social_kwargs(options)
            result = batch_export_recordings(
                records,
                fps=self._fps,
                px_per_cm=self.cleaning_panel.active_px_per_cm(),
                compute_social=bool(options.get("compute_social", True)),
                output_mode=str(options.get("output_mode", "time_file_folder")),
                custom_output_dir=str(options.get("custom_output_dir", "") or ""),
                social_kwargs=social_kwargs,
                fix_identity_from_masks=bool(options.get("fix_identity_from_masks", False)),
                analysis_time_range=analysis_range,
                analysis_frame_range=analysis_frame_range,
                n_jobs=int(options.get("n_jobs", 1) or 0),
                progress_callback=_progress,
            )
            exported = result.get("exported", [])
            summary_details = ""
            if exported and (
                bool(options.get("generate_summary", False))
                or bool(options.get("export_position_maps", True))
            ):
                from dlc_processor.core.metadata_summary import (
                    export_group_position_maps,
                    export_group_summaries,
                    export_group_transitions,
                )

                summary_dir = self._batch_summary_output_dir(options, result)
                summary_result = {"tables": [], "figures": [], "output_dir": str(summary_dir)}
                if bool(options.get("generate_summary", False)):
                    summary_result = export_group_summaries(
                        result.get("summary"),
                        summary_dir,
                        group_by=list(options.get("group_by") or ["animal"]),
                        animal_filter=str(options.get("animal_filter", "") or ""),
                        mouse_id_filter=str(options.get("mouse_id_filter", "") or ""),
                        plot_style=str(options.get("plot_style", "bar_strip") or "bar_strip"),
                        tab10_color_map=options.get("tab10_color_map", ""),
                        comparison_method=str(options.get("comparison_method", "holm_ttest") or "holm_ttest"),
                    )
                    transition_result = export_group_transitions(
                        exported,
                        result.get("summary"),
                        summary_dir,
                        group_by=list(options.get("group_by") or ["animal"]),
                        animal_filter=str(options.get("animal_filter", "") or ""),
                        mouse_id_filter=str(options.get("mouse_id_filter", "") or ""),
                        plot_style=str(options.get("plot_style", "bar_strip") or "bar_strip"),
                        tab10_color_map=options.get("tab10_color_map", ""),
                        comparison_method=str(options.get("comparison_method", "holm_ttest") or "holm_ttest"),
                    )
                else:
                    transition_result = {"tables": [], "figures": []}
                position_result = {"tables": [], "figures": []}
                if bool(options.get("export_position_maps", True)):
                    position_result = export_group_position_maps(
                        exported,
                        result.get("summary"),
                        summary_dir,
                        group_by=list(options.get("group_by") or ["animal"]),
                        animal_filter=str(options.get("animal_filter", "") or ""),
                        mouse_id_filter=str(options.get("mouse_id_filter", "") or ""),
                    )
                n_tables = len(summary_result.get("tables", []))
                n_figures = len(summary_result.get("figures", []))
                n_transition_tables = len(transition_result.get("tables", []))
                n_transition_figures = len(transition_result.get("figures", []))
                n_position_tables = len(position_result.get("tables", []))
                n_position_figures = len(position_result.get("figures", []))
                summary_details = (
                    f"\nSummary: {n_tables + n_transition_tables + n_position_tables} table(s), "
                    f"{n_figures + n_transition_figures + n_position_figures} figure(s) -> "
                    f"{summary_result.get('output_dir')}"
                )
            if exported:
                first = Path(exported[0]["table_path"])
                identity_details = _batch_identity_fix_message(exported)
                message = f"Batch exported {len(exported)} recording(s).\nFirst table: {first}{identity_details}{summary_details}"
                self.export_panel.set_project_batch_result(message)
                self.batch_panel.set_result(message)
                self.social_panel.set_batch_process_result(message)
                self._status_bar.setText(f"Batch metric/social export complete: {len(exported)} recording(s)")
            else:
                self.export_panel.set_project_batch_result("Batch export finished, but no recordings were written.")
                self.batch_panel.set_result("Batch export finished, but no recordings were written.")
                self.social_panel.set_batch_process_result("Batch export finished, but no recordings were written.")
        except Exception as exc:
            logger.exception("Project batch export failed: %s", exc)
            self.export_panel.set_project_batch_result(f"Batch export failed: {exc}")
            self.batch_panel.set_result(f"Batch export failed: {exc}")
            self.social_panel.set_batch_process_result(f"Batch export failed: {exc}")

    def _batch_social_kwargs(self, options: dict) -> dict:
        panel_params = self.social_panel.current_detection_params()
        params = dict(panel_params)
        params.update(dict((options or {}).get("social_params") or {}))

        def _float(name: str, default: float) -> float:
            try:
                return float((options or {}).get(name, params.get(name, default)))
            except (TypeError, ValueError):
                return float(default)

        def _int(name: str, default: int) -> int:
            try:
                return int((options or {}).get(name, params.get(name, default)))
            except (TypeError, ValueError):
                return int(default)

        return {
            "close_tol": _float("close_tol_px", 25.0),
            "side_tol": _float("side_tol_px", 50.0),
            "follow_tol": _float("follow_tol_px", 30.0),
            "follow_window": _int("follow_window", 12),
            "median_filter": _int("median_filter", 6),
            "likelihood_threshold": _float("likelihood_threshold", 0.20),
            "use_masks": bool((options or {}).get("use_masks_for_social", params.get("use_masks", False))),
            "mask_edge_margin_percent": _float(
                "mask_edge_margin_percent",
                float(params.get("mask_edge_margin_percent", self.social_panel.mask_contact_margin_percent()) or 5.0),
            ),
        }

    def _sync_project_batch_count(self) -> None:
        records = getattr(self.load_panel, "paired_records", lambda: [])()
        count = sum(1 for rec in records if rec.get("dlc_path") or rec.get("animal_dfs"))
        self.export_panel.set_project_batch_count(count)
        self.social_panel.set_project_batch_count(count)
        self.batch_panel.set_records(self._project_batch_records())

    def _batch_summary_output_dir(self, options: dict, result: dict) -> Path:
        custom = str(options.get("custom_output_dir", "") or "")
        if custom:
            return Path(custom) / "batch_summary"
        exported = result.get("exported") or []
        if exported:
            return Path(exported[0]["table_path"]).parent / "batch_summary"
        return Path.cwd() / "batch_summary"

    def _project_batch_records(self) -> list[dict]:
        records = []
        for rec in self.load_panel.paired_records():
            dlc_path = str(rec.get("dlc_path", "") or "")
            if not dlc_path and not rec.get("animal_dfs"):
                continue
            out = dict(rec)
            modified = self._get_modified_tracking(dlc_path) if dlc_path else None
            if modified is not None:
                out["animal_dfs"] = modified
                out["tracking_modified"] = True
            elif rec.get("active") and self._source_animal_dfs:
                out["animal_dfs"] = self._source_animal_dfs
                out["tracking_modified"] = self._has_modified_tracking(dlc_path) if dlc_path else False
            else:
                out["tracking_modified"] = False
            video_path = str(out.get("video_path", "") or "")
            out["px_per_cm"] = self._calibration_for_video_path(video_path)
            records.append(out)
        return records

    def _calibration_for_video_path(self, video_path: str) -> float:
        if video_path:
            for key in _video_calibration_keys(video_path):
                value = float(self._video_calibrations_px_per_cm.get(key, 0.0) or 0.0)
                if value > 0:
                    return value
        return float(self.cleaning_panel.active_px_per_cm() or 0.0)

    def _propagate_calibration(self, active_px_per_cm: Optional[float] = None) -> None:
        """Push the active calibration scale into analysis/export panels."""
        px_per_cm = (
            self.cleaning_panel.active_px_per_cm()
            if active_px_per_cm is None
            else active_px_per_cm
        )
        self.kinematics_panel.set_calibration(px_per_cm)
        self.social_panel.set_calibration(px_per_cm)
        self.roi_panel.set_calibration(px_per_cm)
        self.export_panel.set_calibration(px_per_cm)

    def _apply_calibration_for_video(self, video_path: str) -> None:
        """Restore the calibration assigned to a specific video path."""
        px_cm = 0.0
        for key in _video_calibration_keys(video_path):
            px_cm = float(self._video_calibrations_px_per_cm.get(key, 0.0) or 0.0)
            if px_cm > 0:
                break
        if px_cm > 0:
            self.cleaning_panel.set_calibration(px_cm, use_for_analysis=True)
            self._propagate_calibration(px_cm)
            self._status_bar.setText(f"Calibration restored for {Path(video_path).name}: {px_cm:.2f} px/cm")
        else:
            self.cleaning_panel.set_calibration(0.0, use_for_analysis=False)
            self._propagate_calibration(0.0)

    def _apply_time_to_panels(self):
        """Align external frame times to the active tracking window and apply them."""
        aligned = self._panel_time_override
        if aligned is None:
            aligned = align_times_to_dfs(self._custom_time, self._animal_dfs)
        self.plot_panel.set_custom_time(aligned)
        self.export_panel.set_custom_time(aligned)
        if hasattr(self.kinematics_panel, "set_custom_time"):
            self.kinematics_panel.set_custom_time(aligned)
        return aligned

    def _reapply_source_data_if_available(self) -> None:
        if self._source_animal_dfs:
            self._apply_source_animal_dfs()

    def _commit_active_tracking_update(self, updated_dfs: dict, status: str = "") -> None:
        """Persist modified tracking data for the active recording in memory."""
        active_path = getattr(self.load_panel, "_active_path", "")
        source = self._source_animal_dfs or self._animal_dfs
        committed = _merge_active_update_into_source(source, updated_dfs)
        self._source_animal_dfs = committed
        self._source_behavior_arrays = {}
        if active_path:
            self._forget_social_cache(active_path)
        if active_path:
            self._set_modified_tracking(active_path, committed)
            if hasattr(self.load_panel, "replace_loaded_data"):
                self.load_panel.replace_loaded_data(active_path, committed, activate=False)
        self._apply_source_animal_dfs()
        if status:
            self._status_bar.setText(status)

    def _get_modified_tracking(self, path: str):
        """Return cached edits for *path*, accepting equivalent path spellings."""
        if not path:
            return None
        if path in self._modified_dlc_cache:
            return self._modified_dlc_cache[path]
        keys = set(_path_lookup_keys(path))
        for cached_path, dfs in self._modified_dlc_cache.items():
            if keys.intersection(_path_lookup_keys(cached_path)):
                return dfs
        return None

    def _has_modified_tracking(self, path: str) -> bool:
        """Return True when project-save edits exist for *path*."""
        return self._get_modified_tracking(path) is not None

    def _set_modified_tracking(self, path: str, dfs: dict) -> None:
        """Store edited tracking data using a stable existing key when possible."""
        if not path:
            return
        if path in self._modified_dlc_cache:
            self._modified_dlc_cache[path] = dfs
            return
        keys = set(_path_lookup_keys(path))
        for cached_path in list(self._modified_dlc_cache):
            if keys.intersection(_path_lookup_keys(cached_path)):
                self._modified_dlc_cache[cached_path] = dfs
                return
        self._modified_dlc_cache[path] = dfs

    def _forget_modified_tracking(self, path: str) -> None:
        """Discard cached tracking edits for *path* and equivalent spellings."""
        if not path:
            return
        keys = set(_path_lookup_keys(path))
        for cached_path in list(self._modified_dlc_cache):
            if cached_path == path or keys.intersection(_path_lookup_keys(cached_path)):
                self._modified_dlc_cache.pop(cached_path, None)

    def _save_project_tracking_edits(self, project_path: Path) -> dict[str, str]:
        """Write modified in-memory tracking data next to the project file."""
        active_path = getattr(self.load_panel, "_active_path", "")
        if active_path and self._animal_dfs and self._has_modified_tracking(active_path):
            source = self._source_animal_dfs or self._animal_dfs
            self._set_modified_tracking(
                active_path,
                _merge_active_update_into_source(source, self._animal_dfs),
            )
        if not self._modified_dlc_cache:
            return {}
        from dlc_processor.core.data_cleaner import save_cleaned_h5

        edit_dir = project_path.with_suffix("")
        edit_dir = edit_dir.with_name(f"{edit_dir.name}_tracking_edits")
        edit_dir.mkdir(parents=True, exist_ok=True)

        edited: dict[str, str] = {}
        for idx, (original_path, dfs) in enumerate(self._modified_dlc_cache.items(), start=1):
            src = Path(original_path)
            safe_stem = _safe_filename(src.stem or f"tracking_{idx}")
            out_path = edit_dir / f"{idx:03d}_{safe_stem}_edited.h5"
            written_path = save_cleaned_h5(dfs, str(out_path))
            edited[original_path] = str(written_path)
        return edited

    def _load_project_tracking_edits(self, edited_dlc_files: dict) -> None:
        """Load project sidecar tracking edits into the per-recording cache."""
        if not isinstance(edited_dlc_files, dict):
            return
        from dlc_processor.core.dlc_loader import load_dlc_file

        project_dir = Path(self._project_path).parent if self._project_path else Path.cwd()
        for original_path, edited_path in edited_dlc_files.items():
            p = Path(str(edited_path))
            if not p.is_absolute():
                p = project_dir / p
            if not p.exists():
                logger.warning("Project tracking edit missing: %s", p)
                continue
            try:
                self._set_modified_tracking(str(original_path), load_dlc_file(p))
            except Exception:
                logger.exception("Failed to load project tracking edit: %s", p)

    def _remember_current_social_cache(self, visible_arrays: Optional[dict] = None) -> None:
        """Keep the current social detection result for project/session reuse."""
        active_path = getattr(self.load_panel, "_active_path", "")
        if not active_path or not self._animal_dfs:
            return
        all_arrays = self.social_panel.all_behavior_arrays()
        if not all_arrays:
            return
        params = self.social_panel.last_detection_params() or self.social_panel.current_detection_params()
        visible_source = visible_arrays if visible_arrays is not None else self._source_behavior_arrays
        visible_names = list((visible_source or all_arrays).keys())
        entry = {
            "version": 1,
            "context": self._current_social_cache_context(),
            "params": _json_compatible(params),
            "visible_names": visible_names,
            "arrays": {name: _copy_array_for_cache(arr) for name, arr in all_arrays.items()},
        }
        self._social_behavior_cache[_cache_key_for_path(active_path)] = entry

    def _try_restore_social_cache_for_active(self) -> bool:
        """Restore cached social arrays for the active recording if still valid."""
        active_path = getattr(self.load_panel, "_active_path", "")
        if not active_path or not self._animal_dfs:
            return False
        entry = self._find_social_cache(active_path)
        if not entry or not self._social_cache_matches_current(entry):
            return False
        all_arrays = entry.get("arrays") or {}
        if not all_arrays:
            return False

        visible_names = list(entry.get("visible_names") or all_arrays.keys())
        visible = {name: arr for name, arr in all_arrays.items() if name in set(visible_names)}
        self._source_behavior_arrays = dict(visible)
        self._apply_animal_dfs(self._animal_dfs, behavior_arrays=visible)
        self.social_panel.restore_behavior_arrays(
            all_arrays,
            params=entry.get("params") or {},
            visible_names=visible_names,
            emit=False,
        )
        n_bool = sum(1 for a in visible.values() if hasattr(a, "dtype") and a.dtype == bool)
        self._status_bar.setText(f"Loaded cached social detection: {n_bool} behaviours")
        return True

    def _current_social_cache_context(self) -> dict:
        active_path = getattr(self.load_panel, "_active_path", "")
        time_path = getattr(self.load_panel, "_active_time_path", "")
        mask_path = getattr(self.load_panel, "_active_mask_path", "")
        return {
            "active_path_keys": _path_lookup_keys(active_path),
            "active_file": _file_state(active_path),
            "time_path_keys": _path_lookup_keys(time_path),
            "time_file": _file_state(time_path),
            "mask_path_keys": _path_lookup_keys(mask_path),
            "mask_file": _file_state(mask_path),
            "fps": round(float(self._fps), 6),
            "px_per_cm": round(float(self.cleaning_panel.active_px_per_cm() or 0.0), 6),
            "analysis_range": {
                "enabled": bool(self._analysis_time_range_enabled),
                "start_s": round(float(self._analysis_start_s), 6),
                "end_s": round(float(self._analysis_end_s), 6),
            },
            "tracking": _tracking_context_signature(self._animal_dfs),
        }

    def _social_cache_matches_current(self, entry: dict) -> bool:
        params = entry.get("params") or {}
        for aid in (params.get("animal_a"), params.get("animal_b")):
            if aid and aid not in self._animal_dfs:
                return False
        cached = entry.get("context") or {}
        current = self._current_social_cache_context()
        if not _path_sets_intersect(cached.get("active_path_keys"), current.get("active_path_keys")):
            return False
        for path_key in ("time_path_keys", "mask_path_keys"):
            if not _path_sets_intersect(cached.get(path_key), current.get(path_key)):
                return False
        for key in ("active_file", "time_file", "mask_file", "analysis_range", "tracking"):
            if cached.get(key) != current.get(key):
                return False
        return (
            abs(float(cached.get("fps", 0.0)) - float(current.get("fps", 0.0))) <= 1e-6
            and abs(float(cached.get("px_per_cm", 0.0)) - float(current.get("px_per_cm", 0.0))) <= 1e-6
        )

    def _find_social_cache(self, path: str) -> Optional[dict]:
        keys = set(_path_lookup_keys(path))
        direct = self._social_behavior_cache.get(_cache_key_for_path(path))
        if direct is not None:
            return direct
        for cache_key, entry in self._social_behavior_cache.items():
            entry_keys = set(entry.get("context", {}).get("active_path_keys") or [])
            if cache_key in keys or keys.intersection(entry_keys):
                return entry
        return None

    def _forget_social_cache(self, path: str) -> None:
        keys = set(_path_lookup_keys(path))
        for cache_key, entry in list(self._social_behavior_cache.items()):
            entry_keys = set(entry.get("context", {}).get("active_path_keys") or [])
            if cache_key in keys or keys.intersection(entry_keys):
                self._social_behavior_cache.pop(cache_key, None)

    def _save_project_social_cache(self, project_path: Path) -> dict:
        if not self._social_behavior_cache:
            return {}
        import json
        import numpy as np

        cache_path = project_path.with_suffix("")
        cache_path = cache_path.with_name(f"{cache_path.name}_social_cache.npz")
        payload: dict[str, object] = {}
        entries: list[dict] = []
        for entry_idx, (cache_key, entry) in enumerate(self._social_behavior_cache.items()):
            arrays = entry.get("arrays") or {}
            if not arrays:
                continue
            array_map = {}
            for array_idx, (name, arr) in enumerate(arrays.items()):
                array_key = f"entry{entry_idx}_array{array_idx}"
                payload[array_key] = np.asarray(arr)
                array_map[name] = array_key
            entries.append({
                "cache_key": cache_key,
                "version": int(entry.get("version", 1)),
                "context": _json_compatible(entry.get("context") or {}),
                "params": _json_compatible(entry.get("params") or {}),
                "visible_names": list(entry.get("visible_names") or arrays.keys()),
                "arrays": array_map,
            })
        if not entries:
            return {}
        payload["__metadata__"] = np.asarray(json.dumps({"version": 1, "entries": entries}))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **payload)
        return {"version": 1, "path": cache_path.name}

    def _load_project_social_cache(self, cache_info: dict) -> None:
        if not isinstance(cache_info, dict) or not cache_info.get("path"):
            return
        import json
        import numpy as np

        project_dir = Path(self._project_path).parent if self._project_path else Path.cwd()
        cache_path = Path(str(cache_info.get("path", "")))
        if not cache_path.is_absolute():
            cache_path = project_dir / cache_path
        if not cache_path.exists():
            logger.warning("Project social cache missing: %s", cache_path)
            return
        try:
            with np.load(cache_path, allow_pickle=False) as data:
                metadata = json.loads(str(data["__metadata__"].item()))
                for entry in metadata.get("entries", []) or []:
                    arrays = {
                        name: np.asarray(data[array_key]).copy()
                        for name, array_key in (entry.get("arrays") or {}).items()
                        if array_key in data
                    }
                    if arrays:
                        cache_key = str(entry.get("cache_key") or "")
                        if not cache_key:
                            keys = entry.get("context", {}).get("active_path_keys") or []
                            cache_key = str(keys[0]) if keys else f"entry_{len(self._social_behavior_cache)}"
                        self._social_behavior_cache[cache_key] = {
                            "version": int(entry.get("version", 1)),
                            "context": entry.get("context") or {},
                            "params": entry.get("params") or {},
                            "visible_names": list(entry.get("visible_names") or arrays.keys()),
                            "arrays": arrays,
                        }
        except Exception:
            logger.exception("Failed to load project social cache: %s", cache_path)

    def _apply_source_animal_dfs(self) -> None:
        dfs, time_override = self._filter_dfs_to_analysis_range(self._source_animal_dfs)
        self._panel_time_override = time_override
        self._apply_animal_dfs(dfs, behavior_arrays=self._source_behavior_arrays)
        if time_override is not None:
            self._status_bar.setText(
                f"Analysis range: {self._analysis_start_s:.3f}-{self._analysis_end_s:.3f} s "
                f"({len(time_override)} frame(s))"
            )

    def _filter_dfs_to_analysis_range(self, dfs: dict) -> tuple[dict, Optional[object]]:
        if not self._analysis_time_range_enabled or self._analysis_end_s <= self._analysis_start_s or not dfs:
            return dfs, None

        import numpy as np

        first_df = next(iter(dfs.values()), None)
        if first_df is None:
            return dfs, None
        n_rows = len(first_df)
        if n_rows == 0:
            return dfs, None

        times = align_times_to_dfs(self._custom_time, dfs)
        if times is None:
            times = np.arange(n_rows, dtype=np.float64) / max(float(self._fps), 1e-9)
        else:
            times = np.asarray(times, dtype=np.float64).reshape(-1)
            if len(times) < n_rows:
                padded = np.full(n_rows, np.nan, dtype=np.float64)
                padded[: len(times)] = times
                times = padded
            elif len(times) > n_rows:
                times = times[:n_rows]

        mask = (
            np.isfinite(times)
            & (times >= float(self._analysis_start_s))
            & (times <= float(self._analysis_end_s))
        )
        indices = np.flatnonzero(mask)
        if indices.size == 0:
            return {aid: _slice_tracking_df(df, []) for aid, df in dfs.items()}, np.array([], dtype=np.float64)

        filtered = {aid: _slice_tracking_df(df, indices) for aid, df in dfs.items()}
        return filtered, times[indices].copy()

    def _current_rows_to_source_indices(self, start: int, end: int):
        import numpy as np

        if not self._animal_dfs:
            return None
        first_df = next(iter(self._animal_dfs.values()), None)
        if first_df is None:
            return None
        n = len(first_df)
        start = max(0, min(int(start), n))
        end = max(start, min(int(end), n))
        row_indices = np.arange(start, end, dtype=np.int64)
        source_rows = getattr(first_df, "attrs", {}).get("analysis_row_indices")
        if source_rows is not None and len(source_rows) == n:
            return np.asarray(source_rows, dtype=np.int64)[row_indices]
        return row_indices

    def _apply_animal_dfs(self, dfs: dict, behavior_arrays: Optional[dict] = None) -> None:
        """Propagate a new tracking dataset to all dependent panels."""
        self._animal_dfs = dfs
        arrays = _merge_mobility_behavior_arrays(dfs, behavior_arrays or {})
        if dfs:
            n_frames = max(len(df) for df in dfs.values())
            self.cleaning_panel.set_n_frames(n_frames)
        self.cleaning_panel.set_animal_dfs(dfs)
        self.kinematics_panel.set_animal_dfs(dfs)
        self.social_panel.set_animal_dfs(dfs, fps=self._fps)
        self.refinement_panel.set_animal_dfs(dfs)
        self.roi_panel.set_animal_dfs(dfs, fps=self._fps)
        if hasattr(self.roi_panel, "set_frame_numbers"):
            self.roi_panel.set_frame_numbers(_frame_numbers_from_dfs(dfs))
        self.export_panel.set_animal_dfs(dfs)
        self.export_panel.set_behavior_arrays(arrays)
        self.video_panel.set_data(dfs, arrays)
        self.plot_panel.set_animal_dfs(dfs, fps=self._fps)
        self.plot_panel.set_behavior_arrays(arrays)
        self._propagate_calibration()
        self._apply_time_to_panels()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        if not event.mimeData().hasUrls():
            event.ignore()
            return
        self.load_panel.handle_paths(url.toLocalFile() for url in event.mimeData().urls())
        event.acceptProposedAction()


def _frame_numbers_from_dfs(dfs: dict) -> Optional[object]:
    if not dfs:
        return None
    first_df = next(iter(dfs.values()), None)
    if first_df is None:
        return None
    return getattr(first_df, "attrs", {}).get("frame_numbers")


def _batch_identity_fix_message(exported: list[dict]) -> str:
    checked = sum(int(item.get("mask_identity_frames_checked", 0) or 0) for item in exported or [])
    corrected = sum(int(item.get("mask_identity_frames_corrected", 0) or 0) for item in exported or [])
    errors = [str(item.get("mask_identity_error", "") or "") for item in exported or []]
    errors = [text for text in errors if text]
    if checked <= 0 and corrected <= 0 and not errors:
        return ""
    message = f"\nMask identity fix: corrected {corrected}/{checked} frame(s)"
    if errors:
        message += f"; {len(errors)} error(s)"
    return message


def _merge_mobility_behavior_arrays(dfs: dict, behavior_arrays: Optional[dict] = None) -> dict:
    arrays = dict(behavior_arrays or {})
    for aid, df in (dfs or {}).items():
        safe_aid = _safe_behavior_id(aid)
        mobility = _mobility_masks_from_df(df)
        for state, mask in mobility.items():
            arrays.setdefault(f"{safe_aid}__{state}", mask)
    return arrays


def _mobility_masks_from_df(df) -> dict[str, object]:
    import numpy as np
    import pandas as pd

    if df is None or len(df) == 0:
        return {}
    if "mobility_state" in df.columns:
        state = df["mobility_state"].fillna("").astype(str).str.lower()
        immobile = state.eq("immobile").to_numpy(dtype=bool)
        mobile = state.eq("mobile").to_numpy(dtype=bool)
        if immobile.any() or mobile.any():
            return {"immobile": immobile, "mobile": mobile}
    for col in ("is_immobile", "immobile", "freezing"):
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
        immobile = values > 0.5
        return {"immobile": immobile, "mobile": ~immobile}
    return {}


def _safe_behavior_id(text: object) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(text).strip())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or "animal"


def _video_calibration_key(path: str) -> str:
    keys = _video_calibration_keys(path)
    return keys[0] if keys else ""


def _video_calibration_keys(path: str) -> list[str]:
    return _path_lookup_keys(path)


def _path_lookup_keys(path: str) -> list[str]:
    if not path:
        return []
    keys: list[str] = []

    def add(value: object) -> None:
        key = str(value).casefold()
        if key and key not in keys:
            keys.append(key)

    try:
        expanded = Path(path).expanduser()
        add(expanded.resolve(strict=False))
        add(expanded)
    except Exception:
        pass
    add(path)
    return keys


def _safe_filename(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text))
    cleaned = cleaned.strip("._")
    return cleaned or "tracking"


def _cleaned_mask_json_path(path: str) -> str:
    p = Path(path)
    suffix = p.suffix or ".json"
    return str(p.with_name(f"{p.stem}_cleaned{suffix}"))


def _cache_key_for_path(path: str) -> str:
    keys = _path_lookup_keys(path)
    return keys[0] if keys else "active"


def _path_sets_intersect(a, b) -> bool:
    set_a = set(a or [])
    set_b = set(b or [])
    if not set_a and not set_b:
        return True
    return bool(set_a.intersection(set_b))


def _file_state(path: str) -> dict:
    if not path:
        return {}
    try:
        p = Path(path).expanduser().resolve(strict=False)
        stat = p.stat()
    except Exception:
        return {}
    return {
        "size": int(stat.st_size),
        "mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))),
    }


def _tracking_context_signature(dfs: dict) -> list[dict]:
    import numpy as np

    signature = []
    for aid, df in (dfs or {}).items():
        frames = getattr(df, "attrs", {}).get("frame_numbers")
        frame_sig = None
        if frames is not None:
            arr = np.asarray(frames).reshape(-1)
            if arr.size:
                frame_sig = {
                    "len": int(arr.size),
                    "first": _scalar_for_json(arr[0]),
                    "last": _scalar_for_json(arr[-1]),
                }
            else:
                frame_sig = {"len": 0, "first": None, "last": None}
        signature.append({
            "animal": str(aid),
            "rows": int(len(df)),
            "columns": [str(col) for col in getattr(df, "columns", [])],
            "frame_numbers": frame_sig,
        })
    return signature


def _copy_array_for_cache(value):
    import numpy as np

    return np.asarray(value).copy()


def _json_compatible(value):
    import numpy as np

    if isinstance(value, dict):
        return {str(k): _json_compatible(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_compatible(v) for v in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    return value


def _scalar_for_json(value):
    import numpy as np

    if isinstance(value, np.generic):
        return value.item()
    return value


def _slice_tracking_df(df, indices) -> object:
    import numpy as np

    idx = np.asarray(indices, dtype=np.int64)
    out = df.iloc[idx].reset_index(drop=True).copy()
    out.attrs.update(getattr(df, "attrs", {}))
    frames = out.attrs.get("frame_numbers")
    if frames is not None:
        frames_arr = np.asarray(frames, dtype=np.int64)
        if len(frames_arr) == len(df):
            out.attrs["frame_numbers"] = frames_arr[idx].copy()
    out.attrs["analysis_row_indices"] = idx.copy()
    return out


def _source_frames_for_row_indices(dfs: dict, indices) -> Optional[object]:
    import numpy as np

    if indices is None or not dfs:
        return None
    first_df = next(iter(dfs.values()), None)
    if first_df is None:
        return np.asarray(indices, dtype=np.int64)

    row_idx = np.asarray(indices, dtype=np.int64)
    valid = row_idx[(row_idx >= 0) & (row_idx < len(first_df))]
    frames = getattr(first_df, "attrs", {}).get("frame_numbers")
    if frames is not None:
        frame_arr = np.asarray(frames, dtype=np.int64)
        if len(frame_arr) == len(first_df):
            return frame_arr[valid].copy()
    return valid.copy()


def _mask_track_id_for_animal(animal_label: str, dfs: dict, mask_store) -> Optional[int]:
    label = str(animal_label or "").strip()
    if not label:
        return None

    available = _available_mask_track_ids(mask_store)
    labels = [str(aid) for aid in (dfs or {}).keys()]
    if label in labels:
        ordered = sorted(available)
        if _prefer_mask_track_order_for_animals(labels, ordered):
            pos = labels.index(label)
            if pos < len(ordered):
                return int(ordered[pos])

    numeric = re.findall(r"\d+", label)
    if numeric:
        parsed = int(numeric[-1])
        if label.lower().startswith("animal_") and parsed == 0 and 0 not in available:
            shifted = parsed + 1
            if not available or shifted in available:
                return shifted
        if not available or parsed in available:
            return parsed
        shifted = parsed + 1
        if parsed == 0 and shifted in available:
            return shifted

    if label in labels:
        pos = labels.index(label)
        one_based = pos + 1
        zero_based = pos
        if not available or one_based in available:
            return one_based
        if zero_based in available:
            return zero_based
        ordered = sorted(available)
        if pos < len(ordered):
            return int(ordered[pos])
    return None


def _prefer_mask_track_order_for_animals(labels: list[str], ordered_tracks: list[int]) -> bool:
    if len(ordered_tracks) < len(labels) or not labels:
        return False
    suffixes: list[int] = []
    for label in labels:
        found = re.findall(r"\d+", str(label or ""))
        if not found:
            return False
        suffixes.append(int(found[-1]))
    if sorted(suffixes) != list(range(1, len(suffixes) + 1)):
        return False
    if set(suffixes).issubset(set(int(track) for track in ordered_tracks)):
        return False
    return True


def _mask_track_ids_by_animal_order(
    animal_a: str,
    animal_b: str,
    dfs: dict,
    mask_store,
) -> tuple[Optional[int], Optional[int]]:
    labels = [str(aid) for aid in (dfs or {}).keys()]
    if animal_a not in labels or animal_b not in labels:
        return None, None

    available = sorted(_available_mask_track_ids(mask_store))
    if not available:
        return labels.index(animal_a) + 1, labels.index(animal_b) + 1

    pos_a = labels.index(animal_a)
    pos_b = labels.index(animal_b)
    if pos_a >= len(available) or pos_b >= len(available):
        return None, None
    return int(available[pos_a]), int(available[pos_b])


def _available_mask_track_ids(mask_store) -> set[int]:
    frames = getattr(mask_store, "frames", None)
    if not isinstance(frames, dict):
        return set()
    out: set[int] = set()
    for annotations in frames.values():
        for ann in annotations or []:
            try:
                out.add(int(getattr(ann, "track_id")))
            except Exception:
                continue
    return out


def _normalize_single_animal_mask_identity(source_dfs: dict, mask_store):
    """Force a single tracked animal and single mask track to the mouse1 identity."""
    if len(source_dfs or {}) != 1:
        return source_dfs, mask_store, {
            "old_label": "",
            "new_label": "",
            "old_track_id": None,
            "new_track_id": None,
            "tracking_renamed": False,
            "mask_reassigned": False,
        }

    old_label = str(next(iter(source_dfs.keys())))
    fixed = _rename_identity_data(source_dfs, old_label, "mouse1")

    updated_store = mask_store
    old_track = None
    available = sorted(_available_mask_track_ids(mask_store))
    if len(available) == 1:
        old_track = int(available[0])
        if old_track != 1:
            reassigner = getattr(mask_store, "with_reassigned_track_ids", None)
            if reassigner is not None:
                updated_store = reassigner({old_track: 1})
            else:
                swapper = getattr(mask_store, "with_swapped_track_ids", None)
                if swapper is not None:
                    updated_store = swapper(old_track, 1)

    return fixed, updated_store, {
        "old_label": old_label,
        "new_label": "mouse1",
        "old_track_id": old_track,
        "new_track_id": 1 if old_track is not None else None,
        "tracking_renamed": old_label != "mouse1",
        "mask_reassigned": updated_store is not mask_store,
    }


def _swap_identity_data(dfs: dict, animal_a: str, animal_b: str, indices) -> dict:
    import numpy as np
    from dlc_processor.core.dlc_loader import get_bodyparts

    out = {}
    for aid, df in dfs.items():
        copied = df.copy()
        copied.attrs.update(getattr(df, "attrs", {}))
        out[aid] = copied

    df_a = out[animal_a]
    df_b = out[animal_b]
    bps_a = get_bodyparts(df_a)
    bps_b = set(get_bodyparts(df_b))
    common_bps = [bp for bp in bps_a if bp in bps_b]
    if indices is None:
        row_idx = np.arange(min(len(df_a), len(df_b)), dtype=np.int64)
    else:
        row_idx = np.asarray(indices, dtype=np.int64)
        row_idx = row_idx[(row_idx >= 0) & (row_idx < min(len(df_a), len(df_b)))]

    for bp in common_bps:
        for suffix in ("_x", "_y", "_likelihood"):
            col = f"{bp}{suffix}"
            if col not in df_a.columns or col not in df_b.columns:
                continue
            a_col = df_a.columns.get_loc(col)
            b_col = df_b.columns.get_loc(col)
            tmp = df_a.iloc[row_idx, a_col].copy()
            df_a.iloc[row_idx, a_col] = df_b.iloc[row_idx, b_col].to_numpy()
            df_b.iloc[row_idx, b_col] = tmp.to_numpy()
    return out


def _rename_identity_data(dfs: dict, old_label: str, new_label: str) -> dict:
    out = {}
    for aid, df in dfs.items():
        key = new_label if aid == old_label else aid
        copied = df.copy()
        copied.attrs.update(getattr(df, "attrs", {}))
        out[key] = copied
    return out


def _merge_active_update_into_source(source_dfs: dict, updated_dfs: dict) -> dict:
    """Merge a possibly range-filtered update back into full source tracking."""
    import numpy as np

    if not source_dfs:
        return updated_dfs
    if not updated_dfs:
        return source_dfs

    first_updated = next(iter(updated_dfs.values()), None)
    row_indices = getattr(first_updated, "attrs", {}).get("analysis_row_indices") if first_updated is not None else None
    if row_indices is None:
        return updated_dfs

    row_idx = np.asarray(row_indices, dtype=np.int64)
    merged = {}
    for aid, source_df in source_dfs.items():
        src = source_df.copy()
        src.attrs.update(getattr(source_df, "attrs", {}))
        upd = updated_dfs.get(aid)
        if upd is None:
            merged[aid] = src
            continue
        valid = row_idx[(row_idx >= 0) & (row_idx < len(src))]
        count = min(len(valid), len(upd))
        if count > 0:
            valid = valid[:count]
            for col in upd.columns:
                if col in src.columns:
                    src.iloc[valid, src.columns.get_loc(col)] = upd.iloc[:count][col].to_numpy()
        merged[aid] = src
    return merged
