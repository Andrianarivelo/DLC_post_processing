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
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QAction, QColor
from PySide6.QtWidgets import (
    QColorDialog,
    QFileDialog,
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
from dlc_processor.ui.roi_panel        import ROIPanel
from dlc_processor.core.settings_store import load_settings, save_settings, add_recent_path
from dlc_processor.core.project_manager import (
    scan_folder, save_project, load_project,
)

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
        self._fps = 25.0
        self._project_path = ""
        self._settings = load_settings()
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
        self.export_panel     = ExportPanel()
        self.refinement_panel = RefinementPanel()
        self.roi_panel        = ROIPanel()
        self.video_panel      = VideoPanel()
        self.plot_panel       = PlotPanel()

        self.add_activity("load",       "file-plus",  "Load",       "Load Files",        self.load_panel)
        self.add_activity("clean",      "sliders",    "Clean",      "Data Cleaning",     self.cleaning_panel)
        self.add_activity("kinematics", "move",       "Kinematics", "Kinematics",        self.kinematics_panel)
        self.add_activity("social",     "users",      "Social",     "Social Behaviors",  self.social_panel)
        self.add_activity("roi",        "crosshair",  "ROI",        "Regions of Interest", self.roi_panel)
        self.add_bar_separator()
        self.add_activity("refine",     "edit-3",     "Refine",     "ID Refinement",     self.refinement_panel)
        self.add_activity("export",     "download",   "Export",     "Export / Inference", self.export_panel)

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

        # Register shortcuts on the widget so they work when tab is active
        for act in file_menu.actions() + settings_menu.actions():
            self.addAction(act)

    # ── Signal wiring ──────────────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        lp = self.load_panel
        lp.data_loaded.connect(self._on_data_loaded)
        lp.video_loaded.connect(self._on_video_loaded)
        lp.config_loaded.connect(self.export_panel.set_config_path)

        self.cleaning_panel.cleaning_requested.connect(self._on_clean)
        self.cleaning_panel.impossible_fix_requested.connect(self._on_fix_impossible)
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
        self.kinematics_panel.computed.connect(self._on_kinematics)
        self.kinematics_panel.heatmap_requested.connect(self._open_egocentric_heatmap)
        self.social_panel.detected.connect(self._on_social)
        self.social_panel.detected.connect(self.export_panel.set_behavior_arrays)
        self.video_panel.frame_changed.connect(self.refinement_panel.set_current_frame)
        self.video_panel.frame_changed.connect(self.plot_panel.set_frame_cursor)
        self.plot_panel.cursor_moved.connect(self._on_plot_cursor)
        self.refinement_panel.data_changed.connect(self._on_data_refined)
        self.export_panel.inference_requested.connect(self._on_inference_done)
        self.export_panel.skeleton_edges_needed.connect(self._provide_skeleton_edges)
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

    def _save_current_settings(self) -> None:
        """Persist current settings to disk."""
        self._settings["skeleton_edges"] = [
            list(e) for e in self.video_panel._skeleton_edges
        ]
        self._settings["fps"] = self._fps
        if self._project_path:
            self._settings["last_project_path"] = self._project_path
        save_settings(self._settings)

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
        elif p.suffix.lower() in {".h5", ".hdf5", ".csv"}:
            self.load_panel.add_dlc_path(path)
        elif p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm"}:
            self.load_panel.add_video_path(path)
        elif p.is_dir():
            videos, dlc_files = scan_folder(p)
            if videos or dlc_files:
                self.load_panel.load_from_folder(videos, dlc_files)

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
        if video_files or dlc_files:
            self.load_panel.load_from_folder(video_files, dlc_files)

        edges = data.get("skeleton_edges", [])
        if edges:
            self.video_panel._skeleton_edges = [tuple(e) for e in edges]

        px_cm = data.get("calibration_px_per_cm", 0.0)
        if px_cm > 0:
            self.cleaning_panel._px_per_cm = px_cm

        proj_settings = data.get("settings", {})
        if proj_settings:
            self._settings.update(proj_settings)
            self._apply_settings()

        self._fps = data.get("fps", 25.0)

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
        data = {
            "video_files": getattr(self.load_panel, "_video_paths", []),
            "dlc_files": getattr(self.load_panel, "_dlc_paths", []),
            "config_path": getattr(self.export_panel, "_config_path", ""),
            "skeleton_edges": self.video_panel._skeleton_edges,
            "calibration_px_per_cm": getattr(self.cleaning_panel, "_px_per_cm", 0.0),
            "fps": self._fps,
            "settings": {
                k: v for k, v in self._settings.items()
                if k not in ("last_dlc_dir", "last_video_dir", "last_project_path")
            },
        }
        save_project(path, data)
        self._add_to_recent(str(path))
        self._save_current_settings()
        logger.info("Project saved: %s", path)

    def _open_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Open Folder with Videos + DLC Files")
        if not folder:
            return
        videos, dlc_files = scan_folder(Path(folder))
        if videos or dlc_files:
            self.load_panel.load_from_folder(videos, dlc_files)
            self._add_to_recent(folder)
        logger.info("Opened folder: %d videos, %d DLC files", len(videos), len(dlc_files))

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

    # ── Handlers ───────────────────────────────────────────────────────────────

    @Slot(dict, str)
    def _on_data_loaded(self, dfs: dict, path: str) -> None:
        self._apply_animal_dfs(dfs, behavior_arrays={})
        # Save current session files for "Load Last Session"
        self._settings["last_session_dlc_files"] = list(self.load_panel._dlc_paths)
        self._settings["last_session_video_files"] = list(self.load_panel._video_paths)
        save_settings(self._settings)
        logger.info("DLC data loaded: %d animals from %s", len(dfs), path)

    @Slot(str)
    def _on_video_loaded(self, path: str) -> None:
        self.video_panel.set_video(path)
        self.export_panel.set_video_path(path)
        self.cleaning_panel.set_video_path(path)
        self.roi_panel.set_video_info(
            self.video_panel._frame_w, self.video_panel._frame_h,
        )
        # Update session files
        self._settings["last_session_video_files"] = list(self.load_panel._video_paths)
        save_settings(self._settings)

    @Slot(dict)
    def _on_clean(self, params: dict) -> None:
        if not self._animal_dfs:
            return
        from dlc_processor.core.data_cleaner import clean_all_animals
        cleaned = clean_all_animals(self._animal_dfs, **params)
        self._apply_animal_dfs(cleaned, behavior_arrays={})

    @Slot(dict)
    def _on_fix_impossible(self, params: dict) -> None:
        if not self._animal_dfs:
            return
        from dlc_processor.core.data_cleaner import fix_impossible_all_animals
        fixed = fix_impossible_all_animals(self._animal_dfs, **params)
        self._apply_animal_dfs(fixed, behavior_arrays={})

    @Slot(float)
    def _on_calibration(self, px_per_cm: float) -> None:
        self._settings["calibration_px_per_cm"] = px_per_cm
        self._save_current_settings()
        logger.info("Calibration updated: %.2f px/cm", px_per_cm)

    @Slot(dict)
    def _on_kinematics(self, result: dict) -> None:
        self._fps = self.kinematics_panel.fps()
        self.export_panel.set_fps(self._fps)
        self._apply_animal_dfs(result, behavior_arrays={})

    @Slot(dict)
    def _on_social(self, arrays: dict) -> None:
        self.video_panel.set_data(self._animal_dfs, arrays)
        self.plot_panel.set_behavior_arrays(arrays)

    @Slot(dict)
    def _on_data_refined(self, dfs: dict) -> None:
        self._apply_animal_dfs(dfs, behavior_arrays={})

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
            self._animal_dfs, fps=self._fps, current_frame=current_frame, parent=self,
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
        self._apply_animal_dfs(dfs, behavior_arrays={})
        logger.info("Track editor changes applied to all panels")

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

    def _load_last_session(self) -> None:
        """Reload the DLC and video files from the previous session."""
        dlc_files = self._settings.get("last_session_dlc_files", [])
        video_files = self._settings.get("last_session_video_files", [])
        # Filter to files that still exist
        dlc_files = [p for p in dlc_files if Path(p).exists()]
        video_files = [p for p in video_files if Path(p).exists()]
        if not dlc_files and not video_files:
            logger.info("No previous session files found.")
            return
        self.load_panel.load_from_folder(video_files, dlc_files)
        logger.info("Last session restored: %d videos, %d DLC files",
                     len(video_files), len(dlc_files))

    @Slot(list)
    def _on_rois_changed(self, rois: list) -> None:
        """Handle updated ROI definitions from ROI panel."""
        # Pass ROIs to video panel for overlay rendering
        self.video_panel._rois = rois
        if self.video_panel._video_path:
            self.video_panel._render_frame(self.video_panel._current_frame)

    def _apply_animal_dfs(self, dfs: dict, behavior_arrays: Optional[dict] = None) -> None:
        """Propagate a new tracking dataset to all dependent panels."""
        self._animal_dfs = dfs
        arrays = behavior_arrays or {}
        if dfs:
            n_frames = max(len(df) for df in dfs.values())
            self.cleaning_panel.set_n_frames(n_frames)
        self.cleaning_panel.set_animal_dfs(dfs)
        self.kinematics_panel.set_animal_dfs(dfs)
        self.social_panel.set_animal_dfs(dfs, fps=self._fps)
        self.refinement_panel.set_animal_dfs(dfs)
        self.roi_panel.set_animal_dfs(dfs, fps=self._fps)
        self.export_panel.set_animal_dfs(dfs)
        self.export_panel.set_behavior_arrays(arrays)
        self.video_panel.set_data(dfs, arrays)
        self.plot_panel.set_animal_dfs(dfs, fps=self._fps)
        self.plot_panel.set_behavior_arrays(arrays)
