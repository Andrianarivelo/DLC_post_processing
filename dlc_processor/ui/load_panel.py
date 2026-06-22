"""Multi-file loading panel for DLC Processor.

Supports:
  - Side-by-side Video Files and DLC Files lists with positional mapping
  - Add / Remove buttons for each list
  - Dragging DLC files (.h5 / .csv), video files (.mp4 ...), or a DLC config.yaml
    directly onto the panel
  - Programmatic loading via add_video_path / add_dlc_path / load_from_folder
  - Selecting a DLC file loads its data and activates the corresponding video

Signals
-------
data_loaded(dict, str)   -- animal_dfs for the active file, path
video_loaded(str)        -- video path
config_loaded(str)       -- DLC config path
time_loaded(object)       -- FrameTimeData or None
masks_loaded(object, str) -- lazy COCO mask store, path
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

_DLC_EXT   = {".h5", ".hdf5", ".csv"}
_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm"}
_TIME_EXT  = {".csv", ".txt", ".tsv"}
_CFG_NAMES = {"config.yaml", "config.yml"}


class LoadPanel(QGroupBox):
    """Multi-file DLC loader with side-by-side video/DLC lists and drag-and-drop."""

    data_loaded    = Signal(dict, str)   # animal_dfs, file_path
    video_loaded   = Signal(str)
    config_loaded  = Signal(str)
    time_loaded    = Signal(object)    # FrameTimeData or None
    masks_loaded   = Signal(object, str)
    files_changed  = Signal()
    metadata_csv_loaded = Signal(str)
    error          = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Load Files", parent)
        # Ordered path lists -- positional mapping: video[i] <-> dlc[i]
        self._video_paths: list[str] = []
        self._dlc_paths: list[str] = []
        self._mask_paths: list[str] = []
        self._time_paths: list[str] = []
        # Loaded DLC data keyed by path
        self._files: dict[str, dict] = {}
        self._animal_ids_by_path: dict[str, list[str]] = {}
        self._time_files: dict[str, object] = {}
        self._active_path = ""
        self._active_time_path = ""
        self._active_mask_path = ""
        self._video_path  = ""
        self._config_path = ""
        self._mask_store = None

        self.setAcceptDrops(True)
        self._setup_ui()

    # -- UI -------------------------------------------------------------------

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(6)

        # -- Side-by-side lists -----------------------------------------------
        lists_row = QHBoxLayout()
        lists_row.setSpacing(8)

        # Left: Video Files
        left = QVBoxLayout()
        left.setSpacing(4)
        lbl_vid = QLabel("Video Files")
        lbl_vid.setStyleSheet("color:#a6adc8; font-size:11px; font-weight:600;")
        left.addWidget(lbl_vid)

        self._video_list = QListWidget()
        self._video_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._video_list.setMinimumHeight(80)
        self._video_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._video_list.currentItemChanged.connect(self._on_video_item_changed)
        self._video_list.itemDoubleClicked.connect(lambda _item: self._on_load_selected())
        left.addWidget(self._video_list, 1)

        vid_btns = QHBoxLayout()
        vid_btns.setSpacing(4)
        self._btn_add_video = QPushButton("+ Add Video")
        self._btn_add_video.setObjectName("secondary")
        self._btn_add_video.setToolTip("Add a video file to pair with the DLC tracking data")
        self._btn_add_video.clicked.connect(self._pick_video)
        self._btn_remove_video = QPushButton("\u2212 Remove")
        self._btn_remove_video.setObjectName("secondary")
        self._btn_remove_video.clicked.connect(self._remove_selected_video)
        self._btn_remove_video.setEnabled(False)
        vid_btns.addWidget(self._btn_add_video)
        vid_btns.addWidget(self._btn_remove_video)
        vid_btns.addStretch()
        left.addLayout(vid_btns)

        lists_row.addLayout(left, 1)

        # Right: DLC Files
        right = QVBoxLayout()
        right.setSpacing(4)
        lbl_dlc = QLabel("DLC Files")
        lbl_dlc.setStyleSheet("color:#a6adc8; font-size:11px; font-weight:600;")
        right.addWidget(lbl_dlc)

        self._dlc_list = QListWidget()
        self._dlc_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._dlc_list.setMinimumHeight(80)
        self._dlc_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._dlc_list.currentItemChanged.connect(self._on_dlc_item_changed)
        self._dlc_list.itemDoubleClicked.connect(lambda _item: self._on_load_selected())
        right.addWidget(self._dlc_list, 1)

        dlc_btns = QHBoxLayout()
        dlc_btns.setSpacing(4)
        self._btn_add_dlc = QPushButton("+ Add DLC File")
        self._btn_add_dlc.setObjectName("secondary")
        self._btn_add_dlc.setToolTip("Add a DeepLabCut H5 or CSV tracking file")
        self._btn_add_dlc.clicked.connect(self._pick_data)
        self._btn_remove_dlc = QPushButton("\u2212 Remove")
        self._btn_remove_dlc.setObjectName("secondary")
        self._btn_remove_dlc.clicked.connect(self._remove_selected_dlc)
        self._btn_remove_dlc.setEnabled(False)
        dlc_btns.addWidget(self._btn_add_dlc)
        dlc_btns.addWidget(self._btn_remove_dlc)
        dlc_btns.addStretch()
        right.addLayout(dlc_btns)

        lists_row.addLayout(right, 1)

        # Time Files: paired by row with video/tracking.
        time_col = QVBoxLayout()
        time_col.setSpacing(4)
        lbl_time_list = QLabel("Time Files")
        lbl_time_list.setStyleSheet("color:#a6adc8; font-size:11px; font-weight:600;")
        time_col.addWidget(lbl_time_list)

        self._time_list = QListWidget()
        self._time_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._time_list.setMinimumHeight(80)
        self._time_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._time_list.currentItemChanged.connect(self._on_time_item_changed)
        self._time_list.itemDoubleClicked.connect(lambda _item: self._on_load_selected())
        time_col.addWidget(self._time_list, 1)

        time_btns = QHBoxLayout()
        time_btns.setSpacing(4)
        self._btn_add_time = QPushButton("+ Time")
        self._btn_add_time.setObjectName("secondary")
        self._btn_add_time.setToolTip("Add a frame-time CSV/TXT file paired with this video")
        self._btn_add_time.clicked.connect(self._pick_time_file)
        self._btn_remove_time = QPushButton("\u2212 Remove")
        self._btn_remove_time.setObjectName("secondary")
        self._btn_remove_time.clicked.connect(self._remove_selected_time)
        self._btn_remove_time.setEnabled(False)
        time_btns.addWidget(self._btn_add_time)
        time_btns.addWidget(self._btn_remove_time)
        time_btns.addStretch()
        time_col.addLayout(time_btns)
        lists_row.addLayout(time_col, 1)

        # Mask Files: paired by row with video/tracking.
        mask_col = QVBoxLayout()
        mask_col.setSpacing(4)
        lbl_mask_list = QLabel("Mask Files")
        lbl_mask_list.setStyleSheet("color:#a6adc8; font-size:11px; font-weight:600;")
        mask_col.addWidget(lbl_mask_list)

        self._mask_list = QListWidget()
        self._mask_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._mask_list.setMinimumHeight(80)
        self._mask_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._mask_list.currentItemChanged.connect(self._on_mask_item_changed)
        self._mask_list.itemDoubleClicked.connect(lambda _item: self._on_load_selected())
        mask_col.addWidget(self._mask_list, 1)

        mask_btns = QHBoxLayout()
        mask_btns.setSpacing(4)
        self._btn_add_mask = QPushButton("+ Masks")
        self._btn_add_mask.setObjectName("secondary")
        self._btn_add_mask.setToolTip("Add a COCO mask JSON paired with this video")
        self._btn_add_mask.clicked.connect(self._pick_masks)
        self._btn_remove_mask = QPushButton("\u2212 Remove")
        self._btn_remove_mask.setObjectName("secondary")
        self._btn_remove_mask.clicked.connect(self._remove_selected_mask)
        self._btn_remove_mask.setEnabled(False)
        mask_btns.addWidget(self._btn_add_mask)
        mask_btns.addWidget(self._btn_remove_mask)
        mask_btns.addStretch()
        mask_col.addLayout(mask_btns)
        lists_row.addLayout(mask_col, 1)

        root.addLayout(lists_row, 1)

        # -- Mapping status ---------------------------------------------------
        nav_row = QHBoxLayout()
        nav_row.setSpacing(4)
        self._btn_prev_pair = QPushButton("Previous")
        self._btn_prev_pair.setObjectName("secondary")
        self._btn_prev_pair.setToolTip("Activate the previous loaded recording row")
        self._btn_prev_pair.clicked.connect(self.activate_previous)
        self._btn_next_pair = QPushButton("Next")
        self._btn_next_pair.setObjectName("secondary")
        self._btn_next_pair.setToolTip("Activate the next loaded recording row")
        self._btn_next_pair.clicked.connect(self.activate_next)
        nav_row.addWidget(self._btn_prev_pair)
        nav_row.addWidget(self._btn_next_pair)
        nav_row.addStretch()
        root.addLayout(nav_row)

        self._lbl_mapping = QLabel("No files loaded.")
        self._lbl_mapping.setWordWrap(True)
        self._lbl_mapping.setStyleSheet("color:#a6adc8; font-size:11px;")
        root.addWidget(self._lbl_mapping)

        # -- Load Selected button ---------------------------------------------
        btn_load_row = QHBoxLayout()
        btn_load_row.setSpacing(4)
        self._btn_load_selected = QPushButton("Load Selected")
        self._btn_load_selected.setObjectName("secondary")
        self._btn_load_selected.setEnabled(False)
        self._btn_load_selected.setToolTip("Activate the selected DLC file and its paired video row")
        self._btn_load_selected.clicked.connect(self._on_load_selected)
        self._btn_load_selected.setVisible(False)
        btn_load_row.addWidget(self._btn_load_selected)
        btn_load_row.addStretch()
        root.addLayout(btn_load_row)

        # -- Drop hint --------------------------------------------------------
        self._lbl_drop = QLabel("\u2190 or drag files/folders here")
        self._lbl_drop.setStyleSheet("color:#45475a; font-size:10px; font-style:italic;")
        root.addWidget(self._lbl_drop)

        # -- Active file status ------------------------------------------------
        self._lbl_status = QLabel("No file selected.")
        self._lbl_status.setWordWrap(True)
        self._lbl_status.setStyleSheet("color:#a6adc8; font-size:11px;")
        root.addWidget(self._lbl_status)

        # -- Separator --------------------------------------------------------
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#3d3d5c; margin:4px 0;")
        root.addWidget(sep)

        # -- DLC config -------------------------------------------------------
        sep_lbl2 = QLabel("DLC Config (optional)")
        sep_lbl2.setStyleSheet("color:#a6adc8; font-size:11px; font-weight:600; margin-top:6px;")
        root.addWidget(sep_lbl2)

        row_c = QHBoxLayout()
        self._lbl_config = QLabel("No config selected.")
        self._lbl_config.setStyleSheet("color:#6c7086; font-size:11px;")
        self._lbl_config.setWordWrap(True)
        btn_cfg = QPushButton("Browse\u2026")
        btn_cfg.setObjectName("secondary")
        btn_cfg.clicked.connect(self._pick_config)
        row_c.addWidget(self._lbl_config, 1)
        row_c.addWidget(btn_cfg)
        root.addLayout(row_c)

        # -- Active sidecar status -------------------------------------------
        mask_lbl = QLabel("Active Masks")
        mask_lbl.setStyleSheet("color:#a6adc8; font-size:11px; font-weight:600; margin-top:6px;")
        root.addWidget(mask_lbl)

        row_m = QHBoxLayout()
        self._lbl_masks = QLabel("No mask file selected.")
        self._lbl_masks.setStyleSheet("color:#6c7086; font-size:11px;")
        self._lbl_masks.setWordWrap(True)
        row_m.addWidget(self._lbl_masks, 1)
        root.addLayout(row_m)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet("color:#3d3d5c; margin:4px 0;")
        root.addWidget(sep2)

        time_lbl = QLabel("Active Time")
        time_lbl.setStyleSheet("color:#a6adc8; font-size:11px; font-weight:600;")
        time_lbl.setToolTip(
            "Load timestamps from a CSV/TXT file to replace the default FPS-based time axis.\n"
            "Useful for syncing with electrophysiology, respiration, or other recordings.\n"
            "File should have one timestamp (in seconds) per line/row."
        )
        root.addWidget(time_lbl)

        row_t = QHBoxLayout()
        self._lbl_time = QLabel("Using FPS-based time.")
        self._lbl_time.setStyleSheet("color:#6c7086; font-size:11px;")
        self._lbl_time.setWordWrap(True)
        row_t.addWidget(self._lbl_time, 1)
        root.addLayout(row_t)

    # -- Drag and drop --------------------------------------------------------

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
        self.handle_paths(url.toLocalFile() for url in event.mimeData().urls())
        event.acceptProposedAction()
        self._lbl_drop.setText("\u2190 or drag files/folders here")

    def dragLeaveEvent(self, event) -> None:
        self._lbl_drop.setText("\u2190 or drag files/folders here")

    # -- Browse slots ---------------------------------------------------------

    def _pick_data(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open DLC Data File(s)", "",
            "DLC Result Files (*.h5 *.hdf5 *.csv);;All Files (*)"
        )
        for path in paths:
            if path:
                self.add_dlc_path(path)

    def _pick_video(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open Video File(s)", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv *.webm);;All Files (*)"
        )
        for path in paths:
            if path:
                self.add_video_path(path)

    def _pick_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open DLC Config", "",
            "DLC Config (config.yaml);;YAML Files (*.yaml *.yml);;All Files (*)"
        )
        if path:
            self._set_config(path)

    def _pick_time_file(self) -> None:
        """Add external timestamps from a CSV or TXT file."""
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open Time File", "",
            "Time Files (*.csv *.txt *.tsv);;All Files (*)"
        )
        for path in paths:
            if path:
                self.add_time_path(path)

    def _pick_masks(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open COCO Mask JSON", "",
            "COCO Mask JSON (*.json);;All Files (*)",
        )
        if path:
            self.add_mask_path(path)

    def _clear_time_file(self) -> None:
        """Clear external timestamps, reverting to FPS-based time."""
        self._active_time_path = ""
        self._lbl_time.setText("Using FPS-based time.")
        self.time_loaded.emit(None)

    # -- Remove slots ---------------------------------------------------------

    def _remove_selected_video(self) -> None:
        item = self._video_list.currentItem()
        if item is None:
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        row = self._video_list.row(item)
        self._video_list.takeItem(row)
        if path in self._video_paths:
            self._video_paths.remove(path)
        self._btn_remove_video.setEnabled(self._video_list.count() > 0)
        self._update_mapping_label()

    def _remove_selected_dlc(self) -> None:
        item = self._dlc_list.currentItem()
        if item is None:
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        row = self._dlc_list.row(item)
        self._dlc_list.takeItem(row)
        if path in self._dlc_paths:
            self._dlc_paths.remove(path)
        self._files.pop(path, None)
        self._animal_ids_by_path.pop(path, None)
        if path == self._active_path:
            self._active_path = ""
            self._lbl_status.setText("No file selected.")
        self._btn_remove_dlc.setEnabled(self._dlc_list.count() > 0)
        self._update_mapping_label()
        self._update_load_selected_btn()

    def _remove_selected_time(self) -> None:
        item = self._time_list.currentItem()
        if item is None:
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        row = self._time_list.row(item)
        self._time_list.takeItem(row)
        if path in self._time_paths:
            self._time_paths.remove(path)
        self._time_files.pop(path, None)
        if path == self._active_time_path:
            self._active_time_path = ""
            self._lbl_time.setText("Using FPS-based time.")
            self.time_loaded.emit(None)
        self._btn_remove_time.setEnabled(self._time_list.count() > 0)
        self._update_mapping_label()
        self._update_load_selected_btn()

    def _remove_selected_mask(self) -> None:
        item = self._mask_list.currentItem()
        if item is None:
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        row = self._mask_list.row(item)
        self._mask_list.takeItem(row)
        if path in self._mask_paths:
            self._mask_paths.remove(path)
        if path == self._active_mask_path:
            self._clear_masks()
        self._btn_remove_mask.setEnabled(self._mask_list.count() > 0)
        self._update_mapping_label()
        self._update_load_selected_btn()

    # -- List selection slots -------------------------------------------------

    def _on_video_item_changed(self, current: QListWidgetItem, _prev) -> None:
        self._btn_remove_video.setEnabled(current is not None)
        if current is None:
            self._update_load_selected_btn()
            return
        self._activate_row(self._video_list.row(current), origin=self._video_list)

    def _on_dlc_item_changed(self, current: QListWidgetItem, _prev) -> None:
        self._btn_remove_dlc.setEnabled(current is not None)
        if current is None:
            self._update_load_selected_btn()
            return
        self._activate_row(self._dlc_list.row(current), origin=self._dlc_list)

    def _on_time_item_changed(self, current: QListWidgetItem, _prev) -> None:
        self._btn_remove_time.setEnabled(current is not None)
        if current is None:
            self._update_load_selected_btn()
            return
        self._activate_row(self._time_list.row(current), origin=self._time_list)

    def _on_mask_item_changed(self, current: QListWidgetItem, _prev) -> None:
        self._btn_remove_mask.setEnabled(current is not None)
        if current is None:
            self._update_load_selected_btn()
            return
        self._activate_row(self._mask_list.row(current), origin=self._mask_list)

    def _on_load_selected(self) -> None:
        """Load the currently selected DLC + video pair."""
        row = self._selected_row()
        if row >= 0:
            self._activate_row(row, origin=None)

    # -- Internal helpers -----------------------------------------------------

    def _selected_row(self) -> int:
        """Return the row selected in any paired list, preferring tracking data."""
        for widget in (self._dlc_list, self._video_list, self._time_list, self._mask_list):
            row = widget.currentRow()
            if row >= 0:
                return row
        return -1

    def _sync_row_selection(self, idx: int, origin: Optional[QListWidget]) -> None:
        """Select the same row in all paired lists without recursively loading data."""
        if idx < 0:
            return
        for widget in (self._video_list, self._dlc_list, self._time_list, self._mask_list):
            if widget is origin:
                continue
            widget.blockSignals(True)
            if idx < widget.count():
                widget.setCurrentRow(idx)
            else:
                widget.clearSelection()
                widget.setCurrentRow(-1)
            widget.blockSignals(False)
        self._update_load_selected_btn()
        self._update_navigation_buttons()

    def _activate_row(self, idx: int, origin: Optional[QListWidget]) -> None:
        """Activate every file available at a paired row."""
        if idx < 0:
            return
        self._sync_row_selection(idx, origin=origin)
        if 0 <= idx < len(self._video_paths):
            self._set_video(self._video_paths[idx])
        self._set_time_at_row(idx)
        self._set_mask_at_row(idx)
        if 0 <= idx < len(self._dlc_paths):
            self._ensure_loaded_and_activate(self._dlc_paths[idx])

    def _set_time_at_row(self, idx: int) -> None:
        if 0 <= idx < len(self._time_paths):
            self._set_time(self._time_paths[idx])
        else:
            self._clear_time_file()

    def _set_mask_at_row(self, idx: int) -> None:
        if 0 <= idx < len(self._mask_paths):
            try:
                self._set_masks(self._mask_paths[idx])
            except Exception as exc:
                logger.exception("Failed to load masks at row %d: %s", idx, exc)
                self._lbl_masks.setText(f"Error loading masks: {Path(self._mask_paths[idx]).name}: {exc}")
                self.error.emit(str(exc))
        else:
            self._clear_masks()

    def _ensure_loaded_and_activate(self, path: str) -> None:
        """Load a DLC file if not yet loaded, then activate it."""
        if path not in self._files:
            self._load_file(path)
        if path in self._files:
            self._activate(path)

    def _load_file(self, path: str) -> None:
        from dlc_processor.core.dlc_loader import load_dlc_file
        try:
            dfs = load_dlc_file(path)
            self._files[path] = dfs
            self._animal_ids_by_path[path] = [str(aid) for aid in dfs.keys()]
        except Exception as exc:
            logger.exception("Failed to load %s: %s", path, exc)
            self._lbl_status.setText(f"Error loading {Path(path).name}: {exc}")
            self.error.emit(str(exc))

    def _activate(self, path: str) -> None:
        if path not in self._files:
            return
        self._active_path = path
        dfs = self._files[path]
        from dlc_processor.core.dlc_loader import get_bodyparts
        animals  = list(dfs.keys())
        n_frames = next(iter(dfs.values())).__len__()
        bps      = get_bodyparts(next(iter(dfs.values())))
        self._lbl_status.setText(
            f"Active: {Path(path).name}\n"
            f"{len(animals)} animal(s): {', '.join(animals)}\n"
            f"{len(bps)} bodyparts \u00b7 {n_frames} frames"
        )
        self.data_loaded.emit(dfs, path)
        source_video = getattr(next(iter(dfs.values())), "attrs", {}).get("source_video", "")
        if source_video and not self._video_path and Path(str(source_video)).exists():
            had_videos = bool(self._video_paths)
            self.add_video_path(str(source_video))
            if had_videos:
                self._set_video(str(source_video))

    def _set_video(self, path: str) -> None:
        self._video_path = path
        self.video_loaded.emit(path)

    def _set_config(self, path: str) -> None:
        self._config_path = path
        self._lbl_config.setText(Path(path).name)
        self.config_loaded.emit(path)

    def _set_time(self, path: str) -> None:
        from dlc_processor.core.time_loader import load_frame_times

        try:
            time_data = self._time_files.get(path)
            if time_data is None:
                time_data = load_frame_times(path)
                self._time_files[path] = time_data
            self._active_time_path = path
            summary = getattr(time_data, "summary", f"{Path(path).name}")
            self._lbl_time.setText(f"{Path(path).name}: {summary}")
            self.time_loaded.emit(time_data)
        except Exception as exc:
            logger.exception("Failed to load time file %s: %s", path, exc)
            self._active_time_path = ""
            self._lbl_time.setText(f"Error loading time file: {Path(path).name}: {exc}")
            self.time_loaded.emit(None)
            self.error.emit(str(exc))

    def _set_masks(self, path: str) -> None:
        from dlc_processor.core.mask_loader import CocoMaskStore

        store = CocoMaskStore.from_file(path)
        self._mask_store = store
        self._active_mask_path = path
        self._lbl_masks.setText(
            f"{Path(path).name} ({store.frame_count} frames, {store.annotation_count} masks)"
        )
        self.masks_loaded.emit(store, path)

    def _clear_masks(self) -> None:
        self._mask_store = None
        self._active_mask_path = ""
        self._lbl_masks.setText("No mask file selected.")
        self.masks_loaded.emit(None, "")

    def _update_mapping_label(self) -> None:
        nv = len(self._video_paths)
        nd = len(self._dlc_paths)
        nt = len(self._time_paths)
        nm = len(self._mask_paths)
        if nv == 0 and nd == 0 and nt == 0 and nm == 0:
            self._lbl_mapping.setText("No files loaded yet. Video, DLC, time, and mask lists are paired by row.")
            self._lbl_mapping.setStyleSheet("color:#a6adc8; font-size:11px;")
            return
        if nv == nd:
            self._lbl_mapping.setText(
                f"{nv} video(s) \u2194 {nd} DLC file(s) \u00b7 {nt} time file(s) \u00b7 {nm} mask file(s) "
                f"\u00b7 rows are paired by position"
            )
            self._lbl_mapping.setStyleSheet("color:#a6adc8; font-size:11px;")
        else:
            self._lbl_mapping.setText(
                f"\u26a0 {nv} video(s) but {nd} DLC file(s) \u00b7 {nt} time file(s) \u00b7 {nm} mask file(s) "
                f"\u00b7 rows are paired by position, so some files have no partner yet"
            )
            self._lbl_mapping.setStyleSheet("color:#fab387; font-size:11px;")
        self._update_navigation_buttons()
        self.files_changed.emit()

    def _update_load_selected_btn(self) -> None:
        dlc_item = self._dlc_list.currentItem()
        vid_item = self._video_list.currentItem()
        time_item = self._time_list.currentItem()
        mask_item = self._mask_list.currentItem()
        has_selection = any(item is not None for item in (dlc_item, vid_item, time_item, mask_item))
        self._btn_load_selected.setEnabled(has_selection)
        if dlc_item is not None and vid_item is not None:
            self._btn_load_selected.setText("Activate Pair")
            self._btn_load_selected.setToolTip("Load the selected row: DLC, video, time, and masks when available")
        elif dlc_item is not None:
            self._btn_load_selected.setText("Activate DLC")
            self._btn_load_selected.setToolTip("Load the selected DLC file and sync its paired video if available")
        elif vid_item is not None:
            self._btn_load_selected.setText("Activate Video")
            self._btn_load_selected.setToolTip("Load the selected video only")
        elif time_item is not None:
            self._btn_load_selected.setText("Activate Time")
            self._btn_load_selected.setToolTip("Load the selected time file")
        elif mask_item is not None:
            self._btn_load_selected.setText("Activate Masks")
            self._btn_load_selected.setToolTip("Load the selected mask file")
        else:
            self._btn_load_selected.setText("Load Selected")
            self._btn_load_selected.setToolTip("Activate the selected paired row")
        self._update_navigation_buttons()

    def _row_count(self) -> int:
        return max(
            len(self._video_paths),
            len(self._dlc_paths),
            len(self._time_paths),
            len(self._mask_paths),
        )

    def active_row(self) -> int:
        row = self._selected_row()
        if row >= 0:
            return row
        if self._active_path and self._active_path in self._dlc_paths:
            return self._dlc_paths.index(self._active_path)
        if self._video_path and self._video_path in self._video_paths:
            return self._video_paths.index(self._video_path)
        return -1

    def _update_navigation_buttons(self) -> None:
        count = self._row_count()
        row = self.active_row()
        self._btn_prev_pair.setEnabled(count > 1 and row > 0)
        self._btn_next_pair.setEnabled(count > 1 and 0 <= row < count - 1)

    def activate_previous(self) -> None:
        row = self.active_row()
        if row > 0:
            self._activate_row(row - 1, origin=None)

    def activate_next(self) -> None:
        row = self.active_row()
        count = self._row_count()
        if 0 <= row < count - 1:
            self._activate_row(row + 1, origin=None)

    def paired_records(self) -> list[dict]:
        """Return loaded video/DLC/time/mask rows for batch processing."""
        records: list[dict] = []
        count = self._row_count()
        for idx in range(count):
            dlc_path = self._dlc_paths[idx] if idx < len(self._dlc_paths) else ""
            video_path = self._video_paths[idx] if idx < len(self._video_paths) else ""
            time_path = self._time_paths[idx] if idx < len(self._time_paths) else ""
            mask_path = self._mask_paths[idx] if idx < len(self._mask_paths) else ""
            if not any((dlc_path, video_path, time_path, mask_path)):
                continue
            records.append({
                "index": idx,
                "dlc_path": dlc_path,
                "video_path": video_path,
                "time_path": time_path,
                "mask_path": mask_path,
                "animal_dfs": self._files.get(dlc_path) if dlc_path else None,
                "animal_ids": list(self._animal_ids_by_path.get(dlc_path, [])) if dlc_path else [],
                "time_data": self._time_files.get(time_path) if time_path else None,
                "active": idx == self.active_row(),
            })
        return records

    def _add_video_to_list(self, path: str) -> QListWidgetItem:
        """Add a video path to the list widget and return the item."""
        item = QListWidgetItem(Path(path).name)
        item.setData(Qt.ItemDataRole.UserRole, path)
        item.setToolTip(path)
        self._video_list.addItem(item)
        return item

    def _add_dlc_to_list(self, path: str) -> QListWidgetItem:
        """Add a DLC path to the list widget and return the item."""
        item = QListWidgetItem(Path(path).name)
        item.setData(Qt.ItemDataRole.UserRole, path)
        item.setToolTip(path)
        self._dlc_list.addItem(item)
        return item

    def _add_time_to_list(self, path: str) -> QListWidgetItem:
        """Add a frame-time path to the list widget and return the item."""
        item = QListWidgetItem(Path(path).name)
        item.setData(Qt.ItemDataRole.UserRole, path)
        item.setToolTip(path)
        self._time_list.addItem(item)
        return item

    def _add_mask_to_list(self, path: str) -> QListWidgetItem:
        """Add a mask path to the list widget and return the item."""
        item = QListWidgetItem(Path(path).name)
        item.setData(Qt.ItemDataRole.UserRole, path)
        item.setToolTip(path)
        self._mask_list.addItem(item)
        return item

    # -- Public API -----------------------------------------------------------

    def add_video_path(self, path: str) -> None:
        """Add a video file to the video list. Skips duplicates."""
        if path in self._video_paths:
            return
        self._video_paths.append(path)
        item = self._add_video_to_list(path)
        if self._video_list.count() == 1:
            self._video_list.setCurrentItem(item)
        self._btn_remove_video.setEnabled(True)
        self._update_mapping_label()

    def add_dlc_path(self, path: str) -> None:
        """Add a DLC file to the DLC list. Skips duplicates."""
        if path in self._dlc_paths:
            return
        self._dlc_paths.append(path)
        item = self._add_dlc_to_list(path)
        if self._dlc_list.count() == 1:
            self._dlc_list.setCurrentItem(item)
        self._btn_remove_dlc.setEnabled(True)
        self._update_mapping_label()
        self._update_load_selected_btn()

    def add_time_path(self, path: str) -> None:
        """Add a frame-time file to the time list. Skips duplicates."""
        if path in self._time_paths:
            return
        self._time_paths.append(path)
        item = self._add_time_to_list(path)
        if self._time_list.count() == 1:
            self._time_list.setCurrentItem(item)
        elif self._selected_row() == self._time_list.row(item):
            self._set_time(path)
        self._btn_remove_time.setEnabled(True)
        self._update_mapping_label()
        self._update_load_selected_btn()

    def add_mask_path(self, path: str) -> None:
        """Add a COCO mask JSON to the mask list. Skips duplicates."""
        if path in self._mask_paths:
            return
        self._mask_paths.append(path)
        item = self._add_mask_to_list(path)
        if self._mask_list.count() == 1:
            self._mask_list.setCurrentItem(item)
        elif self._selected_row() == self._mask_list.row(item):
            self._set_mask_at_row(self._mask_list.row(item))
        self._btn_remove_mask.setEnabled(True)
        self._update_mapping_label()
        self._update_load_selected_btn()

    def load_from_folder(
        self,
        video_paths: list[str],
        dlc_paths: list[str],
        mask_paths: Optional[list[str]] = None,
        time_paths: Optional[list[str]] = None,
    ) -> None:
        """Populate both lists from a folder scan or project load.

        Clears existing entries first, then adds all paths in order.
        """
        mask_paths = mask_paths or []
        time_paths = time_paths or []

        # Clear existing
        self._video_list.clear()
        self._dlc_list.clear()
        self._time_list.clear()
        self._mask_list.clear()
        self._video_paths.clear()
        self._dlc_paths.clear()
        self._mask_paths.clear()
        self._time_paths.clear()
        self._files.clear()
        self._animal_ids_by_path.clear()
        self._time_files.clear()
        self._mask_store = None
        self._active_path = ""
        self._active_time_path = ""
        self._active_mask_path = ""
        self._lbl_status.setText("No file selected.")
        self._lbl_masks.setText("No mask file selected.")
        self._lbl_time.setText("Using FPS-based time.")

        # Add all videos
        for vp in video_paths:
            self._video_paths.append(vp)
            self._add_video_to_list(vp)

        # Add all DLC files
        for dp in dlc_paths:
            self._dlc_paths.append(dp)
            self._add_dlc_to_list(dp)

        for tp in time_paths:
            self._time_paths.append(tp)
            self._add_time_to_list(tp)

        for mp in mask_paths:
            self._mask_paths.append(mp)
            self._add_mask_to_list(mp)

        self._btn_remove_video.setEnabled(self._video_list.count() > 0)
        self._btn_remove_dlc.setEnabled(self._dlc_list.count() > 0)
        self._btn_remove_time.setEnabled(self._time_list.count() > 0)
        self._btn_remove_mask.setEnabled(self._mask_list.count() > 0)
        self._update_mapping_label()
        self._update_load_selected_btn()

        # Auto-select and activate the first paired row.
        if any(widget.count() > 0 for widget in (self._video_list, self._dlc_list, self._time_list, self._mask_list)):
            self._activate_row(0, origin=None)

    def load_file_programmatic(self, path: str) -> None:
        """Load a file from code (e.g. after DLC inference). Backwards-compatible."""
        self.add_dlc_path(path)
        # Select the item in the list
        for i in range(self._dlc_list.count()):
            item = self._dlc_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == path:
                self._dlc_list.setCurrentItem(item)
                break

    def replace_loaded_data(self, path: str, dfs: dict, activate: bool = True) -> None:
        """Replace cached tracking data for a loaded file."""
        if not path:
            return
        self._files[path] = dfs
        self._animal_ids_by_path[path] = [str(aid) for aid in (dfs or {}).keys()]
        if activate:
            self._activate(path)

    def preload_dlc_metadata(self, progress_callback=None) -> int:
        """Cache animal IDs for listed DLC files without activating each row."""
        from dlc_processor.core.dlc_loader import detect_animals

        loaded = 0
        total = len(self._dlc_paths)
        for idx, path in enumerate(self._dlc_paths, start=1):
            if path in self._animal_ids_by_path:
                continue
            if path in self._files:
                self._animal_ids_by_path[path] = [str(aid) for aid in self._files[path].keys()]
                loaded += 1
                continue
            if progress_callback:
                progress_callback(idx - 1, total, Path(path).name)
            try:
                animals = detect_animals(path)
            except Exception as exc:
                logger.exception("Failed to inspect DLC metadata %s: %s", path, exc)
                self.error.emit(str(exc))
                animals = []
            if animals:
                self._animal_ids_by_path[path] = [str(aid) for aid in animals]
                loaded += 1
        if progress_callback:
            progress_callback(total, total, "")
        if loaded:
            self.files_changed.emit()
        return loaded

    def handle_paths(self, paths) -> None:
        """Load dropped or programmatic file/folder paths."""
        from dlc_processor.core.mask_loader import is_coco_mask_json
        from dlc_processor.core.project_manager import is_metadata_table_file, scan_folder_with_sidecars
        from dlc_processor.core.time_loader import is_frame_time_file

        for raw_path in paths:
            if not raw_path:
                continue
            path = str(raw_path)
            p = Path(path)
            ext = p.suffix.lower()
            name = p.name.lower()

            if p.is_dir():
                videos, dlc_files, mask_files, time_files = scan_folder_with_sidecars(p)
                if videos or dlc_files or mask_files or time_files:
                    self.load_from_folder(videos, dlc_files, mask_files, time_files)
                else:
                    logger.debug("Dropped folder ignored (no supported files): %s", path)
                continue

            if name in _CFG_NAMES:
                self._set_config(path)
            elif is_metadata_table_file(p):
                self.metadata_csv_loaded.emit(path)
            elif is_coco_mask_json(p):
                self.add_mask_path(path)
            elif is_frame_time_file(p):
                self.add_time_path(path)
            elif ext in _VIDEO_EXT:
                self.add_video_path(path)
            elif ext in _DLC_EXT:
                self.add_dlc_path(path)
            else:
                logger.debug("Dropped file ignored (unknown type): %s", path)

    @property
    def animal_dfs(self) -> dict:
        return self._files.get(self._active_path, {}) if self._active_path else {}

    @property
    def video_path(self) -> str:
        return self._video_path

    @property
    def config_path(self) -> str:
        return self._config_path

    def set_config_path(self, path: str) -> None:
        """Public config setter used by project restore and sibling panels."""
        if path:
            self._set_config(path)
            return
        self._config_path = ""
        self._lbl_config.setText("No config selected.")
        self.config_loaded.emit("")
