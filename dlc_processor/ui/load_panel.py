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
_CFG_NAMES = {"config.yaml", "config.yml"}


class LoadPanel(QGroupBox):
    """Multi-file DLC loader with side-by-side video/DLC lists and drag-and-drop."""

    data_loaded   = Signal(dict, str)   # animal_dfs, file_path
    video_loaded  = Signal(str)
    config_loaded = Signal(str)
    error         = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Load Files", parent)
        # Ordered path lists -- positional mapping: video[i] <-> dlc[i]
        self._video_paths: list[str] = []
        self._dlc_paths: list[str] = []
        # Loaded DLC data keyed by path
        self._files: dict[str, dict] = {}
        self._active_path = ""
        self._video_path  = ""
        self._config_path = ""

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
        left.addWidget(self._video_list, 1)

        vid_btns = QHBoxLayout()
        vid_btns.setSpacing(4)
        self._btn_add_video = QPushButton("+ Add Video")
        self._btn_add_video.setObjectName("secondary")
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
        right.addWidget(self._dlc_list, 1)

        dlc_btns = QHBoxLayout()
        dlc_btns.setSpacing(4)
        self._btn_add_dlc = QPushButton("+ Add File")
        self._btn_add_dlc.setObjectName("secondary")
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
        root.addLayout(lists_row, 1)

        # -- Mapping status ---------------------------------------------------
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
        self._btn_load_selected.clicked.connect(self._on_load_selected)
        btn_load_row.addWidget(self._btn_load_selected)
        btn_load_row.addStretch()
        root.addLayout(btn_load_row)

        # -- Drop hint --------------------------------------------------------
        self._lbl_drop = QLabel("\u2190 or drag files here")
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
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if not path:
                continue
            p = Path(path)
            ext = p.suffix.lower()
            name = p.name.lower()

            if ext in _DLC_EXT:
                self.add_dlc_path(path)
            elif ext in _VIDEO_EXT:
                self.add_video_path(path)
            elif name in _CFG_NAMES:
                self._set_config(path)
            else:
                logger.debug("Dropped file ignored (unknown type): %s", path)

        event.acceptProposedAction()
        self._lbl_drop.setText("\u2190 or drag files here")

    def dragLeaveEvent(self, event) -> None:
        self._lbl_drop.setText("\u2190 or drag files here")

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
        if path == self._active_path:
            self._active_path = ""
            self._lbl_status.setText("No file selected.")
        self._btn_remove_dlc.setEnabled(self._dlc_list.count() > 0)
        self._update_mapping_label()
        self._update_load_selected_btn()

    # -- List selection slots -------------------------------------------------

    def _on_video_item_changed(self, current: QListWidgetItem, _prev) -> None:
        self._btn_remove_video.setEnabled(current is not None)
        self._update_load_selected_btn()
        if current is None:
            return
        idx = self._video_list.row(current)
        path = current.data(Qt.ItemDataRole.UserRole)
        self._set_video(path)
        # Sync DLC list selection to same index (if exists)
        if 0 <= idx < self._dlc_list.count():
            self._dlc_list.blockSignals(True)
            self._dlc_list.setCurrentRow(idx)
            self._dlc_list.blockSignals(False)

    def _on_dlc_item_changed(self, current: QListWidgetItem, _prev) -> None:
        self._btn_remove_dlc.setEnabled(current is not None)
        self._update_load_selected_btn()
        if current is None:
            return
        idx = self._dlc_list.row(current)
        path = current.data(Qt.ItemDataRole.UserRole)
        # Load and activate the DLC data
        self._ensure_loaded_and_activate(path)
        # If a video exists at the same index, emit video_loaded
        if 0 <= idx < len(self._video_paths):
            vid_path = self._video_paths[idx]
            self._set_video(vid_path)
            # Sync video list selection
            self._video_list.blockSignals(True)
            self._video_list.setCurrentRow(idx)
            self._video_list.blockSignals(False)

    def _on_load_selected(self) -> None:
        """Load the currently selected DLC + video pair."""
        dlc_item = self._dlc_list.currentItem()
        vid_item = self._video_list.currentItem()

        if dlc_item is not None:
            dlc_path = dlc_item.data(Qt.ItemDataRole.UserRole)
            self._ensure_loaded_and_activate(dlc_path)

        if vid_item is not None:
            vid_path = vid_item.data(Qt.ItemDataRole.UserRole)
            self._set_video(vid_path)

    # -- Internal helpers -----------------------------------------------------

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

    def _set_video(self, path: str) -> None:
        self._video_path = path
        self.video_loaded.emit(path)

    def _set_config(self, path: str) -> None:
        self._config_path = path
        self._lbl_config.setText(Path(path).name)
        self.config_loaded.emit(path)

    def _update_mapping_label(self) -> None:
        nv = len(self._video_paths)
        nd = len(self._dlc_paths)
        if nv == 0 and nd == 0:
            self._lbl_mapping.setText("No files loaded.")
            return
        if nv == nd:
            self._lbl_mapping.setText(f"{nv} video(s) \u2194 {nd} DLC file(s)")
            self._lbl_mapping.setStyleSheet("color:#a6adc8; font-size:11px;")
        else:
            self._lbl_mapping.setText(f"\u26a0 {nv} video(s) but {nd} DLC file(s)")
            self._lbl_mapping.setStyleSheet("color:#fab387; font-size:11px;")

    def _update_load_selected_btn(self) -> None:
        has_selection = (
            self._dlc_list.currentItem() is not None
            or self._video_list.currentItem() is not None
        )
        self._btn_load_selected.setEnabled(has_selection)

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

    def load_from_folder(self, video_paths: list[str], dlc_paths: list[str]) -> None:
        """Populate both lists from a folder scan or project load.

        Clears existing entries first, then adds all paths in order.
        """
        # Clear existing
        self._video_list.clear()
        self._dlc_list.clear()
        self._video_paths.clear()
        self._dlc_paths.clear()
        self._files.clear()
        self._active_path = ""
        self._lbl_status.setText("No file selected.")

        # Add all videos
        for vp in video_paths:
            self._video_paths.append(vp)
            self._add_video_to_list(vp)

        # Add all DLC files
        for dp in dlc_paths:
            self._dlc_paths.append(dp)
            self._add_dlc_to_list(dp)

        self._btn_remove_video.setEnabled(self._video_list.count() > 0)
        self._btn_remove_dlc.setEnabled(self._dlc_list.count() > 0)
        self._update_mapping_label()
        self._update_load_selected_btn()

        # Auto-select first items
        if self._dlc_list.count() > 0:
            self._dlc_list.setCurrentRow(0)
        if self._video_list.count() > 0:
            self._video_list.blockSignals(True)
            self._video_list.setCurrentRow(0)
            self._video_list.blockSignals(False)

    def load_file_programmatic(self, path: str) -> None:
        """Load a file from code (e.g. after DLC inference). Backwards-compatible."""
        self.add_dlc_path(path)
        # Select the item in the list
        for i in range(self._dlc_list.count()):
            item = self._dlc_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == path:
                self._dlc_list.setCurrentItem(item)
                break

    @property
    def animal_dfs(self) -> dict:
        return self._files.get(self._active_path, {}) if self._active_path else {}

    @property
    def video_path(self) -> str:
        return self._video_path

    @property
    def config_path(self) -> str:
        return self._config_path
