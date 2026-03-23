"""Behavior annotation panel with timeline and tag buttons."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QRect, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ethoscore_tab.core.annotation_store import AnnotationStore

logger = logging.getLogger(__name__)

# Distinct colors per behavior index
_COLORS = [
    "#7c3aed", "#e06c75", "#56b6c2", "#e5c07b",
    "#98c379", "#c678dd", "#61afef", "#d19a66",
]


class TimelineBar(QWidget):
    """Miniature colored timeline showing annotation intervals."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(32)
        self._store: Optional[AnnotationStore] = None
        self._n_frames = 1
        self._current = 0

    def set_store(self, store: AnnotationStore, n_frames: int) -> None:
        self._store = store
        self._n_frames = max(n_frames, 1)
        self.update()

    def set_current_frame(self, fi: int) -> None:
        self._current = fi
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        # Background
        painter.fillRect(0, 0, w, h, QColor("#1e1e2e"))

        if not self._store:
            return

        behaviors = self._store.behaviors()
        n = self._n_frames
        row_h = max(6, h // max(len(behaviors), 1))

        for bi, behavior in enumerate(behaviors):
            color = QColor(_COLORS[bi % len(_COLORS)])
            color.setAlpha(180)
            y = bi * row_h

            for start, stop in self._store.intervals(behavior):
                x1 = int(start / n * w)
                x2 = int(stop  / n * w)
                painter.fillRect(QRect(x1, y, max(1, x2 - x1), row_h - 1), color)

        # Playhead
        px = int(self._current / n * w)
        painter.setPen(QPen(QColor("#ffffff"), 1))
        painter.drawLine(px, 0, px, h)


class AnnotationPanel(QWidget):
    """Full annotation interface: behavior list + frame slider + tag buttons."""

    frame_changed = Signal(int)
    store_updated = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._store = AnnotationStore()
        self._n_frames = 0
        self._current_frame = 0
        self._setup_ui()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        # Behavior list
        beh_grp = QGroupBox("Behaviors")
        beh_lay = QVBoxLayout(beh_grp)
        self._list = QListWidget()
        self._list.setMaximumHeight(120)
        beh_lay.addWidget(self._list)

        btn_row = QHBoxLayout()
        btn_add = QPushButton("+ Add")
        btn_add.setObjectName("secondary")
        btn_add.clicked.connect(self._add_behavior)
        btn_del = QPushButton("Remove")
        btn_del.setObjectName("secondary")
        btn_del.clicked.connect(self._remove_behavior)
        btn_row.addWidget(btn_add)
        btn_row.addWidget(btn_del)
        btn_row.addStretch()
        beh_lay.addLayout(btn_row)
        layout.addWidget(beh_grp)

        # Frame slider
        slider_grp = QGroupBox("Frame")
        slider_lay = QVBoxLayout(slider_grp)
        self._lbl_frame = QLabel("Frame: 0 / 0")
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.valueChanged.connect(self._on_slider)
        slider_lay.addWidget(self._lbl_frame)
        slider_lay.addWidget(self._slider)
        layout.addWidget(slider_grp)

        # Tag buttons
        tag_grp = QGroupBox("Tagging")
        tag_lay = QHBoxLayout(tag_grp)
        self._btn_start = QPushButton("▶ Tag Start")
        self._btn_start.clicked.connect(self._tag_start)
        self._btn_stop = QPushButton("⏹ Tag Stop")
        self._btn_stop.clicked.connect(self._tag_stop)
        self._btn_undo = QPushButton("↩ Undo Last")
        self._btn_undo.setObjectName("secondary")
        self._btn_undo.clicked.connect(self._undo_last)
        tag_lay.addWidget(self._btn_start)
        tag_lay.addWidget(self._btn_stop)
        tag_lay.addWidget(self._btn_undo)
        layout.addWidget(tag_grp)

        # Timeline
        self._timeline = TimelineBar()
        layout.addWidget(self._timeline)

        # Save / Load
        io_row = QHBoxLayout()
        btn_save = QPushButton("Save CSV")
        btn_save.setObjectName("secondary")
        btn_save.clicked.connect(self._save)
        btn_load = QPushButton("Load CSV")
        btn_load.setObjectName("secondary")
        btn_load.clicked.connect(self._load)
        io_row.addWidget(btn_save)
        io_row.addWidget(btn_load)
        io_row.addStretch()
        layout.addLayout(io_row)

    # ── Public API ────────────────────────────────────────────────────────────

    def set_n_frames(self, n: int) -> None:
        self._n_frames = n
        self._store.set_n_frames(n)
        self._slider.setMaximum(max(0, n - 1))
        self._timeline.set_store(self._store, n)

    def store(self) -> AnnotationStore:
        return self._store

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_slider(self, value: int) -> None:
        self._current_frame = value
        self._lbl_frame.setText(f"Frame: {value} / {self._n_frames}")
        self._timeline.set_current_frame(value)
        self.frame_changed.emit(value)

    def _add_behavior(self) -> None:
        name, ok = QInputDialog.getText(self, "New Behavior", "Behavior name:")
        if ok and name.strip():
            self._store.add_behavior(name.strip())
            self._list.addItem(name.strip())
            self._timeline.update()

    def _remove_behavior(self) -> None:
        row = self._list.currentRow()
        if row < 0:
            return
        name = self._list.item(row).text()
        self._store.remove_behavior(name)
        self._list.takeItem(row)
        self._timeline.update()

    def _selected_behavior(self) -> Optional[str]:
        item = self._list.currentItem()
        return item.text() if item else None

    def _tag_start(self) -> None:
        b = self._selected_behavior()
        if b:
            self._store.tag_start(b, self._current_frame)

    def _tag_stop(self) -> None:
        b = self._selected_behavior()
        if b:
            self._store.tag_stop(b, self._current_frame)
            self._timeline.update()
            self.store_updated.emit()

    def _undo_last(self) -> None:
        b = self._selected_behavior()
        if b:
            self._store.remove_last_interval(b)
            self._timeline.update()
            self.store_updated.emit()

    def _save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save Annotations", "", "CSV (*.csv)")
        if path:
            self._store.save_csv(path)

    def _load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load Annotations", "", "CSV (*.csv)")
        if path:
            self._store.load_csv(path)
            self._list.clear()
            for b in self._store.behaviors():
                self._list.addItem(b)
            self._timeline.update()
            self.store_updated.emit()
