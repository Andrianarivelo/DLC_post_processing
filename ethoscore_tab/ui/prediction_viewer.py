"""Ethogram-style prediction viewer.

Displays per-frame predicted behavior labels as a color-coded horizontal bar
(one row per behavior class, filled where that class is predicted).  A vertical
cursor tracks the current frame.  Clicking the bar seeks to that frame.

Signals
-------
frame_seeked(int) — emitted when the user clicks the bar to seek
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

# ── Palette for predicted classes ─────────────────────────────────────────────

_CLASS_COLORS = [
    QColor("#cba6f7"),  # mauve
    QColor("#89b4fa"),  # blue
    QColor("#a6e3a1"),  # green
    QColor("#f38ba8"),  # red
    QColor("#fab387"),  # peach
    QColor("#f9e2af"),  # yellow
    QColor("#94e2d5"),  # teal
    QColor("#eba0ac"),  # pink
]
_BG_COLOR    = QColor("#1e1e2e")
_CURSOR_PEN  = QPen(QColor("#ffffff"), 1, Qt.PenStyle.SolidLine)
_ROW_H       = 20   # px per behavior row


class _EthogramBar(QWidget):
    """Raw painter widget; embedded inside PredictionViewer."""

    frame_seeked = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._predictions: Optional[np.ndarray] = None
        self._class_names: list[str] = []
        self._n_frames: int = 0
        self._current_frame: int = 0
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_predictions(self, predictions: np.ndarray, class_names: list[str]) -> None:
        self._predictions = predictions
        self._class_names = class_names
        self._n_frames    = len(predictions)
        n_rows = len(class_names)
        self.setFixedHeight(max(n_rows * _ROW_H, 1))
        self.update()

    def set_frame(self, frame_idx: int) -> None:
        self._current_frame = frame_idx
        self.update()

    def paintEvent(self, event) -> None:
        if self._predictions is None or self._n_frames == 0:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        w = self.width()
        h = self.height()

        # Background
        p.fillRect(0, 0, w, h, _BG_COLOR)

        n_classes = len(self._class_names)
        row_h = h // max(n_classes, 1)

        # Draw each class row
        for cls_idx, name in enumerate(self._class_names):
            color = _CLASS_COLORS[cls_idx % len(_CLASS_COLORS)]
            dim_color = QColor(color)
            dim_color.setAlpha(40)
            y0 = cls_idx * row_h

            # Dim background for this row
            p.fillRect(0, y0, w, row_h - 1, dim_color)

            # Filled segments where this class is predicted
            mask = (self._predictions == cls_idx)
            if not mask.any():
                continue

            color.setAlpha(200)
            # Run-length encode for efficiency
            in_run = False
            run_start = 0
            for fi in range(self._n_frames + 1):
                active = fi < self._n_frames and mask[fi]
                if active and not in_run:
                    run_start = fi
                    in_run = True
                elif not active and in_run:
                    x0 = int(run_start / self._n_frames * w)
                    x1 = int(fi / self._n_frames * w)
                    p.fillRect(x0, y0, max(x1 - x0, 1), row_h - 1, color)
                    in_run = False

        # Current-frame cursor
        cx = int(self._current_frame / max(self._n_frames - 1, 1) * w)
        p.setPen(_CURSOR_PEN)
        p.drawLine(cx, 0, cx, h)
        p.end()

    def mousePressEvent(self, event) -> None:
        if self._n_frames == 0:
            return
        frac = event.position().x() / max(self.width(), 1)
        frame = int(frac * self._n_frames)
        frame = max(0, min(frame, self._n_frames - 1))
        self.frame_seeked.emit(frame)


class PredictionViewer(QWidget):
    """Full prediction viewer: legend + ethogram bar."""

    frame_seeked = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 2, 4, 2)
        root.setSpacing(4)

        # Header
        hdr = QHBoxLayout()
        lbl = QLabel("Predicted Behaviors")
        lbl.setStyleSheet("color: #a6adc8; font-size: 11px; font-weight: 600;")
        hdr.addWidget(lbl)
        hdr.addStretch()
        self._lbl_frame = QLabel("")
        self._lbl_frame.setStyleSheet("color: #6c7086; font-size: 10px;")
        hdr.addWidget(self._lbl_frame)
        root.addLayout(hdr)

        # Legend
        self._legend = QHBoxLayout()
        self._legend.setSpacing(10)
        root.addLayout(self._legend)

        # Ethogram bar (scrollable horizontally for long videos)
        self._bar = _EthogramBar()
        self._bar.frame_seeked.connect(self._on_seek)

        scroll = QScrollArea()
        scroll.setWidget(self._bar)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setFixedHeight(_ROW_H * 6 + 16)
        root.addWidget(scroll)

        self._placeholder = QLabel("No predictions — run inference first.")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet("color: #45475a; font-style: italic;")
        root.addWidget(self._placeholder)

        self._class_names: list[str] = []

    # ── Public API ─────────────────────────────────────────────────────────────

    def set_predictions(self, predictions: np.ndarray, class_names: list[str]) -> None:
        self._class_names = class_names
        self._bar.set_predictions(predictions, class_names)
        self._placeholder.hide()
        self._bar.parentWidget().show()

        # Rebuild legend
        while self._legend.count():
            item = self._legend.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for i, name in enumerate(class_names):
            color = _CLASS_COLORS[i % len(_CLASS_COLORS)]
            dot = QLabel("●")
            dot.setStyleSheet(f"color: {color.name()}; font-size: 14px;")
            txt = QLabel(name)
            txt.setStyleSheet("color: #cdd6f4; font-size: 11px;")
            self._legend.addWidget(dot)
            self._legend.addWidget(txt)
        self._legend.addStretch()

        # Resize scroll area to fit rows
        n_rows = len(class_names)
        scroll = self._bar.parentWidget().parentWidget()
        if scroll:
            scroll.setFixedHeight(n_rows * _ROW_H + 16)

    def set_current_frame(self, frame_idx: int) -> None:
        self._bar.set_frame(frame_idx)
        n = self._bar._n_frames
        if n > 0:
            cls_idx = int(self._bar._predictions[min(frame_idx, n - 1)])
            name = self._class_names[cls_idx] if cls_idx < len(self._class_names) else str(cls_idx)
            self._lbl_frame.setText(f"Frame {frame_idx}: {name}")

    def _on_seek(self, frame: int) -> None:
        self._bar.set_frame(frame)
        self.frame_seeked.emit(frame)
