"""Interactive calibration dialog — draw a line of known length on a video frame."""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal, QPointF
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QMouseEvent
from PySide6.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class _FrameLabel(QLabel):
    """QLabel that supports interactive line drawing."""

    line_drawn = Signal(float)  # pixel distance of drawn line

    def __init__(self, parent=None):
        super().__init__(parent)
        self._drawing = False
        self._start: Optional[QPointF] = None
        self._end: Optional[QPointF] = None
        self._pixmap_orig: Optional[QPixmap] = None
        self._scale_x = 1.0
        self._scale_y = 1.0

    def set_frame_pixmap(self, pixmap: QPixmap) -> None:
        self._pixmap_orig = pixmap
        self._update_display()

    def enable_drawing(self, enabled: bool) -> None:
        self._drawing = enabled
        self.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)
        if enabled:
            self._start = None
            self._end = None
            self._update_display()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if not self._drawing or event.button() != Qt.MouseButton.LeftButton:
            return
        if self._start is None:
            self._start = event.position()
        else:
            self._end = event.position()
            self._drawing = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self._update_display()
            # Compute pixel distance in original image coords
            if self._pixmap_orig and self.pixmap():
                displayed = self.pixmap()
                sx = self._pixmap_orig.width() / max(displayed.width(), 1)
                sy = self._pixmap_orig.height() / max(displayed.height(), 1)
                # Account for centering offset
                offset_x = (self.width() - displayed.width()) / 2
                offset_y = (self.height() - displayed.height()) / 2
                x1 = (self._start.x() - offset_x) * sx
                y1 = (self._start.y() - offset_y) * sy
                x2 = (self._end.x() - offset_x) * sx
                y2 = (self._end.y() - offset_y) * sy
                dist = np.hypot(x2 - x1, y2 - y1)
                self.line_drawn.emit(dist)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drawing and self._start is not None:
            self._end = event.position()
            self._update_display()

    def _update_display(self) -> None:
        if self._pixmap_orig is None:
            return
        scaled = self._pixmap_orig.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        if self._start is not None and self._end is not None:
            painter = QPainter(scaled)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            pen = QPen(QColor("#f38ba8"), 2)
            painter.setPen(pen)
            # Map widget coords to scaled pixmap coords
            offset_x = (self.width() - scaled.width()) / 2
            offset_y = (self.height() - scaled.height()) / 2
            p1 = QPointF(self._start.x() - offset_x, self._start.y() - offset_y)
            p2 = QPointF(self._end.x() - offset_x, self._end.y() - offset_y)
            painter.drawLine(p1, p2)
            # Draw endpoint circles
            painter.setBrush(QColor("#f38ba8"))
            painter.drawEllipse(p1, 5, 5)
            painter.drawEllipse(p2, 5, 5)
            painter.end()
        self.setPixmap(scaled)


class CalibrationDialog(QDialog):
    """Modal dialog for interactive spatial calibration."""

    calibration_done = Signal(float)  # px_per_cm

    def __init__(self, video_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Spatial Calibration")
        self.setMinimumSize(700, 550)
        self.setStyleSheet(
            "QDialog { background: #1e1e2e; }"
            "QLabel { color: #cdd6f4; }"
            "QPushButton { color: #cdd6f4; background: #313244; border: 1px solid #45475a;"
            "  border-radius: 4px; padding: 6px 14px; }"
            "QPushButton:hover { background: #45475a; }"
            "QDoubleSpinBox { color: #cdd6f4; background: #313244; border: 1px solid #45475a;"
            "  border-radius: 4px; padding: 4px; }"
        )

        self._video_path = video_path
        self._total_frames = 0
        self._px_distance = 0.0
        self._px_per_cm = 0.0

        self._build_ui()
        self._load_video_info()
        self._show_frame(0)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Instructions
        instr = QLabel("Navigate to a frame, click 'Draw Line', then click two points "
                       "of known distance. Enter the real-world length to compute the scale.")
        instr.setWordWrap(True)
        instr.setStyleSheet("color: #a6adc8; font-size: 11px; padding: 4px;")
        layout.addWidget(instr)

        # Frame display
        self._frame_label = _FrameLabel()
        self._frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._frame_label.setMinimumSize(640, 360)
        self._frame_label.setStyleSheet("background: #0a0a14; border-radius: 4px;")
        self._frame_label.line_drawn.connect(self._on_line_drawn)
        layout.addWidget(self._frame_label, 1)

        # Slider
        slider_row = QHBoxLayout()
        self._lbl_frame = QLabel("0 / 0")
        self._lbl_frame.setMinimumWidth(80)
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.valueChanged.connect(self._on_slider)
        slider_row.addWidget(self._lbl_frame)
        slider_row.addWidget(self._slider, 1)
        layout.addLayout(slider_row)

        # Controls
        ctrl_row = QHBoxLayout()

        self._btn_draw = QPushButton("Draw Line")
        self._btn_draw.clicked.connect(lambda: self._frame_label.enable_drawing(True))
        ctrl_row.addWidget(self._btn_draw)

        ctrl_row.addWidget(QLabel("Known length:"))
        self._spin_length = QDoubleSpinBox()
        self._spin_length.setRange(0.1, 10000.0)
        self._spin_length.setValue(10.0)
        self._spin_length.setSuffix(" cm")
        self._spin_length.setDecimals(2)
        ctrl_row.addWidget(self._spin_length)

        self._btn_compute = QPushButton("Compute Scale")
        self._btn_compute.clicked.connect(self._compute)
        self._btn_compute.setEnabled(False)
        ctrl_row.addWidget(self._btn_compute)

        ctrl_row.addStretch()
        layout.addLayout(ctrl_row)

        # Result
        self._lbl_result = QLabel("")
        self._lbl_result.setStyleSheet("color: #a6e3a1; font-size: 12px; font-weight: 600;")
        layout.addWidget(self._lbl_result)

        # OK / Cancel
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_ok = QPushButton("Apply")
        btn_ok.clicked.connect(self._apply)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_ok)
        btn_row.addWidget(btn_cancel)
        layout.addLayout(btn_row)

    def _load_video_info(self) -> None:
        cap = cv2.VideoCapture(self._video_path)
        if cap.isOpened():
            self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._slider.setMaximum(max(0, self._total_frames - 1))
        cap.release()

    def _show_frame(self, fi: int) -> None:
        cap = cv2.VideoCapture(self._video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, w * ch, QImage.Format.Format_RGB888).copy()
        self._frame_label.set_frame_pixmap(QPixmap.fromImage(qimg))

    def _on_slider(self, value: int) -> None:
        self._lbl_frame.setText(f"{value} / {self._total_frames}")
        self._show_frame(value)

    def _on_line_drawn(self, px_dist: float) -> None:
        self._px_distance = px_dist
        self._btn_compute.setEnabled(True)
        self._lbl_result.setText(f"Line: {px_dist:.1f} px")

    def _compute(self) -> None:
        known_cm = self._spin_length.value()
        if known_cm > 0 and self._px_distance > 0:
            self._px_per_cm = self._px_distance / known_cm
            self._lbl_result.setText(
                f"Line: {self._px_distance:.1f} px = {known_cm:.2f} cm  →  "
                f"Scale: {self._px_per_cm:.2f} px/cm"
            )

    def _apply(self) -> None:
        self._compute()
        if self._px_per_cm <= 0:
            QMessageBox.warning(
                self,
                "Calibration incomplete",
                "Draw a line and set its known length before applying calibration.",
            )
            return
        self.calibration_done.emit(float(self._px_per_cm))
        self.accept()

    def px_per_cm(self) -> float:
        return self._px_per_cm
