"""Interactive mask-contact margin calibration dialog."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
)

from dlc_processor.core.mask_social import (
    _animal_center,
    _assign_two_masks,
    _contact_gap_px,
    _frame_numbers,
    _masks_touch,
)
from dlc_processor.ui.video_panel import _ZoomableVideoLabel
from dlc_processor.workers.overlay_worker import _to_qimage


class MaskContactCalibrationDialog(QDialog):
    """Preview mask outlines and the margin used for social mask contact."""

    def __init__(
        self,
        *,
        video_path: str,
        mask_store,
        animal_dfs: dict,
        aid_a: str,
        aid_b: str,
        current_row: int = 0,
        margin_percent: float = 5.0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Mask Contact Calibration")
        self.resize(920, 720)

        self._video_path = str(video_path or "")
        self._mask_store = mask_store
        self._animal_dfs = animal_dfs or {}
        self._aid_a = aid_a
        self._aid_b = aid_b
        self._cap: Optional[cv2.VideoCapture] = None
        self._total_frames = 0
        self._frame_w = 0
        self._frame_h = 0

        self._load_video_info()
        self._setup_ui(float(margin_percent), int(current_row))
        self._render_current()

    def margin_percent(self) -> float:
        return float(self._spin_margin.value())

    def closeEvent(self, event) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        super().closeEvent(event)

    def _load_video_info(self) -> None:
        self._cap = cv2.VideoCapture(self._video_path)
        if self._cap is None or not self._cap.isOpened():
            return
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._frame_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._frame_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def _setup_ui(self, margin_percent: float, current_row: int) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(6)

        self._viewer = _ZoomableVideoLabel()
        self._viewer.setMinimumSize(640, 420)
        self._viewer.setText("No video frame")
        root.addWidget(self._viewer, 1)

        frame_row = QHBoxLayout()
        self._lbl_frame = QLabel("0 / 0")
        self._lbl_frame.setMinimumWidth(140)
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(max(0, self._row_count() - 1))
        self._slider.setValue(max(0, min(current_row, self._slider.maximum())))
        self._slider.valueChanged.connect(lambda _v: self._render_current())
        frame_row.addWidget(self._lbl_frame)
        frame_row.addWidget(self._slider, 1)
        root.addLayout(frame_row)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Mask edge margin:"))
        self._spin_margin = QDoubleSpinBox()
        self._spin_margin.setRange(0.0, 30.0)
        self._spin_margin.setDecimals(1)
        self._spin_margin.setSingleStep(0.5)
        self._spin_margin.setSuffix(" %")
        self._spin_margin.setValue(max(0.0, min(30.0, margin_percent)))
        self._spin_margin.setToolTip("Margin as a percent of the mean selected-mask bounding-box diagonal")
        self._spin_margin.valueChanged.connect(lambda _v: self._render_current())
        controls.addWidget(self._spin_margin)

        btn_zoom_out = QPushButton("-")
        btn_zoom_out.setFixedWidth(28)
        btn_zoom_out.clicked.connect(self._viewer.zoom_out)
        btn_zoom_in = QPushButton("+")
        btn_zoom_in.setFixedWidth(28)
        btn_zoom_in.clicked.connect(self._viewer.zoom_in)
        btn_fit = QPushButton("Fit")
        btn_fit.setObjectName("secondary")
        btn_fit.clicked.connect(self._viewer.reset_view)
        controls.addSpacing(12)
        controls.addWidget(btn_zoom_out)
        controls.addWidget(btn_zoom_in)
        controls.addWidget(btn_fit)
        controls.addStretch()
        root.addLayout(controls)

        self._lbl_status = QLabel("")
        self._lbl_status.setWordWrap(True)
        self._lbl_status.setStyleSheet("color:#a6adc8; font-size:11px;")
        root.addWidget(self._lbl_status)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _row_count(self) -> int:
        if self._animal_dfs:
            return max((len(df) for df in self._animal_dfs.values()), default=0)
        return max(0, self._total_frames)

    def _source_frame_for_row(self, row: int) -> int:
        df = self._animal_dfs.get(self._aid_a)
        if df is not None:
            frames = getattr(df, "attrs", {}).get("frame_numbers")
            if frames is not None and 0 <= row < len(frames):
                return int(np.asarray(frames, dtype=np.int64)[row])
        return int(row)

    def _read_frame(self, source_frame: int) -> Optional[np.ndarray]:
        if self._cap is None or not self._cap.isOpened():
            return None
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, int(source_frame))
        ok, frame = self._cap.read()
        return frame if ok else None

    def _render_current(self) -> None:
        row = int(self._slider.value())
        source_frame = self._source_frame_for_row(row)
        frame = self._read_frame(source_frame)
        if frame is None:
            if self._frame_w > 0 and self._frame_h > 0:
                frame = np.zeros((self._frame_h, self._frame_w, 3), dtype=np.uint8)
            else:
                return

        rendered, status = self._draw_preview(frame, row, source_frame)
        qimg = _to_qimage(rendered)
        self._viewer.set_source_pixmap(QPixmap.fromImage(qimg), smooth=True)
        self._lbl_frame.setText(f"{row} / {max(0, self._row_count() - 1)} (video {source_frame})")
        self._lbl_status.setText(status)

    def _draw_preview(self, frame: np.ndarray, row: int, source_frame: int) -> tuple[np.ndarray, str]:
        out = frame.copy()
        masks = self._mask_store.masks_for_frame(int(source_frame)) if self._mask_store is not None else []
        if len(masks) < 2:
            return out, "No pair of masks on this frame."

        idx_a, idx_b = self._assigned_mask_indices(masks, row)
        if idx_a is None or idx_b is None or idx_a == idx_b:
            return _draw_all_mask_outlines(out, masks), "Could not assign two masks to the selected animals."

        bbox_a = masks[idx_a][2]
        bbox_b = masks[idx_b][2]
        margin_px = _contact_gap_px(bbox_a, bbox_b, 0, float(self._spin_margin.value()))
        contact = _masks_touch(
            masks[idx_a][1],
            masks[idx_b][1],
            max_edge_gap_px=margin_px,
            bbox_a=bbox_a,
            bbox_b=bbox_b,
        )

        out = _draw_all_mask_outlines(out, masks)
        out = _draw_contact_margin(out, masks[idx_a][1], margin_px, (80, 220, 255))
        out = _draw_contact_margin(out, masks[idx_b][1], margin_px, (255, 160, 80))
        out = _draw_selected_mask(out, masks[idx_a][1], (40, 220, 255), f"{self._aid_a}")
        out = _draw_selected_mask(out, masks[idx_b][1], (255, 200, 40), f"{self._aid_b}")

        status = (
            f"{'CONTACT' if contact else 'no contact'} | "
            f"margin {self._spin_margin.value():.1f}% = {margin_px} px on this frame"
        )
        return out, status

    def _assigned_mask_indices(self, masks, row: int) -> tuple[int | None, int | None]:
        df_a = self._animal_dfs.get(self._aid_a)
        df_b = self._animal_dfs.get(self._aid_b)
        if df_a is None or df_b is None:
            return 0, 1 if len(masks) > 1 else None
        centers_a = _animal_center(df_a, len(df_a))
        centers_b = _animal_center(df_b, len(df_b))
        if row >= len(centers_a) or row >= len(centers_b):
            return 0, 1 if len(masks) > 1 else None
        return _assign_two_masks(masks, centers_a[row], centers_b[row])


def _draw_all_mask_outlines(frame: np.ndarray, masks) -> np.ndarray:
    out = frame.copy()
    for track_id, mask, _bbox, _score in masks:
        mask_arr = _frame_sized_mask(mask, out.shape[:2])
        contours, _ = cv2.findContours(mask_arr.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(out, contours, -1, _mask_color(int(track_id)), 1)
    return out


def _draw_contact_margin(frame: np.ndarray, mask: np.ndarray, margin_px: int, color: tuple[int, int, int]) -> np.ndarray:
    out = frame.copy()
    mask_arr = _frame_sized_mask(mask, out.shape[:2])
    if margin_px <= 0:
        return out
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin_px + 1, 2 * margin_px + 1))
    dilated = cv2.dilate(mask_arr.astype(np.uint8), kernel, iterations=1).astype(bool)
    ring = dilated & ~mask_arr
    overlay = out.copy()
    overlay[ring] = (0.45 * np.asarray(color, dtype=np.float32) + 0.55 * overlay[ring].astype(np.float32)).astype(np.uint8)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)
    contours, _ = cv2.findContours(dilated.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(out, contours, -1, color, 2)
    return out


def _draw_selected_mask(frame: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], label: str) -> np.ndarray:
    out = frame.copy()
    mask_arr = _frame_sized_mask(mask, out.shape[:2])
    contours, _ = cv2.findContours(mask_arr.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(out, contours, -1, color, 3)
        pts = np.vstack(contours).reshape(-1, 2)
        x, y = np.min(pts, axis=0)
        cv2.putText(out, label, (int(x), max(16, int(y) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return out


def _frame_sized_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask_arr = np.asarray(mask, dtype=bool)
    if mask_arr.shape[:2] == (h, w):
        return mask_arr
    return cv2.resize(mask_arr.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)


def _mask_color(track_id: int) -> tuple[int, int, int]:
    colors = [
        (255, 120, 50),
        (50, 200, 255),
        (80, 240, 80),
        (80, 80, 255),
        (240, 100, 240),
        (240, 240, 80),
    ]
    return colors[max(0, track_id - 1) % len(colors)]

