"""Video viewer with keypoint overlay for DLC Processor.

Performance notes
-----------------
- A persistent cv2.VideoCapture is kept open to avoid re-opening per frame.
- During playback, sequential reads (cap.read()) are used — no seeking.
- Seeking only happens on slider drag or single-frame jumps.
- QPixmap scaling uses FastTransformation during playback, Smooth on pause.
- Overlay rendering is done inline (cheap for keypoints + skeleton).
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from dlc_processor.core.dlc_loader import get_bodyparts

logger = logging.getLogger(__name__)


class VideoPanel(QGroupBox):
    """Frame display + playback slider with overlay rendering."""

    frame_changed = Signal(int)  # emitted when slider moves

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Video Preview", parent)
        self._video_path   = ""
        self._animal_dfs   = {}
        self._behavior_arrays: dict = {}
        self._skeleton_edges: list[tuple[str, str]] = []
        self._total_frames = 0
        self._current_frame = 0
        self._frame_w = 0
        self._frame_h = 0
        self._cap: Optional[cv2.VideoCapture] = None
        self._cap_pos = -1  # frame index the capture is positioned at
        self._playing = False
        self._overlay_cache = None  # reusable lightweight overlay helper
        self._rois: list = []  # list of ROIDef objects for overlay
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(4)

        # Frame display
        self._lbl_frame = QLabel()
        self._lbl_frame.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_frame.setMinimumSize(400, 280)
        self._lbl_frame.setStyleSheet("background: #0a0a14; border-radius: 4px;")
        self._lbl_frame.setText("<span style='color:#45475a;'>No video loaded</span>")
        layout.addWidget(self._lbl_frame, 1)

        # Slider
        slider_row = QHBoxLayout()
        self._lbl_pos = QLabel("0 / 0")
        self._lbl_pos.setMinimumWidth(80)
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.valueChanged.connect(self._on_slider)
        slider_row.addWidget(self._lbl_pos)
        slider_row.addWidget(self._slider, 1)
        layout.addLayout(slider_row)

        # Controls
        ctrl_row = QHBoxLayout()
        self._btn_play = QPushButton("▶ Play")
        self._btn_play.setObjectName("secondary")
        self._btn_play.setCheckable(True)
        self._btn_play.toggled.connect(self._toggle_play)
        self._chk_skel  = QCheckBox("Skeleton")
        self._chk_skel.setChecked(True)
        self._chk_labels = QCheckBox("Labels")
        self._chk_labels.setChecked(True)
        self._chk_behavior = QCheckBox("Behavior")
        self._chk_behavior.setChecked(True)
        self._chk_skel_only = QCheckBox("Skeleton Only")
        self._btn_edit_skel = QPushButton("Edit Skeleton")
        self._btn_edit_skel.setObjectName("secondary")
        self._btn_edit_skel.clicked.connect(self._open_skeleton_editor)
        ctrl_row.addWidget(self._btn_play)
        ctrl_row.addWidget(self._chk_skel)
        ctrl_row.addWidget(self._chk_labels)
        ctrl_row.addWidget(self._chk_behavior)
        ctrl_row.addWidget(self._chk_skel_only)
        ctrl_row.addWidget(self._btn_edit_skel)
        ctrl_row.addStretch()
        layout.addLayout(ctrl_row)

        # Timer for playback
        from PySide6.QtCore import QTimer
        self._timer = QTimer(self)
        self._timer.setInterval(40)   # ~25 fps
        self._timer.timeout.connect(self._advance_frame)

        # Debounce timer for slider scrubbing (avoids expensive seeks per pixel)
        self._render_debounce = QTimer(self)
        self._render_debounce.setSingleShot(True)
        self._render_debounce.setInterval(35)
        self._render_debounce.timeout.connect(self._render_deferred)
        self._pending_render_frame = 0

        self._chk_skel_only.toggled.connect(
            lambda: self._render_frame(self._current_frame) if self._video_path else None
        )

    # ── Video capture management ──────────────────────────────────────────────

    def _ensure_cap(self) -> Optional[cv2.VideoCapture]:
        """Return the persistent VideoCapture, opening if needed."""
        if self._cap is not None and self._cap.isOpened():
            return self._cap
        if not self._video_path:
            return None
        self._cap = cv2.VideoCapture(self._video_path)
        self._cap_pos = -1
        return self._cap if self._cap.isOpened() else None

    def _release_cap(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self._cap_pos = -1

    def _read_frame(self, fi: int) -> Optional[np.ndarray]:
        """Read frame *fi* from the persistent capture.

        Uses sequential read if fi == cap_pos (fast path during playback).
        Falls back to seek + read otherwise.
        """
        cap = self._ensure_cap()
        if cap is None:
            return None

        if fi != self._cap_pos:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)

        ret, frame = cap.read()
        if ret:
            self._cap_pos = fi + 1
            return frame
        return None

    # ── Public API ────────────────────────────────────────────────────────────

    def set_video(self, path: str) -> None:
        self._release_cap()
        self._video_path = path
        cap = self._ensure_cap()
        if cap is None:
            return
        self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._slider.setMaximum(max(0, self._total_frames - 1))
        self._slider.setValue(0)
        self._render_frame(0)

    def set_data(self, animal_dfs: dict, behavior_arrays: Optional[dict] = None) -> None:
        self._animal_dfs = animal_dfs
        self._behavior_arrays = behavior_arrays or {}
        self._overlay_cache = None  # invalidate
        if self._video_path:
            self._render_frame(self._current_frame)

    # ── Playback ──────────────────────────────────────────────────────────────

    @Slot(int)
    def _on_slider(self, value: int) -> None:
        self._current_frame = value
        self._lbl_pos.setText(f"{value} / {self._total_frames}")
        self.frame_changed.emit(value)
        if self._playing:
            # During playback sequential reads are fast — render immediately
            self._render_frame(value)
        else:
            # Debounce during scrubbing to avoid expensive seeks per pixel
            self._pending_render_frame = value
            self._render_debounce.start()

    def _render_deferred(self) -> None:
        """Render the last pending frame after debounce timeout."""
        self._render_frame(self._pending_render_frame)

    def _toggle_play(self, checked: bool) -> None:
        self._playing = checked
        if checked:
            self._btn_play.setText("⏸ Pause")
            self._timer.start()
        else:
            self._btn_play.setText("▶ Play")
            self._timer.stop()
            # Re-render with smooth scaling on pause
            self._render_frame(self._current_frame)

    def _advance_frame(self) -> None:
        next_f = self._current_frame + 1
        if next_f >= self._total_frames:
            self._btn_play.setChecked(False)
            return
        self._slider.setValue(next_f)

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _render_frame(self, fi: int) -> None:
        if not self._video_path:
            return

        skel_only = self._chk_skel_only.isChecked()

        if skel_only:
            frame = np.zeros((self._frame_h, self._frame_w, 3), dtype=np.uint8)
        else:
            frame = self._read_frame(fi)
            if frame is None:
                return

        # Build or reuse overlay helper
        from dlc_processor.workers.overlay_worker import OverlayWorker, _to_qimage
        if self._overlay_cache is None:
            w = OverlayWorker.__new__(OverlayWorker)
            self._overlay_cache = w
        else:
            w = self._overlay_cache

        w.animal_dfs      = self._animal_dfs
        w.behavior_arrays = self._behavior_arrays
        w.draw_skeleton   = self._chk_skel.isChecked()
        w.draw_labels     = self._chk_labels.isChecked()
        w.draw_behaviors  = self._chk_behavior.isChecked()
        w.skeleton_edges  = self._skeleton_edges or None
        w.fill_body       = skel_only

        try:
            rendered = w._render(frame, fi, list(self._animal_dfs.keys()))
        except Exception:
            rendered = frame

        # Draw ROI overlays
        if self._rois:
            rendered = _draw_rois(rendered, self._rois)

        qimg   = _to_qimage(rendered)
        pixmap = QPixmap.fromImage(qimg)

        # Fast scaling during playback, smooth when paused/scrubbing
        transform = (
            Qt.TransformationMode.FastTransformation
            if self._playing
            else Qt.TransformationMode.SmoothTransformation
        )
        scaled = pixmap.scaled(
            self._lbl_frame.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            transform,
        )
        self._lbl_frame.setPixmap(scaled)

    # ── Skeleton editor ───────────────────────────────────────────────────────

    def _open_skeleton_editor(self) -> None:
        """Open the skeleton editor dialog with current bodypart names."""
        all_bps: list[str] = []
        for df in self._animal_dfs.values():
            for bp in get_bodyparts(df):
                if bp not in all_bps:
                    all_bps.append(bp)

        if not all_bps:
            return

        from dlc_processor.ui.skeleton_editor import SkeletonEditorDialog
        dlg = SkeletonEditorDialog(all_bps, self._skeleton_edges or None, parent=self)
        if dlg.exec() == SkeletonEditorDialog.DialogCode.Accepted:
            self._skeleton_edges = dlg.get_edges()
            self._overlay_cache = None  # force rebuild
            if self._video_path:
                self._render_frame(self._current_frame)


def _draw_rois(frame: np.ndarray, rois: list) -> np.ndarray:
    """Draw ROI shapes on the frame with semi-transparent fill and labels."""
    out = frame.copy()
    for roi in rois:
        # Parse hex color to BGR
        hex_c = getattr(roi, "color", "#e74c3c")
        r_c = int(hex_c[1:3], 16)
        g_c = int(hex_c[3:5], 16)
        b_c = int(hex_c[5:7], 16)
        bgr = (b_c, g_c, r_c)

        poly = roi.as_polygon()
        if len(poly) < 3:
            continue
        pts = np.array(poly, dtype=np.int32)

        # Semi-transparent fill
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], bgr)
        cv2.addWeighted(overlay, 0.2, out, 0.8, 0, out)

        # Border
        cv2.polylines(out, [pts], isClosed=True, color=bgr, thickness=2)

        # Label
        cx = int(np.mean(pts[:, 0]))
        cy = int(np.mean(pts[:, 1]))
        cv2.putText(
            out, roi.name, (cx - 20, cy - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1, cv2.LINE_AA,
        )
    return out
