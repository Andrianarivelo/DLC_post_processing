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
from PySide6.QtCore import QPointF, QRectF, Qt, Signal, Slot
from PySide6.QtGui import QImage, QKeySequence, QMouseEvent, QPainter, QPixmap, QShortcut, QWheelEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from dlc_processor.core.dlc_loader import get_bodyparts

logger = logging.getLogger(__name__)


class _ZoomableVideoLabel(QLabel):
    """Pixmap viewer with wheel zoom and drag panning."""

    zoom_changed = Signal(float)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._source = QPixmap()
        self._zoom = 1.0
        self._offset = QPointF(0.0, 0.0)
        self._drag_pos: Optional[QPointF] = None
        self._smooth = True
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def set_source_pixmap(self, pixmap: QPixmap, *, smooth: bool = True) -> None:
        self._source = pixmap
        self._smooth = smooth
        self._clamp_offset()
        self.update()

    def reset_view(self) -> None:
        self._zoom = 1.0
        self._offset = QPointF(0.0, 0.0)
        self.zoom_changed.emit(self._zoom)
        self.update()

    def zoom_in(self) -> None:
        self.set_zoom(self._zoom * 1.25)

    def zoom_out(self) -> None:
        self.set_zoom(self._zoom / 1.25)

    def set_zoom(self, value: float, anchor: Optional[QPointF] = None) -> None:
        old_zoom = self._zoom
        new_zoom = max(1.0, min(12.0, float(value)))
        if abs(new_zoom - old_zoom) < 1e-6:
            return
        if anchor is None:
            anchor = QPointF(self.width() / 2.0, self.height() / 2.0)
        view_center = QPointF(self.width() / 2.0, self.height() / 2.0)
        old_center = view_center + self._offset
        scale = new_zoom / old_zoom
        rel = anchor - old_center
        new_center = QPointF(anchor.x() - rel.x() * scale, anchor.y() - rel.y() * scale)
        self._zoom = new_zoom
        self._offset = new_center - view_center
        self._clamp_offset()
        self.zoom_changed.emit(self._zoom)
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        if self._source.isNull():
            painter.setPen(Qt.GlobalColor.gray)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.text())
            painter.end()
            return

        if self._smooth:
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        src_w = max(1, self._source.width())
        src_h = max(1, self._source.height())
        base = min(self.width() / src_w, self.height() / src_h)
        scale = base * self._zoom
        draw_w = src_w * scale
        draw_h = src_h * scale
        center = QPointF(self.width() / 2.0, self.height() / 2.0) + self._offset
        target = QRectF(center.x() - draw_w / 2.0, center.y() - draw_h / 2.0, draw_w, draw_h)
        painter.drawPixmap(target, self._source, QRectF(self._source.rect()))
        painter.end()

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._source.isNull():
            return
        factor = 1.25 if event.angleDelta().y() > 0 else 1.0 / 1.25
        self.set_zoom(self._zoom * factor, event.position())
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._zoom > 1.0:
            self._drag_pos = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_pos is not None:
            self._offset += event.position() - self._drag_pos
            self._drag_pos = event.position()
            self._clamp_offset()
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._drag_pos is not None:
            self._drag_pos = None
            self.setCursor(Qt.CursorShape.OpenHandCursor if self._zoom > 1.0 else Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.reset_view()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def resizeEvent(self, event) -> None:
        self._clamp_offset()
        super().resizeEvent(event)

    def _clamp_offset(self) -> None:
        if self._source.isNull() or self._zoom <= 1.0:
            self._offset = QPointF(0.0, 0.0)
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return

        src_w = max(1, self._source.width())
        src_h = max(1, self._source.height())
        base = min(self.width() / src_w, self.height() / src_h)
        draw_w = src_w * base * self._zoom
        draw_h = src_h * base * self._zoom
        max_x = max(0.0, (draw_w - self.width()) / 2.0)
        max_y = max(0.0, (draw_h - self.height()) / 2.0)
        self._offset.setX(max(-max_x, min(max_x, self._offset.x())))
        self._offset.setY(max(-max_y, min(max_y, self._offset.y())))
        self.setCursor(Qt.CursorShape.OpenHandCursor)


class VideoPanel(QGroupBox):
    """Frame display + playback slider with overlay rendering."""

    frame_changed = Signal(int)  # emitted when slider moves
    range_start_flag_requested = Signal(int)
    range_end_flag_requested = Signal(int)
    range_clear_requested = Signal()
    identity_swap_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Video Preview", parent)
        self._video_path   = ""
        self._animal_dfs   = {}
        self._behavior_arrays: dict = {}
        self._mask_store = None
        self._frame_numbers: Optional[np.ndarray] = None
        self._skeleton_edges: list[tuple[str, str]] = []
        self._total_frames = 0
        self._current_frame = 0
        self._frame_w = 0
        self._frame_h = 0
        self._cap: Optional[cv2.VideoCapture] = None
        self._cap_pos = -1  # source-video frame index the capture is positioned at
        self._playing = False
        self._playback_speed = 1.0
        self._show_subtitles = True
        self._overlay_cache = None  # reusable lightweight overlay helper
        self._rois: list = []  # list of ROIDef objects for overlay
        self._setup_ui()
        self._setup_shortcuts()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(4)

        # Frame display
        self._lbl_frame = _ZoomableVideoLabel()
        self._lbl_frame.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_frame.setMinimumSize(400, 280)
        self._lbl_frame.setStyleSheet("background: #0a0a14; border-radius: 4px;")
        self._lbl_frame.setText("No video loaded")
        self._lbl_frame.zoom_changed.connect(self._on_zoom_changed)
        layout.addWidget(self._lbl_frame, 1)

        # Slider
        slider_row = QHBoxLayout()
        self._lbl_pos = QLabel("0 / 0")
        self._lbl_pos.setMinimumWidth(80)
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.valueChanged.connect(self._on_slider)
        self._slider.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._slider.customContextMenuRequested.connect(self._show_slider_context_menu)
        slider_row.addWidget(self._lbl_pos)
        slider_row.addWidget(self._slider, 1)
        layout.addLayout(slider_row)

        # Playback controls row
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(4)

        self._btn_prev = QPushButton("\u23EE")
        self._btn_prev.setFixedWidth(28)
        self._btn_prev.setToolTip("Previous frame (Left arrow)")
        self._btn_prev.clicked.connect(lambda: self._step_frames(-1))
        ctrl_row.addWidget(self._btn_prev)

        self._btn_play = QPushButton("\u25B6 Play")
        self._btn_play.setObjectName("secondary")
        self._btn_play.setCheckable(True)
        self._btn_play.setToolTip("Play / Pause (Space)")
        self._btn_play.toggled.connect(self._toggle_play)
        ctrl_row.addWidget(self._btn_play)

        self._btn_next = QPushButton("\u23ED")
        self._btn_next.setFixedWidth(28)
        self._btn_next.setToolTip("Next frame (Right arrow)")
        self._btn_next.clicked.connect(lambda: self._step_frames(1))
        ctrl_row.addWidget(self._btn_next)

        self._combo_speed = QComboBox()
        self._combo_speed.setToolTip("Playback speed")
        self._combo_speed.setFixedWidth(70)
        for label, val in [("0.25x", 0.25), ("0.5x", 0.5), ("1x", 1.0), ("2x", 2.0), ("4x", 4.0)]:
            self._combo_speed.addItem(label, val)
        self._combo_speed.setCurrentIndex(2)  # 1x
        self._combo_speed.currentIndexChanged.connect(self._on_speed_changed)
        ctrl_row.addWidget(self._combo_speed)

        ctrl_row.addSpacing(8)

        self._btn_zoom_out = QPushButton("-")
        self._btn_zoom_out.setFixedWidth(28)
        self._btn_zoom_out.setToolTip("Zoom out")
        self._btn_zoom_out.clicked.connect(self._lbl_frame.zoom_out)
        self._lbl_zoom = QLabel("100%")
        self._lbl_zoom.setMinimumWidth(44)
        self._lbl_zoom.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._btn_zoom_in = QPushButton("+")
        self._btn_zoom_in.setFixedWidth(28)
        self._btn_zoom_in.setToolTip("Zoom in")
        self._btn_zoom_in.clicked.connect(self._lbl_frame.zoom_in)
        self._btn_zoom_reset = QPushButton("Fit")
        self._btn_zoom_reset.setObjectName("secondary")
        self._btn_zoom_reset.setToolTip("Reset zoom")
        self._btn_zoom_reset.clicked.connect(self._lbl_frame.reset_view)
        ctrl_row.addWidget(self._btn_zoom_out)
        ctrl_row.addWidget(self._lbl_zoom)
        ctrl_row.addWidget(self._btn_zoom_in)
        ctrl_row.addWidget(self._btn_zoom_reset)

        ctrl_row.addSpacing(8)

        self._chk_masks = QCheckBox("Masks")
        self._chk_masks.setChecked(True)
        self._chk_masks.setToolTip("Show instance masks when a COCO mask file is loaded")
        self._chk_skel = QCheckBox("Skeleton")
        self._chk_skel.setChecked(True)
        self._chk_skel.setToolTip("Show skeleton overlay (S)")
        self._chk_labels = QCheckBox("Labels")
        self._chk_labels.setChecked(True)
        self._chk_labels.setToolTip("Show bodypart labels")
        self._chk_behavior = QCheckBox("Behavior")
        self._chk_behavior.setChecked(True)
        self._chk_behavior.setToolTip("Show behavior labels on frame")
        self._chk_skel_only = QCheckBox("Skeleton Only")
        self._chk_skel_only.setToolTip("Hide video, show skeleton on black background")
        self._chk_subtitle = QCheckBox("Info")
        self._chk_subtitle.setChecked(True)
        self._chk_subtitle.setToolTip("Show frame info and active behaviors as overlay text")
        self._chk_subtitle.toggled.connect(
            lambda: self._render_frame(self._current_frame) if self._video_path else None
        )
        self._btn_edit_skel = QPushButton("Edit Skeleton")
        self._btn_edit_skel.setObjectName("secondary")
        self._btn_edit_skel.setToolTip("Open the skeleton editor to define bodypart connections")
        self._btn_edit_skel.clicked.connect(self._open_skeleton_editor)

        ctrl_row.addWidget(self._chk_masks)
        ctrl_row.addWidget(self._chk_skel)
        ctrl_row.addWidget(self._chk_labels)
        ctrl_row.addWidget(self._chk_behavior)
        ctrl_row.addWidget(self._chk_skel_only)
        ctrl_row.addWidget(self._chk_subtitle)
        ctrl_row.addWidget(self._btn_edit_skel)
        ctrl_row.addStretch()
        layout.addLayout(ctrl_row)

        # Timer for playback
        from PySide6.QtCore import QTimer
        self._timer = QTimer(self)
        self._timer.setInterval(33)   # ~30 fps default
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
        for chk in (
            self._chk_masks,
            self._chk_skel,
            self._chk_labels,
            self._chk_behavior,
            self._chk_skel_only,
            self._chk_subtitle,
        ):
            chk.toggled.connect(self._request_rerender)
        self._update_ui_state()

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

    def _source_frame_for_row(self, row_idx: int) -> int:
        if self._frame_numbers is not None and 0 <= row_idx < len(self._frame_numbers):
            return int(self._frame_numbers[row_idx])
        return int(row_idx)

    def _read_frame(self, row_idx: int) -> Optional[np.ndarray]:
        """Read the source-video frame for tracking row *row_idx*.

        Uses sequential read if source_frame == cap_pos (fast path during playback).
        Falls back to seek + read otherwise.
        """
        cap = self._ensure_cap()
        if cap is None:
            return None

        source_frame = self._source_frame_for_row(row_idx)
        if source_frame != self._cap_pos:
            cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame)

        ret, frame = cap.read()
        if ret:
            self._cap_pos = source_frame + 1
            return frame
        return None

    # ── Public API ────────────────────────────────────────────────────────────

    def set_video(self, path: str) -> None:
        self._release_cap()
        self._video_path = path
        cap = self._ensure_cap()
        if cap is None:
            self._update_ui_state()
            return
        self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._sync_slider_range()
        self._update_ui_state()
        self._render_frame(self._slider.value())

    def set_data(self, animal_dfs: dict, behavior_arrays: Optional[dict] = None) -> None:
        self._animal_dfs = animal_dfs
        self._behavior_arrays = behavior_arrays or {}
        self._frame_numbers = _frame_numbers_from_dfs(animal_dfs)
        self._sync_slider_range()
        self._update_ui_state()
        self._overlay_cache = None  # invalidate
        # Auto-apply default skeleton if none set by user
        if not self._skeleton_edges and animal_dfs:
            self._auto_apply_default_skeleton(animal_dfs)
        if self._video_path:
            self._render_frame(self._current_frame)

    def set_masks(self, mask_store) -> None:
        self._mask_store = mask_store
        self._chk_masks.setEnabled(mask_store is not None)
        if self._video_path:
            self._render_frame(self._current_frame)

    def _request_rerender(self, *_args) -> None:
        if self._video_path:
            self._render_frame(self._current_frame)

    def _update_ui_state(self) -> None:
        has_video = bool(self._video_path) and self._total_frames > 0
        has_tracks = bool(self._animal_dfs)
        has_behaviors = bool(self._behavior_arrays)
        has_masks = self._mask_store is not None

        self._slider.setEnabled(has_video)
        self._btn_prev.setEnabled(has_video)
        self._btn_next.setEnabled(has_video)
        self._btn_play.setEnabled(has_video)
        self._combo_speed.setEnabled(has_video)
        self._btn_zoom_out.setEnabled(has_video)
        self._btn_zoom_in.setEnabled(has_video)
        self._btn_zoom_reset.setEnabled(has_video)
        self._chk_skel_only.setEnabled(has_video)
        self._chk_masks.setEnabled(has_video and has_masks)
        self._chk_subtitle.setEnabled(has_video)
        self._chk_skel.setEnabled(has_video and has_tracks)
        self._chk_labels.setEnabled(has_video and has_tracks)
        self._chk_behavior.setEnabled(has_video and has_behaviors)
        self._btn_edit_skel.setEnabled(has_video and has_tracks)

        if not has_video:
            self._btn_play.setChecked(False)
            self._btn_play.setText("▶ Play")
            self._lbl_pos.setText("0 / 0")
            self._lbl_frame.setText("No video loaded")
            self._lbl_frame.reset_view()

    def _sync_slider_range(self) -> None:
        """Use tracking rows as timeline when data is loaded."""
        if self._frame_numbers is not None and len(self._frame_numbers) > 0:
            max_row = len(self._frame_numbers) - 1
        else:
            max_row = max(0, self._total_frames - 1)
        self._slider.setMinimum(0)
        self._slider.setMaximum(max(0, max_row))
        if self._current_frame > max_row:
            self._current_frame = 0
            self._slider.setValue(0)
        self._lbl_pos.setText(self._format_position(self._current_frame))

    def _auto_apply_default_skeleton(self, animal_dfs: dict) -> None:
        """Set skeleton edges from the default template, filtered to available bodyparts."""
        from dlc_processor.core.dlc_loader import get_bodyparts
        from dlc_processor.workers.overlay_worker import _SKELETON_EDGES, resolve_skeleton_edges
        first_df = next(iter(animal_dfs.values()))
        bps = get_bodyparts(first_df)
        resolved = resolve_skeleton_edges(list(_SKELETON_EDGES), bps)
        if resolved:
            self._skeleton_edges = resolved

    # ── Playback ──────────────────────────────────────────────────────────────

    @Slot(int)
    def _on_slider(self, value: int) -> None:
        self._current_frame = value
        self._lbl_pos.setText(self._format_position(value))
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
        if next_f > self._slider.maximum():
            self._btn_play.setChecked(False)
            return
        self._slider.setValue(next_f)

    def _frame_from_slider_pos(self, pos) -> int:
        if self._slider.maximum() <= self._slider.minimum():
            return int(self._slider.value())
        width = max(1, self._slider.width())
        frac = max(0.0, min(1.0, float(pos.x()) / float(width)))
        lo = self._slider.minimum()
        hi = self._slider.maximum()
        return int(round(lo + frac * (hi - lo)))

    def _show_slider_context_menu(self, pos) -> None:
        if not self._slider.isEnabled():
            return
        frame = self._frame_from_slider_pos(pos)
        menu = QMenu(self)

        act_start = menu.addAction(f"Set start flag at frame {frame}")
        act_start.triggered.connect(
            lambda _checked=False, frame=frame: self.range_start_flag_requested.emit(frame)
        )

        act_end = menu.addAction(f"Set end flag at frame {frame}")
        act_end.triggered.connect(
            lambda _checked=False, frame=frame: self.range_end_flag_requested.emit(frame)
        )

        menu.addSeparator()

        act_swap = menu.addAction("Swap Mouse Identity")
        act_swap.triggered.connect(lambda _checked=False: self.identity_swap_requested.emit())

        act_clear = menu.addAction("Clear flagged range")
        act_clear.triggered.connect(lambda _checked=False: self.range_clear_requested.emit())

        menu.exec(self._slider.mapToGlobal(pos))

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

        if self._chk_masks.isChecked() and self._mask_store is not None:
            frame = _draw_masks(
                frame,
                self._mask_store,
                self._source_frame_for_row(fi),
                animal_dfs=self._animal_dfs,
                row_idx=fi,
            )

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
        w.frame_index_mode = "row"

        try:
            rendered = w._render(frame, fi, list(self._animal_dfs.keys()))
        except Exception:
            logger.warning("Overlay rendering failed for frame %d", fi, exc_info=True)
            rendered = frame

        # Draw ROI overlays
        if self._rois:
            rendered = _draw_rois(rendered, self._rois)

        # Subtitle overlay
        if self._chk_subtitle.isChecked():
            rendered = self._draw_subtitle(rendered, fi)

        qimg   = _to_qimage(rendered)
        pixmap = QPixmap.fromImage(qimg)

        self._lbl_frame.set_source_pixmap(pixmap, smooth=not self._playing)

    # ── Shortcuts ─────────────────────────────────────────────────────────────

    def _setup_shortcuts(self) -> None:
        """Register keyboard shortcuts on this widget."""
        sc = QShortcut
        sc(QKeySequence(Qt.Key.Key_Space), self, self._shortcut_play_pause)
        sc(QKeySequence(Qt.Key.Key_Left), self, lambda: self._step_frames(-1))
        sc(QKeySequence(Qt.Key.Key_Right), self, lambda: self._step_frames(1))
        sc(QKeySequence("Ctrl+Left"), self, lambda: self._step_frames(-10))
        sc(QKeySequence("Ctrl+Right"), self, lambda: self._step_frames(10))
        sc(QKeySequence("Shift+Left"), self, lambda: self._step_frames(-30))
        sc(QKeySequence("Shift+Right"), self, lambda: self._step_frames(30))
        sc(QKeySequence(Qt.Key.Key_Home), self, lambda: self._slider.setValue(0))
        sc(QKeySequence(Qt.Key.Key_End), self, lambda: self._slider.setValue(self._slider.maximum()))
        sc(QKeySequence(Qt.Key.Key_S), self, lambda: self._chk_skel.toggle())
        sc(QKeySequence(Qt.Key.Key_I), self, lambda: self._chk_subtitle.toggle())
        sc(QKeySequence("Ctrl++"), self, self._lbl_frame.zoom_in)
        sc(QKeySequence("Ctrl+="), self, self._lbl_frame.zoom_in)
        sc(QKeySequence("Ctrl+-"), self, self._lbl_frame.zoom_out)
        sc(QKeySequence("Ctrl+0"), self, self._lbl_frame.reset_view)

    def _on_zoom_changed(self, zoom: float) -> None:
        self._lbl_zoom.setText(f"{int(round(zoom * 100))}%")

    def _shortcut_play_pause(self) -> None:
        self._btn_play.setChecked(not self._btn_play.isChecked())

    def _step_frames(self, delta: int) -> None:
        """Step forward/backward by *delta* frames."""
        if self._playing:
            self._btn_play.setChecked(False)
        new_f = max(self._slider.minimum(), min(self._slider.maximum(), self._current_frame + delta))
        self._slider.setValue(new_f)

    def _on_speed_changed(self, idx: int) -> None:
        self._playback_speed = self._combo_speed.currentData() or 1.0
        interval = max(8, int(round(33 / self._playback_speed)))
        self._timer.setInterval(interval)

    def _format_position(self, row_idx: int) -> str:
        if self._frame_numbers is not None and len(self._frame_numbers) > 0:
            source_frame = self._source_frame_for_row(row_idx)
            return (
                f"{row_idx} / {len(self._frame_numbers)} "
                f"(video {source_frame} / {self._total_frames})"
            )
        return f"{row_idx} / {self._total_frames}"

    # ── Subtitle overlay ─────────────────────────────────────────────────────

    def _draw_subtitle(self, frame: np.ndarray, fi: int) -> np.ndarray:
        """Draw frame info + active behaviors as translucent overlay text."""
        h, w = frame.shape[:2]
        lines: list[str] = [self._format_position(fi)]

        # Active behaviors at this frame
        if self._behavior_arrays:
            active = []
            for name, arr in self._behavior_arrays.items():
                a = np.asarray(arr)
                if fi < len(a) and a.dtype == bool and a[fi]:
                    active.append(_behavior_label(name))
            if active:
                lines.append(" | ".join(active))

        # Draw background bar + text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.4, min(0.55, w / 1400))
        thickness = 1
        y_offset = h - 8
        for line in reversed(lines):
            (tw, th), _ = cv2.getTextSize(line, font, font_scale, thickness)
            # Semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay, (4, y_offset - th - 6), (tw + 12, y_offset + 4), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
            cv2.putText(frame, line, (8, y_offset - 2), font, font_scale, (220, 220, 230), thickness, cv2.LINE_AA)
            y_offset -= th + 10

        return frame

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


def _draw_masks(
    frame: np.ndarray,
    mask_store,
    source_frame: int,
    animal_dfs: Optional[dict] = None,
    row_idx: Optional[int] = None,
) -> np.ndarray:
    masks = mask_store.masks_for_frame(source_frame)
    if not masks:
        return frame

    animals = list((animal_dfs or {}).keys())
    assignments = _assign_masks_to_animals(masks, animal_dfs or {}, row_idx)

    out = frame.copy()
    overlay = out.copy()
    h, w = out.shape[:2]
    for mask_idx, (track_id, mask, bbox, _score) in enumerate(masks):
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape[:2] != (h, w):
            mask_arr = cv2.resize(
                mask_arr.astype(np.uint8),
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        animal_id = assignments.get(mask_idx)
        if animal_id is not None and animal_id in animals:
            animal_index = animals.index(animal_id)
            color = _animal_color(animal_index)
            label = animal_id
        else:
            color = _mask_color(int(track_id))
            label = f"id {track_id}"
        overlay[mask_arr] = (
            0.35 * np.asarray(color, dtype=np.float32)
            + 0.65 * overlay[mask_arr].astype(np.float32)
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_arr.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if contours:
            cv2.drawContours(out, contours, -1, color, 2)
        x, y, bw, bh = bbox
        if bw > 0 and bh > 0:
            cv2.putText(
                out,
                label,
                (int(x), max(14, int(y) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)
    return out


def _assign_masks_to_animals(masks, animal_dfs: dict, row_idx: Optional[int]) -> dict[int, str]:
    if not animal_dfs or not masks:
        return {}

    assignments: dict[int, str] = {}
    animals = [str(animal_id) for animal_id in animal_dfs.keys()]
    used_animals: set[str] = set()
    available_tracks = sorted({int(track_id) for track_id, _mask, _bbox, _score in masks})

    for mask_idx, (track_id, _mask, _bbox, _score) in enumerate(masks):
        animal_id = _animal_for_mask_track(int(track_id), animals, available_tracks)
        if animal_id is None or animal_id in used_animals:
            continue
        assignments[mask_idx] = animal_id
        used_animals.add(animal_id)
    return assignments


def _animal_for_mask_track(track_id: int, animals: list[str], available_tracks: list[int]) -> Optional[str]:
    import re

    for animal_id in animals:
        numbers = re.findall(r"\d+", str(animal_id))
        if not numbers:
            continue
        parsed = int(numbers[-1])
        if parsed == track_id:
            return animal_id
        if parsed == 0 and track_id == 1:
            return animal_id

    if track_id in available_tracks:
        pos = available_tracks.index(track_id)
        if 0 <= pos < len(animals):
            return animals[pos]
    return None


def _animal_center_for_row(df, row_idx: int) -> Optional[np.ndarray]:
    if df is None or row_idx < 0 or row_idx >= len(df):
        return None

    row = df.iloc[row_idx]
    for bp in ("body_center", "body_centre", "center", "centre", "centroid", "neck"):
        point = _point_from_row(row, bp)
        if point is not None:
            return point

    nose = _first_point_from_row(row, ("nose", "snout"))
    tail = _first_point_from_row(row, ("tailbase", "tail_base", "tail"))
    if nose is not None and tail is not None:
        return (nose + tail) / 2.0

    points = []
    for bp in get_bodyparts(df):
        point = _point_from_row(row, bp)
        if point is not None:
            points.append(point)
    if not points:
        return None
    return np.mean(np.vstack(points), axis=0)


def _first_point_from_row(row, bodyparts: tuple[str, ...]) -> Optional[np.ndarray]:
    for bp in bodyparts:
        point = _point_from_row(row, bp)
        if point is not None:
            return point
    return None


def _point_from_row(row, bodypart: str) -> Optional[np.ndarray]:
    x = row.get(f"{bodypart}_x", np.nan)
    y = row.get(f"{bodypart}_y", np.nan)
    try:
        x = float(x)
        y = float(y)
    except Exception:
        return None
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    return np.asarray([x, y], dtype=np.float64)


def _mask_center(mask, bbox) -> Optional[np.ndarray]:
    x, y, bw, bh = bbox
    try:
        x = float(x)
        y = float(y)
        bw = float(bw)
        bh = float(bh)
    except Exception:
        x = y = bw = bh = 0.0
    if bw > 0 and bh > 0:
        return np.asarray([x + bw / 2.0, y + bh / 2.0], dtype=np.float64)

    mask_arr = np.asarray(mask, dtype=bool)
    if not mask_arr.any():
        return None
    ys, xs = np.nonzero(mask_arr)
    return np.asarray([float(np.mean(xs)), float(np.mean(ys))], dtype=np.float64)


def _animal_color(animal_index: int) -> tuple[int, int, int]:
    colors = [
        (255, 120, 50),
        (50, 200, 255),
        (80, 240, 80),
        (80, 80, 255),
        (240, 100, 240),
        (240, 240, 80),
        (150, 80, 240),
        (80, 200, 200),
    ]
    return colors[max(0, int(animal_index)) % len(colors)]


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


def _behavior_label(name: str) -> str:
    if "__" in name:
        animal, metric = name.split("__", 1)
        if metric in {"immobile", "is_immobile"}:
            return f"{animal} immobile"
        if metric in {"mobile", "is_mobile"}:
            return f"{animal} mobile"
    return name.replace("_", " ")


def _frame_numbers_from_dfs(animal_dfs: dict) -> Optional[np.ndarray]:
    if not animal_dfs:
        return None
    first_df = next(iter(animal_dfs.values()), None)
    if first_df is None:
        return None
    frames = getattr(first_df, "attrs", {}).get("frame_numbers")
    if frames is None:
        return None
    arr = np.asarray(frames, dtype=np.int64).reshape(-1)
    if len(arr) != len(first_df):
        return None
    return arr
