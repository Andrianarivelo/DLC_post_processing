"""Interactive plot panel: metric time series + behaviour lanes."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

_PLOT_COLORS = [
    "#f38ba8",
    "#89b4fa",
    "#a6e3a1",
    "#fab387",
    "#cba6f7",
    "#f9e2af",
    "#94e2d5",
    "#f5c2e7",
    "#74c7ec",
    "#b4befe",
]

_GANTT_COLORS = [
    (243, 139, 168, 200),
    (137, 180, 250, 200),
    (166, 227, 161, 200),
    (250, 179, 135, 200),
    (203, 166, 247, 200),
    (249, 226, 175, 200),
    (148, 226, 213, 200),
    (245, 194, 231, 200),
    (116, 199, 236, 200),
]

_METRIC_COLUMNS = [
    "body_speed_px_s",
    "body_accel_px_s2",
    "body_jerk_px_s3",
    "body_orientation_deg",
    "body_angle_rate_deg_fr",
    "distance_traveled_px",
    "path_tortuosity",
    "body_elongation_px",
    "trajectory_curvature_1_px",
    "head_direction_deg",
    "heading_body_angle_diff_deg",
    "inter_animal_dist_px",
    "approach_speed_px_s",
    "relative_heading_deg",
]


class PlotPanel(QWidget):
    """Interactive plot panel with playback-friendly metric rendering."""

    cursor_moved = Signal(int)
    cleaning_region_changed = Signal(int, int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._animal_dfs: dict = {}
        self._behavior_arrays: dict = {}
        self._fps: float = 25.0
        self._n_frames: int = 0
        self._current_frame: int = 0
        self._follow_playback: bool = True
        self._show_cleaning_region: bool = False
        self._cleaning_range: tuple[int, int] = (0, 0)
        self._cleaning_region: Optional[pg.LinearRegionItem] = None
        self._cleaning_region_gantt: Optional[pg.LinearRegionItem] = None
        self._metric_checks: dict[str, QCheckBox] = {}
        self._metric_full_series: dict[str, tuple[np.ndarray, str]] = {}
        self._metric_curves: dict[str, pg.PlotDataItem] = {}
        self._metric_view_signature: Optional[tuple] = None
        self._updating_metric_view: bool = False
        self._setup_ui()

    def _setup_ui(self) -> None:
        pg.setConfigOptions(
            antialias=False,
            background="#1e1e2e",
            foreground="#cdd6f4",
            imageAxisOrder="row-major",
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        toolbar = QWidget()
        toolbar.setStyleSheet("background: #181825; border-bottom: 1px solid #313244;")
        tb_lay = QHBoxLayout(toolbar)
        tb_lay.setContentsMargins(6, 2, 6, 2)
        tb_lay.setSpacing(6)

        lbl = QLabel("Metrics:")
        lbl.setStyleSheet("color: #a6adc8; font-size: 11px; font-weight: 600;")
        tb_lay.addWidget(lbl)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(30)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        chk_container = QWidget()
        self._metric_checks_layout = QHBoxLayout(chk_container)
        self._metric_checks_layout.setContentsMargins(0, 0, 0, 0)
        self._metric_checks_layout.setSpacing(4)
        for col in _METRIC_COLUMNS:
            self._ensure_metric_checkbox(col)
        self._metric_checks_layout.addStretch()
        scroll.setWidget(chk_container)
        tb_lay.addWidget(scroll, 1)

        tb_lay.addWidget(QLabel("Animal:"))
        self._combo_animal = QComboBox()
        self._combo_animal.setMinimumWidth(96)
        self._combo_animal.currentIndexChanged.connect(self._refresh_metric_plot)
        tb_lay.addWidget(self._combo_animal)

        self._chk_follow = QCheckBox("Follow")
        self._chk_follow.setChecked(True)
        self._chk_follow.setStyleSheet("color: #cdd6f4; font-size: 10px;")
        self._chk_follow.toggled.connect(self._on_follow_toggled)
        tb_lay.addWidget(self._chk_follow)

        self._chk_auto_y = QCheckBox("Auto Y")
        self._chk_auto_y.setChecked(True)
        self._chk_auto_y.setStyleSheet("color: #cdd6f4; font-size: 10px;")
        self._chk_auto_y.toggled.connect(lambda _checked: self._update_metric_view(force=True))
        tb_lay.addWidget(self._chk_auto_y)

        btn_reset = QPushButton("Reset")
        btn_reset.setToolTip("Reset plot view")
        btn_reset.setFixedHeight(24)
        btn_reset.setStyleSheet(
            "color: #cdd6f4; background: #313244; border: 1px solid #45475a;"
            " border-radius: 4px; padding: 0 8px;"
        )
        btn_reset.clicked.connect(self._reset_zoom)
        tb_lay.addWidget(btn_reset)

        root.addWidget(toolbar)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setStyleSheet("QSplitter::handle { background: #313244; height: 3px; }")

        self._metric_plot = pg.PlotWidget()
        self._metric_plot.setLabel("bottom", "Frame")
        self._metric_plot.setLabel("left", "Value")
        self._metric_plot.showGrid(x=True, y=True, alpha=0.16)
        self._metric_plot.getPlotItem().getViewBox().setMouseMode(pg.ViewBox.RectMode)
        self._metric_plot.getPlotItem().getViewBox().sigRangeChanged.connect(
            self._on_metric_range_changed
        )
        self._metric_plot.getPlotItem().getViewBox().sigRangeChangedManually.connect(
            self._on_manual_zoom
        )
        self._metric_plot.scene().sigMouseClicked.connect(self._on_metric_click)
        self._cursor_line_metric = pg.InfiniteLine(
            pos=0,
            angle=90,
            pen=pg.mkPen("#f9e2af", width=1, style=Qt.PenStyle.DashLine),
        )
        self._metric_marker = pg.ScatterPlotItem(size=7, pen=pg.mkPen("#11111b", width=1))
        self._metric_plot.addItem(self._cursor_line_metric)
        self._metric_plot.addItem(self._metric_marker)
        splitter.addWidget(self._metric_plot)

        self._gantt_widget = pg.PlotWidget()
        self._gantt_widget.setLabel("bottom", "Frame")
        self._gantt_widget.setLabel("left", "")
        self._gantt_widget.showGrid(x=True, y=False, alpha=0.14)
        self._gantt_widget.getPlotItem().getViewBox().setMouseMode(pg.ViewBox.RectMode)
        self._gantt_widget.getPlotItem().getViewBox().invertY(True)
        self._gantt_widget.scene().sigMouseClicked.connect(self._on_gantt_click)
        self._cursor_line_gantt = pg.InfiniteLine(
            pos=0,
            angle=90,
            pen=pg.mkPen("#f9e2af", width=1, style=Qt.PenStyle.DashLine),
        )
        self._gantt_widget.addItem(self._cursor_line_gantt)
        self._gantt_widget.setXLink(self._metric_plot)
        splitter.addWidget(self._gantt_widget)

        splitter.setSizes([230, 170])
        root.addWidget(splitter, 1)

        if "body_speed_px_s" in self._metric_checks:
            self._metric_checks["body_speed_px_s"].setChecked(True)

    def set_animal_dfs(self, dfs: dict, fps: float = 25.0) -> None:
        self._animal_dfs = dfs
        self._fps = fps
        self._n_frames = max((len(df) for df in dfs.values()), default=0)

        prev = self._combo_animal.currentText()
        self._combo_animal.blockSignals(True)
        self._combo_animal.clear()
        for aid in dfs:
            self._combo_animal.addItem(str(aid))
        if prev:
            idx = self._combo_animal.findText(prev)
            if idx >= 0:
                self._combo_animal.setCurrentIndex(idx)
        self._combo_animal.blockSignals(False)

        if dfs:
            first_df = next(iter(dfs.values()))
            for col in first_df.columns:
                if col.endswith(("_px_s", "_px_s2", "_px_s3", "_px", "_deg", "_deg_fr", "_1_px")):
                    self._ensure_metric_checkbox(col)

        self._refresh_metric_plot()
        self._refresh_gantt()

    def set_behavior_arrays(self, arrays: dict) -> None:
        self._behavior_arrays = arrays
        for name, arr in arrays.items():
            if not _is_boolean_like(arr):
                self._ensure_metric_checkbox(name, checked=False)
        self._refresh_gantt()
        self._refresh_metric_plot()

    def set_frame_cursor(self, frame_idx: int) -> None:
        self._current_frame = frame_idx
        self._cursor_line_metric.setValue(frame_idx)
        self._cursor_line_gantt.setValue(frame_idx)
        self._update_metric_marker()

        if not (self._follow_playback and self._chk_follow.isChecked() and self._n_frames > 0):
            return

        vb = self._metric_plot.getPlotItem().getViewBox()
        xmin, xmax = vb.viewRange()[0]
        window = max(0.0, xmax - xmin)
        target_window = float(self._default_window())

        if window < 2:
            self._set_default_x_range(center_on=frame_idx)
            return

        if window > target_window * 1.5:
            self._set_default_x_range(center_on=frame_idx)
            return

        if frame_idx < xmin + window * 0.15 or frame_idx > xmin + window * 0.85:
            new_xmin = frame_idx - window * 0.2
            max_xmin = max(0.0, self._n_frames - window)
            new_xmin = float(np.clip(new_xmin, 0.0, max_xmin))
            self._metric_plot.setXRange(new_xmin, new_xmin + window, padding=0)

    def _refresh_metric_plot(self) -> None:
        prev_range = self._metric_plot.getPlotItem().getViewBox().viewRange()
        plot_item = self._metric_plot.getPlotItem()
        legend = getattr(plot_item, "legend", None)
        if legend is not None:
            try:
                legend.scene().removeItem(legend)
            except Exception:
                pass
            plot_item.legend = None

        self._metric_plot.clear()
        self._metric_plot.addItem(self._cursor_line_metric)
        self._metric_plot.addItem(self._metric_marker)
        self._metric_full_series = {}
        self._metric_curves = {}
        self._metric_view_signature = None

        aid = self._combo_animal.currentText()
        if not aid and self._animal_dfs:
            aid = str(next(iter(self._animal_dfs)))
        legend = None
        color_idx = 0

        for col, chk in self._metric_checks.items():
            if not chk.isChecked():
                continue
            y = self._metric_array_for(col, aid)
            if y is None:
                continue

            color = _PLOT_COLORS[color_idx % len(_PLOT_COLORS)]
            curve = pg.PlotDataItem(
                pen=pg.mkPen(color, width=1.5),
                connect="finite",
                name=_short_name(col),
            )
            if hasattr(curve, "setSkipFiniteCheck"):
                curve.setSkipFiniteCheck(True)
            self._metric_plot.addItem(curve)
            if legend is None:
                legend = self._metric_plot.addLegend(
                    offset=(10, 10),
                    labelTextColor="#cdd6f4",
                    brush=pg.mkBrush(30, 30, 46, 180),
                )
            legend.addItem(curve, _short_name(col))
            self._metric_curves[col] = curve
            self._metric_full_series[col] = (y, color)
            color_idx += 1

        if self._n_frames > 0:
            xmin, xmax = prev_range[0]
            if np.isfinite(xmin) and np.isfinite(xmax) and xmax > xmin:
                xmax = min(float(self._n_frames), float(xmax))
                xmin = max(0.0, min(float(xmin), max(0.0, xmax - 2.0)))
                self._metric_plot.setXRange(xmin, xmax, padding=0)
            else:
                self._set_default_x_range()

        if self._show_cleaning_region:
            self._add_metric_cleaning_region()
        self._update_metric_view(force=True)

    def _refresh_gantt(self) -> None:
        prev_range = self._gantt_widget.getPlotItem().getViewBox().viewRange()
        self._gantt_widget.clear()
        self._gantt_widget.addItem(self._cursor_line_gantt)

        bool_behaviors = [
            (name, np.asarray(arr, dtype=bool))
            for name, arr in self._behavior_arrays.items()
            if _is_boolean_like(arr)
        ]
        if not bool_behaviors:
            self._gantt_widget.getPlotItem().getAxis("left").setTicks([[]])
            return

        names = [name for name, _arr in bool_behaviors]
        width = max((len(arr) for _name, arr in bool_behaviors), default=self._n_frames)
        width = max(width, self._n_frames, 1)
        image = np.zeros((len(names), width, 4), dtype=np.uint8)

        for row_idx, (_name, arr) in enumerate(bool_behaviors):
            rgba = np.array(_GANTT_COLORS[row_idx % len(_GANTT_COLORS)], dtype=np.uint8)
            base = rgba.copy()
            base[3] = 34
            active = rgba.copy()
            active[3] = 188
            image[row_idx, :, :] = base
            if len(arr):
                image[row_idx, : len(arr), :][arr] = active

        gantt_image = pg.ImageItem(image)
        gantt_image.setRect(QRectF(0, -0.5, width, len(names)))
        self._gantt_widget.addItem(gantt_image)

        # Color squares next to labels — small coloured scatter at x=0
        sq_y = np.arange(len(names), dtype=float)
        sq_brushes = [
            pg.mkBrush(*_GANTT_COLORS[i % len(_GANTT_COLORS)])
            for i in range(len(names))
        ]
        squares = pg.ScatterPlotItem(
            x=np.zeros(len(names)), y=sq_y,
            size=10, symbol="s", pen=pg.mkPen(None), brush=sq_brushes,
        )
        squares.setZValue(10)
        self._gantt_widget.addItem(squares)

        axis = self._gantt_widget.getPlotItem().getAxis("left")
        axis.setTicks([[(row_idx, _short_name(name)) for row_idx, name in enumerate(names)]])
        self._gantt_widget.setYRange(-0.5, len(names) - 0.5, padding=0.03)

        if self._n_frames > 0:
            xmin, xmax = prev_range[0]
            if np.isfinite(xmin) and np.isfinite(xmax) and xmax > xmin:
                self._gantt_widget.setXRange(max(0.0, xmin), min(float(self._n_frames), xmax), padding=0)
            else:
                self._gantt_widget.setXRange(0, self._default_window(), padding=0)

        if self._show_cleaning_region:
            self._add_gantt_cleaning_region()

    def _metric_array_for(self, col: str, aid: str) -> Optional[np.ndarray]:
        if aid and aid in self._animal_dfs and col in self._animal_dfs[aid].columns:
            return self._animal_dfs[aid][col].to_numpy(dtype=np.float64)
        arr = self._behavior_arrays.get(col)
        if arr is None or _is_boolean_like(arr):
            return None
        return np.asarray(arr, dtype=np.float64)

    def _on_follow_toggled(self, checked: bool) -> None:
        self._follow_playback = checked
        if checked and self._n_frames > 0:
            self._set_default_x_range(center_on=self._current_frame)

    def _on_manual_zoom(self, *_args) -> None:
        self._follow_playback = False
        if self._chk_follow.isChecked():
            self._chk_follow.blockSignals(True)
            self._chk_follow.setChecked(False)
            self._chk_follow.blockSignals(False)

    def _on_metric_range_changed(self, *_args) -> None:
        if not self._updating_metric_view:
            self._update_metric_view()

    def _reset_zoom(self) -> None:
        if self._n_frames <= 0:
            return
        self._follow_playback = True
        if not self._chk_follow.isChecked():
            self._chk_follow.blockSignals(True)
            self._chk_follow.setChecked(True)
            self._chk_follow.blockSignals(False)
        self._set_default_x_range(center_on=self._current_frame)
        self._update_metric_view(force=True)

    def show_cleaning_region(self, start: int = 0, end: int = 0) -> None:
        if start > 0 or end > 0:
            self._cleaning_range = (start, end)
        elif self._cleaning_range == (0, 0):
            self._cleaning_range = (0, min(self._default_window(), self._n_frames))
        self._show_cleaning_region = True
        self._add_metric_cleaning_region()
        self._add_gantt_cleaning_region()
        self.cleaning_region_changed.emit(*self._cleaning_range)

    def hide_cleaning_region(self) -> None:
        self._show_cleaning_region = False
        if self._cleaning_region is not None:
            self._metric_plot.removeItem(self._cleaning_region)
            self._cleaning_region = None
        if self._cleaning_region_gantt is not None:
            self._gantt_widget.removeItem(self._cleaning_region_gantt)
            self._cleaning_region_gantt = None

    def set_cleaning_region(self, start: int, end: int) -> None:
        if start <= 0 and end <= 0:
            self.hide_cleaning_region()
            return
        self._cleaning_range = (start, end)
        if self._cleaning_region is not None:
            self._cleaning_region.blockSignals(True)
            self._cleaning_region.setRegion([start, end])
            self._cleaning_region.blockSignals(False)
        if self._cleaning_region_gantt is not None:
            self._cleaning_region_gantt.blockSignals(True)
            self._cleaning_region_gantt.setRegion([start, end])
            self._cleaning_region_gantt.blockSignals(False)

    def _add_metric_cleaning_region(self) -> None:
        s, e = self._cleaning_range
        if e <= s:
            return
        if self._cleaning_region is not None:
            try:
                self._metric_plot.removeItem(self._cleaning_region)
            except Exception:
                pass
        self._cleaning_region = pg.LinearRegionItem(
            values=[s, e],
            brush=pg.mkBrush(124, 58, 237, 30),
            pen=pg.mkPen("#7c3aed", width=1),
        )
        self._cleaning_region.sigRegionChangeFinished.connect(self._on_cleaning_region_moved)
        self._cleaning_region.sigRegionChanged.connect(self._sync_gantt_region)
        self._metric_plot.addItem(self._cleaning_region)

    def _add_gantt_cleaning_region(self) -> None:
        s, e = self._cleaning_range
        if e <= s:
            return
        if self._cleaning_region_gantt is not None:
            try:
                self._gantt_widget.removeItem(self._cleaning_region_gantt)
            except Exception:
                pass
        self._cleaning_region_gantt = pg.LinearRegionItem(
            values=[s, e],
            brush=pg.mkBrush(124, 58, 237, 30),
            pen=pg.mkPen("#7c3aed", width=1),
        )
        self._cleaning_region_gantt.sigRegionChangeFinished.connect(self._on_cleaning_region_moved)
        self._cleaning_region_gantt.sigRegionChanged.connect(self._sync_metric_region)
        self._gantt_widget.addItem(self._cleaning_region_gantt)

    def _sync_gantt_region(self) -> None:
        if self._cleaning_region is not None and self._cleaning_region_gantt is not None:
            self._cleaning_region_gantt.blockSignals(True)
            self._cleaning_region_gantt.setRegion(self._cleaning_region.getRegion())
            self._cleaning_region_gantt.blockSignals(False)

    def _sync_metric_region(self) -> None:
        if self._cleaning_region is not None and self._cleaning_region_gantt is not None:
            self._cleaning_region.blockSignals(True)
            self._cleaning_region.setRegion(self._cleaning_region_gantt.getRegion())
            self._cleaning_region.blockSignals(False)

    def _on_cleaning_region_moved(self) -> None:
        rgn = self._cleaning_region or self._cleaning_region_gantt
        if rgn is None:
            return
        lo, hi = rgn.getRegion()
        start = max(0, int(round(lo)))
        end = min(self._n_frames, int(round(hi)))
        self._cleaning_range = (start, end)
        self.cleaning_region_changed.emit(start, end)

    def _on_metric_click(self, event) -> None:
        self._seek_from_scene(self._metric_plot.getPlotItem().getViewBox(), event.scenePos())

    def _on_gantt_click(self, event) -> None:
        self._seek_from_scene(self._gantt_widget.getPlotItem().getViewBox(), event.scenePos())

    def _seek_from_scene(self, view_box: pg.ViewBox, scene_pos) -> None:
        mouse_point = view_box.mapSceneToView(scene_pos)
        frame = int(round(mouse_point.x()))
        if 0 <= frame < self._n_frames:
            self.set_frame_cursor(frame)
            self.cursor_moved.emit(frame)

    def _set_default_x_range(self, center_on: Optional[int] = None) -> None:
        if self._n_frames <= 0:
            return
        window = float(self._default_window())
        if center_on is None:
            xmin = 0.0
        else:
            xmin = float(center_on) - window * 0.2
        max_xmin = max(0.0, float(self._n_frames) - window)
        xmin = float(np.clip(xmin, 0.0, max_xmin))
        self._metric_plot.setXRange(xmin, xmin + window, padding=0)

    def _default_window(self) -> int:
        if self._n_frames <= 0:
            return 500
        window = max(300, int(round(self._fps * 20.0)))
        return int(min(window, self._n_frames))

    def _update_metric_view(self, force: bool = False) -> None:
        if not self._metric_curves:
            self._metric_marker.setData([], [])
            return

        vb = self._metric_plot.getPlotItem().getViewBox()
        xmin, xmax = vb.viewRange()[0]
        if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
            xmin, xmax = 0.0, float(self._default_window())

        signature = (
            int(np.floor(xmin)),
            int(np.ceil(xmax)),
            tuple(self._metric_curves.keys()),
            bool(self._chk_auto_y.isChecked()),
        )
        if not force and signature == self._metric_view_signature:
            self._update_metric_marker()
            return
        self._metric_view_signature = signature

        ymin = np.inf
        ymax = -np.inf
        self._updating_metric_view = True
        try:
            for name, curve in self._metric_curves.items():
                full_series, _color = self._metric_full_series[name]
                xs, ys, cur_min, cur_max = _visible_series(full_series, xmin, xmax)
                curve.setData(xs, ys, connect="finite")
                if cur_min is not None and cur_max is not None:
                    ymin = min(ymin, cur_min)
                    ymax = max(ymax, cur_max)

            if self._chk_auto_y.isChecked() and np.isfinite(ymin) and np.isfinite(ymax):
                if np.isclose(ymin, ymax):
                    pad = max(1.0, abs(ymin) * 0.05)
                else:
                    pad = max((ymax - ymin) * 0.08, 1e-3)
                self._metric_plot.setYRange(ymin - pad, ymax + pad, padding=0)
        finally:
            self._updating_metric_view = False

        self._update_metric_marker()

    def _update_metric_marker(self) -> None:
        if self._n_frames <= 0 or not self._metric_full_series:
            self._metric_marker.setData([], [])
            return

        frame = int(np.clip(self._current_frame, 0, self._n_frames - 1))
        xs: list[int] = []
        ys: list[float] = []
        brushes = []
        for full_series, color in self._metric_full_series.values():
            if frame >= len(full_series):
                continue
            val = full_series[frame]
            if not np.isfinite(val):
                continue
            xs.append(frame)
            ys.append(float(val))
            brushes.append(pg.mkBrush(color))

        if xs:
            self._metric_marker.setData(x=xs, y=ys, brush=brushes)
        else:
            self._metric_marker.setData([], [])

    def _ensure_metric_checkbox(self, col: str, checked: bool = False) -> None:
        if col in self._metric_checks:
            return
        chk = QCheckBox(_short_name(col))
        chk.setToolTip(col)
        chk.setChecked(checked)
        chk.setStyleSheet("color: #cdd6f4; font-size: 10px;")
        chk.toggled.connect(self._refresh_metric_plot)
        self._metric_checks[col] = chk
        idx = max(0, self._metric_checks_layout.count() - 1)
        self._metric_checks_layout.insertWidget(idx, chk)


def _visible_series(
    full_series: np.ndarray,
    xmin: float,
    xmax: float,
    max_points: int = 2200,
) -> tuple[np.ndarray, np.ndarray, Optional[float], Optional[float]]:
    start = max(0, int(np.floor(xmin)) - 1)
    stop = min(len(full_series), int(np.ceil(xmax)) + 2)
    if stop <= start:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64), None, None

    xs = np.arange(start, stop, dtype=np.float64)
    ys = np.asarray(full_series[start:stop], dtype=np.float64)
    finite = ys[np.isfinite(ys)]
    if finite.size:
        ymin = float(np.min(finite))
        ymax = float(np.max(finite))
    else:
        ymin = None
        ymax = None

    if len(xs) <= max_points:
        return xs, ys, ymin, ymax

    return (*_downsample_minmax(xs, ys, max_points=max_points), ymin, ymax)


def _downsample_minmax(
    xs: np.ndarray,
    ys: np.ndarray,
    max_points: int = 2200,
) -> tuple[np.ndarray, np.ndarray]:
    if len(xs) <= max_points or max_points < 8:
        return xs, ys

    bucket_count = max(1, max_points // 2)
    bucket_size = int(np.ceil(len(xs) / bucket_count))
    out_x: list[float] = []
    out_y: list[float] = []

    for start in range(0, len(xs), bucket_size):
        stop = min(len(xs), start + bucket_size)
        chunk_x = xs[start:stop]
        chunk_y = ys[start:stop]
        finite_mask = np.isfinite(chunk_y)
        if not finite_mask.any():
            out_x.append(float(chunk_x[0]))
            out_y.append(np.nan)
            continue

        finite_idx = np.flatnonzero(finite_mask)
        valid_y = chunk_y[finite_mask]
        rel_min = finite_idx[int(np.argmin(valid_y))]
        rel_max = finite_idx[int(np.argmax(valid_y))]
        order = (rel_min, rel_max) if rel_min <= rel_max else (rel_max, rel_min)
        for rel_idx in order:
            x_val = float(chunk_x[rel_idx])
            y_val = float(chunk_y[rel_idx])
            if out_x and np.isclose(out_x[-1], x_val) and (
                (np.isnan(out_y[-1]) and np.isnan(y_val)) or np.isclose(out_y[-1], y_val)
            ):
                continue
            out_x.append(x_val)
            out_y.append(y_val)

    return np.asarray(out_x, dtype=np.float64), np.asarray(out_y, dtype=np.float64)


def _short_name(col: str) -> str:
    pretty = {
        "a_nose2anogenital_b": "A\u2192B anogenital",
        "b_nose2anogenital_a": "B\u2192A anogenital",
        "a_nose2body_b": "A\u2192B body",
        "b_nose2body_a": "B\u2192A body",
        "a_following_b": "A follows B",
        "b_following_a": "B follows A",
        "a_oriented_toward_b": "A oriented\u2192B",
        "b_oriented_toward_a": "B oriented\u2192A",
        "passive_anogenital": "passive anogenital",
        "passive_investigation": "passive investigation",
        "passive_being_followed": "passive followed",
        "inter_animal_dist_px": "inter-animal dist",
        "approach_speed_px_s": "approach speed",
        "relative_heading_deg": "relative heading",
        "rearing": "rearing",
    }
    if col in pretty:
        return pretty[col]

    return (
        col.replace("body_", "")
        .replace("_px_s2", " acc")
        .replace("_px_s3", " jerk")
        .replace("_px_s", " speed")
        .replace("_px", "")
        .replace("_deg_fr", " deg/fr")
        .replace("_deg", " deg")
        .replace("_1_px", " curvature")
        .replace("_", " ")
    )


def _is_boolean_like(arr: np.ndarray) -> bool:
    arr = np.asarray(arr)
    if np.issubdtype(arr.dtype, np.bool_):
        return True
    if not np.issubdtype(arr.dtype, np.number):
        return False
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return False
    return set(np.unique(valid).tolist()).issubset({0.0, 1.0})
