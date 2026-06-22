"""Interactive plot panel: metric time series + behaviour Gantt chart.

Layout
------
  toolbar:  [Metrics...] [Animal ▾] [Follow] [Auto Y] [Reset] [Export]
  splitter: metric plot  (hideable)
            gantt chart  (hideable)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QBrush, QColor, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QGraphicsRectItem,
    QHBoxLayout,
    QLabel,
    QMenu,
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
    "body_speed_cm_s",
    "body_accel_px_s2",
    "body_jerk_px_s3",
    "body_orientation_deg",
    "body_angle_rate_deg_fr",
    "distance_traveled_px",
    "distance_traveled_cm",
    "path_tortuosity",
    "body_elongation_px",
    "trajectory_curvature_1_px",
    "head_direction_deg",
    "heading_body_angle_diff_deg",
    "inter_animal_dist_px",
    "inter_animal_dist_cm",
    "approach_speed_px_s",
    "relative_heading_deg",
    "partner_distance_px",
    "partner_distance_cm",
    "partner_angle_deg",
    "partner_proximity_index",
]

# ── Toolbar button stylesheet ─────────────────────────────────────────────────

_BTN_QSS = (
    "color: #cdd6f4; background: #313244; border: 1px solid #45475a;"
    " border-radius: 4px; padding: 2px 10px; font-size: 11px;"
)

_BTN_ACTIVE_QSS = (
    "color: #1e1e2e; background: #cba6f7; border: 1px solid #cba6f7;"
    " border-radius: 4px; padding: 2px 10px; font-size: 11px; font-weight: 600;"
)

_MENU_QSS = (
    "QMenu { background: #1e1e2e; color: #cdd6f4; border: 1px solid #313244; }"
    "QMenu::item:selected { background: #313244; }"
)


class PlotPanel(QWidget):
    """Interactive plot panel with playback-friendly metric rendering."""

    cursor_moved = Signal(int)
    cleaning_region_changed = Signal(int, int)
    range_start_flag_requested = Signal(int)
    range_end_flag_requested = Signal(int)
    range_clear_requested = Signal()
    identity_swap_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._animal_dfs: dict = {}
        self._behavior_arrays: dict = {}
        self._fps: float = 30.0
        self._n_frames: int = 0
        self._current_frame: int = 0
        self._follow_playback: bool = True
        self._show_cleaning_region: bool = False
        self._cleaning_range: tuple[int, int] = (0, 0)
        self._cleaning_region: Optional[pg.LinearRegionItem] = None
        self._cleaning_region_gantt: Optional[pg.LinearRegionItem] = None
        self._custom_time = None  # Optional np.ndarray of timestamps
        self._metric_checks: dict[str, bool] = {}  # col -> checked
        self._metric_full_series: dict[str, tuple[np.ndarray, str]] = {}
        self._metric_curves: dict[str, pg.PlotDataItem] = {}
        self._metric_view_signature: Optional[tuple] = None
        self._updating_metric_view: bool = False
        self._gantt_labels: list[pg.TextItem] = []  # floating lane labels
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

        # ── Toolbar ──────────────────────────────────────────────────────────
        toolbar = QWidget()
        toolbar.setFixedHeight(32)
        toolbar.setStyleSheet(
            "background: #181825; border-bottom: 1px solid #313244;"
        )
        tb = QHBoxLayout(toolbar)
        tb.setContentsMargins(8, 0, 8, 0)
        tb.setSpacing(6)

        # Metrics popup button
        self._btn_metrics = QPushButton("Metrics\u2026")
        self._btn_metrics.setFixedHeight(24)
        self._btn_metrics.setStyleSheet(_BTN_QSS)
        self._btn_metrics.setToolTip("Choose which metrics to plot")
        self._btn_metrics.clicked.connect(self._open_metric_selector)
        tb.addWidget(self._btn_metrics)

        # Active metric summary label
        self._lbl_active = QLabel("none selected")
        self._lbl_active.setStyleSheet("color: #6c7086; font-size: 10px;")
        tb.addWidget(self._lbl_active)

        tb.addStretch()

        # Animal selector
        lbl_a = QLabel("Animal:")
        lbl_a.setStyleSheet("color: #a6adc8; font-size: 11px;")
        tb.addWidget(lbl_a)
        self._combo_animal = QComboBox()
        self._combo_animal.setMinimumWidth(96)
        self._combo_animal.setFixedHeight(22)
        self._combo_animal.currentIndexChanged.connect(self._refresh_metric_plot)
        tb.addWidget(self._combo_animal)

        tb.addSpacing(8)

        # Follow / Auto Y
        self._chk_follow = QCheckBox("Follow")
        self._chk_follow.setChecked(True)
        self._chk_follow.setStyleSheet("color: #cdd6f4; font-size: 10px;")
        self._chk_follow.setToolTip("Auto-scroll the plot to follow the video playback cursor")
        self._chk_follow.toggled.connect(self._on_follow_toggled)
        tb.addWidget(self._chk_follow)

        self._chk_auto_y = QCheckBox("Auto Y")
        self._chk_auto_y.setChecked(True)
        self._chk_auto_y.setStyleSheet("color: #cdd6f4; font-size: 10px;")
        self._chk_auto_y.setToolTip("Automatically scale the Y axis to fit visible data")
        self._chk_auto_y.toggled.connect(lambda _: self._update_metric_view(force=True))
        tb.addWidget(self._chk_auto_y)

        tb.addSpacing(4)

        # Show/hide toggles for sub-panels
        self._btn_toggle_metric = QPushButton("Metric")
        self._btn_toggle_metric.setFixedHeight(22)
        self._btn_toggle_metric.setCheckable(True)
        self._btn_toggle_metric.setChecked(True)
        self._btn_toggle_metric.setStyleSheet(_BTN_ACTIVE_QSS)
        self._btn_toggle_metric.setToolTip("Show / hide the metric time-series plot")
        self._btn_toggle_metric.toggled.connect(self._toggle_metric_panel)
        tb.addWidget(self._btn_toggle_metric)

        self._btn_toggle_gantt = QPushButton("Gantt")
        self._btn_toggle_gantt.setFixedHeight(22)
        self._btn_toggle_gantt.setCheckable(True)
        self._btn_toggle_gantt.setChecked(True)
        self._btn_toggle_gantt.setStyleSheet(_BTN_ACTIVE_QSS)
        self._btn_toggle_gantt.setToolTip("Show / hide the behaviour Gantt chart")
        self._btn_toggle_gantt.toggled.connect(self._toggle_gantt_panel)
        tb.addWidget(self._btn_toggle_gantt)

        tb.addSpacing(4)

        btn_reset = QPushButton("Reset")
        btn_reset.setToolTip("Reset plot view to default zoom")
        btn_reset.setFixedHeight(22)
        btn_reset.setStyleSheet(_BTN_QSS)
        btn_reset.clicked.connect(self._reset_zoom)
        tb.addWidget(btn_reset)

        btn_export = QPushButton("Export")
        btn_export.setToolTip("Export plot as PNG or SVG")
        btn_export.setFixedHeight(22)
        btn_export.setStyleSheet(_BTN_QSS)
        btn_export.clicked.connect(self._export_plot)
        tb.addWidget(btn_export)

        root.addWidget(toolbar)

        # ── Plots ────────────────────────────────────────────────────────────
        self._splitter = QSplitter(Qt.Orientation.Vertical)
        self._splitter.setStyleSheet(
            "QSplitter::handle { background: #313244; height: 3px; }"
        )

        # Metric plot
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
        self._metric_plot.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._metric_plot.customContextMenuRequested.connect(self._show_plot_context_menu)
        self._metric_plot.scene().sigMouseClicked.connect(self._on_metric_click)
        self._cursor_line_metric = pg.InfiniteLine(
            pos=0, angle=90,
            pen=pg.mkPen("#f9e2af", width=1, style=Qt.PenStyle.DashLine),
        )
        self._metric_marker = pg.ScatterPlotItem(size=7, pen=pg.mkPen("#11111b", width=1))
        self._metric_plot.addItem(self._cursor_line_metric)
        self._metric_plot.addItem(self._metric_marker)
        self._splitter.addWidget(self._metric_plot)

        # Gantt chart
        self._gantt_widget = pg.PlotWidget()
        self._gantt_widget.setLabel("bottom", "Frame")
        self._gantt_widget.setLabel("left", "")
        self._gantt_widget.showGrid(x=True, y=False, alpha=0.14)
        self._gantt_widget.getPlotItem().getViewBox().setMouseMode(pg.ViewBox.RectMode)
        self._gantt_widget.getPlotItem().getViewBox().invertY(True)
        self._gantt_widget.getPlotItem().getViewBox().sigRangeChanged.connect(
            self._update_gantt_label_positions
        )
        self._gantt_widget.scene().sigMouseClicked.connect(self._on_gantt_click)
        self._gantt_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._gantt_widget.customContextMenuRequested.connect(self._show_gantt_context_menu)
        self._cursor_line_gantt = pg.InfiniteLine(
            pos=0, angle=90,
            pen=pg.mkPen("#f9e2af", width=1, style=Qt.PenStyle.DashLine),
        )
        self._gantt_widget.addItem(self._cursor_line_gantt)
        self._gantt_widget.setXLink(self._metric_plot)
        self._splitter.addWidget(self._gantt_widget)

        self._splitter.setSizes([230, 170])
        root.addWidget(self._splitter, 1)

        # Pre-populate known metrics (all unchecked by default)
        for col in _METRIC_COLUMNS:
            if col not in self._metric_checks:
                self._metric_checks[col] = False
        self._update_ui_state()

    # ── Panel toggles ────────────────────────────────────────────────────────

    def _toggle_metric_panel(self, visible: bool) -> None:
        self._metric_plot.setVisible(visible)
        self._btn_toggle_metric.setStyleSheet(
            _BTN_ACTIVE_QSS if visible else _BTN_QSS
        )

    def _toggle_gantt_panel(self, visible: bool) -> None:
        self._gantt_widget.setVisible(visible)
        self._btn_toggle_gantt.setStyleSheet(
            _BTN_ACTIVE_QSS if visible else _BTN_QSS
        )

    # ── Metric selector dialog ───────────────────────────────────────────────

    def _open_metric_selector(self) -> None:
        dlg = _MetricSelectorDialog(self._metric_checks, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._metric_checks = dlg.result_checks()
            self._update_active_label()
            self._refresh_metric_plot()

    def _update_active_label(self) -> None:
        active = [_short_name(c) for c, v in self._metric_checks.items() if v]
        if not active:
            self._lbl_active.setText("none selected")
        elif len(active) <= 3:
            self._lbl_active.setText(", ".join(active))
        else:
            self._lbl_active.setText(f"{', '.join(active[:2])} +{len(active)-2} more")

    # ── Public API ───────────────────────────────────────────────────────────

    def set_custom_time(self, times) -> None:
        self._custom_time = times
        label = "Frame (external time loaded)" if times is not None else "Frame"
        self._metric_plot.setLabel("bottom", label)
        self._gantt_widget.setLabel("bottom", label)

    def set_animal_dfs(self, dfs: dict, fps: float = 30.0) -> None:
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
                if col.endswith((
                    "_px_s", "_px_s2", "_px_s3", "_px", "_deg", "_deg_fr", "_1_px",
                    "_cm_s", "_cm_s2", "_cm_s3", "_cm", "_1_cm",
                )):
                    if col not in self._metric_checks:
                        self._metric_checks[col] = False
            self._ensure_default_metric_selected(first_df)

        self._update_active_label()
        self._update_ui_state()
        self._refresh_metric_plot()
        self._refresh_gantt()

    def set_behavior_arrays(self, arrays: dict) -> None:
        self._behavior_arrays = arrays
        for name, arr in arrays.items():
            if not _is_boolean_like(arr):
                if name not in self._metric_checks:
                    self._metric_checks[name] = False
        self._update_ui_state()
        self._refresh_gantt()
        self._refresh_metric_plot()

    def _ensure_default_metric_selected(self, df) -> None:
        if any(self._metric_checks.values()):
            return
        for candidate in (
            "body_speed_cm_s",
            "body_speed_px_s",
            "distance_traveled_cm",
            "distance_traveled_px",
            "body_orientation_deg",
        ):
            if candidate in df.columns:
                self._metric_checks[candidate] = True
                return

    def _update_ui_state(self) -> None:
        has_animals = bool(self._animal_dfs)
        has_any_plot = has_animals or bool(self._behavior_arrays)
        self._btn_metrics.setEnabled(has_animals)
        self._combo_animal.setEnabled(has_animals)
        self._chk_follow.setEnabled(has_animals)
        self._chk_auto_y.setEnabled(has_animals)
        self._btn_toggle_metric.setEnabled(has_any_plot)
        self._btn_toggle_gantt.setEnabled(bool(self._behavior_arrays))
        if not has_animals:
            self._lbl_active.setText("load data to plot")

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

    # ── Refresh plots ────────────────────────────────────────────────────────

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

        for col, checked in self._metric_checks.items():
            if not checked:
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
        n_rows = len(names)

        # Draw behaviour bouts as explicit bars so sparse events stay readable.
        no_pen = QPen(Qt.PenStyle.NoPen)
        lane_span = float(width)
        for row_idx, (_name, arr) in enumerate(bool_behaviors):
            r, g, b, _a = _GANTT_COLORS[row_idx % len(_GANTT_COLORS)]
            bg = QGraphicsRectItem(0.0, row_idx - 0.36, lane_span, 0.72)
            bg.setBrush(QBrush(QColor(r, g, b, 24)))
            bg.setPen(no_pen)
            bg.setZValue(0)
            self._gantt_widget.addItem(bg)

            for start, stop in _bool_bouts(arr):
                if stop <= start:
                    continue
                rect = QGraphicsRectItem(float(start), row_idx - 0.34, float(stop - start), 0.68)
                rect.setBrush(QBrush(QColor(r, g, b, 215)))
                rect.setPen(no_pen)
                rect.setZValue(10)
                self._gantt_widget.addItem(rect)

        # Lane separator lines — subtle but clear
        for i in range(n_rows + 1):
            line = pg.InfiniteLine(
                pos=i - 0.5, angle=0,
                pen=pg.mkPen("#585b70", width=1, style=Qt.PenStyle.SolidLine),
            )
            line.setZValue(5)
            self._gantt_widget.addItem(line)

        # Y-axis tick labels — plain text (pyqtgraph AxisItem doesn't render HTML)
        axis = self._gantt_widget.getPlotItem().getAxis("left")
        axis.setStyle(tickLength=0)
        axis.setWidth(160)
        ticks = [(row_idx, _short_name(name)) for row_idx, name in enumerate(names)]
        axis.setTicks([ticks])
        axis.setTextPen(pg.mkPen("#cdd6f4"))

        # Colour swatch labels — pinned to left edge of viewport
        self._gantt_labels = []
        for row_idx, name in enumerate(names):
            r, g, b, _a = _GANTT_COLORS[row_idx % len(_GANTT_COLORS)]
            txt = pg.TextItem(text="\u2588", color=(r, g, b, 220), anchor=(0, 0.5))
            txt.setFont(pg.QtGui.QFont("sans-serif", 11))
            txt.setZValue(15)
            self._gantt_widget.addItem(txt)
            self._gantt_labels.append(txt)

        self._gantt_widget.setYRange(-0.5, n_rows - 0.5, padding=0.05)

        if self._n_frames > 0:
            xmin, xmax = prev_range[0]
            if np.isfinite(xmin) and np.isfinite(xmax) and xmax > xmin:
                self._gantt_widget.setXRange(
                    max(0.0, xmin), min(float(width), xmax), padding=0
                )
            else:
                self._gantt_widget.setXRange(0, min(float(width), float(self._default_window())), padding=0)

        if self._show_cleaning_region:
            self._add_gantt_cleaning_region()

        # Position swatch labels at current view
        self._update_gantt_label_positions()

    def _update_gantt_label_positions(self, *_args) -> None:
        """Pin floating lane labels to the left edge of the visible viewport."""
        if not self._gantt_labels:
            return
        vb = self._gantt_widget.getPlotItem().getViewBox()
        xmin, _xmax = vb.viewRange()[0]
        # Small offset so text doesn't sit right on the axis edge
        x_offset = (_xmax - xmin) * 0.005 + xmin
        for row_idx, txt in enumerate(self._gantt_labels):
            txt.setPos(x_offset, row_idx)

    # ── Data helpers ─────────────────────────────────────────────────────────

    def _metric_array_for(self, col: str, aid: str) -> Optional[np.ndarray]:
        if aid and aid in self._animal_dfs and col in self._animal_dfs[aid].columns:
            return self._animal_dfs[aid][col].to_numpy(dtype=np.float64)
        arr = self._behavior_arrays.get(col)
        if arr is None or _is_boolean_like(arr):
            return None
        return np.asarray(arr, dtype=np.float64)

    # ── Interaction ──────────────────────────────────────────────────────────

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

    def _on_metric_click(self, event) -> None:
        if hasattr(event, "button") and event.button() != Qt.MouseButton.LeftButton:
            return
        self._seek_from_scene(self._metric_plot.getPlotItem().getViewBox(), event.scenePos())

    def _on_gantt_click(self, event) -> None:
        if hasattr(event, "button") and event.button() != Qt.MouseButton.LeftButton:
            return
        self._seek_from_scene(self._gantt_widget.getPlotItem().getViewBox(), event.scenePos())

    def _seek_from_scene(self, view_box: pg.ViewBox, scene_pos) -> None:
        mouse_point = view_box.mapSceneToView(scene_pos)
        frame = int(round(mouse_point.x()))
        if 0 <= frame < self._n_frames:
            self.set_frame_cursor(frame)
            self.cursor_moved.emit(frame)

    # ── Cleaning region ──────────────────────────────────────────────────────

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

    # ── View helpers ─────────────────────────────────────────────────────────

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

    # ── Context menu ─────────────────────────────────────────────────────────

    def _frame_from_plot_pos(self, plot_widget: pg.PlotWidget, pos) -> int:
        if self._n_frames <= 0:
            return int(self._current_frame)
        view_box = plot_widget.getPlotItem().getViewBox()
        scene_pos = plot_widget.mapToScene(pos)
        mouse_point = view_box.mapSceneToView(scene_pos)
        frame = int(round(mouse_point.x()))
        return max(0, min(self._n_frames - 1, frame))

    def _add_range_actions_to_menu(self, menu: QMenu, frame: int) -> None:
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

    def _show_plot_context_menu(self, pos) -> None:
        frame = self._frame_from_plot_pos(self._metric_plot, pos)
        menu = QMenu(self)
        menu.setStyleSheet(_MENU_QSS)

        self._add_range_actions_to_menu(menu, frame)

        menu.addSeparator()

        act_metrics = QAction("Select metrics\u2026", self)
        act_metrics.triggered.connect(self._open_metric_selector)
        menu.addAction(act_metrics)

        menu.addSeparator()

        act_export_png = QAction("Export as PNG\u2026", self)
        act_export_png.triggered.connect(lambda: self._export_plot("png"))
        menu.addAction(act_export_png)

        act_export_svg = QAction("Export as SVG\u2026", self)
        act_export_svg.triggered.connect(lambda: self._export_plot("svg"))
        menu.addAction(act_export_svg)

        menu.addSeparator()

        act_reset = QAction("Reset view", self)
        act_reset.triggered.connect(self._reset_zoom)
        menu.addAction(act_reset)

        menu.exec(self._metric_plot.mapToGlobal(pos))

    def _show_gantt_context_menu(self, pos) -> None:
        frame = self._frame_from_plot_pos(self._gantt_widget, pos)
        menu = QMenu(self)
        menu.setStyleSheet(_MENU_QSS)

        self._add_range_actions_to_menu(menu, frame)

        menu.addSeparator()

        act_reset = QAction("Reset view", self)
        act_reset.triggered.connect(self._reset_zoom)
        menu.addAction(act_reset)

        menu.exec(self._gantt_widget.mapToGlobal(pos))

    def _export_plot(self, fmt: str = "") -> None:
        """Export the metric plot as PNG or SVG."""
        import pyqtgraph.exporters as exporters

        if not fmt:
            path, _ = QFileDialog.getSaveFileName(
                self, "Export Plot", "",
                "PNG Image (*.png);;SVG Image (*.svg)",
            )
            if not path:
                return
            fmt = "svg" if path.lower().endswith(".svg") else "png"
        else:
            ext = f".{fmt}"
            filt_map = {"png": "PNG Image (*.png)", "svg": "SVG Image (*.svg)"}
            path, _ = QFileDialog.getSaveFileName(
                self, "Export Plot", "", filt_map.get(fmt, ""),
            )
            if not path:
                return
            if not path.lower().endswith(ext):
                path += ext

        scene = self._metric_plot.getPlotItem().scene()
        if fmt == "svg":
            exporter = exporters.SVGExporter(scene)
        else:
            exporter = exporters.ImageExporter(scene)
            exporter.parameters()["width"] = 1920
        exporter.export(path)


# ══════════════════════════════════════════════════════════════════════════════
# Metric Selector Dialog
# ══════════════════════════════════════════════════════════════════════════════

# Categories for the metric selector
_METRIC_CATEGORIES: list[tuple[str, list[str]]] = [
    ("Speed & Motion", [
        "body_speed_px_s", "body_speed_cm_s",
        "body_accel_px_s2", "body_accel_cm_s2",
        "body_jerk_px_s3", "body_jerk_cm_s3",
    ]),
    ("Distance & Path", [
        "distance_traveled_px", "distance_traveled_cm",
        "path_tortuosity",
    ]),
    ("Orientation & Posture", [
        "body_orientation_deg", "body_angle_rate_deg_fr",
        "head_direction_deg", "heading_body_angle_diff_deg",
        "body_elongation_px", "body_elongation_cm",
        "trajectory_curvature_1_px", "trajectory_curvature_1_cm",
    ]),
    ("Partner Metrics", [
        "partner_distance_px", "partner_distance_cm",
        "partner_angle_deg", "partner_proximity_index",
        "inter_animal_dist_px", "inter_animal_dist_cm",
        "approach_speed_px_s", "approach_speed_cm_s",
        "relative_heading_deg",
    ]),
]


class _MetricSelectorDialog(QDialog):
    """Popup dialog with categorised metric checkboxes."""

    def __init__(self, current: dict[str, bool], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select Metrics to Plot")
        self.setMinimumWidth(420)
        self.setStyleSheet(
            "QDialog { background: #1e1e2e; color: #cdd6f4; }"
            "QGroupBox { border: 1px solid #313244; border-radius: 4px;"
            " margin-top: 8px; padding-top: 14px; font-weight: 600;"
            " color: #a6adc8; }"
            "QGroupBox::title { subcontrol-origin: margin;"
            " subcontrol-position: top left; left: 10px; }"
            "QCheckBox { color: #cdd6f4; font-size: 11px; padding: 2px 0; }"
            "QPushButton { color: #cdd6f4; background: #313244;"
            " border: 1px solid #45475a; border-radius: 4px;"
            " padding: 4px 14px; }"
            "QPushButton:hover { background: #45475a; }"
        )
        self._checks: dict[str, QCheckBox] = {}
        self._build_ui(current)

    def _build_ui(self, current: dict[str, bool]) -> None:
        lay = QVBoxLayout(self)
        lay.setSpacing(6)

        # Quick actions
        top = QHBoxLayout()
        btn_all = QPushButton("All")
        btn_all.clicked.connect(lambda: self._set_all(True))
        btn_none = QPushButton("None")
        btn_none.clicked.connect(lambda: self._set_all(False))
        top.addWidget(btn_all)
        top.addWidget(btn_none)
        top.addStretch()
        lay.addLayout(top)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setMaximumHeight(400)
        container = QWidget()
        grid_outer = QVBoxLayout(container)
        grid_outer.setSpacing(4)

        # Categorised metrics
        categorised = set()
        for cat_name, cols in _METRIC_CATEGORIES:
            grp = QGroupBox(cat_name)
            g = QGridLayout(grp)
            g.setSpacing(3)
            col_idx = 0
            for c in cols:
                categorised.add(c)
                chk = QCheckBox(_short_name(c))
                chk.setToolTip(c)
                chk.setChecked(current.get(c, False))
                self._checks[c] = chk
                g.addWidget(chk, col_idx // 2, col_idx % 2)
                col_idx += 1
            grid_outer.addWidget(grp)

        # Uncategorised (dynamic columns discovered at runtime)
        uncategorised = [c for c in current if c not in categorised]
        if uncategorised:
            grp = QGroupBox("Other")
            g = QGridLayout(grp)
            g.setSpacing(3)
            for i, c in enumerate(uncategorised):
                chk = QCheckBox(_short_name(c))
                chk.setToolTip(c)
                chk.setChecked(current.get(c, False))
                self._checks[c] = chk
                g.addWidget(chk, i // 2, i % 2)
            grid_outer.addWidget(grp)

        grid_outer.addStretch()
        scroll.setWidget(container)
        lay.addWidget(scroll, 1)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def _set_all(self, checked: bool) -> None:
        for chk in self._checks.values():
            chk.setChecked(checked)

    def result_checks(self) -> dict[str, bool]:
        return {c: chk.isChecked() for c, chk in self._checks.items()}


# ══════════════════════════════════════════════════════════════════════════════
# Module-level helpers
# ══════════════════════════════════════════════════════════════════════════════

def _bool_bouts(arr: np.ndarray) -> list[tuple[int, int]]:
    """Return [start, stop) runs for True values."""
    values = np.asarray(arr, dtype=bool).reshape(-1)
    if values.size == 0:
        return []
    padded = np.concatenate([[False], values, [False]])
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return [(int(s), int(e)) for s, e in zip(starts, stops) if e > s]


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
    """Return a clean, print-friendly display name. No arrows or unicode symbols."""
    if "__" in col:
        animal, metric = col.split("__", 1)
        if metric in {"immobile", "is_immobile"}:
            return f"{animal} immobile"
        if metric in {"mobile", "is_mobile"}:
            return f"{animal} mobile"
        if metric == "mobility_state":
            return f"{animal} mobility"
    pretty = {
        # Social behaviors — NO arrows, plain text
        "nose2nose": "nose-to-nose",
        "mask_contact": "mask contact",
        "fighting": "fighting",
        "attacks": "attacks",
        "a_nose2anogenital_b": "A to B anogenital",
        "b_nose2anogenital_a": "B to A anogenital",
        "a_nose2body_b": "A to B body",
        "b_nose2body_a": "B to A body",
        "a_following_b": "A follows B",
        "b_following_a": "B follows A",
        "a_chasing_b": "A chases B",
        "b_chasing_a": "B chases A",
        "a_approaches_b": "A approaches B",
        "b_approaches_a": "B approaches A",
        "a_withdraws_from_b": "A withdraws from B",
        "b_withdraws_from_a": "B withdraws from A",
        "a_escapes_b": "A escapes B",
        "b_escapes_a": "B escapes A",
        "a_withdrawal_after_contact_b": "A withdraws after contact B",
        "b_withdrawal_after_contact_a": "B withdraws after contact A",
        "a_oriented_toward_b": "A oriented to B",
        "b_oriented_toward_a": "B oriented to A",
        "sidebyside": "side-by-side",
        "sidereside": "side-reverse",
        "passive_anogenital": "passive anogenital",
        "passive_investigation": "passive investigation",
        "passive_being_followed": "passive followed",
        "passive_being_chased": "passive chased",
        "passive_withdrawal": "passive withdrawal",
        # Social metrics
        "inter_animal_dist_px": "inter-animal dist (px)",
        "inter_animal_dist_cm": "inter-animal dist (cm)",
        "approach_speed_px_s": "approach speed (px/s)",
        "approach_speed_cm_s": "approach speed (cm/s)",
        "relative_heading_deg": "relative heading",
        "partner_distance_px": "partner dist (px)",
        "partner_distance_cm": "partner dist (cm)",
        "partner_angle_deg": "partner angle",
        "partner_proximity_index": "proximity index",
        # Individual
        "rearing": "rearing",
        "immobile": "immobile",
        "freezing": "immobile",
        "is_immobile": "immobile",
        # Kinematics
        "body_speed_px_s": "speed (px/s)",
        "body_speed_cm_s": "speed (cm/s)",
        "body_accel_px_s2": "accel (px/s2)",
        "body_accel_cm_s2": "accel (cm/s2)",
        "body_jerk_px_s3": "jerk (px/s3)",
        "body_jerk_cm_s3": "jerk (cm/s3)",
        "body_orientation_deg": "orientation (deg)",
        "body_angle_rate_deg_fr": "angular vel (deg/fr)",
        "distance_traveled_px": "cum. distance (px)",
        "distance_traveled_cm": "cum. distance (cm)",
        "path_tortuosity": "tortuosity",
        "body_elongation_px": "elongation (px)",
        "body_elongation_cm": "elongation (cm)",
        "trajectory_curvature_1_px": "curvature (1/px)",
        "trajectory_curvature_1_cm": "curvature (1/cm)",
        "head_direction_deg": "head direction (deg)",
        "heading_body_angle_diff_deg": "crab-walk angle",
        # Mobility
        "mobility_state": "mobility state",
    }
    if col in pretty:
        return pretty[col]

    # Fallback: strip common suffixes
    s = col
    for old, new in [
        ("_cm_s3", " jerk"), ("_cm_s2", " acc"), ("_cm_s", " speed"),
        ("_px_s3", " jerk"), ("_px_s2", " acc"), ("_px_s", " speed"),
        ("_deg_fr", " (deg/fr)"), ("_deg", " (deg)"),
        ("_1_cm", " curvature"), ("_1_px", " curvature"),
        ("_cm", " (cm)"), ("_px", " (px)"),
    ]:
        if s.endswith(old):
            s = s[: -len(old)] + new
            break
    return s.replace("body_", "").replace("_", " ").strip()


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
