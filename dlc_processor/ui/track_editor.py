"""Track Editor — interactive X-Y trajectory viewer and editor.

Layout::

    ┌──────────┬──────────────────────────────┬──────────┐
    │          │  X-coordinates plot           │          │
    │ Bodypart │──────────────────────────────│  Tools   │
    │ selector │  Y-coordinates plot           │  panel   │
    │          │──────────────────────────────│          │
    │          │  frame slider + controls      │          │
    └──────────┴──────────────────────────────┴──────────┘

Tools (right panel):
  - Selection (rect region) → delete / interpolate / smooth
  - Swap identities in range
  - Auto-fix proximity (deduce nose/tail from neighbours)
  - Save cleaned H5/CSV
"""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from dlc_processor.core.dlc_loader import get_bodyparts

logger = logging.getLogger(__name__)

# Catppuccin palette for bodypart traces
_TRACE_COLORS = [
    "#f38ba8", "#89b4fa", "#a6e3a1", "#fab387", "#cba6f7",
    "#f9e2af", "#94e2d5", "#f5c2e7", "#74c7ec", "#b4befe",
    "#eba0ac", "#89dceb", "#a6e3a1", "#fab387", "#cba6f7",
]

_DIALOG_QSS = """
QDialog { background: #1e1e2e; color: #cdd6f4; }
QGroupBox {
    color: #cdd6f4; border: 1px solid #313244;
    border-radius: 4px; margin-top: 6px; padding-top: 14px;
    font-weight: 600;
}
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
QCheckBox { color: #cdd6f4; font-size: 11px; }
QLabel { color: #cdd6f4; }
QPushButton {
    background: #313244; color: #cdd6f4; border: 1px solid #45475a;
    border-radius: 4px; padding: 5px 12px;
}
QPushButton:hover { background: #45475a; }
QPushButton#accent {
    background: #7c3aed; border-color: #6d28d9;
}
QPushButton#accent:hover { background: #6d28d9; }
QPushButton#danger {
    background: #c0392b; border-color: #a93226;
}
QPushButton#danger:hover { background: #a93226; }
QSlider::groove:horizontal {
    height: 4px; background: #313244; border-radius: 2px;
}
QSlider::handle:horizontal {
    width: 14px; height: 14px; margin: -5px 0;
    background: #7c3aed; border-radius: 7px;
}
QComboBox {
    background: #313244; color: #cdd6f4; border: 1px solid #45475a;
    border-radius: 4px; padding: 3px 6px;
}
QSpinBox, QDoubleSpinBox {
    background: #313244; color: #cdd6f4; border: 1px solid #45475a;
    border-radius: 4px; padding: 2px 4px;
}
QScrollArea { border: none; }
"""


class TrackEditorDialog(QDialog):
    """Full-featured track editor for DLC data.

    Parameters
    ----------
    animal_dfs : dict[str, pd.DataFrame]
        Per-animal flat DataFrames with <bp>_x, <bp>_y, <bp>_likelihood columns.
    parent : QWidget, optional
    """

    data_changed = Signal(dict)  # emitted when user clicks Apply

    def __init__(
        self,
        animal_dfs: dict[str, pd.DataFrame],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Track Editor")
        self.setMinimumSize(1200, 700)
        self.setStyleSheet(_DIALOG_QSS)

        # Working copy — edits happen here; originals kept for undo
        self._original_dfs = animal_dfs
        self._working_dfs = deepcopy(animal_dfs)
        self._n_frames = max((len(df) for df in animal_dfs.values()), default=0)

        # State
        self._current_frame = 0
        self._selection_region: Optional[pg.LinearRegionItem] = None

        self._build_ui()
        self._populate_bodyparts()
        self._refresh_plots()

    # ── UI Construction ──────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # ── Left panel: animal & bodypart selector ───────────────────────────
        left = QWidget()
        left.setFixedWidth(180)
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(4, 4, 4, 4)
        left_lay.setSpacing(4)

        left_lay.addWidget(QLabel("Animal:"))
        self._combo_animal = QComboBox()
        for aid in self._working_dfs:
            self._combo_animal.addItem(str(aid))
        self._combo_animal.currentIndexChanged.connect(self._on_animal_changed)
        left_lay.addWidget(self._combo_animal)

        left_lay.addWidget(QLabel("Bodyparts:"))

        # Select all / none
        sel_row = QHBoxLayout()
        btn_all = QPushButton("All")
        btn_all.clicked.connect(self._select_all_bps)
        btn_none = QPushButton("None")
        btn_none.clicked.connect(self._select_no_bps)
        sel_row.addWidget(btn_all)
        sel_row.addWidget(btn_none)
        left_lay.addLayout(sel_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._bp_container = QWidget()
        self._bp_layout = QVBoxLayout(self._bp_container)
        self._bp_layout.setContentsMargins(2, 2, 2, 2)
        self._bp_layout.setSpacing(2)
        scroll.setWidget(self._bp_container)
        left_lay.addWidget(scroll, 1)

        self._bp_checks: dict[str, QCheckBox] = {}

        root.addWidget(left)

        # ── Center: plots + frame slider ─────────────────────────────────────
        center = QWidget()
        center_lay = QVBoxLayout(center)
        center_lay.setContentsMargins(0, 0, 0, 0)
        center_lay.setSpacing(2)

        pg.setConfigOptions(antialias=True, background="#1e1e2e", foreground="#cdd6f4")

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setStyleSheet(
            "QSplitter::handle { background: #313244; height: 3px; }"
        )

        # X-coordinates plot
        self._plot_x = pg.PlotWidget(title="X coordinate")
        self._plot_x.setLabel("left", "X (px)")
        self._plot_x.showGrid(x=True, y=True, alpha=0.15)
        self._plot_x.getPlotItem().getViewBox().setMouseMode(pg.ViewBox.RectMode)
        self._cursor_x = pg.InfiniteLine(
            pos=0, angle=90,
            pen=pg.mkPen("#f9e2af", width=1, style=Qt.PenStyle.DashLine),
        )
        self._plot_x.addItem(self._cursor_x)
        splitter.addWidget(self._plot_x)

        # Y-coordinates plot
        self._plot_y = pg.PlotWidget(title="Y coordinate")
        self._plot_y.setLabel("left", "Y (px)")
        self._plot_y.setLabel("bottom", "Frame")
        self._plot_y.showGrid(x=True, y=True, alpha=0.15)
        self._plot_y.getPlotItem().getViewBox().setMouseMode(pg.ViewBox.RectMode)
        self._plot_y.setXLink(self._plot_x)
        self._cursor_y = pg.InfiniteLine(
            pos=0, angle=90,
            pen=pg.mkPen("#f9e2af", width=1, style=Qt.PenStyle.DashLine),
        )
        self._plot_y.addItem(self._cursor_y)
        splitter.addWidget(self._plot_y)

        splitter.setSizes([300, 300])
        center_lay.addWidget(splitter, 1)

        # Frame navigation
        nav = QWidget()
        nav.setStyleSheet("background: #181825; border-top: 1px solid #313244;")
        nav_lay = QHBoxLayout(nav)
        nav_lay.setContentsMargins(8, 4, 8, 4)
        nav_lay.setSpacing(6)

        self._btn_prev10 = QPushButton("<<")
        self._btn_prev10.setFixedWidth(32)
        self._btn_prev10.clicked.connect(lambda: self._step_frame(-10))
        self._btn_prev = QPushButton("<")
        self._btn_prev.setFixedWidth(32)
        self._btn_prev.clicked.connect(lambda: self._step_frame(-1))
        self._btn_next = QPushButton(">")
        self._btn_next.setFixedWidth(32)
        self._btn_next.clicked.connect(lambda: self._step_frame(1))
        self._btn_next10 = QPushButton(">>")
        self._btn_next10.setFixedWidth(32)
        self._btn_next10.clicked.connect(lambda: self._step_frame(10))

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(max(0, self._n_frames - 1))
        self._slider.valueChanged.connect(self._on_slider)

        self._lbl_frame = QLabel(f"0 / {self._n_frames}")
        self._lbl_frame.setMinimumWidth(100)

        nav_lay.addWidget(self._btn_prev10)
        nav_lay.addWidget(self._btn_prev)
        nav_lay.addWidget(self._slider, 1)
        nav_lay.addWidget(self._btn_next)
        nav_lay.addWidget(self._btn_next10)
        nav_lay.addWidget(self._lbl_frame)

        center_lay.addWidget(nav)
        root.addWidget(center, 1)

        # Click on plot → move cursor
        self._plot_x.scene().sigMouseClicked.connect(self._on_plot_click)

        # ── Right panel: tools ───────────────────────────────────────────────
        right = QWidget()
        right.setFixedWidth(240)
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        tools = QWidget()
        tools_lay = QVBoxLayout(tools)
        tools_lay.setContentsMargins(4, 4, 4, 4)
        tools_lay.setSpacing(6)

        # --- Selection tool ---
        sel_group = QGroupBox("Selection")
        sel_lay = QVBoxLayout(sel_group)
        sel_lay.setSpacing(4)

        btn_add_sel = QPushButton("Add Region")
        btn_add_sel.setToolTip("Add a rectangular selection region on the plots")
        btn_add_sel.clicked.connect(self._add_selection_region)
        sel_lay.addWidget(btn_add_sel)

        btn_clear_sel = QPushButton("Clear Selection")
        btn_clear_sel.clicked.connect(self._clear_selection)
        sel_lay.addWidget(btn_clear_sel)

        tools_lay.addWidget(sel_group)

        # --- Delete tool ---
        del_group = QGroupBox("Delete")
        del_lay = QVBoxLayout(del_group)
        del_lay.setSpacing(4)

        btn_del = QPushButton("Delete Selected")
        btn_del.setObjectName("danger")
        btn_del.setToolTip("Set selected bodypart data to NaN in the selection range")
        btn_del.clicked.connect(self._delete_selected)
        del_lay.addWidget(btn_del)

        tools_lay.addWidget(del_group)

        # --- Interpolation tool ---
        interp_group = QGroupBox("Interpolation")
        interp_lay = QVBoxLayout(interp_group)
        interp_lay.setSpacing(4)

        interp_form = QFormLayout()
        self._spin_interp_gap = QSpinBox()
        self._spin_interp_gap.setRange(1, 500)
        self._spin_interp_gap.setValue(30)
        self._spin_interp_gap.setSuffix(" fr")
        interp_form.addRow("Max gap:", self._spin_interp_gap)
        interp_lay.addLayout(interp_form)

        btn_interp = QPushButton("Interpolate Selected")
        btn_interp.setToolTip("Fill NaN gaps in selection range")
        btn_interp.clicked.connect(self._interpolate_selected)
        interp_lay.addWidget(btn_interp)

        tools_lay.addWidget(interp_group)

        # --- Smoothing tool ---
        smooth_group = QGroupBox("Smoothing")
        smooth_lay = QVBoxLayout(smooth_group)
        smooth_lay.setSpacing(4)

        smooth_form = QFormLayout()
        self._spin_smooth_win = QSpinBox()
        self._spin_smooth_win.setRange(3, 101)
        self._spin_smooth_win.setSingleStep(2)
        self._spin_smooth_win.setValue(11)
        smooth_form.addRow("Window:", self._spin_smooth_win)
        self._spin_smooth_ord = QSpinBox()
        self._spin_smooth_ord.setRange(1, 9)
        self._spin_smooth_ord.setValue(3)
        smooth_form.addRow("Order:", self._spin_smooth_ord)
        smooth_lay.addLayout(smooth_form)

        btn_smooth = QPushButton("Smooth Selected")
        btn_smooth.setToolTip("Apply Savitzky-Golay filter to selection")
        btn_smooth.clicked.connect(self._smooth_selected)
        smooth_lay.addWidget(btn_smooth)

        tools_lay.addWidget(smooth_group)

        # --- Swap identities ---
        swap_group = QGroupBox("Swap Identities")
        swap_lay = QVBoxLayout(swap_group)
        swap_lay.setSpacing(4)

        swap_form = QFormLayout()
        self._combo_swap_a = QComboBox()
        self._combo_swap_b = QComboBox()
        for aid in self._working_dfs:
            self._combo_swap_a.addItem(str(aid))
            self._combo_swap_b.addItem(str(aid))
        if len(self._working_dfs) >= 2:
            self._combo_swap_b.setCurrentIndex(1)
        swap_form.addRow("A:", self._combo_swap_a)
        swap_form.addRow("B:", self._combo_swap_b)
        swap_lay.addLayout(swap_form)

        btn_swap = QPushButton("Swap in Selection")
        btn_swap.setToolTip("Swap all bodypart data between two animals in selection range")
        btn_swap.clicked.connect(self._swap_identities)
        swap_lay.addWidget(btn_swap)

        tools_lay.addWidget(swap_group)

        # --- Auto-fix proximity ---
        fix_group = QGroupBox("Auto-Fix Proximity")
        fix_lay = QVBoxLayout(fix_group)
        fix_lay.setSpacing(4)

        fix_lay.addWidget(QLabel(
            "Deduce missing nose from ears/neck\n"
            "and tail from hips when animals\n"
            "are too close."
        ))

        btn_fix = QPushButton("Auto-Fix All")
        btn_fix.setToolTip("Fix missing nose/tail by deduction from neighboring bodyparts")
        btn_fix.clicked.connect(self._auto_fix_proximity)
        fix_lay.addWidget(btn_fix)

        tools_lay.addWidget(fix_group)

        # --- Spacer + Apply / Save ---
        tools_lay.addStretch()

        btn_undo = QPushButton("Undo All Changes")
        btn_undo.clicked.connect(self._undo_all)
        tools_lay.addWidget(btn_undo)

        btn_apply = QPushButton("Apply Changes")
        btn_apply.setObjectName("accent")
        btn_apply.clicked.connect(self._apply_changes)
        tools_lay.addWidget(btn_apply)

        btn_save = QPushButton("Save Cleaned File...")
        btn_save.clicked.connect(self._save_cleaned)
        tools_lay.addWidget(btn_save)

        right_scroll.setWidget(tools)
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.addWidget(right_scroll)
        root.addWidget(right)

    # ── Bodypart population ──────────────────────────────────────────────────

    def _populate_bodyparts(self) -> None:
        """Populate bodypart checkboxes for the current animal."""
        # Clear old
        for chk in self._bp_checks.values():
            chk.setParent(None)
        self._bp_checks.clear()

        aid = self._combo_animal.currentText()
        if aid not in self._working_dfs:
            return

        bps = get_bodyparts(self._working_dfs[aid])
        for i, bp in enumerate(bps):
            chk = QCheckBox(bp)
            chk.setChecked(True)
            color = _TRACE_COLORS[i % len(_TRACE_COLORS)]
            chk.setStyleSheet(f"color: {color};")
            chk.toggled.connect(self._refresh_plots)
            self._bp_layout.addWidget(chk)
            self._bp_checks[bp] = chk

    def _on_animal_changed(self) -> None:
        self._populate_bodyparts()
        self._refresh_plots()

    def _select_all_bps(self) -> None:
        for chk in self._bp_checks.values():
            chk.setChecked(True)

    def _select_no_bps(self) -> None:
        for chk in self._bp_checks.values():
            chk.setChecked(False)

    # ── Plot rendering ───────────────────────────────────────────────────────

    def _refresh_plots(self) -> None:
        """Redraw X and Y plots for selected bodyparts."""
        self._plot_x.clear()
        self._plot_x.addItem(self._cursor_x)
        self._plot_y.clear()
        self._plot_y.addItem(self._cursor_y)

        # Re-add selection region if present
        if self._selection_region is not None:
            self._plot_x.addItem(self._selection_region)
            # Clone for Y plot
            rgn = self._selection_region.getRegion()
            self._sel_region_y = pg.LinearRegionItem(
                values=rgn,
                brush=pg.mkBrush(124, 58, 237, 40),
                pen=pg.mkPen("#7c3aed", width=1),
            )
            self._sel_region_y.sigRegionChanged.connect(
                lambda: self._selection_region.setRegion(self._sel_region_y.getRegion())
            )
            self._selection_region.sigRegionChanged.connect(
                lambda: self._sel_region_y.setRegion(self._selection_region.getRegion())
            )
            self._plot_y.addItem(self._sel_region_y)

        aid = self._combo_animal.currentText()
        if aid not in self._working_dfs:
            return

        df = self._working_dfs[aid]
        frames = np.arange(len(df))

        legend_x = self._plot_x.addLegend(
            offset=(10, 10), labelTextColor="#cdd6f4",
            brush=pg.mkBrush(30, 30, 46, 180),
        )
        legend_y = self._plot_y.addLegend(
            offset=(10, 10), labelTextColor="#cdd6f4",
            brush=pg.mkBrush(30, 30, 46, 180),
        )

        color_idx = 0
        for bp, chk in self._bp_checks.items():
            if not chk.isChecked():
                continue
            color = _TRACE_COLORS[color_idx % len(_TRACE_COLORS)]
            pen = pg.mkPen(color, width=1.2)
            color_idx += 1

            x_col = f"{bp}_x"
            y_col = f"{bp}_y"
            if x_col in df.columns:
                x_data = df[x_col].to_numpy(dtype=np.float64)
                self._plot_x.plot(frames, x_data, pen=pen, name=bp)
            if y_col in df.columns:
                y_data = df[y_col].to_numpy(dtype=np.float64)
                self._plot_y.plot(frames, y_data, pen=pen, name=bp)

        # Set default zoom to ~1000 frames around current position
        window = min(1000, self._n_frames)
        start = max(0, self._current_frame - window // 2)
        end = min(self._n_frames, start + window)
        self._plot_x.setXRange(start, end, padding=0.01)

    # ── Frame navigation ─────────────────────────────────────────────────────

    def _on_slider(self, value: int) -> None:
        self._current_frame = value
        self._lbl_frame.setText(f"{value} / {self._n_frames}")
        self._cursor_x.setValue(value)
        self._cursor_y.setValue(value)

    def _step_frame(self, delta: int) -> None:
        new_val = max(0, min(self._n_frames - 1, self._current_frame + delta))
        self._slider.setValue(new_val)

    def _on_plot_click(self, event) -> None:
        pos = event.scenePos()
        vb = self._plot_x.getPlotItem().getViewBox()
        pt = vb.mapSceneToView(pos)
        frame = int(round(pt.x()))
        if 0 <= frame < self._n_frames:
            self._slider.setValue(frame)

    # ── Selection ────────────────────────────────────────────────────────────

    def _add_selection_region(self) -> None:
        """Add a draggable region selector to the plots."""
        if self._selection_region is not None:
            self._clear_selection()

        # Default: 200 frames around current position
        half = 100
        lo = max(0, self._current_frame - half)
        hi = min(self._n_frames, self._current_frame + half)

        self._selection_region = pg.LinearRegionItem(
            values=[lo, hi],
            brush=pg.mkBrush(124, 58, 237, 40),
            pen=pg.mkPen("#7c3aed", width=1),
        )
        self._refresh_plots()

    def _clear_selection(self) -> None:
        if self._selection_region is not None:
            self._plot_x.removeItem(self._selection_region)
            self._selection_region = None
        if hasattr(self, "_sel_region_y") and self._sel_region_y is not None:
            self._plot_y.removeItem(self._sel_region_y)
            self._sel_region_y = None

    def _get_selection_range(self) -> tuple[int, int]:
        """Return (start, end) frame indices of current selection, or full range."""
        if self._selection_region is not None:
            lo, hi = self._selection_region.getRegion()
            return max(0, int(round(lo))), min(self._n_frames, int(round(hi)))
        return 0, self._n_frames

    def _selected_bodyparts(self) -> list[str]:
        """Return list of currently checked bodyparts."""
        return [bp for bp, chk in self._bp_checks.items() if chk.isChecked()]

    # ── Tools ────────────────────────────────────────────────────────────────

    def _delete_selected(self) -> None:
        """Set selected bodypart data to NaN in the selection range."""
        aid = self._combo_animal.currentText()
        if aid not in self._working_dfs:
            return
        start, end = self._get_selection_range()
        bps = self._selected_bodyparts()
        df = self._working_dfs[aid]
        for bp in bps:
            for coord in ("x", "y"):
                col = f"{bp}_{coord}"
                if col in df.columns:
                    df.loc[start:end - 1, col] = np.nan
        logger.info("Deleted %d bodyparts in frames %d-%d", len(bps), start, end)
        self._refresh_plots()

    def _interpolate_selected(self) -> None:
        """Interpolate NaN gaps in the selection range."""
        from dlc_processor.core.data_cleaner import _interpolate_with_gap_limit

        aid = self._combo_animal.currentText()
        if aid not in self._working_dfs:
            return
        start, end = self._get_selection_range()
        max_gap = self._spin_interp_gap.value()
        bps = self._selected_bodyparts()
        df = self._working_dfs[aid]

        for bp in bps:
            for coord in ("x", "y"):
                col = f"{bp}_{coord}"
                if col not in df.columns:
                    continue
                arr = df[col].to_numpy(dtype=np.float64)
                segment = arr[start:end].copy()
                filled = _interpolate_with_gap_limit(segment, max_gap)
                arr[start:end] = filled
                df[col] = arr

        logger.info("Interpolated %d bodyparts in frames %d-%d (max_gap=%d)",
                     len(bps), start, end, max_gap)
        self._refresh_plots()

    def _smooth_selected(self) -> None:
        """Apply Savitzky-Golay smoothing to the selection range."""
        from scipy.signal import savgol_filter

        aid = self._combo_animal.currentText()
        if aid not in self._working_dfs:
            return
        start, end = self._get_selection_range()
        window = self._spin_smooth_win.value()
        if window % 2 == 0:
            window += 1
        polyorder = self._spin_smooth_ord.value()
        window = max(window, polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3)
        bps = self._selected_bodyparts()
        df = self._working_dfs[aid]

        for bp in bps:
            for coord in ("x", "y"):
                col = f"{bp}_{coord}"
                if col not in df.columns:
                    continue
                arr = df[col].to_numpy(dtype=np.float64)
                segment = arr[start:end].copy()
                valid = ~np.isnan(segment)
                if valid.sum() >= window:
                    # Smooth only valid contiguous segments
                    smoothed = segment.copy()
                    # Find contiguous valid runs
                    padded = np.concatenate(([False], valid, [False]))
                    seg_starts = np.where(np.diff(padded.astype(int)) == 1)[0]
                    seg_ends = np.where(np.diff(padded.astype(int)) == -1)[0]
                    for s, e in zip(seg_starts, seg_ends):
                        if (e - s) >= window:
                            smoothed[s:e] = savgol_filter(segment[s:e], window, polyorder)
                    arr[start:end] = smoothed
                    df[col] = arr

        logger.info("Smoothed %d bodyparts in frames %d-%d (win=%d, order=%d)",
                     len(bps), start, end, window, polyorder)
        self._refresh_plots()

    def _swap_identities(self) -> None:
        """Swap all bodypart data between two animals in the selection range."""
        aid_a = self._combo_swap_a.currentText()
        aid_b = self._combo_swap_b.currentText()
        if aid_a == aid_b:
            return
        if aid_a not in self._working_dfs or aid_b not in self._working_dfs:
            return

        start, end = self._get_selection_range()
        df_a = self._working_dfs[aid_a]
        df_b = self._working_dfs[aid_b]

        bps_a = get_bodyparts(df_a)
        bps_b = get_bodyparts(df_b)
        # Swap columns that exist in both
        common_bps = [bp for bp in bps_a if bp in bps_b]

        for bp in common_bps:
            for suffix in ("_x", "_y", "_likelihood"):
                col = f"{bp}{suffix}"
                if col in df_a.columns and col in df_b.columns:
                    tmp = df_a.loc[start:end - 1, col].copy()
                    df_a.loc[start:end - 1, col] = df_b.loc[start:end - 1, col].values
                    df_b.loc[start:end - 1, col] = tmp.values

        logger.info("Swapped %s <-> %s in frames %d-%d (%d bodyparts)",
                     aid_a, aid_b, start, end, len(common_bps))
        self._refresh_plots()

    def _auto_fix_proximity(self) -> None:
        """Fix missing nose/tail by deduction from neighbouring bodyparts.

        Logic:
        - If nose is NaN but ears + neck exist → nose = 2*neck - midpoint(ears)
          (extrapolate along the neck→nose axis)
        - If tail is NaN but hips exist → tail = midpoint(hips) extended away
          from neck direction
        """
        for aid, df in self._working_dfs.items():
            bps = get_bodyparts(df)
            n_fixed = _auto_fix_animal_proximity(df, bps)
            if n_fixed > 0:
                logger.info("Auto-fixed %d frames for animal %s", n_fixed, aid)

        self._refresh_plots()

    def _undo_all(self) -> None:
        """Reset working data to original."""
        self._working_dfs = deepcopy(self._original_dfs)
        self._populate_bodyparts()
        self._refresh_plots()
        logger.info("All track editor changes undone")

    def _apply_changes(self) -> None:
        """Emit the working data and close."""
        self.data_changed.emit(self._working_dfs)
        self.accept()

    def _save_cleaned(self) -> None:
        """Save current working data as H5 or CSV."""
        path, filt = QFileDialog.getSaveFileName(
            self, "Save Cleaned Data", "",
            "HDF5 Files (*.h5);;CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return

        p = Path(path)
        if p.suffix.lower() in (".h5", ".hdf5"):
            from dlc_processor.core.data_cleaner import save_cleaned_h5
            save_cleaned_h5(self._working_dfs, path)
        elif p.suffix.lower() == ".csv":
            _save_cleaned_csv(self._working_dfs, path)
        else:
            from dlc_processor.core.data_cleaner import save_cleaned_h5
            save_cleaned_h5(self._working_dfs, path)

        logger.info("Saved cleaned data to %s", path)


# ── Auto-fix proximity helpers ──────────────────────────────────────────────

_NOSE_CANDIDATES = ["nose", "Nose", "snout", "Snout"]
_NECK_CANDIDATES = ["neck", "Neck", "spine1", "Spine_1", "Spine1"]
_LEFT_EAR_CANDIDATES = ["left_ear", "Left_ear", "leftear", "LeftEar", "ear_left", "Ear_left"]
_RIGHT_EAR_CANDIDATES = ["right_ear", "Right_ear", "rightear", "RightEar", "ear_right", "Ear_right"]
_TAIL_CANDIDATES = ["tail", "Tail", "tailbase", "Tailbase", "tail_base", "TailBase", "tail_tip"]
_LEFT_HIP_CANDIDATES = ["left_hip", "Left_hip", "lefthip", "LeftHip", "hip_left"]
_RIGHT_HIP_CANDIDATES = ["right_hip", "Right_hip", "righthip", "RightHip", "hip_right"]


def _find_bp(bps: list[str], candidates: list[str]) -> Optional[str]:
    """Return the first bodypart name matching any candidate (case-insensitive)."""
    for cand in candidates:
        for bp in bps:
            if bp.lower() == cand.lower():
                return bp
    return None


def _auto_fix_animal_proximity(df: pd.DataFrame, bps: list[str]) -> int:
    """Fix missing nose/tail for a single animal DataFrame.

    Returns the total number of fixed frames.
    """
    n_fixed = 0

    # ── Fix nose ─────────────────────────────────────────────────────────────
    nose = _find_bp(bps, _NOSE_CANDIDATES)
    neck = _find_bp(bps, _NECK_CANDIDATES)
    left_ear = _find_bp(bps, _LEFT_EAR_CANDIDATES)
    right_ear = _find_bp(bps, _RIGHT_EAR_CANDIDATES)

    if nose and neck:
        nose_x = df[f"{nose}_x"].to_numpy(dtype=np.float64)
        nose_y = df[f"{nose}_y"].to_numpy(dtype=np.float64)
        neck_x = df[f"{neck}_x"].to_numpy(dtype=np.float64)
        neck_y = df[f"{neck}_y"].to_numpy(dtype=np.float64)

        nose_nan = np.isnan(nose_x)
        neck_valid = ~np.isnan(neck_x)

        if left_ear and right_ear:
            # Nose = extrapolation: neck + (neck - ear_midpoint)
            le_x = df[f"{left_ear}_x"].to_numpy(dtype=np.float64)
            le_y = df[f"{left_ear}_y"].to_numpy(dtype=np.float64)
            re_x = df[f"{right_ear}_x"].to_numpy(dtype=np.float64)
            re_y = df[f"{right_ear}_y"].to_numpy(dtype=np.float64)

            ear_mid_x = (le_x + re_x) / 2.0
            ear_mid_y = (le_y + re_y) / 2.0
            ears_valid = ~np.isnan(le_x) & ~np.isnan(re_x)

            fixable = nose_nan & neck_valid & ears_valid
            n = int(fixable.sum())
            if n > 0:
                # Direction: ears → neck → nose (extrapolate past neck)
                dx = neck_x - ear_mid_x
                dy = neck_y - ear_mid_y
                length = np.hypot(dx, dy)
                length[length == 0] = 1.0
                # Normalize and extend by typical nose-neck distance
                # Use median of valid nose-neck distances as reference
                valid_both = ~np.isnan(nose_x) & ~np.isnan(neck_x)
                if valid_both.sum() > 5:
                    nose_neck_dist = np.nanmedian(
                        np.hypot(nose_x[valid_both] - neck_x[valid_both],
                                 nose_y[valid_both] - neck_y[valid_both])
                    )
                else:
                    nose_neck_dist = length[fixable].mean() * 0.8

                unit_x = dx / length
                unit_y = dy / length
                df.loc[fixable, f"{nose}_x"] = neck_x[fixable] + unit_x[fixable] * nose_neck_dist
                df.loc[fixable, f"{nose}_y"] = neck_y[fixable] + unit_y[fixable] * nose_neck_dist
                df.loc[fixable, f"{nose}_likelihood"] = 0.3  # low confidence marker
                n_fixed += n
                logger.debug("Fixed %d nose frames from ears+neck for animal", n)

        elif neck_valid.any():
            # Fallback: use neck + previous valid nose-neck offset
            fixable = nose_nan & neck_valid
            n = int(fixable.sum())
            if n > 0:
                valid_both = ~np.isnan(nose_x) & ~np.isnan(neck_x)
                if valid_both.sum() > 5:
                    med_dx = np.nanmedian(nose_x[valid_both] - neck_x[valid_both])
                    med_dy = np.nanmedian(nose_y[valid_both] - neck_y[valid_both])
                    df.loc[fixable, f"{nose}_x"] = neck_x[fixable] + med_dx
                    df.loc[fixable, f"{nose}_y"] = neck_y[fixable] + med_dy
                    df.loc[fixable, f"{nose}_likelihood"] = 0.2
                    n_fixed += n

    # ── Fix tail ─────────────────────────────────────────────────────────────
    tail = _find_bp(bps, _TAIL_CANDIDATES)
    left_hip = _find_bp(bps, _LEFT_HIP_CANDIDATES)
    right_hip = _find_bp(bps, _RIGHT_HIP_CANDIDATES)

    if tail and left_hip and right_hip:
        tail_x = df[f"{tail}_x"].to_numpy(dtype=np.float64)
        tail_y = df[f"{tail}_y"].to_numpy(dtype=np.float64)
        lh_x = df[f"{left_hip}_x"].to_numpy(dtype=np.float64)
        lh_y = df[f"{left_hip}_y"].to_numpy(dtype=np.float64)
        rh_x = df[f"{right_hip}_x"].to_numpy(dtype=np.float64)
        rh_y = df[f"{right_hip}_y"].to_numpy(dtype=np.float64)

        tail_nan = np.isnan(tail_x)
        hips_valid = ~np.isnan(lh_x) & ~np.isnan(rh_x)

        hip_mid_x = (lh_x + rh_x) / 2.0
        hip_mid_y = (lh_y + rh_y) / 2.0

        fixable = tail_nan & hips_valid
        n = int(fixable.sum())
        if n > 0:
            # Direction: neck → hip_midpoint → tail (extrapolate past hips)
            if neck:
                neck_x = df[f"{neck}_x"].to_numpy(dtype=np.float64)
                neck_y = df[f"{neck}_y"].to_numpy(dtype=np.float64)
                neck_valid = ~np.isnan(neck_x)
                fixable = fixable & neck_valid

                dx = hip_mid_x - neck_x
                dy = hip_mid_y - neck_y
                length = np.hypot(dx, dy)
                length[length == 0] = 1.0

                valid_both = ~np.isnan(tail_x) & ~np.isnan(hip_mid_x)
                if valid_both.sum() > 5:
                    tail_hip_dist = np.nanmedian(
                        np.hypot(tail_x[valid_both] - hip_mid_x[valid_both],
                                 tail_y[valid_both] - hip_mid_y[valid_both])
                    )
                else:
                    tail_hip_dist = length[fixable].mean() * 0.6 if fixable.sum() > 0 else 20.0

                unit_x = dx / length
                unit_y = dy / length

                n = int(fixable.sum())
                if n > 0:
                    df.loc[fixable, f"{tail}_x"] = hip_mid_x[fixable] + unit_x[fixable] * tail_hip_dist
                    df.loc[fixable, f"{tail}_y"] = hip_mid_y[fixable] + unit_y[fixable] * tail_hip_dist
                    df.loc[fixable, f"{tail}_likelihood"] = 0.3
                    n_fixed += n
                    logger.debug("Fixed %d tail frames from hips+neck for animal", n)
            else:
                # No neck: use median tail-hip offset
                valid_both = ~np.isnan(tail_x) & hips_valid
                if valid_both.sum() > 5:
                    med_dx = np.nanmedian(tail_x[valid_both] - hip_mid_x[valid_both])
                    med_dy = np.nanmedian(tail_y[valid_both] - hip_mid_y[valid_both])
                    fixable2 = tail_nan & hips_valid
                    n2 = int(fixable2.sum())
                    if n2 > 0:
                        df.loc[fixable2, f"{tail}_x"] = hip_mid_x[fixable2] + med_dx
                        df.loc[fixable2, f"{tail}_y"] = hip_mid_y[fixable2] + med_dy
                        df.loc[fixable2, f"{tail}_likelihood"] = 0.2
                        n_fixed += n2

    return n_fixed


def _save_cleaned_csv(
    animal_dfs: dict[str, pd.DataFrame],
    output_path: str,
) -> None:
    """Save cleaned DataFrames as a flat CSV file."""
    frames = []
    for aid, df in animal_dfs.items():
        renamed = {}
        for col in df.columns:
            renamed[col] = f"{aid}_{col}"
        frames.append(df.rename(columns=renamed))
    combined = pd.concat(frames, axis=1)
    combined.to_csv(output_path, index=True)
    logger.info("Saved cleaned CSV -> %s", output_path)
