"""Data cleaning controls panel."""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from dlc_processor.core.data_cleaner import save_cleaned_h5
from dlc_processor.ui.calibration_widget import CalibrationDialog

logger = logging.getLogger(__name__)


class CleaningPanel(QGroupBox):
    """Controls for confidence filtering, gap interpolation, and SG smoothing."""

    cleaning_requested = Signal(dict)   # params dict
    impossible_fix_requested = Signal(dict)  # params dict
    calibration_changed = Signal(float)  # px_per_cm
    bodyparts_added = Signal(dict)       # emitted after computed bodyparts are added
    track_editor_requested = Signal()    # open track editor dialog
    range_selection_requested = Signal() # ask plot to show cleaning region
    range_changed = Signal(int, int)     # start, end frame changed

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Data Cleaning", parent)
        self._video_path: str = ""
        self._px_per_cm: float = 0.0
        self._computed_defs: list[dict] = []
        self._animal_dfs: dict = {}
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        # --- Calibration section ---
        cal_group = QGroupBox("Calibration")
        cal_layout = QVBoxLayout(cal_group)
        cal_layout.setSpacing(4)

        cal_row = QHBoxLayout()
        self._btn_calibrate = QPushButton("Calibrate\u2026")
        self._btn_calibrate.clicked.connect(self._open_calibration)
        cal_row.addWidget(self._btn_calibrate)
        self._lbl_scale = QLabel("Not calibrated")
        cal_row.addWidget(self._lbl_scale, 1)
        cal_layout.addLayout(cal_row)

        self._chk_apply_cal = QCheckBox("Apply calibration (convert px \u2192 cm)")
        self._chk_apply_cal.setEnabled(False)
        cal_layout.addWidget(self._chk_apply_cal)

        layout.addWidget(cal_group)

        # --- Computed Bodyparts section ---
        comp_group = QGroupBox("Computed Bodyparts")
        comp_layout = QVBoxLayout(comp_group)
        comp_layout.setSpacing(4)

        comp_form = QFormLayout()
        comp_form.setSpacing(4)

        self._combo_operation = QComboBox()
        self._combo_operation.addItems(["mean", "midpoint"])
        comp_form.addRow("Operation:", self._combo_operation)

        self._edit_bp_name = QLineEdit()
        self._edit_bp_name.setPlaceholderText("e.g. body_center")
        comp_form.addRow("Name:", self._edit_bp_name)

        self._edit_sources = QLineEdit()
        self._edit_sources.setPlaceholderText("e.g. neck, left_hip, right_hip")
        comp_form.addRow("Sources:", self._edit_sources)

        comp_layout.addLayout(comp_form)

        comp_btn_row = QHBoxLayout()
        btn_add_bp = QPushButton("Add")
        btn_add_bp.clicked.connect(self._add_computed_bp)
        comp_btn_row.addWidget(btn_add_bp)
        btn_remove_bp = QPushButton("Remove")
        btn_remove_bp.clicked.connect(self._remove_computed_bp)
        comp_btn_row.addWidget(btn_remove_bp)
        comp_btn_row.addStretch()
        comp_layout.addLayout(comp_btn_row)

        self._computed_list = QListWidget()
        self._computed_list.setMaximumHeight(80)
        comp_layout.addWidget(self._computed_list)

        layout.addWidget(comp_group)

        # --- Cleaning stages ---
        form = QFormLayout()
        form.setSpacing(5)

        # Stage 1 — Confidence filter
        self._chk_conf = QCheckBox("Enable")
        self._chk_conf.setChecked(True)
        self._spin_conf = QDoubleSpinBox()
        self._spin_conf.setRange(0.0, 1.0)
        self._spin_conf.setSingleStep(0.05)
        self._spin_conf.setValue(0.60)
        self._spin_conf.setDecimals(2)
        row1 = QHBoxLayout()
        row1.addWidget(self._chk_conf)
        row1.addWidget(QLabel("Threshold:"))
        row1.addWidget(self._spin_conf)
        row1.addStretch()
        form.addRow("Conf. filter:", row1)

        # Stage 2 — Interpolation
        self._chk_interp = QCheckBox("Enable")
        self._chk_interp.setChecked(True)
        self._spin_gap = QSpinBox()
        self._spin_gap.setRange(1, 200)
        self._spin_gap.setValue(15)
        self._spin_gap.setSuffix(" fr")
        row2 = QHBoxLayout()
        row2.addWidget(self._chk_interp)
        row2.addWidget(QLabel("Max gap:"))
        row2.addWidget(self._spin_gap)
        row2.addStretch()
        form.addRow("Interpolation:", row2)

        # Stage 3 — SG smoothing
        self._chk_smooth = QCheckBox("Enable")
        self._chk_smooth.setChecked(True)
        self._spin_win = QSpinBox()
        self._spin_win.setRange(3, 101)
        self._spin_win.setSingleStep(2)
        self._spin_win.setValue(11)
        self._spin_order = QSpinBox()
        self._spin_order.setRange(1, 9)
        self._spin_order.setValue(3)
        row3 = QHBoxLayout()
        row3.addWidget(self._chk_smooth)
        row3.addWidget(QLabel("Win:"))
        row3.addWidget(self._spin_win)
        row3.addWidget(QLabel("Order:"))
        row3.addWidget(self._spin_order)
        row3.addStretch()
        form.addRow("SG smooth:", row3)

        layout.addLayout(form)

        # --- Cleaning Range ---
        range_group = QGroupBox("Cleaning Range")
        range_lay = QVBoxLayout(range_group)
        range_lay.setSpacing(4)

        self._chk_use_range = QCheckBox("Apply to selected range only")
        self._chk_use_range.toggled.connect(self._on_range_toggled)
        range_lay.addWidget(self._chk_use_range)

        range_form = QFormLayout()
        range_form.setSpacing(4)
        self._spin_range_start = QSpinBox()
        self._spin_range_start.setRange(0, 999999)
        self._spin_range_start.setValue(0)
        self._spin_range_start.setSuffix(" fr")
        self._spin_range_start.setEnabled(False)
        self._spin_range_start.valueChanged.connect(self._emit_range_changed)
        range_form.addRow("Start:", self._spin_range_start)

        self._spin_range_end = QSpinBox()
        self._spin_range_end.setRange(0, 999999)
        self._spin_range_end.setValue(1000)
        self._spin_range_end.setSuffix(" fr")
        self._spin_range_end.setEnabled(False)
        self._spin_range_end.valueChanged.connect(self._emit_range_changed)
        range_form.addRow("End:", self._spin_range_end)
        range_lay.addLayout(range_form)

        self._btn_select_on_plot = QPushButton("Select on Plot")
        self._btn_select_on_plot.setEnabled(False)
        self._btn_select_on_plot.clicked.connect(
            lambda: self.range_selection_requested.emit()
        )
        range_lay.addWidget(self._btn_select_on_plot)

        layout.addWidget(range_group)

        btn_row = QHBoxLayout()
        btn_apply = QPushButton("Apply Cleaning")
        btn_apply.clicked.connect(self._emit_request)
        btn_row.addWidget(btn_apply)
        btn_fix = QPushButton("Fix Impossible Conditions")
        btn_fix.clicked.connect(self._emit_impossible_fix_request)
        btn_row.addWidget(btn_fix)
        layout.addLayout(btn_row)

        # --- Track Editor ---
        self._btn_track_editor = QPushButton("Track Editor\u2026")
        self._btn_track_editor.setToolTip(
            "Open interactive track editor to visualize, edit, fix and save tracks"
        )
        self._btn_track_editor.clicked.connect(self._open_track_editor)
        layout.addWidget(self._btn_track_editor)

        # --- Save Cleaned H5 ---
        btn_save = QPushButton("Save Cleaned H5\u2026")
        btn_save.clicked.connect(self._save_cleaned_h5)
        layout.addWidget(btn_save)

    def _emit_request(self) -> None:
        p = self.params()
        if self._computed_defs:
            p["computed_bodyparts"] = self._computed_defs
        self.cleaning_requested.emit(p)

    def _emit_impossible_fix_request(self) -> None:
        p = {
            "max_gap_frames": self._spin_gap.value(),
        }
        if self._computed_defs:
            p["computed_bodyparts"] = self._computed_defs
        if self._chk_use_range.isChecked():
            p["start_frame"] = self._spin_range_start.value()
            p["end_frame"] = self._spin_range_end.value()
        self.impossible_fix_requested.emit(p)

    # --- Calibration helpers ---

    def set_video_path(self, path: str) -> None:
        """Set the video path used by the calibration dialog."""
        self._video_path = path

    def px_per_cm(self) -> float:
        """Return the current calibration scale (0.0 if not calibrated)."""
        return self._px_per_cm

    def _open_calibration(self) -> None:
        if not self._video_path:
            logger.warning("No video path set — cannot open calibration dialog.")
            return
        dlg = CalibrationDialog(self._video_path, parent=self)
        dlg.calibration_done.connect(self._on_calibration_done)
        dlg.exec()

    def _on_calibration_done(self, px_per_cm: float) -> None:
        self._px_per_cm = px_per_cm
        self._lbl_scale.setText(f"Scale: {px_per_cm:.2f} px/cm")
        self._chk_apply_cal.setEnabled(True)
        self._chk_apply_cal.setChecked(True)
        self.calibration_changed.emit(px_per_cm)

    # --- Range helpers ---

    def _on_range_toggled(self, checked: bool) -> None:
        self._spin_range_start.setEnabled(checked)
        self._spin_range_end.setEnabled(checked)
        self._btn_select_on_plot.setEnabled(checked)
        if checked:
            self.range_selection_requested.emit()
        else:
            self.range_changed.emit(0, 0)

    def _emit_range_changed(self) -> None:
        if self._chk_use_range.isChecked():
            self.range_changed.emit(
                self._spin_range_start.value(),
                self._spin_range_end.value(),
            )

    def set_range(self, start: int, end: int) -> None:
        """Update range spinboxes from plot selection (no re-emit)."""
        self._spin_range_start.blockSignals(True)
        self._spin_range_end.blockSignals(True)
        self._spin_range_start.setValue(start)
        self._spin_range_end.setValue(end)
        self._spin_range_start.blockSignals(False)
        self._spin_range_end.blockSignals(False)

    def set_n_frames(self, n: int) -> None:
        """Update spinbox maximums when data changes."""
        self._spin_range_start.setMaximum(max(0, n - 1))
        self._spin_range_end.setMaximum(n)

    # --- Params ---

    def params(self) -> dict:
        p = {
            "apply_conf":       self._chk_conf.isChecked(),
            "conf_threshold":   self._spin_conf.value(),
            "apply_interp":     self._chk_interp.isChecked(),
            "max_gap_frames":   self._spin_gap.value(),
            "apply_smooth":     self._chk_smooth.isChecked(),
            "sg_window":        self._spin_win.value(),
            "sg_polyorder":     self._spin_order.value(),
        }
        if self._chk_apply_cal.isChecked() and self._px_per_cm > 0:
            p["px_per_cm"] = self._px_per_cm
        if self._chk_use_range.isChecked():
            p["start_frame"] = self._spin_range_start.value()
            p["end_frame"] = self._spin_range_end.value()
        return p

    # --- Computed bodyparts ---

    def computed_bodyparts(self) -> list[dict]:
        """Return the current list of computed bodypart definitions."""
        return list(self._computed_defs)

    def _add_computed_bp(self) -> None:
        name = self._edit_bp_name.text().strip()
        sources_text = self._edit_sources.text().strip()
        operation = self._combo_operation.currentText()
        if not name or not sources_text:
            logger.warning("Computed bodypart name and sources are required.")
            return
        sources = [s.strip() for s in sources_text.split(",") if s.strip()]
        if operation == "midpoint" and len(sources) != 2:
            logger.warning("Midpoint requires exactly 2 sources, got %d.", len(sources))
            return
        defn = {"name": name, "sources": sources, "operation": operation}
        self._computed_defs.append(defn)
        self._computed_list.addItem(f"{name} = {operation}({', '.join(sources)})")
        self._edit_bp_name.clear()
        self._edit_sources.clear()
        logger.info("Added computed bodypart definition: %s", defn)

    def _remove_computed_bp(self) -> None:
        row = self._computed_list.currentRow()
        if row < 0:
            return
        self._computed_list.takeItem(row)
        removed = self._computed_defs.pop(row)
        logger.info("Removed computed bodypart definition: %s", removed)

    # --- Track Editor ---

    def _open_track_editor(self) -> None:
        self.track_editor_requested.emit()

    # --- Animal data / Save H5 ---

    def set_animal_dfs(self, dfs: dict) -> None:
        """Store the current animal DataFrames for saving."""
        self._animal_dfs = dfs

    def _save_cleaned_h5(self) -> None:
        if not self._animal_dfs:
            logger.warning("No animal data available to save.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Cleaned H5", "", "HDF5 Files (*.h5);;All Files (*)"
        )
        if not path:
            return
        try:
            save_cleaned_h5(self._animal_dfs, path)
            logger.info("Saved cleaned H5 to %s", path)
        except Exception:
            logger.exception("Failed to save cleaned H5")
