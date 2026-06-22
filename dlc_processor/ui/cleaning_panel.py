"""Data cleaning controls panel."""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import Qt, Signal
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
    QMenu,
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
    identity_swap_requested = Signal(dict)  # animal_a, animal_b, optional range
    identity_rename_requested = Signal(dict)  # old_label, new_label
    mask_identity_fix_requested = Signal(dict)  # use mask track ids as identity ground truth
    reset_cleaning_requested = Signal()  # restore original tracking and masks

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Data Cleaning", parent)
        self._video_path: str = ""
        self._px_per_cm: float = 0.0
        self._n_frames: int = 0
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
        self._btn_calibrate.setToolTip("Draw a known distance on the video to set the px/cm scale")
        self._btn_calibrate.clicked.connect(self._open_calibration)
        cal_row.addWidget(self._btn_calibrate)
        self._lbl_scale = QLabel("Not calibrated")
        cal_row.addWidget(self._lbl_scale, 1)
        cal_layout.addLayout(cal_row)

        self._chk_apply_cal = QCheckBox("Use calibrated units in analyses / export")
        self._chk_apply_cal.setToolTip("When checked, kinematics and export will include cm-based columns alongside px-based ones")
        self._chk_apply_cal.setEnabled(False)
        self._chk_apply_cal.toggled.connect(self._emit_active_calibration_changed)
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
        self._combo_operation.setToolTip("mean: average of all sources; midpoint: halfway between exactly 2 sources")
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
        btn_add_bp.setToolTip("Create a virtual bodypart computed from the listed sources")
        btn_add_bp.clicked.connect(self._add_computed_bp)
        comp_btn_row.addWidget(btn_add_bp)
        btn_remove_bp = QPushButton("Remove")
        btn_remove_bp.setToolTip("Remove the selected computed bodypart definition")
        btn_remove_bp.clicked.connect(self._remove_computed_bp)
        comp_btn_row.addWidget(btn_remove_bp)
        comp_btn_row.addStretch()
        comp_layout.addLayout(comp_btn_row)

        self._computed_list = QListWidget()
        self._computed_list.setMaximumHeight(80)
        comp_layout.addWidget(self._computed_list)

        layout.addWidget(comp_group)

        # --- Cleaning stages ---
        stages_group = QGroupBox("Cleaning Stages")
        stages_lay = QVBoxLayout(stages_group)
        stages_lay.setSpacing(4)
        form = QFormLayout()
        form.setSpacing(5)

        # Stage 1 — Confidence filter
        self._chk_conf = QCheckBox("Enable")
        self._chk_conf.setChecked(True)
        self._chk_conf.setToolTip("Mark low-confidence frames as missing data for interpolation")
        self._spin_conf = QDoubleSpinBox()
        self._spin_conf.setRange(0.0, 1.0)
        self._spin_conf.setSingleStep(0.05)
        self._spin_conf.setValue(0.60)
        self._spin_conf.setDecimals(2)
        self._spin_conf.setToolTip("DLC likelihood threshold (0.6 is typical; lower keeps more data)")
        row1 = QHBoxLayout()
        row1.addWidget(self._chk_conf)
        row1.addWidget(QLabel("Threshold:"))
        row1.addWidget(self._spin_conf)
        row1.addStretch()
        form.addRow("Conf. filter:", row1)

        # Stage 2 — Interpolation
        self._chk_interp = QCheckBox("Enable")
        self._chk_interp.setChecked(True)
        self._chk_interp.setToolTip("Fill NaN gaps with linear interpolation")
        self._spin_gap = QSpinBox()
        self._spin_gap.setRange(1, 200)
        self._spin_gap.setValue(15)
        self._spin_gap.setSuffix(" frames")
        self._spin_gap.setToolTip("Maximum consecutive missing frames to interpolate (longer gaps stay empty)")
        row2 = QHBoxLayout()
        row2.addWidget(self._chk_interp)
        row2.addWidget(QLabel("Max gap:"))
        row2.addWidget(self._spin_gap)
        row2.addStretch()
        form.addRow("Interpolation:", row2)

        # Stage 3 — SG smoothing
        self._chk_smooth = QCheckBox("Enable")
        self._chk_smooth.setChecked(True)
        self._chk_smooth.setToolTip("Apply Savitzky-Golay filter to smooth trajectories")
        self._spin_win = QSpinBox()
        self._spin_win.setRange(3, 101)
        self._spin_win.setSingleStep(2)
        self._spin_win.setValue(11)
        self._spin_win.setToolTip("Window size in frames (must be odd; larger = smoother)")
        self._spin_order = QSpinBox()
        self._spin_order.setRange(1, 9)
        self._spin_order.setValue(3)
        self._spin_order.setToolTip("Polynomial order (lower = smoother; typically 2-5)")
        row3 = QHBoxLayout()
        row3.addWidget(self._chk_smooth)
        row3.addWidget(QLabel("Window:"))
        row3.addWidget(self._spin_win)
        row3.addWidget(QLabel("Poly order:"))
        row3.addWidget(self._spin_order)
        row3.addStretch()
        form.addRow("SG smooth:", row3)

        stages_lay.addLayout(form)
        layout.addWidget(stages_group)

        # --- Cleaning Range ---
        range_group = QGroupBox("Cleaning Range")
        range_lay = QVBoxLayout(range_group)
        range_lay.setSpacing(4)

        self._chk_use_range = QCheckBox("Apply to selected range only")
        self._chk_use_range.setToolTip("Restrict cleaning to a frame range instead of the entire recording")
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

        range_btn_row = QHBoxLayout()
        self._btn_select_on_plot = QPushButton("Select on Plot")
        self._btn_select_on_plot.setEnabled(False)
        self._btn_select_on_plot.clicked.connect(
            lambda: self.range_selection_requested.emit()
        )
        range_btn_row.addWidget(self._btn_select_on_plot)

        self._btn_select_all = QPushButton("Select All")
        self._btn_select_all.setEnabled(False)
        self._btn_select_all.setToolTip("Set the range to all frames in the active recording")
        self._btn_select_all.clicked.connect(self._select_all_range)
        range_btn_row.addWidget(self._btn_select_all)
        range_btn_row.addStretch()
        range_lay.addLayout(range_btn_row)

        layout.addWidget(range_group)

        # --- Identity swap ---
        swap_group = QGroupBox("Identity Swap")
        swap_lay = QVBoxLayout(swap_group)
        swap_lay.setSpacing(4)

        swap_form = QFormLayout()
        swap_form.setSpacing(4)
        self._combo_swap_a = QComboBox()
        self._combo_swap_b = QComboBox()
        swap_form.addRow("Mouse A:", self._combo_swap_a)
        swap_form.addRow("Mouse B:", self._combo_swap_b)
        swap_lay.addLayout(swap_form)

        self._lbl_swap_hint = QLabel(
            "Uses the Cleaning Range when enabled; otherwise swaps the whole active recording."
        )
        self._lbl_swap_hint.setWordWrap(True)
        self._lbl_swap_hint.setStyleSheet("color:#a6adc8; font-size:11px;")
        swap_lay.addWidget(self._lbl_swap_hint)

        self._btn_swap_ids = QPushButton("Swap Mouse Identity")
        self._btn_swap_ids.setEnabled(False)
        self._btn_swap_ids.setToolTip(
            "Swap all shared bodypart coordinates between two animals in the selected recording"
        )
        self._btn_swap_ids.clicked.connect(self._emit_identity_swap_request)
        self._btn_swap_ids.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._btn_swap_ids.customContextMenuRequested.connect(self._show_swap_button_menu)
        swap_lay.addWidget(self._btn_swap_ids)

        self._btn_fix_mask_ids = QPushButton("Fix Identity From Masks")
        self._btn_fix_mask_ids.setEnabled(False)
        self._btn_fix_mask_ids.setToolTip(
            "Use COCO mask track IDs as ground truth and repair swapped keypoint identities over the whole recording"
        )
        self._btn_fix_mask_ids.clicked.connect(self._emit_mask_identity_fix_request)
        swap_lay.addWidget(self._btn_fix_mask_ids)

        layout.addWidget(swap_group)

        # --- Identity rename ---
        rename_group = QGroupBox("Identity Labels")
        rename_lay = QVBoxLayout(rename_group)
        rename_lay.setSpacing(4)

        rename_form = QFormLayout()
        rename_form.setSpacing(4)
        self._combo_rename_from = QComboBox()
        rename_form.addRow("Current:", self._combo_rename_from)
        self._edit_rename_to = QLineEdit()
        self._edit_rename_to.setPlaceholderText("e.g. mouse1")
        rename_form.addRow("Rename to:", self._edit_rename_to)
        rename_lay.addLayout(rename_form)

        self._btn_rename_id = QPushButton("Rename Mouse Label")
        self._btn_rename_id.setEnabled(False)
        self._btn_rename_id.setToolTip("Rename an animal identity label in the active recording")
        self._btn_rename_id.clicked.connect(self._emit_identity_rename_request)
        rename_lay.addWidget(self._btn_rename_id)

        layout.addWidget(rename_group)

        btn_row = QHBoxLayout()
        btn_apply = QPushButton("Apply Cleaning")
        btn_apply.setToolTip("Run the enabled cleaning stages: confidence filter, then interpolation, then smoothing")
        btn_apply.clicked.connect(self._emit_request)
        btn_row.addWidget(btn_apply)
        btn_fix = QPushButton("Fix Impossible Conditions")
        btn_fix.setToolTip("Detect and fix anatomically impossible poses (e.g. nose behind neck, ears on same side)")
        btn_fix.clicked.connect(self._emit_impossible_fix_request)
        btn_row.addWidget(btn_fix)
        layout.addLayout(btn_row)

        self._btn_reset_cleaning = QPushButton("Reset to Original")
        self._btn_reset_cleaning.setToolTip("Reload the original DLC tracking and COCO masks for this recording")
        self._btn_reset_cleaning.clicked.connect(self.reset_cleaning_requested.emit)
        layout.addWidget(self._btn_reset_cleaning)

        # --- Track Editor ---
        self._btn_track_editor = QPushButton("Track Editor\u2026")
        self._btn_track_editor.setToolTip(
            "Open interactive track editor to visualize, edit, fix and save tracks"
        )
        self._btn_track_editor.clicked.connect(self._open_track_editor)
        layout.addWidget(self._btn_track_editor)

        # --- Save Cleaned H5 ---
        btn_save = QPushButton("Save Cleaned H5\u2026")
        btn_save.setToolTip("Save the cleaned data as a new HDF5 file")
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

    def _emit_identity_swap_request(self) -> None:
        animal_a = self._combo_swap_a.currentText()
        animal_b = self._combo_swap_b.currentText()
        if not animal_a or not animal_b or animal_a == animal_b:
            return
        p = {
            "animal_a": animal_a,
            "animal_b": animal_b,
            "write_csv": True,
        }
        if self._chk_use_range.isChecked():
            if self._spin_range_end.value() <= self._spin_range_start.value():
                return
            p["start_frame"] = self._spin_range_start.value()
            p["end_frame"] = self._spin_range_end.value()
        self.identity_swap_requested.emit(p)

    def _emit_mask_identity_fix_request(self) -> None:
        self.mask_identity_fix_requested.emit({"write_csv": True})

    def _show_swap_button_menu(self, pos) -> None:
        menu = QMenu(self)
        act_swap = menu.addAction("Swap Mouse Identity")
        act_swap.setEnabled(self._btn_swap_ids.isEnabled())
        act_swap.triggered.connect(lambda _checked=False: self._emit_identity_swap_request())
        menu.exec(self._btn_swap_ids.mapToGlobal(pos))

    def _emit_identity_rename_request(self) -> None:
        old_label = self._combo_rename_from.currentText()
        new_label = self._edit_rename_to.text().strip()
        if not old_label or not new_label or old_label == new_label:
            return
        self.identity_rename_requested.emit(
            {"old_label": old_label, "new_label": new_label, "write_csv": True}
        )

    # --- Calibration helpers ---

    def set_video_path(self, path: str) -> None:
        """Set the video path used by the calibration dialog."""
        self._video_path = path

    def video_path(self) -> str:
        """Return the video path used by the calibration dialog."""
        return self._video_path

    def px_per_cm(self) -> float:
        """Return the current calibration scale (0.0 if not calibrated)."""
        return self._px_per_cm

    def active_px_per_cm(self) -> float:
        """Return the scale currently applied to analyses."""
        if self._chk_apply_cal.isChecked() and self._px_per_cm > 0:
            return self._px_per_cm
        return 0.0

    def set_calibration(self, px_per_cm: float, use_for_analysis: bool = True) -> None:
        """Restore/update calibration UI state without reopening the dialog."""
        self._px_per_cm = float(max(px_per_cm, 0.0))
        if self._px_per_cm > 0:
            self._lbl_scale.setText(f"Scale: {self._px_per_cm:.2f} px/cm")
            self._chk_apply_cal.setEnabled(True)
            self._chk_apply_cal.blockSignals(True)
            self._chk_apply_cal.setChecked(use_for_analysis)
            self._chk_apply_cal.blockSignals(False)
        else:
            self._lbl_scale.setText("Not calibrated")
            self._chk_apply_cal.blockSignals(True)
            self._chk_apply_cal.setChecked(False)
            self._chk_apply_cal.blockSignals(False)
            self._chk_apply_cal.setEnabled(False)

    def _open_calibration(self) -> None:
        if not self._video_path:
            logger.warning("No video path set — cannot open calibration dialog.")
            return
        dlg = CalibrationDialog(self._video_path, parent=self)
        dlg.calibration_done.connect(self._on_calibration_done)
        dlg.exec()

    def _on_calibration_done(self, px_per_cm: float) -> None:
        self.set_calibration(px_per_cm, use_for_analysis=True)
        self.calibration_changed.emit(self.active_px_per_cm())

    def _emit_active_calibration_changed(self, _checked: bool) -> None:
        self.calibration_changed.emit(self.active_px_per_cm())

    # --- Range helpers ---

    def _set_range_controls_enabled(self, checked: bool) -> None:
        self._spin_range_start.setEnabled(checked)
        self._spin_range_end.setEnabled(checked)
        self._btn_select_on_plot.setEnabled(checked)
        self._btn_select_all.setEnabled(self._n_frames > 0)

    def _on_range_toggled(self, checked: bool) -> None:
        self._set_range_controls_enabled(checked)
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

    def set_range(self, start: int, end: int, enable: Optional[bool] = None) -> None:
        """Update range spinboxes from plot selection or flag actions."""
        if enable is not None:
            self._chk_use_range.blockSignals(True)
            self._chk_use_range.setChecked(bool(enable))
            self._chk_use_range.blockSignals(False)
            self._set_range_controls_enabled(bool(enable))
        self._spin_range_start.blockSignals(True)
        self._spin_range_end.blockSignals(True)
        self._spin_range_start.setValue(start)
        self._spin_range_end.setValue(end)
        self._spin_range_start.blockSignals(False)
        self._spin_range_end.blockSignals(False)

    def flag_range_start(self, frame: int) -> None:
        """Set the selected-range start border from a timeline context menu."""
        start = self._clamp_start_frame(frame)
        end = self._spin_range_end.value() if self._chk_use_range.isChecked() else start + 1
        if end <= start:
            end = start + 1
        end = self._clamp_end_frame(end)
        self._apply_flagged_range(start, end)

    def flag_range_end(self, frame: int) -> None:
        """Set the selected-range end border from a timeline context menu."""
        end = self._clamp_end_frame(int(frame) + 1)
        start = self._spin_range_start.value() if self._chk_use_range.isChecked() else max(0, end - 1)
        if start >= end:
            start = max(0, end - 1)
        start = self._clamp_start_frame(start)
        self._apply_flagged_range(start, end)

    def clear_range(self) -> None:
        """Disable the selected range and hide the plot selection."""
        end = min(self._n_frames, 1000) if self._n_frames > 0 else 0
        self.set_range(0, end, enable=False)
        self.range_changed.emit(0, 0)

    def request_identity_swap(self) -> None:
        """Run the normal identity swap action from an external context menu."""
        self._emit_identity_swap_request()

    def _apply_flagged_range(self, start: int, end: int) -> None:
        self.set_range(start, end, enable=True)
        self.range_selection_requested.emit()
        self.range_changed.emit(start, end)

    def _clamp_start_frame(self, frame: int) -> int:
        frame = max(0, int(frame))
        if self._n_frames > 0:
            frame = min(frame, self._n_frames - 1)
        return frame

    def _clamp_end_frame(self, frame: int) -> int:
        frame = max(0, int(frame))
        if self._n_frames > 0:
            frame = min(frame, self._n_frames)
        return frame

    def set_n_frames(self, n: int) -> None:
        """Update spinbox maximums when data changes."""
        self._n_frames = max(0, int(n))
        self._spin_range_start.setMaximum(max(0, n - 1))
        self._spin_range_end.setMaximum(n)
        self._btn_select_all.setEnabled(self._n_frames > 0)

    def _select_all_range(self) -> None:
        """Set manual range controls to the full active recording."""
        if self._n_frames <= 0:
            return
        if not self._chk_use_range.isChecked():
            self._chk_use_range.setChecked(True)
        self.set_range(0, self._n_frames)
        self.range_changed.emit(0, self._n_frames)

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
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Missing fields", "Both name and sources are required.")
            return
        sources = [s.strip() for s in sources_text.split(",") if s.strip()]
        if operation == "midpoint" and len(sources) != 2:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, "Invalid sources",
                f"Midpoint requires exactly 2 sources, but got {len(sources)}.",
            )
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
        animals = list(dfs.keys())
        current_a = self._combo_swap_a.currentText()
        current_b = self._combo_swap_b.currentText()
        current_rename = self._combo_rename_from.currentText()
        for combo, current in (
            (self._combo_swap_a, current_a),
            (self._combo_swap_b, current_b),
            (self._combo_rename_from, current_rename),
        ):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(animals)
            if current in animals:
                combo.setCurrentText(current)
            combo.blockSignals(False)
        if len(animals) >= 2 and self._combo_swap_a.currentText() == self._combo_swap_b.currentText():
            self._combo_swap_b.setCurrentIndex(1)
        self._btn_swap_ids.setEnabled(len(animals) >= 2)
        self._btn_fix_mask_ids.setEnabled(len(animals) >= 1)
        self._btn_rename_id.setEnabled(len(animals) >= 1)

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
            written_path = save_cleaned_h5(self._animal_dfs, path)
            logger.info("Saved cleaned data to %s", written_path)
        except Exception:
            logger.exception("Failed to save cleaned H5")
