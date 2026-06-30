"""Social behaviour detection panel."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from shared.ui_kit import COLORS, Card, hint

# Behaviour colors — matches Gantt chart palette in plot_panel
logger = logging.getLogger(__name__)

# Form alignment + a fixed, calm field width so numeric inputs sit in a tidy
# right-aligned column instead of stretching across the whole card.
_FORM_LABEL_ALIGN = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
_FORM_ALIGN = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
_FIELD_WIDTH = 110


def _compact_field(widget: QWidget) -> None:
    """Constrain a numeric input to a fixed, premium-looking column width."""
    widget.setFixedWidth(_FIELD_WIDTH)
    widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

_BEHAVIOR_COLORS = [
    (243, 139, 168),
    (137, 180, 250),
    (166, 227, 161),
    (250, 179, 135),
    (203, 166, 247),
    (249, 226, 175),
    (148, 226, 213),
    (245, 194, 231),
    (116, 199, 236),
]

_CONTINUOUS = {
    "inter_animal_dist_px",
    "inter_animal_dist_cm",
    "approach_speed_px_s",
    "approach_speed_cm_s",
    "relative_heading_deg",
}


class SocialPanel(QGroupBox):
    """Controls and results for vectorised social behaviour detection."""

    detected = Signal(dict)
    mask_contact_calibration_requested = Signal(float)
    batch_process_requested = Signal(dict)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Social Behaviours", parent)
        self._animal_dfs: dict = {}
        self._all_behavior_arrays: dict = {}
        self._behavior_arrays: dict = {}
        self._behavior_checks: dict[str, QCheckBox] = {}
        self._last_detection_params: dict = {}
        self._fps = 25.0
        self._px_per_cm = 0.0
        self._mask_store = None
        self._mask_contact_cache: dict[tuple, np.ndarray] = {}
        self._detecting = False
        self._batch_processing = False
        self._project_batch_count = 0
        self._distance_unit_mode = "px"
        self._last_summary_context: tuple[int, str, str] | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # ── Animal pair ───────────────────────────────────────────────────
        pair_card = Card("Animal pair", "Subject A vs partner B", accent=COLORS["teal"])

        pair_row = QHBoxLayout()
        pair_row.setSpacing(8)
        lbl_a = QLabel("A")
        lbl_a.setStyleSheet(f"color: {COLORS['text_muted']}; font-weight: 700;")
        pair_row.addWidget(lbl_a)
        self._combo_a = QComboBox()
        self._combo_a.currentIndexChanged.connect(self._update_ui_state)
        self._combo_a.setToolTip("Subject animal — directional behaviours (e.g. 'A follows B') are relative to this animal")
        pair_row.addWidget(self._combo_a, 1)
        lbl_vs = QLabel("vs B")
        lbl_vs.setStyleSheet(f"color: {COLORS['text_muted']}; font-weight: 700;")
        pair_row.addWidget(lbl_vs)
        self._combo_b = QComboBox()
        self._combo_b.currentIndexChanged.connect(self._update_ui_state)
        self._combo_b.setToolTip("Partner animal — the target of directional behaviours")
        pair_row.addWidget(self._combo_b, 1)
        self._btn_swap = QPushButton("Swap")
        self._btn_swap.setObjectName("secondary")
        self._btn_swap.setToolTip("Swap Animal A and Animal B")
        self._btn_swap.clicked.connect(self._swap_animals)
        pair_row.addWidget(self._btn_swap)
        pair_card.body.addLayout(pair_row)
        layout.addWidget(pair_card)

        # ── Detection thresholds ──────────────────────────────────────────
        thresh_card = Card("Detection thresholds", "Distance and timing tolerances")

        form = QFormLayout()
        form.setSpacing(8)
        form.setHorizontalSpacing(12)
        form.setLabelAlignment(_FORM_LABEL_ALIGN)
        form.setFormAlignment(_FORM_ALIGN)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)

        self._spin_close = QDoubleSpinBox()
        self._spin_close.setRange(0, 500)
        self._spin_close.setValue(25)
        self._spin_close.setSuffix(" px")
        self._spin_close.setToolTip("Max distance for nose-to-nose, nose-to-anogenital, and body contact events (set to 0 for adaptive)")
        _compact_field(self._spin_close)
        form.addRow("Close contact tol", self._spin_close)

        self._spin_side = QDoubleSpinBox()
        self._spin_side.setRange(1, 500)
        self._spin_side.setValue(50)
        self._spin_side.setSuffix(" px")
        self._spin_side.setToolTip("Max distance for side-by-side and side-reverse detections")
        _compact_field(self._spin_side)
        form.addRow("Side contact tol", self._spin_side)

        self._spin_follow = QDoubleSpinBox()
        self._spin_follow.setRange(1, 500)
        self._spin_follow.setValue(30)
        self._spin_follow.setSuffix(" px")
        self._spin_follow.setToolTip("Max distance between follower's nose and leader's tail for following detection")
        _compact_field(self._spin_follow)
        form.addRow("Follow tol", self._spin_follow)

        self._spin_fwin = QSpinBox()
        self._spin_fwin.setRange(1, 120)
        self._spin_fwin.setValue(12)
        self._spin_fwin.setSuffix(" frames")
        self._spin_fwin.setToolTip("Sliding window size for following detection — longer windows require more sustained following")
        _compact_field(self._spin_fwin)
        form.addRow("Follow window", self._spin_fwin)

        self._spin_median = QSpinBox()
        self._spin_median.setRange(1, 30)
        self._spin_median.setValue(6)
        self._spin_median.setSuffix(" frames")
        self._spin_median.setToolTip("Temporal median filter to remove brief false detections (higher = more smoothing)")
        _compact_field(self._spin_median)
        form.addRow("Median filter", self._spin_median)

        self._spin_likelihood = QDoubleSpinBox()
        self._spin_likelihood.setRange(0.0, 1.0)
        self._spin_likelihood.setSingleStep(0.05)
        self._spin_likelihood.setDecimals(2)
        self._spin_likelihood.setValue(0.20)
        self._spin_likelihood.setToolTip("Ignore keypoints below this DLC likelihood when likelihood columns are present")
        _compact_field(self._spin_likelihood)
        form.addRow("Min likelihood", self._spin_likelihood)

        thresh_card.body.addLayout(form)

        self._chk_use_masks = QCheckBox("Use masks for contact")
        self._chk_use_masks.setChecked(False)
        self._chk_use_masks.setEnabled(False)
        self._chk_use_masks.setToolTip("Use segmentation masks to confirm physical contact; slower on long videos")
        thresh_card.body.addWidget(self._chk_use_masks)

        mask_form = QFormLayout()
        mask_form.setSpacing(8)
        mask_form.setHorizontalSpacing(12)
        mask_form.setLabelAlignment(_FORM_LABEL_ALIGN)
        mask_form.setFormAlignment(_FORM_ALIGN)
        mask_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)

        mask_row = QHBoxLayout()
        mask_row.setSpacing(8)
        self._spin_mask_margin = QDoubleSpinBox()
        self._spin_mask_margin.setRange(0.0, 30.0)
        self._spin_mask_margin.setDecimals(1)
        self._spin_mask_margin.setSingleStep(0.5)
        self._spin_mask_margin.setValue(5.0)
        self._spin_mask_margin.setSuffix(" %")
        self._spin_mask_margin.setToolTip("Mask edge contact margin as a percent of the selected mask size")
        self._spin_mask_margin.setEnabled(False)
        _compact_field(self._spin_mask_margin)
        self._btn_mask_calibrate = QPushButton("Calibrate")
        self._btn_mask_calibrate.setObjectName("secondary")
        self._btn_mask_calibrate.setEnabled(False)
        self._btn_mask_calibrate.setToolTip("Preview mask outlines and tune the contact margin")
        self._btn_mask_calibrate.clicked.connect(
            lambda: self.mask_contact_calibration_requested.emit(float(self._spin_mask_margin.value()))
        )
        mask_row.addWidget(self._spin_mask_margin)
        mask_row.addWidget(self._btn_mask_calibrate)
        mask_row.addStretch()
        mask_form.addRow("Mask margin", mask_row)
        thresh_card.body.addLayout(mask_form)
        layout.addWidget(thresh_card)

        # ── Primary actions ───────────────────────────────────────────────
        actions_card = Card("Run detection", accent=COLORS["accent"])

        self._btn_detect = QPushButton("Detect Behaviours")
        self._btn_detect.setToolTip("Run social behaviour detection on the selected animal pair")
        self._btn_detect.clicked.connect(self._detect)
        actions_card.body.addWidget(self._btn_detect)

        self._btn_process_all = QPushButton("Process All Loaded Videos")
        self._btn_process_all.setObjectName("secondary")
        self._btn_process_all.setToolTip(
            "Compute framewise behavior metrics and social behaviours for every loaded recording row"
        )
        self._btn_process_all.clicked.connect(self._request_batch_process)
        actions_card.body.addWidget(self._btn_process_all)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setVisible(False)
        actions_card.body.addWidget(self._progress)

        self._lbl_status = QLabel("")
        self._lbl_status.setWordWrap(True)
        self._lbl_status.setObjectName("hint")
        actions_card.body.addWidget(self._lbl_status)
        layout.addWidget(actions_card)

        # ── Visible behaviours ────────────────────────────────────────────
        visible_group = Card("Visible behaviours", "Toggle which detections appear in the Gantt view")

        vis_btn_row = QHBoxLayout()
        vis_btn_row.setSpacing(8)
        btn_all = QPushButton("All")
        btn_all.setObjectName("secondary")
        btn_all.clicked.connect(lambda: self._set_all_behavior_checks(True))
        btn_none = QPushButton("None")
        btn_none.setObjectName("secondary")
        btn_none.clicked.connect(lambda: self._set_all_behavior_checks(False))
        vis_btn_row.addWidget(btn_all)
        vis_btn_row.addWidget(btn_none)
        vis_btn_row.addStretch()
        visible_group.body.addLayout(vis_btn_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(140)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; }")
        container = QWidget()
        container.setStyleSheet("background: transparent;")
        self._behavior_checks_layout = QVBoxLayout(container)
        self._behavior_checks_layout.setContentsMargins(0, 0, 0, 0)
        self._behavior_checks_layout.setSpacing(4)
        self._behavior_checks_layout.addStretch()
        scroll.setWidget(container)
        visible_group.body.addWidget(scroll)
        layout.addWidget(visible_group)

        # ── Results ───────────────────────────────────────────────────────
        results_card = Card("Results", "Per-behaviour frame counts and metrics")

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Behaviour", "Pair", "Frames", "Time %"])
        self._table.setMinimumHeight(180)
        self._table.setMaximumHeight(320)
        self._table.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.verticalHeader().setVisible(False)
        results_card.body.addWidget(self._table)
        results_card.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        layout.addWidget(results_card, 1)

        self._update_ui_state()

    def set_animal_dfs(self, dfs: dict, fps: float = 25.0) -> None:
        self._animal_dfs = dfs
        self._fps = fps
        self._all_behavior_arrays = {}
        self._behavior_arrays = {}
        self._last_detection_params = {}
        self._last_detection_cache_signature = None
        self._last_summary_context = None
        self._mask_contact_cache.clear()
        self._table.setRowCount(0)
        self._clear_behavior_checks()

        animals = list(dfs.keys())
        self._combo_a.clear()
        self._combo_b.clear()
        for animal in animals:
            self._combo_a.addItem(animal)
            self._combo_b.addItem(animal)
        if len(animals) >= 2:
            self._combo_b.setCurrentIndex(1)
        self._update_ui_state()

    def set_calibration(self, px_per_cm: float) -> None:
        new_scale = float(max(px_per_cm, 0.0))
        old_scale = self._px_per_cm
        if np.isclose(new_scale, old_scale):
            self._px_per_cm = new_scale
            self._update_ui_state()
            return

        if new_scale > 0 and self._distance_unit_mode == "px":
            self._convert_distance_spinboxes(1.0 / new_scale)
            self._distance_unit_mode = "cm"
        elif new_scale <= 0 and old_scale > 0 and self._distance_unit_mode == "cm":
            self._convert_distance_spinboxes(old_scale)
            self._distance_unit_mode = "px"

        self._px_per_cm = new_scale
        suffix = " cm" if self._distance_unit_mode == "cm" else " px"
        for spin in (self._spin_close, self._spin_side, self._spin_follow):
            spin.setSuffix(suffix)
        self._update_ui_state()

    def set_mask_store(self, mask_store) -> None:
        """Set optional segmentation masks used to confirm physical contact."""
        self._mask_store = mask_store
        self._mask_contact_cache.clear()
        has_masks = mask_store is not None
        self._chk_use_masks.setEnabled(has_masks)
        self._spin_mask_margin.setEnabled(has_masks)
        self._btn_mask_calibrate.setEnabled(has_masks)
        if not has_masks:
            self._chk_use_masks.setChecked(False)
        self._update_ui_state()

    def set_project_batch_count(self, count: int) -> None:
        """Update how many loaded recording rows can be batch processed."""
        self._project_batch_count = max(0, int(count))
        self._update_ui_state()

    def behavior_arrays(self) -> dict:
        return self._behavior_arrays

    def all_behavior_arrays(self) -> dict:
        return self._all_behavior_arrays

    def last_detection_params(self) -> dict:
        return dict(self._last_detection_params)

    def current_detection_params(self) -> dict:
        return {
            "animal_a": self._combo_a.currentText(),
            "animal_b": self._combo_b.currentText(),
            "fps": float(self._fps),
            "close_tol_px": float(self._distance_to_px(self._spin_close.value())),
            "side_tol_px": float(self._distance_to_px(self._spin_side.value())),
            "follow_tol_px": float(self._distance_to_px(self._spin_follow.value())),
            "follow_window": int(self._spin_fwin.value()),
            "median_filter": int(self._spin_median.value()),
            "likelihood_threshold": float(self._spin_likelihood.value()),
            "px_per_cm": float(self._px_per_cm),
            "has_masks": self._mask_store is not None,
            "use_masks": bool(self._mask_store is not None and self._chk_use_masks.isChecked()),
            "mask_edge_margin_percent": float(self._spin_mask_margin.value()),
        }

    def current_pair(self) -> tuple[str, str]:
        return self._combo_a.currentText(), self._combo_b.currentText()

    def mask_contact_margin_percent(self) -> float:
        return float(self._spin_mask_margin.value())

    def set_mask_contact_margin_percent(self, value: float) -> None:
        new_value = max(self._spin_mask_margin.minimum(), min(self._spin_mask_margin.maximum(), float(value)))
        if not np.isclose(float(self._spin_mask_margin.value()), new_value):
            self._mask_contact_cache.clear()
        self._spin_mask_margin.setValue(new_value)

    def set_batch_process_running(self, running: bool) -> None:
        self._batch_processing = bool(running)
        self._progress.setVisible(running)
        if running:
            self._progress.setValue(0)
        self._update_ui_state()

    def set_batch_process_progress(self, pct: int, text: str) -> None:
        self._progress.setVisible(True)
        self._progress.setValue(max(0, min(100, int(pct))))
        self._lbl_status.setText(text)
        QApplication.processEvents()

    def set_batch_process_result(self, text: str) -> None:
        self._batch_processing = False
        self._progress.setValue(100)
        self._progress.setVisible(False)
        self._lbl_status.setText(text)
        self._update_ui_state(preserve_status=True)

    def restore_behavior_arrays(
        self,
        arrays: dict,
        *,
        params: Optional[dict] = None,
        visible_names: Optional[list[str]] = None,
        emit: bool = False,
    ) -> None:
        """Restore cached social arrays without recomputing them."""
        self._apply_detection_params(params or {})
        self._all_behavior_arrays = dict(arrays or {})
        self._last_detection_params = dict(params or self.current_detection_params())
        self._last_detection_cache_signature = self._detection_cache_signature() if self._animal_dfs else None
        if self._animal_dfs:
            aid_a = self._combo_a.currentText()
            aid_b = self._combo_b.currentText()
            self._last_summary_context = (len(next(iter(self._animal_dfs.values()))), aid_a, aid_b)
        else:
            self._last_summary_context = None
        self._rebuild_behavior_checks()
        if visible_names is not None:
            visible = set(visible_names)
            for name, chk in self._behavior_checks.items():
                chk.blockSignals(True)
                chk.setChecked(name in visible)
                chk.blockSignals(False)
        self._apply_visibility_filters(emit=emit)

    def _swap_animals(self) -> None:
        a_idx = self._combo_a.currentIndex()
        b_idx = self._combo_b.currentIndex()
        if a_idx < 0 or b_idx < 0:
            return
        self._combo_a.setCurrentIndex(b_idx)
        self._combo_b.setCurrentIndex(a_idx)
        self._update_ui_state()

    def _update_ui_state(self, *args, preserve_status: bool = False) -> None:
        animals = list(self._animal_dfs)
        has_pair = len(animals) >= 2
        aid_a = self._combo_a.currentText()
        aid_b = self._combo_b.currentText()
        distinct_pair = bool(has_pair and aid_a and aid_b and aid_a != aid_b)
        busy = self._detecting or self._batch_processing
        self._btn_detect.setEnabled(distinct_pair and not busy)
        self._btn_swap.setEnabled(has_pair and not busy)
        self._btn_process_all.setEnabled(self._project_batch_count > 0 and not busy)
        if busy:
            return
        if preserve_status:
            return

        if not has_pair:
            if self._project_batch_count > 0:
                self._lbl_status.setText(
                    f"{self._project_batch_count} loaded recording row(s) ready for all-video metrics + social processing."
                )
                return
            self._lbl_status.setText("Load at least 2 animals to run social behaviour detection.")
            return

        unit_hint = "Thresholds are displayed in cm." if self._distance_unit_mode == "cm" else "Thresholds are displayed in px."
        if not distinct_pair:
            self._lbl_status.setText(f"Choose 2 different animals to detect behaviours. {unit_hint}")
            return

        visible_count = sum(
            1
            for name, arr in self._behavior_arrays.items()
            if name not in _CONTINUOUS and _is_boolean_like(arr)
        )
        if self._behavior_arrays:
            self._lbl_status.setText(
                f"Ready: {aid_a} vs {aid_b}. Showing {visible_count} visible boolean behaviours plus continuous social metrics. {unit_hint}"
            )
        else:
            self._lbl_status.setText(
                f"Ready: {aid_a} vs {aid_b}. Run detection to populate behaviours and the Gantt view. {unit_hint}"
            )

    def _request_batch_process(self) -> None:
        params = self.current_detection_params()
        self.batch_process_requested.emit({
            "compute_social": True,
            "use_masks_for_social": bool(params.get("use_masks", False)),
            "social_params": params,
        })

    def _convert_distance_spinboxes(self, factor: float) -> None:
        for spin in (self._spin_close, self._spin_side, self._spin_follow):
            spin.blockSignals(True)
            spin.setValue(max(spin.minimum(), spin.value() * factor))
            spin.blockSignals(False)

    def _distance_to_px(self, value: float) -> float:
        if self._distance_unit_mode == "cm" and self._px_per_cm > 0:
            return value * self._px_per_cm
        return value

    def _distance_from_px(self, value: float) -> float:
        if self._distance_unit_mode == "cm" and self._px_per_cm > 0:
            return float(value) / self._px_per_cm
        return float(value)

    def _apply_detection_params(self, params: dict) -> None:
        if not params:
            return
        for combo, key in ((self._combo_a, "animal_a"), (self._combo_b, "animal_b")):
            value = str(params.get(key, ""))
            idx = combo.findText(value)
            if idx >= 0:
                combo.blockSignals(True)
                combo.setCurrentIndex(idx)
                combo.blockSignals(False)

        for spin, key in (
            (self._spin_close, "close_tol_px"),
            (self._spin_side, "side_tol_px"),
            (self._spin_follow, "follow_tol_px"),
        ):
            if key in params:
                spin.blockSignals(True)
                spin.setValue(max(spin.minimum(), self._distance_from_px(float(params.get(key, spin.value())))))
                spin.blockSignals(False)

        int_spins = (
            (self._spin_fwin, "follow_window"),
            (self._spin_median, "median_filter"),
        )
        for spin, key in int_spins:
            if key in params:
                spin.blockSignals(True)
                spin.setValue(int(params.get(key, spin.value())))
                spin.blockSignals(False)

        if "likelihood_threshold" in params:
            self._spin_likelihood.blockSignals(True)
            self._spin_likelihood.setValue(float(params.get("likelihood_threshold", self._spin_likelihood.value())))
            self._spin_likelihood.blockSignals(False)
        if "use_masks" in params:
            self._chk_use_masks.blockSignals(True)
            self._chk_use_masks.setChecked(bool(params.get("use_masks")) and self._mask_store is not None)
            self._chk_use_masks.blockSignals(False)
        if "mask_edge_margin_percent" in params:
            self._spin_mask_margin.blockSignals(True)
            self._spin_mask_margin.setValue(float(params.get("mask_edge_margin_percent", self._spin_mask_margin.value())))
            self._spin_mask_margin.blockSignals(False)
        self._update_ui_state()

    def _detect(self) -> None:
        if len(self._animal_dfs) < 2:
            return

        from dlc_processor.core.social_behaviors import SocialBehaviors

        aid_a = self._combo_a.currentText()
        aid_b = self._combo_b.currentText()
        if aid_a == aid_b or aid_a not in self._animal_dfs or aid_b not in self._animal_dfs:
            return

        params = self.current_detection_params()
        cache_signature = self._detection_cache_signature()
        if (
            self._all_behavior_arrays
            and self._last_detection_params == params
            and getattr(self, "_last_detection_cache_signature", None) == cache_signature
        ):
            self._rebuild_behavior_checks()
            self._apply_visibility_filters(emit=True)
            self._lbl_status.setText("Reused cached social behaviour detection.")
            return

        self._set_detection_running(True)
        try:
            self._set_detection_progress(0, f"Preparing {aid_a} vs {aid_b}")
            sb = SocialBehaviors(
                df_a=self._animal_dfs[aid_a],
                df_b=self._animal_dfs[aid_b],
                fps=self._fps,
                close_tol=params["close_tol_px"],
                side_tol=params["side_tol_px"],
                follow_tol=params["follow_tol_px"],
                follow_window=params["follow_window"],
                median_filter=params["median_filter"],
                likelihood_threshold=params["likelihood_threshold"],
                mask_contact=None,
            )

            if self._mask_store is not None and self._chk_use_masks.isChecked():
                mask_contact = self._cached_or_compute_mask_contact(sb, aid_a, aid_b)
                sb.set_mask_contact(mask_contact)
                start_pct, span_pct = 42, 56
            else:
                start_pct, span_pct = 2, 96

            def _behaviour_progress(pct: int, msg: str) -> None:
                mapped = start_pct + int((max(0, min(100, int(pct))) / 100.0) * span_pct)
                self._set_detection_progress(mapped, msg)

            self._all_behavior_arrays = sb.compute_all(progress_callback=_behaviour_progress)
            self._set_detection_progress(99, "Updating social behaviour display")
            if self._px_per_cm > 0:
                if "inter_animal_dist_px" in self._all_behavior_arrays:
                    self._all_behavior_arrays["inter_animal_dist_cm"] = (
                        np.asarray(self._all_behavior_arrays["inter_animal_dist_px"], dtype=np.float64)
                        / self._px_per_cm
                    )
                if "approach_speed_px_s" in self._all_behavior_arrays:
                    self._all_behavior_arrays["approach_speed_cm_s"] = (
                        np.asarray(self._all_behavior_arrays["approach_speed_px_s"], dtype=np.float64)
                        / self._px_per_cm
                    )
            self._last_detection_cache_signature = cache_signature
            self._last_detection_params = params
            self._last_summary_context = (len(next(iter(self._animal_dfs.values()))), aid_a, aid_b)
            self._rebuild_behavior_checks()
            self._apply_visibility_filters(emit=True)
            self._set_detection_progress(100, "Social behaviour detection complete")
        except Exception as exc:
            logger.exception("Social behaviour detection failed")
            self._set_detection_running(False)
            self._lbl_status.setText(f"Social detection failed: {exc}")
            return
        self._set_detection_running(False)

    def _cached_or_compute_mask_contact(self, sb, aid_a: str, aid_b: str) -> np.ndarray:
        from dlc_processor.core.mask_social import pair_mask_contact

        key = self._mask_contact_cache_key(aid_a, aid_b)
        cached = self._mask_contact_cache.get(key)
        if cached is not None:
            self._set_detection_progress(40, "Reusing cached mask contact")
            return cached

        self._set_detection_progress(4, "Selecting plausible mask-contact frames")
        candidates = sb.mask_contact_candidate_frames()
        n_candidates = int(np.count_nonzero(candidates))
        n_total = int(len(candidates))
        self._set_detection_progress(6, f"Mask contact candidates: {n_candidates}/{n_total} frames")

        def _mask_progress(pct: int, msg: str) -> None:
            mapped = 8 + int((max(0, min(100, int(pct))) / 100.0) * 32)
            self._set_detection_progress(mapped, msg)

        mask_contact = pair_mask_contact(
            self._mask_store,
            self._animal_dfs[aid_a],
            self._animal_dfs[aid_b],
            max_edge_gap_px=0,
            max_edge_gap_percent=float(self._spin_mask_margin.value()),
            candidate_frames=candidates,
            exact_masks=False,
            progress_callback=_mask_progress,
        )
        self._mask_contact_cache[key] = np.asarray(mask_contact, dtype=bool).copy()
        return mask_contact

    def _mask_contact_cache_key(self, aid_a: str, aid_b: str) -> tuple:
        n_frames = max((len(df) for df in self._animal_dfs.values()), default=0)
        return (
            str(aid_a),
            str(aid_b),
            int(n_frames),
            round(float(self._spin_mask_margin.value()), 3),
            id(self._mask_store),
        )

    def _detection_cache_signature(self) -> tuple:
        first_df = next(iter(self._animal_dfs.values()), None)
        frames = getattr(first_df, "attrs", {}).get("frame_numbers") if first_df is not None else None
        frame_sig = ()
        if frames is not None and len(frames):
            arr = np.asarray(frames)
            frame_sig = (int(arr[0]), int(arr[-1]), int(len(arr)))
        return (self._mask_contact_cache_key(self._combo_a.currentText(), self._combo_b.currentText()), frame_sig)

    def _set_detection_running(self, running: bool) -> None:
        self._detecting = bool(running)
        self._progress.setVisible(running)
        if running:
            self._progress.setValue(0)
        self._update_ui_state()

    def _set_detection_progress(self, pct: int, text: str) -> None:
        self._progress.setVisible(True)
        self._progress.setValue(max(0, min(100, int(pct))))
        self._lbl_status.setText(text)
        QApplication.processEvents()

    def _rebuild_behavior_checks(self) -> None:
        previous = {name: chk.isChecked() for name, chk in self._behavior_checks.items()}
        self._clear_behavior_checks()

        color_idx = 0
        for name, arr in self._all_behavior_arrays.items():
            if name in _CONTINUOUS or not _is_boolean_like(arr):
                continue
            r, g, b = _BEHAVIOR_COLORS[color_idx % len(_BEHAVIOR_COLORS)]
            chk = QCheckBox(_display_behavior_name(name))
            chk.setToolTip(name)
            chk.setChecked(previous.get(name, True))
            chk.setStyleSheet(
                f"QCheckBox {{ color: rgb({r},{g},{b}); font-weight: bold; }}"
            )
            chk.toggled.connect(self._on_visibility_changed)
            self._behavior_checks[name] = chk
            self._behavior_checks_layout.insertWidget(self._behavior_checks_layout.count() - 1, chk)
            color_idx += 1

    def _clear_behavior_checks(self) -> None:
        for chk in self._behavior_checks.values():
            chk.deleteLater()
        self._behavior_checks.clear()

    def _set_all_behavior_checks(self, checked: bool) -> None:
        if not self._behavior_checks:
            return
        for chk in self._behavior_checks.values():
            chk.blockSignals(True)
            chk.setChecked(checked)
            chk.blockSignals(False)
        self._apply_visibility_filters(emit=True)

    def _on_visibility_changed(self) -> None:
        self._apply_visibility_filters(emit=True)

    def _apply_visibility_filters(self, emit: bool) -> None:
        visible = {}
        for name, arr in self._all_behavior_arrays.items():
            if self._px_per_cm > 0 and name == "inter_animal_dist_px" and "inter_animal_dist_cm" in self._all_behavior_arrays:
                continue
            if self._px_per_cm > 0 and name == "approach_speed_px_s" and "approach_speed_cm_s" in self._all_behavior_arrays:
                continue
            if name in _CONTINUOUS or not _is_boolean_like(arr):
                visible[name] = arr
                continue
            chk = self._behavior_checks.get(name)
            if chk is None or chk.isChecked():
                visible[name] = arr

        self._behavior_arrays = visible
        if self._last_summary_context is not None:
            self._populate_table(*self._last_summary_context)
        self._update_ui_state()
        if emit:
            self.detected.emit(self._behavior_arrays)

    def _populate_table(self, n_frames: int, aid_a: str, aid_b: str) -> None:
        pair_ab = f"{aid_a} - {aid_b}"
        pair_a_b = f"{aid_a} to {aid_b}"
        pair_b_a = f"{aid_b} to {aid_a}"
        pair_labels = {
            "nose2nose": pair_ab,
            "mask_contact": pair_ab,
            "sidebyside": pair_ab,
            "sidereside": pair_ab,
            "fighting": pair_ab,
            "attacks": pair_ab,
            "a_nose2anogenital_b": pair_a_b,
            "b_nose2anogenital_a": pair_b_a,
            "a_nose2body_b": pair_a_b,
            "b_nose2body_a": pair_b_a,
            "a_following_b": pair_a_b,
            "b_following_a": pair_b_a,
            "a_chasing_b": pair_a_b,
            "b_chasing_a": pair_b_a,
            "a_approaches_b": pair_a_b,
            "b_approaches_a": pair_b_a,
            "a_withdraws_from_b": pair_a_b,
            "b_withdraws_from_a": pair_b_a,
            "a_escapes_b": pair_a_b,
            "b_escapes_a": pair_b_a,
            "a_withdrawal_after_contact_b": pair_a_b,
            "b_withdrawal_after_contact_a": pair_b_a,
            "a_oriented_toward_b": pair_a_b,
            "b_oriented_toward_a": pair_b_a,
            "passive_anogenital": pair_b_a,
            "passive_investigation": pair_b_a,
            "passive_being_followed": pair_b_a,
            "passive_being_chased": pair_b_a,
            "passive_withdrawal": pair_b_a,
            "inter_animal_dist_px": pair_ab,
            "inter_animal_dist_cm": pair_ab,
            "approach_speed_px_s": pair_ab,
            "approach_speed_cm_s": pair_ab,
            "relative_heading_deg": pair_ab,
        }

        rows = []       # (label, pair, val, detail, color_or_None)
        bool_idx = 0
        for name, arr in self._behavior_arrays.items():
            pair = pair_labels.get(name, "")
            label = _display_behavior_name(name)
            if name in _CONTINUOUS:
                valid = arr[~np.isnan(arr)] if np.issubdtype(arr.dtype, np.floating) else arr
                if len(valid):
                    rows.append((label, pair, f"{np.nanmean(arr):.1f}", f"±{np.nanstd(arr):.1f}", None))
                else:
                    rows.append((label, pair, "N/A", "N/A", None))
            else:
                frames_in = int(np.nansum(arr))
                pct = 100.0 * frames_in / max(n_frames, 1)
                color = _BEHAVIOR_COLORS[bool_idx % len(_BEHAVIOR_COLORS)]
                rows.append((label, pair, str(frames_in), f"{pct:.1f}%", color))
                bool_idx += 1

        self._table.setHorizontalHeaderLabels(["Behaviour", "Pair", "Value", "Detail"])
        self._table.setRowCount(len(rows))
        for row_idx, (label, pair, val, detail, color) in enumerate(rows):
            for col_idx, text in enumerate((label, pair, val, detail)):
                item = QTableWidgetItem(text)
                if color is not None and col_idx == 0:
                    from PySide6.QtGui import QColor as _QC
                    item.setForeground(_QC(*color))
                self._table.setItem(row_idx, col_idx, item)


def _display_behavior_name(name: str) -> str:
    pretty = {
        "nose2nose": "nose-to-nose",
        "mask_contact": "mask contact",
        "sidebyside": "side-by-side",
        "sidereside": "side-reverse",
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
        "passive_anogenital": "passive anogenital",
        "passive_investigation": "passive investigation",
        "passive_being_followed": "passive followed",
        "passive_being_chased": "passive chased",
        "passive_withdrawal": "passive withdrawal",
        "inter_animal_dist_px": "inter-animal dist (px)",
        "inter_animal_dist_cm": "inter-animal dist (cm)",
        "approach_speed_px_s": "approach speed (px/s)",
        "approach_speed_cm_s": "approach speed (cm/s)",
        "relative_heading_deg": "relative heading",
        "rearing": "rearing",
    }
    return pretty.get(name, name.replace("_", " "))


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
