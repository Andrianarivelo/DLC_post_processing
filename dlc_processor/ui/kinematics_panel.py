"""Kinematics computation panel."""

from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class KinematicsPanel(QGroupBox):
    """Controls for computing and displaying kinematic metrics."""

    computed = Signal(dict)   # {animal_id: enriched_df}
    heatmap_requested = Signal()  # open egocentric heatmap dialog

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Kinematics", parent)
        self._animal_dfs: dict = {}
        self._result_dfs: dict = {}
        self._custom_time = None
        self._px_per_cm: float = 0.0
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        form = QFormLayout()
        form.setSpacing(5)

        self._spin_fps = QDoubleSpinBox()
        self._spin_fps.setRange(1.0, 1000.0)
        self._spin_fps.setValue(30.0)
        self._spin_fps.setSuffix(" fps")
        self._spin_fps.setToolTip("Video acquisition frame rate used to convert frame-based metrics to time-based units")
        form.addRow("Frame rate:", self._spin_fps)

        self._chk_perbp = QCheckBox("Per-bodypart speed/accel")
        self._chk_perbp.setChecked(False)
        self._chk_perbp.setToolTip("Compute speed and acceleration for each individual bodypart")
        self._chk_body = QCheckBox("Body-centre speed")
        self._chk_body.setChecked(True)
        self._chk_body.setToolTip("Instantaneous speed of the body centre (neck or centroid) in px/s and cm/s")
        self._chk_orient = QCheckBox("Body orientation")
        self._chk_orient.setChecked(True)
        self._chk_orient.setToolTip("Body axis angle (tail to nose) in degrees, plus angular velocity")
        self._chk_accel = QCheckBox("Acceleration + jerk")
        self._chk_accel.setChecked(True)
        self._chk_accel.setToolTip("First and second derivatives of speed (acceleration and jerk)")
        self._chk_distance = QCheckBox("Cumulative distance")
        self._chk_distance.setChecked(True)
        self._chk_distance.setToolTip("Total distance traveled from the start of the recording")
        self._chk_immobile = QCheckBox("Immobile detection")
        self._chk_immobile.setChecked(True)
        self._chk_immobile.setToolTip("Binary immobility state based on speed threshold + minimum duration")
        self._chk_tortuosity = QCheckBox("Path tortuosity")
        self._chk_tortuosity.setChecked(True)
        self._chk_tortuosity.setToolTip("Ratio of path length to straight-line displacement over a sliding window")
        self._chk_elongation = QCheckBox("Body elongation")
        self._chk_elongation.setChecked(True)
        self._chk_elongation.setToolTip("Distance between nose and tail, useful as a stretching/compression proxy")
        self._chk_curvature = QCheckBox("Trajectory curvature")
        self._chk_curvature.setChecked(True)
        self._chk_curvature.setToolTip("How sharply the path curves, where higher values mean tighter turns")
        self._chk_headdir = QCheckBox("Head direction + crab-walk")
        self._chk_headdir.setChecked(True)
        self._chk_headdir.setToolTip("Head direction angle and heading-body angle difference")
        self._chk_rearing = QCheckBox("Rearing detection")
        self._chk_rearing.setChecked(True)
        self._chk_rearing.setToolTip("Detect when the animal stands on hind legs")
        self._chk_partner = QCheckBox("Partner metrics (distance, angle, proximity)")
        self._chk_partner.setChecked(False)
        self._chk_partner.setToolTip("Egocentric distance, angle, and proximity index to the partner animal")

        layout.addLayout(form)
        layout.addWidget(self._chk_perbp)
        layout.addWidget(self._chk_body)
        layout.addWidget(self._chk_orient)
        layout.addWidget(self._chk_accel)
        layout.addWidget(self._chk_distance)
        layout.addWidget(self._chk_immobile)
        layout.addWidget(self._chk_tortuosity)
        layout.addWidget(self._chk_elongation)
        layout.addWidget(self._chk_curvature)
        layout.addWidget(self._chk_headdir)
        layout.addWidget(self._chk_rearing)
        layout.addWidget(self._chk_partner)

        preset_row = QHBoxLayout()
        btn_core = QPushButton("Core Preset")
        btn_core.setObjectName("secondary")
        btn_core.setToolTip("Enable the most common metrics for a quick first pass")
        btn_core.clicked.connect(lambda: self._apply_metric_preset("core"))
        preset_row.addWidget(btn_core)

        btn_all = QPushButton("All Metrics")
        btn_all.setObjectName("secondary")
        btn_all.setToolTip("Enable every available kinematics metric")
        btn_all.clicked.connect(lambda: self._apply_metric_preset("all"))
        preset_row.addWidget(btn_all)

        btn_min = QPushButton("Minimal")
        btn_min.setObjectName("secondary")
        btn_min.setToolTip("Keep only the essential locomotion/orientation metrics enabled")
        btn_min.clicked.connect(lambda: self._apply_metric_preset("minimal"))
        preset_row.addWidget(btn_min)
        preset_row.addStretch()
        layout.addLayout(preset_row)

        btn_row = QHBoxLayout()
        self._btn_compute = QPushButton("Compute Kinematics")
        self._btn_compute.clicked.connect(self._compute)
        btn_row.addWidget(self._btn_compute)

        self._btn_heatmap = QPushButton("Position Map")
        self._btn_heatmap.setToolTip("Egocentric position heatmap for the partner animal")
        self._btn_heatmap.clicked.connect(self.heatmap_requested.emit)
        btn_row.addWidget(self._btn_heatmap)
        layout.addLayout(btn_row)

        self._lbl_status = QLabel("")
        self._lbl_status.setWordWrap(True)
        self._lbl_status.setStyleSheet("color: #a6adc8; font-size: 11px;")
        layout.addWidget(self._lbl_status)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Animal", "Metric", "Mean", "Max"])
        self._table.setMaximumHeight(160)
        self._table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._table)
        self._update_ui_state()

    def fps(self) -> float:
        """Return the current FPS value from the spin box."""
        return self._spin_fps.value()

    def set_animal_dfs(self, dfs: dict) -> None:
        self._animal_dfs = dfs
        self._update_ui_state()

    def set_custom_time(self, times) -> None:
        self._custom_time = None if times is None else np.asarray(times, dtype=np.float64).reshape(-1)
        self._update_ui_state()

    def result_dfs(self) -> dict:
        return self._result_dfs

    def set_calibration(self, px_per_cm: float) -> None:
        self._px_per_cm = float(max(px_per_cm, 0.0))
        self._update_ui_state()

    def _apply_metric_preset(self, preset: str) -> None:
        checks = {
            "perbp": self._chk_perbp,
            "body": self._chk_body,
            "orient": self._chk_orient,
            "accel": self._chk_accel,
            "distance": self._chk_distance,
            "immobile": self._chk_immobile,
            "tortuosity": self._chk_tortuosity,
            "elongation": self._chk_elongation,
            "curvature": self._chk_curvature,
            "headdir": self._chk_headdir,
            "rearing": self._chk_rearing,
            "partner": self._chk_partner,
        }
        presets = {
            "minimal": {
                "body": True,
                "orient": True,
                "distance": True,
                "immobile": True,
            },
            "core": {
                "body": True,
                "orient": True,
                "accel": True,
                "distance": True,
                "immobile": True,
                "tortuosity": True,
                "elongation": True,
                "headdir": True,
            },
            "all": {name: True for name in checks},
        }
        enabled = presets.get(preset, presets["core"])
        for name, chk in checks.items():
            chk.setChecked(enabled.get(name, False))
        self._update_ui_state()

    def _update_ui_state(self) -> None:
        has_data = bool(self._animal_dfs)
        n_animals = len(self._animal_dfs)
        self._btn_compute.setEnabled(has_data)
        self._btn_heatmap.setEnabled(n_animals >= 2)
        if not has_data:
            self._lbl_status.setText("Load tracking data to compute kinematics.")
            return

        unit_hint = (
            f"Calibration active: {self._px_per_cm:.2f} px/cm. Recompute to refresh cm-based outputs."
            if self._px_per_cm > 0
            else "Working in pixel units. Calibrate first if you want cm-based outputs."
        )
        heatmap_hint = (
            "Position map available for multi-animal data."
            if n_animals >= 2
            else "Position map needs at least 2 animals."
        )
        self._lbl_status.setText(
            f"Ready: {n_animals} animal(s) loaded. Time source: {self._time_source_label()}. "
            f"{unit_hint} {heatmap_hint}"
        )

    def _compute(self) -> None:
        if not self._animal_dfs:
            return

        from dlc_processor.core.kinematics import compute_kinematics, compute_partner_kinematics

        fps = self._spin_fps.value()
        time_s = self._custom_time
        result = {}
        for aid, df in self._animal_dfs.items():
            result[aid] = compute_kinematics(
                df,
                fps=fps,
                time_s=time_s,
                per_bodypart=self._chk_perbp.isChecked(),
                body_speed=self._chk_body.isChecked(),
                orientation=self._chk_orient.isChecked(),
                acceleration=self._chk_accel.isChecked(),
                distance_traveled=self._chk_distance.isChecked(),
                immobile=self._chk_immobile.isChecked(),
                path_tortuosity=self._chk_tortuosity.isChecked(),
                body_elongation=self._chk_elongation.isChecked(),
                curvature=self._chk_curvature.isChecked(),
                head_direction=self._chk_headdir.isChecked(),
                rearing=self._chk_rearing.isChecked(),
                px_per_cm=self._px_per_cm,
            )

        if self._chk_partner.isChecked() and len(result) >= 2:
            aids = list(result.keys())
            for i, aid in enumerate(aids):
                partner_id = aids[(i + 1) % len(aids)]
                result[aid] = compute_partner_kinematics(
                    result[aid],
                    self._animal_dfs[partner_id],
                    fps=fps,
                    px_per_cm=self._px_per_cm,
                )

        self._result_dfs = result
        self._populate_table(result)
        unit_msg = "px + cm" if self._px_per_cm > 0 else "px only"
        self._lbl_status.setText(
            f"Computed kinematics for {len(result)} animal(s). Time source: {self._time_source_label()}. Output units: {unit_msg}."
        )
        self.computed.emit(result)

    def _time_source_label(self) -> str:
        if self._custom_time is not None and len(self._custom_time) >= max((len(df) for df in self._animal_dfs.values()), default=0):
            return "loaded frame times"
        return f"{self._spin_fps.value():.2f} fps fallback"

    def _populate_table(self, result: dict) -> None:
        summary_cols = [
            ("body_speed_cm_s", "body_speed_px_s"),
            ("body_accel_cm_s2", "body_accel_px_s2"),
            ("body_jerk_cm_s3", "body_jerk_px_s3"),
            ("body_orientation_deg", None),
            ("body_angle_rate_deg_fr", None),
            ("distance_traveled_cm", "distance_traveled_px"),
            ("path_tortuosity", None),
            ("body_elongation_cm", "body_elongation_px"),
            ("trajectory_curvature_1_cm", "trajectory_curvature_1_px"),
            ("head_direction_deg", None),
            ("heading_body_angle_diff_deg", None),
            ("partner_distance_cm", "partner_distance_px"),
            ("partner_angle_deg", None),
            ("partner_proximity_index", None),
        ]

        rows = []
        for aid, df in result.items():
            for preferred, fallback in summary_cols:
                col = preferred if preferred in df.columns else fallback
                if not col or col not in df.columns:
                    continue
                arr = df[col].to_numpy(dtype=float)
                arr = arr[~np.isnan(arr)]
                if len(arr):
                    rows.append((aid, col, f"{arr.mean():.2f}", f"{arr.max():.2f}"))

            immobile_col = "immobile" if "immobile" in df.columns else "freezing"
            if immobile_col in df.columns:
                total = len(df)
                immobile_frames = int(df[immobile_col].sum())
                pct = f"{100.0 * immobile_frames / max(total, 1):.1f}%"
                rows.append((aid, "immobile", pct, f"{immobile_frames} fr"))

        self._table.setRowCount(len(rows))
        self._table.setMaximumHeight(max(160, 22 * len(rows) + 30))
        for row_idx, row in enumerate(rows):
            for col_idx, value in enumerate(row):
                self._table.setItem(row_idx, col_idx, QTableWidgetItem(value))
