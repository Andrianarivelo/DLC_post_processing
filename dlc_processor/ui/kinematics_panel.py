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
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        form = QFormLayout()
        form.setSpacing(5)

        self._spin_fps = QDoubleSpinBox()
        self._spin_fps.setRange(1.0, 1000.0)
        self._spin_fps.setValue(25.0)
        self._spin_fps.setSuffix(" fps")
        form.addRow("Frame rate:", self._spin_fps)

        self._chk_perbp       = QCheckBox("Per-bodypart speed/accel")
        self._chk_perbp.setChecked(False)
        self._chk_body        = QCheckBox("Body-centre speed")
        self._chk_body.setChecked(True)
        self._chk_orient      = QCheckBox("Body orientation")
        self._chk_orient.setChecked(True)
        self._chk_accel       = QCheckBox("Acceleration + jerk")
        self._chk_accel.setChecked(True)
        self._chk_distance    = QCheckBox("Cumulative distance")
        self._chk_distance.setChecked(True)
        self._chk_freezing    = QCheckBox("Freezing detection")
        self._chk_freezing.setChecked(True)
        self._chk_tortuosity  = QCheckBox("Path tortuosity")
        self._chk_tortuosity.setChecked(True)
        self._chk_elongation  = QCheckBox("Body elongation")
        self._chk_elongation.setChecked(True)
        self._chk_curvature   = QCheckBox("Trajectory curvature")
        self._chk_curvature.setChecked(True)
        self._chk_headdir     = QCheckBox("Head direction + crab-walk")
        self._chk_headdir.setChecked(True)
        self._chk_rearing     = QCheckBox("Rearing detection")
        self._chk_rearing.setChecked(True)
        self._chk_partner     = QCheckBox("Partner metrics (distance, angle, proximity)")
        self._chk_partner.setChecked(False)
        self._chk_partner.setToolTip("Egocentric distance, angle, and proximity index to partner (multi-animal only)")

        layout.addLayout(form)
        layout.addWidget(self._chk_perbp)
        layout.addWidget(self._chk_body)
        layout.addWidget(self._chk_orient)
        layout.addWidget(self._chk_accel)
        layout.addWidget(self._chk_distance)
        layout.addWidget(self._chk_freezing)
        layout.addWidget(self._chk_tortuosity)
        layout.addWidget(self._chk_elongation)
        layout.addWidget(self._chk_curvature)
        layout.addWidget(self._chk_headdir)
        layout.addWidget(self._chk_rearing)
        layout.addWidget(self._chk_partner)

        btn_row = QHBoxLayout()
        btn = QPushButton("Compute Kinematics")
        btn.clicked.connect(self._compute)
        btn_row.addWidget(btn)

        btn_heatmap = QPushButton("Position Map")
        btn_heatmap.setToolTip("Egocentric position heatmap — where does the other animal spend time?")
        btn_heatmap.clicked.connect(self.heatmap_requested.emit)
        btn_row.addWidget(btn_heatmap)
        layout.addLayout(btn_row)

        # Summary table
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Animal", "Metric", "Mean", "Max"])
        self._table.setMaximumHeight(160)
        self._table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._table)

    def fps(self) -> float:
        """Return the current FPS value from the spin box."""
        return self._spin_fps.value()

    def set_animal_dfs(self, dfs: dict) -> None:
        self._animal_dfs = dfs

    def result_dfs(self) -> dict:
        return self._result_dfs

    def _compute(self) -> None:
        if not self._animal_dfs:
            return
        from dlc_processor.core.kinematics import compute_kinematics, compute_partner_kinematics
        fps = self._spin_fps.value()
        result = {}
        for aid, df in self._animal_dfs.items():
            result[aid] = compute_kinematics(
                df,
                fps=fps,
                per_bodypart=self._chk_perbp.isChecked(),
                body_speed=self._chk_body.isChecked(),
                orientation=self._chk_orient.isChecked(),
                acceleration=self._chk_accel.isChecked(),
                distance_traveled=self._chk_distance.isChecked(),
                freezing=self._chk_freezing.isChecked(),
                path_tortuosity=self._chk_tortuosity.isChecked(),
                body_elongation=self._chk_elongation.isChecked(),
                curvature=self._chk_curvature.isChecked(),
                head_direction=self._chk_headdir.isChecked(),
                rearing=self._chk_rearing.isChecked(),
            )

        # Partner metrics (multi-animal only)
        if self._chk_partner.isChecked() and len(result) >= 2:
            aids = list(result.keys())
            for i, aid in enumerate(aids):
                # Pair with next animal (circular)
                partner_id = aids[(i + 1) % len(aids)]
                result[aid] = compute_partner_kinematics(
                    result[aid], self._animal_dfs[partner_id], fps=fps,
                )

        self._result_dfs = result
        self._populate_table(result)
        self.computed.emit(result)

    def _populate_table(self, result: dict) -> None:
        _SUMMARY_COLS = [
            "body_speed_px_s", "body_accel_px_s2", "body_jerk_px_s3",
            "body_orientation_deg", "body_angle_rate_deg_fr",
            "distance_traveled_px", "path_tortuosity",
            "body_elongation_px", "trajectory_curvature_1_px",
            "head_direction_deg", "heading_body_angle_diff_deg",
            "partner_distance_px", "partner_angle_deg",
            "partner_proximity_index",
        ]
        rows = []
        for aid, df in result.items():
            for col in _SUMMARY_COLS:
                if col in df.columns:
                    arr = df[col].to_numpy(dtype=float)
                    arr = arr[~np.isnan(arr)]
                    if len(arr):
                        rows.append((aid, col, f"{arr.mean():.2f}", f"{arr.max():.2f}"))
            # Freezing: show total frames + percentage
            if "freezing" in df.columns:
                total = len(df)
                frozen = int(df["freezing"].sum())
                pct = f"{100.0 * frozen / max(total, 1):.1f}%"
                rows.append((aid, "freezing", pct, f"{frozen} fr"))

        self._table.setRowCount(len(rows))
        self._table.setMaximumHeight(max(160, 22 * len(rows) + 30))
        for r, (aid, metric, mean, mx) in enumerate(rows):
            for c, val in enumerate((aid, metric, mean, mx)):
                self._table.setItem(r, c, QTableWidgetItem(val))
