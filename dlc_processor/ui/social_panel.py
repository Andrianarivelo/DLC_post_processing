"""Social behaviour detection panel."""

from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

_CONTINUOUS = {"inter_animal_dist_px", "approach_speed_px_s", "relative_heading_deg"}


class SocialPanel(QGroupBox):
    """Controls and results for vectorised social behaviour detection."""

    detected = Signal(dict)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Social Behaviours", parent)
        self._animal_dfs: dict = {}
        self._all_behavior_arrays: dict = {}
        self._behavior_arrays: dict = {}
        self._behavior_checks: dict[str, QCheckBox] = {}
        self._fps = 25.0
        self._last_summary_context: tuple[int, str, str] | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        pair_row = QHBoxLayout()
        pair_row.addWidget(QLabel("Animal A:"))
        self._combo_a = QComboBox()
        pair_row.addWidget(self._combo_a)
        pair_row.addWidget(QLabel("↔ B:"))
        self._combo_b = QComboBox()
        pair_row.addWidget(self._combo_b)
        pair_row.addStretch()
        layout.addLayout(pair_row)

        form = QFormLayout()
        form.setSpacing(5)

        self._spin_close = QDoubleSpinBox()
        self._spin_close.setRange(1, 500)
        self._spin_close.setValue(25)
        self._spin_close.setSuffix(" px")
        form.addRow("Close contact tol:", self._spin_close)

        self._spin_side = QDoubleSpinBox()
        self._spin_side.setRange(1, 500)
        self._spin_side.setValue(50)
        self._spin_side.setSuffix(" px")
        form.addRow("Side contact tol:", self._spin_side)

        self._spin_follow = QDoubleSpinBox()
        self._spin_follow.setRange(1, 500)
        self._spin_follow.setValue(30)
        self._spin_follow.setSuffix(" px")
        form.addRow("Follow tol:", self._spin_follow)

        self._spin_fwin = QSpinBox()
        self._spin_fwin.setRange(1, 120)
        self._spin_fwin.setValue(12)
        self._spin_fwin.setSuffix(" fr")
        form.addRow("Follow window:", self._spin_fwin)

        self._spin_median = QSpinBox()
        self._spin_median.setRange(1, 30)
        self._spin_median.setValue(6)
        self._spin_median.setSuffix(" fr")
        form.addRow("Median filter:", self._spin_median)

        layout.addLayout(form)

        btn = QPushButton("Detect Behaviours")
        btn.clicked.connect(self._detect)
        layout.addWidget(btn)

        visible_group = QGroupBox("Visible Behaviours")
        visible_lay = QVBoxLayout(visible_group)
        visible_lay.setSpacing(4)

        vis_btn_row = QHBoxLayout()
        btn_all = QPushButton("All")
        btn_all.clicked.connect(lambda: self._set_all_behavior_checks(True))
        btn_none = QPushButton("None")
        btn_none.clicked.connect(lambda: self._set_all_behavior_checks(False))
        vis_btn_row.addWidget(btn_all)
        vis_btn_row.addWidget(btn_none)
        vis_btn_row.addStretch()
        visible_lay.addLayout(vis_btn_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(120)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        container = QWidget()
        self._behavior_checks_layout = QVBoxLayout(container)
        self._behavior_checks_layout.setContentsMargins(0, 0, 0, 0)
        self._behavior_checks_layout.setSpacing(2)
        self._behavior_checks_layout.addStretch()
        scroll.setWidget(container)
        visible_lay.addWidget(scroll)
        layout.addWidget(visible_group)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Behaviour", "Pair", "Frames", "Time %"])
        self._table.setMaximumHeight(220)
        self._table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._table)

    def set_animal_dfs(self, dfs: dict, fps: float = 25.0) -> None:
        self._animal_dfs = dfs
        self._fps = fps
        self._all_behavior_arrays = {}
        self._behavior_arrays = {}
        self._last_summary_context = None
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

    def behavior_arrays(self) -> dict:
        return self._behavior_arrays

    def _detect(self) -> None:
        if len(self._animal_dfs) < 2:
            return

        from dlc_processor.core.social_behaviors import SocialBehaviors

        aid_a = self._combo_a.currentText()
        aid_b = self._combo_b.currentText()
        if aid_a == aid_b or aid_a not in self._animal_dfs or aid_b not in self._animal_dfs:
            return

        sb = SocialBehaviors(
            df_a=self._animal_dfs[aid_a],
            df_b=self._animal_dfs[aid_b],
            fps=self._fps,
            close_tol=self._spin_close.value(),
            side_tol=self._spin_side.value(),
            follow_tol=self._spin_follow.value(),
            follow_window=self._spin_fwin.value(),
            median_filter=self._spin_median.value(),
        )
        self._all_behavior_arrays = sb.compute_all()
        self._last_summary_context = (len(next(iter(self._animal_dfs.values()))), aid_a, aid_b)
        self._rebuild_behavior_checks()
        self._apply_visibility_filters(emit=True)

    def _rebuild_behavior_checks(self) -> None:
        previous = {name: chk.isChecked() for name, chk in self._behavior_checks.items()}
        self._clear_behavior_checks()

        for name, arr in self._all_behavior_arrays.items():
            if name in _CONTINUOUS or not _is_boolean_like(arr):
                continue
            chk = QCheckBox(_display_behavior_name(name))
            chk.setToolTip(name)
            chk.setChecked(previous.get(name, True))
            chk.toggled.connect(self._on_visibility_changed)
            self._behavior_checks[name] = chk
            self._behavior_checks_layout.insertWidget(self._behavior_checks_layout.count() - 1, chk)

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
            if name in _CONTINUOUS or not _is_boolean_like(arr):
                visible[name] = arr
                continue
            chk = self._behavior_checks.get(name)
            if chk is None or chk.isChecked():
                visible[name] = arr

        self._behavior_arrays = visible
        if self._last_summary_context is not None:
            self._populate_table(*self._last_summary_context)
        if emit:
            self.detected.emit(self._behavior_arrays)

    def _populate_table(self, n_frames: int, aid_a: str, aid_b: str) -> None:
        pair_ab = f"{aid_a}↔{aid_b}"
        pair_a_b = f"{aid_a}→{aid_b}"
        pair_b_a = f"{aid_b}→{aid_a}"
        passive_label = f"{aid_a}←{aid_b}"
        pair_labels = {
            "nose2nose": pair_ab,
            "sidebyside": pair_ab,
            "sidereside": pair_ab,
            "a_nose2anogenital_b": pair_a_b,
            "b_nose2anogenital_a": pair_b_a,
            "a_nose2body_b": pair_a_b,
            "b_nose2body_a": pair_b_a,
            "a_following_b": pair_a_b,
            "b_following_a": pair_b_a,
            "a_oriented_toward_b": pair_a_b,
            "b_oriented_toward_a": pair_b_a,
            "passive_anogenital": passive_label,
            "passive_investigation": passive_label,
            "passive_being_followed": passive_label,
            "inter_animal_dist_px": pair_ab,
            "approach_speed_px_s": pair_ab,
            "relative_heading_deg": pair_ab,
        }

        rows = []
        for name, arr in self._behavior_arrays.items():
            pair = pair_labels.get(name, "")
            label = _display_behavior_name(name)
            if name in _CONTINUOUS:
                valid = arr[~np.isnan(arr)] if np.issubdtype(arr.dtype, np.floating) else arr
                if len(valid):
                    rows.append((label, pair, f"{np.nanmean(arr):.1f}", f"±{np.nanstd(arr):.1f}"))
                else:
                    rows.append((label, pair, "N/A", "N/A"))
            else:
                frames_in = int(np.nansum(arr))
                pct = 100.0 * frames_in / max(n_frames, 1)
                rows.append((label, pair, str(frames_in), f"{pct:.1f}%"))

        self._table.setHorizontalHeaderLabels(["Behaviour", "Pair", "Value", "Detail"])
        self._table.setRowCount(len(rows))
        for row_idx, row in enumerate(rows):
            for col_idx, val in enumerate(row):
                self._table.setItem(row_idx, col_idx, QTableWidgetItem(val))


def _display_behavior_name(name: str) -> str:
    pretty = {
        "nose2nose": "nose\u2194nose",
        "sidebyside": "side-by-side",
        "sidereside": "side\u2194rear",
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
