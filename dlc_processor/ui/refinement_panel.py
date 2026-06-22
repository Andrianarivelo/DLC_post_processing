"""Manual refinement panel: ID swap and keypoint correction for a frame range."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class RefinementPanel(QGroupBox):
    """Swap two animals' tracking data for a given frame range."""

    data_changed = Signal(dict)   # updated animal_dfs

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Manual Refinement", parent)
        self._animal_dfs: dict = {}
        self._mark_from: Optional[int] = None
        self._mark_to:   Optional[int] = None
        self._current_frame = 0
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        # Current frame display
        frame_row = QHBoxLayout()
        frame_row.addWidget(QLabel("Current frame:"))
        self._lbl_frame = QLabel("—")
        frame_row.addWidget(self._lbl_frame)
        frame_row.addStretch()
        layout.addLayout(frame_row)

        # --- Frame Range Selection ---
        range_group = QGroupBox("Frame Range")
        range_lay = QVBoxLayout(range_group)
        range_lay.setSpacing(4)

        range_row = QHBoxLayout()
        self._btn_from = QPushButton("Mark From")
        self._btn_from.setObjectName("secondary")
        self._btn_from.setToolTip("Set the start of the swap range to the current frame")
        self._btn_from.clicked.connect(self._mark_from_clicked)
        self._btn_to = QPushButton("Mark To")
        self._btn_to.setObjectName("secondary")
        self._btn_to.setToolTip("Set the end of the swap range to the current frame")
        self._btn_to.clicked.connect(self._mark_to_clicked)
        self._lbl_range = QLabel("Range: not set")
        range_row.addWidget(self._btn_from)
        range_row.addWidget(self._btn_to)
        range_row.addWidget(self._lbl_range)
        range_row.addStretch()
        range_lay.addLayout(range_row)

        spin_row = QHBoxLayout()
        spin_row.addWidget(QLabel("From:"))
        self._spin_from = QSpinBox()
        self._spin_from.setRange(0, 10_000_000)
        self._spin_from.setToolTip("Start frame of the range (inclusive)")
        spin_row.addWidget(self._spin_from)
        spin_row.addWidget(QLabel("To:"))
        self._spin_to = QSpinBox()
        self._spin_to.setRange(0, 10_000_000)
        self._spin_to.setToolTip("End frame of the range (inclusive)")
        spin_row.addWidget(self._spin_to)
        spin_row.addStretch()
        range_lay.addLayout(spin_row)
        layout.addWidget(range_group)

        # --- Identity Swap ---
        swap_group = QGroupBox("Identity Swap")
        swap_lay = QVBoxLayout(swap_group)
        swap_lay.setSpacing(4)

        swap_row = QHBoxLayout()
        swap_row.addWidget(QLabel("Swap:"))
        self._combo_a = QComboBox()
        self._combo_a.setToolTip("First animal to swap")
        self._combo_b = QComboBox()
        self._combo_b.setToolTip("Second animal to swap")
        swap_row.addWidget(self._combo_a)
        swap_row.addWidget(QLabel("<>"))
        swap_row.addWidget(self._combo_b)
        swap_row.addStretch()
        swap_lay.addLayout(swap_row)

        btn_swap = QPushButton("Apply Swap")
        btn_swap.setToolTip("Swap the tracking data of the two selected animals within the frame range")
        btn_swap.clicked.connect(self._apply_swap)
        swap_lay.addWidget(btn_swap)

        self._lbl_status = QLabel("")
        swap_lay.addWidget(self._lbl_status)
        layout.addWidget(swap_group)

    def set_animal_dfs(self, dfs: dict) -> None:
        self._animal_dfs = dfs
        animals = list(dfs.keys())
        self._combo_a.clear()
        self._combo_b.clear()
        for a in animals:
            self._combo_a.addItem(a)
            self._combo_b.addItem(a)
        if len(animals) >= 2:
            self._combo_b.setCurrentIndex(1)

    def set_current_frame(self, fi: int) -> None:
        self._current_frame = fi
        self._lbl_frame.setText(str(fi))

    def _mark_from_clicked(self) -> None:
        self._mark_from = self._current_frame
        self._spin_from.setValue(self._current_frame)
        self._update_range_label()

    def _mark_to_clicked(self) -> None:
        self._mark_to = self._current_frame
        self._spin_to.setValue(self._current_frame)
        self._update_range_label()

    def _update_range_label(self) -> None:
        f = self._mark_from
        t = self._mark_to
        if f is None and t is None:
            self._lbl_range.setText("Range: not set")
        elif f is not None and t is None:
            self._lbl_range.setText(f"Range: {f} - ?")
        elif f is None and t is not None:
            self._lbl_range.setText(f"Range: ? - {t}")
        else:
            self._lbl_range.setText(f"Range: {f} - {t}")

    def _apply_swap(self) -> None:
        aid_a = self._combo_a.currentText()
        aid_b = self._combo_b.currentText()
        if aid_a == aid_b or aid_a not in self._animal_dfs or aid_b not in self._animal_dfs:
            self._lbl_status.setText("✗ Select two different animals.")
            return

        fr = self._spin_from.value()
        to = self._spin_to.value()
        if fr > to:
            fr, to = to, fr

        df_a = self._animal_dfs[aid_a]
        df_b = self._animal_dfs[aid_b]

        idx = slice(fr, to + 1)
        # Swap all columns
        tmp_a = df_a.iloc[idx].copy()
        df_a.iloc[idx] = df_b.iloc[idx].values
        df_b.iloc[idx] = tmp_a.values

        self._animal_dfs[aid_a] = df_a
        self._animal_dfs[aid_b] = df_b

        self._lbl_status.setText(f"Swapped {aid_a} <> {aid_b} for frames {fr}-{to}")
        logger.info("ID swap: %s ↔ %s  frames %d–%d", aid_a, aid_b, fr, to)
        self.data_changed.emit(self._animal_dfs)
