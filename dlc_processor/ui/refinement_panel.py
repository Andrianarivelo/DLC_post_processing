"""Manual refinement panel: ID swap and keypoint correction for a frame range."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from shared.ui_kit import COLORS, Card, hint

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
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # ── Frame Range card ─────────────────────────────────────────────────
        range_card = Card(
            "Frame Range",
            "Mark a start and end frame to define the correction window.",
            accent=COLORS["teal"],
        )

        # Current-frame chip + mark controls.
        frame_row = QHBoxLayout()
        frame_row.setSpacing(8)
        cap_frame = QLabel("Current frame")
        cap_frame.setObjectName("hint")
        self._lbl_frame = QLabel("—")
        self._lbl_frame.setStyleSheet(
            f"color:{COLORS['text']}; font-weight:700; font-size:14px;"
        )
        frame_row.addWidget(cap_frame, 0)
        frame_row.addWidget(self._lbl_frame, 0)
        frame_row.addStretch(1)
        range_card.body.addLayout(frame_row)

        # Mark From / Mark To buttons + live range readout.
        mark_row = QHBoxLayout()
        mark_row.setSpacing(8)
        self._btn_from = QPushButton("Mark From")
        self._btn_from.setObjectName("secondary")
        self._btn_from.setToolTip("Set the start of the swap range to the current frame")
        self._btn_from.clicked.connect(self._mark_from_clicked)
        self._btn_to = QPushButton("Mark To")
        self._btn_to.setObjectName("secondary")
        self._btn_to.setToolTip("Set the end of the swap range to the current frame")
        self._btn_to.clicked.connect(self._mark_to_clicked)
        self._lbl_range = QLabel("Range: not set")
        self._lbl_range.setObjectName("hint")
        mark_row.addWidget(self._btn_from)
        mark_row.addWidget(self._btn_to)
        mark_row.addWidget(self._lbl_range, 1)
        range_card.body.addLayout(mark_row)

        # Precise numeric From / To entry, kept compact.
        spin_row = QHBoxLayout()
        spin_row.setSpacing(8)
        cap_from = QLabel("From")
        cap_from.setObjectName("hint")
        self._spin_from = QSpinBox()
        self._spin_from.setRange(0, 10_000_000)
        self._spin_from.setToolTip("Start frame of the range (inclusive)")
        self._spin_from.setFixedWidth(110)
        cap_to = QLabel("To")
        cap_to.setObjectName("hint")
        self._spin_to = QSpinBox()
        self._spin_to.setRange(0, 10_000_000)
        self._spin_to.setToolTip("End frame of the range (inclusive)")
        self._spin_to.setFixedWidth(110)
        spin_row.addWidget(cap_from, 0)
        spin_row.addWidget(self._spin_from, 0)
        spin_row.addSpacing(8)
        spin_row.addWidget(cap_to, 0)
        spin_row.addWidget(self._spin_to, 0)
        spin_row.addStretch(1)
        range_card.body.addLayout(spin_row)

        layout.addWidget(range_card)

        # ── Identity Swap card ───────────────────────────────────────────────
        swap_card = Card(
            "Identity Swap",
            "Exchange two animals' tracks across the selected frame range.",
            accent=COLORS["accent"],
        )

        swap_row = QHBoxLayout()
        swap_row.setSpacing(8)
        self._combo_a = QComboBox()
        self._combo_a.setToolTip("First animal to swap")
        self._combo_a.setMinimumWidth(130)
        self._combo_a.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        swap_arrow = QLabel("⇄")
        swap_arrow.setAlignment(Qt.AlignmentFlag.AlignCenter)
        swap_arrow.setStyleSheet(f"color:{COLORS['text_muted']}; font-size:15px;")
        self._combo_b = QComboBox()
        self._combo_b.setToolTip("Second animal to swap")
        self._combo_b.setMinimumWidth(130)
        self._combo_b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        swap_row.addWidget(self._combo_a, 1)
        swap_row.addWidget(swap_arrow, 0)
        swap_row.addWidget(self._combo_b, 1)
        swap_card.body.addLayout(swap_row)

        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        btn_swap = QPushButton("Apply Swap")
        btn_swap.setToolTip("Swap the tracking data of the two selected animals within the frame range")
        btn_swap.clicked.connect(self._apply_swap)
        action_row.addStretch(1)
        action_row.addWidget(btn_swap, 0)
        swap_card.body.addLayout(action_row)

        self._lbl_status = hint("")
        swap_card.body.addWidget(self._lbl_status)

        layout.addWidget(swap_card)
        layout.addStretch(1)

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
