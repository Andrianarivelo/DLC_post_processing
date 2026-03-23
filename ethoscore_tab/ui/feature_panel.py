"""Feature matrix construction panel."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QAbstractItemView,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import numpy as np

from ethoscore_tab.core.feature_builder import FeatureBuilder


class FeaturePanel(QWidget):
    """Column selector and feature matrix builder."""

    matrix_built = Signal(object, list)   # np.ndarray, feature_names

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._builder = FeatureBuilder()
        self._matrix: Optional[np.ndarray] = None
        self._names: list[str] = []
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        grp = QGroupBox("Feature Matrix")
        grp_lay = QVBoxLayout(grp)

        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        grp_lay.addWidget(self._list)

        btn_row = QHBoxLayout()
        btn_all  = QPushButton("Select All")
        btn_all.setObjectName("secondary")
        btn_all.clicked.connect(self._list.selectAll)
        btn_none = QPushButton("Clear")
        btn_none.setObjectName("secondary")
        btn_none.clicked.connect(self._list.clearSelection)
        btn_row.addWidget(btn_all)
        btn_row.addWidget(btn_none)
        btn_row.addStretch()
        grp_lay.addLayout(btn_row)

        self._lbl_shape = QLabel("No matrix built.")
        grp_lay.addWidget(self._lbl_shape)

        btn_build = QPushButton("Build Feature Matrix")
        btn_build.clicked.connect(self._build)
        grp_lay.addWidget(btn_build)

        layout.addWidget(grp)

    def set_builder(self, builder: FeatureBuilder) -> None:
        self._builder = builder
        self._refresh_columns()

    def set_sources(self, animal_dfs: dict, behavior_arrays: dict) -> None:
        self._builder.set_animal_dfs(animal_dfs)
        self._builder.set_behavior_arrays(behavior_arrays)
        self._refresh_columns()

    def _refresh_columns(self) -> None:
        self._list.clear()
        for col in self._builder.available_columns():
            self._list.addItem(col)
        self._list.selectAll()

    def _build(self) -> None:
        selected = [self._list.item(i).text()
                    for i in range(self._list.count())
                    if self._list.item(i).isSelected()]
        if not selected:
            return
        try:
            matrix, names = self._builder.build(selected=selected)
            self._matrix = matrix
            self._names  = names
            self._lbl_shape.setText(f"Matrix: {matrix.shape[0]} frames × {matrix.shape[1]} features")
            self.matrix_built.emit(matrix, names)
        except Exception as exc:
            self._lbl_shape.setText(f"Error: {exc}")

    def matrix(self) -> Optional[np.ndarray]:
        return self._matrix

    def feature_names(self) -> list[str]:
        return self._names
