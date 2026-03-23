"""Training results display: fold table, metrics, checkpoint export."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class ResultsPanel(QWidget):
    """Display cross-validation results and allow checkpoint export."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._results: Optional[dict] = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        grp = QGroupBox("Training Results")
        grp_lay = QVBoxLayout(grp)

        # Summary labels
        self._lbl_f1  = QLabel("Mean F1:  —")
        self._lbl_acc = QLabel("Mean Acc: —")
        self._lbl_cls = QLabel("Classes: —")
        for lbl in (self._lbl_f1, self._lbl_acc, self._lbl_cls):
            grp_lay.addWidget(lbl)

        # Per-fold table
        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Fold", "F1", "Accuracy"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setMaximumHeight(200)
        grp_lay.addWidget(self._table)

        # Export button
        btn_row = QHBoxLayout()
        self._btn_export = QPushButton("Export Best Checkpoint…")
        self._btn_export.setEnabled(False)
        self._btn_export.clicked.connect(self._export_checkpoint)
        btn_row.addWidget(self._btn_export)
        btn_row.addStretch()
        grp_lay.addLayout(btn_row)

        self._lbl_export = QLabel("")
        grp_lay.addWidget(self._lbl_export)

        layout.addWidget(grp)

    @Slot(dict)
    def set_results(self, results: dict) -> None:
        self._results = results

        self._lbl_f1.setText(
            f"Mean F1:  {results['mean_f1']:.3f} ± {results['std_f1']:.3f}"
        )
        self._lbl_acc.setText(
            f"Mean Acc: {results['mean_acc']:.3f} ± {results['std_acc']:.3f}"
        )
        self._lbl_cls.setText(f"Classes: {results.get('class_names', [])}")

        folds = results.get("fold_results", [])
        self._table.setRowCount(len(folds))
        for r, fold in enumerate(folds):
            self._table.setItem(r, 0, QTableWidgetItem(str(fold["fold"])))
            self._table.setItem(r, 1, QTableWidgetItem(f"{fold['f1']:.3f}"))
            self._table.setItem(r, 2, QTableWidgetItem(f"{fold['acc']:.3f}"))

        self._btn_export.setEnabled(bool(results.get("best_checkpoint")))

    def _export_checkpoint(self) -> None:
        if not self._results:
            return
        src = self._results.get("best_checkpoint", "")
        if not src or not Path(src).exists():
            self._lbl_export.setText("✗ Checkpoint file not found.")
            return
        dst, _ = QFileDialog.getSaveFileName(
            self, "Save Checkpoint", "best_model.pt", "PyTorch (*.pt)"
        )
        if dst:
            shutil.copy2(src, dst)
            self._lbl_export.setText(f"✔ Saved → {Path(dst).name}")
