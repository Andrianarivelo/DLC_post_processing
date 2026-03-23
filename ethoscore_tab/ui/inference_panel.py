"""Inference panel for Ethoscore tab.

Allows loading a saved model checkpoint and running inference on the current
feature matrix (built via FeaturePanel).  Predictions are emitted as a
(N_frames, class_names) pair.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class InferencePanel(QWidget):
    """Checkpoint loader + run inference button."""

    # Emits (predictions: np.ndarray (N,), class_names: list[str])
    prediction_done = Signal(object, list)
    error           = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._checkpoint_path: str = ""
        self._feature_matrix: Optional[np.ndarray] = None
        self._worker = None
        self._setup_ui()

    # ── UI ─────────────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Checkpoint file
        grp_ckpt = QGroupBox("Model Checkpoint")
        ckpt_lay = QVBoxLayout(grp_ckpt)

        row = QHBoxLayout()
        self._lbl_ckpt = QLabel("No checkpoint selected.")
        self._lbl_ckpt.setStyleSheet("color: #6c7086; font-size: 11px;")
        self._lbl_ckpt.setWordWrap(True)
        btn_browse = QPushButton("Browse…")
        btn_browse.setObjectName("secondary")
        btn_browse.clicked.connect(self._pick_checkpoint)
        row.addWidget(self._lbl_ckpt, 1)
        row.addWidget(btn_browse)
        ckpt_lay.addLayout(row)
        layout.addWidget(grp_ckpt)

        # Feature matrix info
        grp_feat = QGroupBox("Input Features")
        feat_lay = QVBoxLayout(grp_feat)
        self._lbl_matrix = QLabel("No feature matrix — build one in the Features panel first.")
        self._lbl_matrix.setStyleSheet("color: #6c7086; font-size: 11px;")
        self._lbl_matrix.setWordWrap(True)
        feat_lay.addWidget(self._lbl_matrix)
        layout.addWidget(grp_feat)

        # Run button
        self._btn_run = QPushButton("▶  Run Inference")
        self._btn_run.setEnabled(False)
        self._btn_run.clicked.connect(self._run)
        layout.addWidget(self._btn_run)

        # Status
        self._lbl_status = QLabel("")
        self._lbl_status.setWordWrap(True)
        self._lbl_status.setStyleSheet("color: #a6adc8; font-size: 11px;")
        layout.addWidget(self._lbl_status)

        layout.addStretch()

    # ── Public API ─────────────────────────────────────────────────────────────

    def set_feature_matrix(self, matrix: np.ndarray) -> None:
        self._feature_matrix = matrix
        self._lbl_matrix.setText(
            f"{matrix.shape[0]} frames × {matrix.shape[1]} features"
        )
        self._refresh_run_btn()

    # ── Slots ──────────────────────────────────────────────────────────────────

    def _pick_checkpoint(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Model Checkpoint", "",
            "PyTorch checkpoint (*.pt *.pth);;All Files (*)"
        )
        if not path:
            return
        self._checkpoint_path = path
        self._lbl_ckpt.setText(Path(path).name)
        self._refresh_run_btn()

    def _run(self) -> None:
        if self._worker and self._worker.isRunning():
            return
        self._btn_run.setEnabled(False)
        self._lbl_status.setText("Running inference…")

        from ethoscore_tab.workers.inference_worker import InferenceWorker
        self._worker = InferenceWorker(
            checkpoint_path=self._checkpoint_path,
            feature_matrix=self._feature_matrix,
            parent=self,
        )
        self._worker.prediction_done.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    @Slot(object, list)
    def _on_done(self, predictions: np.ndarray, class_names: list) -> None:
        self._lbl_status.setText(
            f"Done — {len(predictions)} frames · {len(class_names)} classes: "
            + ", ".join(class_names)
        )
        self._refresh_run_btn()
        self.prediction_done.emit(predictions, class_names)

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        self._lbl_status.setText(f"Error: {msg}")
        self._refresh_run_btn()
        self.error.emit(msg)

    def _refresh_run_btn(self) -> None:
        ready = bool(self._checkpoint_path) and self._feature_matrix is not None
        self._btn_run.setEnabled(ready)
