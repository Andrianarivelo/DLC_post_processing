"""Training configuration and launch panel."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class TrainingPanel(QWidget):
    """Hyper-parameter controls + train button for PyTorch classifier."""

    training_finished = Signal(dict)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._matrix: Optional[np.ndarray] = None
        self._labels: Optional[np.ndarray] = None
        self._worker = None
        self._ckpt_dir = tempfile.mkdtemp()
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        grp = QGroupBox("Model Training")
        grp_lay = QVBoxLayout(grp)

        form = QFormLayout()
        form.setSpacing(5)

        self._combo_type = QComboBox()
        self._combo_type.addItems(["MLP", "LSTM"])
        form.addRow("Model:", self._combo_type)

        self._spin_hidden = QSpinBox()
        self._spin_hidden.setRange(16, 1024)
        self._spin_hidden.setValue(128)
        form.addRow("Hidden units:", self._spin_hidden)

        self._spin_seq = QSpinBox()
        self._spin_seq.setRange(4, 256)
        self._spin_seq.setValue(16)
        self._spin_seq.setEnabled(False)
        form.addRow("Seq len (LSTM):", self._spin_seq)
        self._combo_type.currentTextChanged.connect(
            lambda t: self._spin_seq.setEnabled(t == "LSTM")
        )

        self._spin_epochs = QSpinBox()
        self._spin_epochs.setRange(1, 500)
        self._spin_epochs.setValue(30)
        form.addRow("Epochs:", self._spin_epochs)

        self._spin_lr = QDoubleSpinBox()
        self._spin_lr.setRange(1e-5, 0.1)
        self._spin_lr.setValue(1e-3)
        self._spin_lr.setDecimals(5)
        self._spin_lr.setSingleStep(1e-4)
        form.addRow("Learning rate:", self._spin_lr)

        self._spin_train = QDoubleSpinBox()
        self._spin_train.setRange(0.5, 0.95)
        self._spin_train.setValue(0.80)
        self._spin_train.setSingleStep(0.05)
        form.addRow("Train fraction:", self._spin_train)

        self._spin_folds = QSpinBox()
        self._spin_folds.setRange(1, 10)
        self._spin_folds.setValue(5)
        form.addRow("CV folds:", self._spin_folds)

        self._spin_batch = QSpinBox()
        self._spin_batch.setRange(16, 512)
        self._spin_batch.setValue(128)
        form.addRow("Batch size:", self._spin_batch)

        grp_lay.addLayout(form)

        # Checkpoint dir
        ckpt_row = QPushButton("Set checkpoint dir…")
        ckpt_row.setObjectName("secondary")
        ckpt_row.clicked.connect(self._pick_ckpt_dir)
        grp_lay.addWidget(ckpt_row)
        self._lbl_ckpt = QLabel(f"→ {self._ckpt_dir}")
        self._lbl_ckpt.setWordWrap(True)
        grp_lay.addWidget(self._lbl_ckpt)

        # Status
        self._lbl_status = QLabel("")
        grp_lay.addWidget(self._lbl_status)

        # Progress bar
        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        grp_lay.addWidget(self._bar)

        # Train / Stop buttons
        self._btn_train = QPushButton("▶ Train")
        self._btn_train.clicked.connect(self._start_training)
        grp_lay.addWidget(self._btn_train)

        layout.addWidget(grp)

    # ── Public API ────────────────────────────────────────────────────────────

    def set_data(self, matrix: np.ndarray, labels: np.ndarray) -> None:
        self._matrix = matrix
        self._labels = labels
        n_labeled = int((labels > 0).sum())
        self._lbl_status.setText(
            f"Ready — {matrix.shape[0]} frames, {matrix.shape[1]} features, "
            f"{n_labeled} labeled frames"
        )

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _pick_ckpt_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Checkpoint Directory")
        if d:
            self._ckpt_dir = d
            self._lbl_ckpt.setText(f"→ {d}")

    def _start_training(self) -> None:
        if self._matrix is None or self._labels is None:
            self._lbl_status.setText("✗ Build a feature matrix and annotate first.")
            return
        if self._worker and self._worker.isRunning():
            self._worker.abort()
            self._btn_train.setText("▶ Train")
            return

        from ethoscore_tab.workers.training_worker import TrainingWorker
        self._worker = TrainingWorker(
            feature_matrix=self._matrix,
            labels=self._labels,
            model_type=self._combo_type.currentText(),
            hidden=self._spin_hidden.value(),
            seq_len=self._spin_seq.value(),
            epochs=self._spin_epochs.value(),
            lr=self._spin_lr.value(),
            train_frac=self._spin_train.value(),
            n_folds=self._spin_folds.value(),
            batch_size=self._spin_batch.value(),
            checkpoint_dir=self._ckpt_dir,
        )
        self._worker.epoch_done.connect(self._on_epoch)
        self._worker.training_finished.connect(self._on_done)
        self._worker.error.connect(lambda e: self._lbl_status.setText(f"✗ {e}"))
        self._bar.setValue(0)
        self._btn_train.setText("⏹ Stop")
        self._worker.start()

    @Slot(int, float, float)
    def _on_epoch(self, epoch: int, loss: float, f1: float) -> None:
        total = self._spin_epochs.value()
        self._bar.setValue(int(100 * epoch / max(total, 1)))
        self._lbl_status.setText(f"Epoch {epoch}/{total}  loss={loss:.4f}  val_F1={f1:.3f}")

    @Slot(dict)
    def _on_done(self, results: dict) -> None:
        self._btn_train.setText("▶ Train")
        self._bar.setValue(100)
        self._lbl_status.setText(
            f"✔ Done — mean F1={results['mean_f1']:.3f} ± {results['std_f1']:.3f}"
        )
        self.training_finished.emit(results)
