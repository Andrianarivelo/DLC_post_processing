"""QThread worker for PyTorch model training."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PySide6.QtCore import QThread, Signal

logger = logging.getLogger(__name__)


class TrainingWorker(QThread):
    epoch_done        = Signal(int, float, float)   # epoch, train_loss, val_f1
    training_finished = Signal(dict)                # results dict
    error             = Signal(str)

    def __init__(
        self,
        feature_matrix: np.ndarray,
        labels: np.ndarray,
        model_type: str = "MLP",
        hidden: int = 128,
        seq_len: int = 16,
        epochs: int = 30,
        lr: float = 1e-3,
        train_frac: float = 0.80,
        n_folds: int = 5,
        batch_size: int = 128,
        checkpoint_dir: Optional[str] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.feature_matrix  = feature_matrix
        self.labels          = labels
        self.model_type      = model_type
        self.hidden          = hidden
        self.seq_len         = seq_len
        self.epochs          = epochs
        self.lr              = lr
        self.train_frac      = train_frac
        self.n_folds         = n_folds
        self.batch_size      = batch_size
        self.checkpoint_dir  = checkpoint_dir
        self._abort          = False

    def abort(self) -> None:
        self._abort = True

    def run(self) -> None:
        from ethoscore_tab.core.ml_trainer import train_model
        try:
            results = train_model(
                feature_matrix=self.feature_matrix,
                labels=self.labels,
                model_type=self.model_type,
                hidden=self.hidden,
                seq_len=self.seq_len,
                epochs=self.epochs,
                lr=self.lr,
                train_frac=self.train_frac,
                n_folds=self.n_folds,
                batch_size=self.batch_size,
                checkpoint_dir=self.checkpoint_dir,
                progress_callback=self._on_progress,
            )
            self.training_finished.emit(results)
        except Exception as exc:
            logger.exception("Training error: %s", exc)
            self.error.emit(str(exc))

    def _on_progress(self, epoch: int, train_loss: float, val_f1: float) -> None:
        if not self._abort:
            self.epoch_done.emit(epoch, train_loss, val_f1)
