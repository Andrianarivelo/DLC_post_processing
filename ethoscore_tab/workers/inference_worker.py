"""QThread wrapper for behavior inference."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PySide6.QtCore import QThread, Signal

logger = logging.getLogger(__name__)


class InferenceWorker(QThread):
    prediction_done = Signal(object, list)   # np.ndarray (N,), class_names list[str]
    error           = Signal(str)

    def __init__(
        self,
        checkpoint_path: str,
        feature_matrix: np.ndarray,
        seq_len: int = 16,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.checkpoint_path = checkpoint_path
        self.feature_matrix  = feature_matrix
        self.seq_len         = seq_len

    def run(self) -> None:
        from ethoscore_tab.core.predictor import predict
        try:
            preds, class_names = predict(
                self.checkpoint_path,
                self.feature_matrix,
                seq_len=self.seq_len,
            )
            self.prediction_done.emit(preds, class_names)
        except Exception as exc:
            logger.exception("Inference error: %s", exc)
            self.error.emit(str(exc))
