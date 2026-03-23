"""Load a saved ML checkpoint and run frame-level behavior predictions.

Checkpoint format (saved by ml_trainer.py)
------------------------------------------
{
    "state_dict": OrderedDict,   # model weights
    "model_type": "MLP" | "LSTM",
    "input_dim":  int,
    "hidden":     int,
    "n_classes":  int,
    "le_classes": list[int | str],  # original label values
}
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def predict(
    checkpoint_path: str,
    feature_matrix: np.ndarray,
    seq_len: int = 16,
) -> tuple[np.ndarray, list]:
    """Load *checkpoint_path* and return per-frame predictions.

    Parameters
    ----------
    checkpoint_path : str
        Path to a .pt file saved by ml_trainer.train_model().
    feature_matrix  : (N, F) float array
        The same feature columns used during training (already normalised).
    seq_len         : int
        Window length used when training an LSTM (ignored for MLP).

    Returns
    -------
    predictions : (N,) int array — predicted class index per frame
    class_names : list of str / int — le_classes from the checkpoint
    """
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_type = ckpt["model_type"]
    input_dim  = ckpt["input_dim"]
    hidden     = ckpt["hidden"]
    n_classes  = ckpt["n_classes"]
    le_classes = ckpt["le_classes"]

    # ── Rebuild model ─────────────────────────────────────────────────────────
    if model_type == "LSTM":
        from ethoscore_tab.core.ml_trainer import _LSTMClassifier
        model = _LSTMClassifier(input_dim, hidden, n_classes)
    else:
        from ethoscore_tab.core.ml_trainer import _build_mlp
        model = _build_mlp(input_dim, hidden, n_classes)

    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # ── Prepare features ──────────────────────────────────────────────────────
    N = len(feature_matrix)
    X = feature_matrix.astype(np.float32)

    if model_type == "LSTM":
        # Sliding window — frames before seq_len get the first window's label
        pad = np.tile(X[0:1], (seq_len - 1, 1))
        X_padded = np.concatenate([pad, X], axis=0)
        n_windows = N
        windows = np.lib.stride_tricks.sliding_window_view(
            X_padded, (seq_len, X.shape[1])
        )[:, :, 0, :]  # (N, seq_len, F)
        X_tensor = torch.from_numpy(windows.astype(np.float32))
    else:
        X_tensor = torch.from_numpy(X)

    # ── Inference ─────────────────────────────────────────────────────────────
    with torch.no_grad():
        logits = model(X_tensor)
        preds  = logits.argmax(dim=1).numpy().astype(np.int32)

    return preds, [str(c) for c in le_classes]
