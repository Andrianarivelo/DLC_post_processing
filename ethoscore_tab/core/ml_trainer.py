"""PyTorch-based behavior classifier training.

Supports two architectures:
  MLP   — simple multi-layer perceptron (frame-level)
  LSTM  — sequence model (windowed frames, captures temporal context)

Training procedure
------------------
1. Optional k-fold cross-validation (sklearn StratifiedKFold).
2. Adam optimiser, CrossEntropyLoss for multi-class.
3. Per-epoch validation F1 + accuracy logged.
4. Best checkpoint saved by validation F1.

The module avoids top-level torch imports to skip CUDA init at startup.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Model definitions ─────────────────────────────────────────────────────────

def _build_mlp(input_dim: int, hidden: int, n_classes: int):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden, hidden // 2),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden // 2, n_classes),
    )


class _LSTMClassifier:
    """Thin wrapper so LSTM follows same .forward() interface as MLP."""

    def __new__(cls, input_dim: int, hidden: int, n_classes: int, n_layers: int = 2):
        import torch.nn as nn

        class _LSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(input_dim, hidden, n_layers,
                                    batch_first=True, dropout=0.3)
                self.head = nn.Linear(hidden, n_classes)

            def forward(self, x):
                # x: (batch, seq_len, features)
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :])  # last time step

        return _LSTM()


# ── Public training API ───────────────────────────────────────────────────────

def train_model(
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
    progress_callback=None,   # fn(epoch, train_loss, val_f1)
) -> dict:
    """Train a classifier and return performance metrics.

    Parameters
    ----------
    feature_matrix : (N, F) float array
    labels         : (N,) int array (0 = background / unlabelled)
    model_type     : "MLP" or "LSTM"
    hidden         : hidden layer size
    seq_len        : sequence length for LSTM windowing
    epochs         : number of training epochs
    lr             : Adam learning rate
    train_frac     : fraction of data used for training (when n_folds=1)
    n_folds        : number of cross-validation folds (1 = single split)
    batch_size     : mini-batch size
    checkpoint_dir : directory for best model checkpoint; uses tempdir if None
    progress_callback : optional fn(epoch_int, train_loss_float, val_f1_float)

    Returns
    -------
    dict with keys: fold_results, mean_f1, std_f1, mean_acc, std_acc,
                    best_checkpoint, n_classes, class_names
    """
    import torch
    import torch.nn as nn
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score, accuracy_score
    from sklearn.preprocessing import LabelEncoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on %s", device)

    # ── Prepare data ──────────────────────────────────────────────────────────
    X = feature_matrix.astype(np.float32)
    y_raw = labels.astype(np.int64)

    # Encode labels to 0..N-1
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    n_classes = len(le.classes_)
    input_dim = X.shape[1]

    logger.info("Classes: %s  n=%d  features=%d", le.classes_, len(X), input_dim)

    if model_type == "LSTM":
        X, y = _make_windows(X, y, seq_len)

    # ── Cross-validation ──────────────────────────────────────────────────────
    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else Path(tempfile.mkdtemp())
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = str(ckpt_dir / "best_model.pt")
    best_global_f1 = -1.0

    fold_results = []

    if n_folds <= 1:
        splits = _single_split(y, train_frac)
    else:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        splits = list(skf.split(X, y))

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx],  y[val_idx]

        # Build model
        if model_type == "LSTM":
            model = _LSTMClassifier(input_dim, hidden, n_classes)
        else:
            model = _build_mlp(input_dim, hidden, n_classes)
        model = model.to(device)

        optim = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        X_tr_t  = torch.from_numpy(X_tr).to(device)
        y_tr_t  = torch.from_numpy(y_tr).to(device)
        X_val_t = torch.from_numpy(X_val).to(device)
        y_val_t = torch.from_numpy(y_val).to(device)

        best_val_f1 = -1.0
        best_state  = None

        for epoch in range(epochs):
            model.train()
            # Mini-batch shuffle
            perm = torch.randperm(len(X_tr_t))
            total_loss = 0.0
            for i in range(0, len(perm), batch_size):
                idx_b = perm[i:i+batch_size]
                xb = X_tr_t[idx_b]
                yb = y_tr_t[idx_b]
                optim.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optim.step()
                total_loss += loss.item() * len(idx_b)
            train_loss = total_loss / max(len(X_tr_t), 1)

            # Validation
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_t)
                val_preds  = val_logits.argmax(dim=1).cpu().numpy()
            val_f1  = f1_score(y_val, val_preds, average="macro", zero_division=0)
            val_acc = accuracy_score(y_val, val_preds)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if progress_callback:
                progress_callback(epoch + 1, train_loss, val_f1)

            logger.debug(
                "Fold %d  epoch %d/%d  loss=%.4f  val_f1=%.3f  val_acc=%.3f",
                fold_idx + 1, epoch + 1, epochs, train_loss, val_f1, val_acc,
            )

        # Restore best weights for this fold's final eval
        if best_state:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            final_preds = val_logits.argmax(dim=1).cpu().numpy()
        final_f1  = f1_score(y_val, final_preds, average="macro", zero_division=0)
        final_acc = accuracy_score(y_val, final_preds)
        fold_results.append({"fold": fold_idx + 1, "f1": final_f1, "acc": final_acc})

        # Save globally best checkpoint
        if final_f1 > best_global_f1:
            best_global_f1 = final_f1
            torch.save({"state_dict": best_state, "model_type": model_type,
                        "input_dim": input_dim, "hidden": hidden,
                        "n_classes": n_classes, "le_classes": le.classes_.tolist()},
                       best_ckpt)

        logger.info("Fold %d — F1=%.3f  Acc=%.3f", fold_idx + 1, final_f1, final_acc)

    f1s  = [r["f1"] for r in fold_results]
    accs = [r["acc"] for r in fold_results]

    return {
        "fold_results":     fold_results,
        "mean_f1":          float(np.mean(f1s)),
        "std_f1":           float(np.std(f1s)),
        "mean_acc":         float(np.mean(accs)),
        "std_acc":          float(np.std(accs)),
        "best_checkpoint":  best_ckpt,
        "n_classes":        n_classes,
        "class_names":      le.classes_.tolist(),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_windows(X: np.ndarray, y: np.ndarray, seq_len: int):
    """Reshape (N, F) + (N,) → (N_windows, seq_len, F) + (N_windows,)."""
    n = len(X)
    indices = np.arange(seq_len - 1, n)
    windows = np.lib.stride_tricks.sliding_window_view(X, (seq_len, X.shape[1]))[..., 0, :]
    labels  = y[indices]
    return windows, labels


def _single_split(y: np.ndarray, train_frac: float):
    n = len(y)
    n_train = int(n * train_frac)
    train_idx = np.arange(n_train)
    val_idx   = np.arange(n_train, n)
    return [(train_idx, val_idx)]
