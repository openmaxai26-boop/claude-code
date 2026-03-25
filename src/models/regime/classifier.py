"""
Deep-Learning Regime Classifier (PyTorch)
Four-class classifier: BULL=0, BEAR=1, RANGE=2, HIGH_VOL=3
Includes label generation from returns + volatility for supervised training.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


# -----------------------------------------------------------------------
# Label generation helper
# -----------------------------------------------------------------------

def label_regimes_from_returns(
    returns: np.ndarray,
    volatility: np.ndarray,
) -> np.ndarray:
    """
    Generate regime labels from return and volatility arrays using
    threshold-based heuristics (designed to match HMM output labels).

    Labels
    ------
    0 : BULL     – positive return, below-median vol
    1 : BEAR     – negative return, above-median vol
    2 : RANGE    – small |return|, below-median vol
    3 : HIGH_VOL – above 75th-pct vol regardless of direction

    Parameters
    ----------
    returns : np.ndarray shape (T,), log returns
    volatility : np.ndarray shape (T,), realized vol (annualized)

    Returns
    -------
    np.ndarray shape (T,) of int labels
    """
    returns = np.asarray(returns, dtype=float)
    volatility = np.asarray(volatility, dtype=float)

    labels = np.full(len(returns), 2, dtype=int)  # default RANGE

    vol_median = np.nanmedian(volatility)
    vol_75 = np.nanpercentile(volatility, 75)
    ret_threshold = 0.005  # 0.5 % daily

    # HIGH_VOL first (overrides others)
    labels[volatility > vol_75] = 3

    # BULL: positive return, below-median vol
    mask_bull = (returns > ret_threshold) & (volatility <= vol_median)
    labels[mask_bull] = 0

    # BEAR: negative return, above-median vol (but not HIGH_VOL)
    mask_bear = (returns < -ret_threshold) & (volatility > vol_median) & (volatility <= vol_75)
    labels[mask_bear] = 1

    return labels


# -----------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------

class RegimeClassifier(nn.Module):
    """
    Feed-forward network for supervised regime classification.

    Architecture:
        input_dim -> Linear(128) -> BN -> ReLU -> Dropout(0.3)
                  -> Linear(256) -> BN -> ReLU -> Dropout(0.3)
                  -> Linear(128) -> BN -> ReLU -> Dropout(0.3)
                  -> Linear(4)   -> softmax (implied by CrossEntropyLoss)

    Class labels: 0=BULL, 1=BEAR, 2=RANGE, 3=HIGH_VOL
    """

    LABEL_NAMES = ["BULL", "BEAR", "RANGE", "HIGH_VOL"]

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 4),
        )

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self._device)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _to_tensor(self, X: np.ndarray, dtype=torch.float32) -> torch.Tensor:
        return torch.tensor(X, dtype=dtype).to(self._device)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 256,
        patience: int = 10,
    ) -> Dict[str, list]:
        """
        Train with Adam optimizer, CrossEntropyLoss, CosineAnnealingLR and
        early stopping on validation loss.

        Returns
        -------
        history dict with 'train_loss', 'val_loss', 'val_acc'
        """
        X_tr = self._to_tensor(X_train)
        y_tr = self._to_tensor(y_train, dtype=torch.long)
        X_v = self._to_tensor(X_val)
        y_v = self._to_tensor(y_val, dtype=torch.long)

        dataset = torch.utils.data.TensorDataset(X_tr, y_tr)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )

        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
        criterion = nn.CrossEntropyLoss()

        history: Dict[str, list] = {"train_loss": [], "val_loss": [], "val_acc": []}

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(1, epochs + 1):
            self.train()
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                logits = self(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(X_batch)

            epoch_loss /= len(X_train)
            scheduler.step()

            # Validation
            self.eval()
            with torch.no_grad():
                val_logits = self(X_v)
                val_loss = criterion(val_logits, y_v).item()
                val_preds = val_logits.argmax(dim=1).cpu().numpy()

            val_acc = accuracy_score(y_val, val_preds)

            history["train_loss"].append(epoch_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_state is not None:
            self.load_state_dict(best_state)

        return history

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class probabilities.

        Returns
        -------
        np.ndarray shape (n_samples, 4)
        """
        self.eval()
        with torch.no_grad():
            logits = self(self._to_tensor(X))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return integer regime labels.

        Returns
        -------
        np.ndarray shape (n_samples,) of int in {0,1,2,3}
        """
        return self.predict_proba(X).argmax(axis=1)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict:
        """
        Compute accuracy, macro-F1 and confusion matrix.

        Returns
        -------
        dict with keys: accuracy, f1_macro, f1_per_class, confusion_matrix
        """
        preds = self.predict(X_test)
        acc = float(accuracy_score(y_test, preds))
        f1_macro = float(f1_score(y_test, preds, average="macro", zero_division=0))
        f1_per = f1_score(y_test, preds, average=None, zero_division=0).tolist()
        cm = confusion_matrix(y_test, preds).tolist()

        return {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_per_class": {
                self.LABEL_NAMES[i]: f1_per[i]
                for i in range(min(len(f1_per), 4))
            },
            "confusion_matrix": cm,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Save model weights and hyper-parameters."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "input_dim": self.input_dim,
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "RegimeClassifier":
        """Load a saved RegimeClassifier."""
        checkpoint = torch.load(path, map_location="cpu")
        obj = cls(input_dim=checkpoint["input_dim"])
        obj.load_state_dict(checkpoint["state_dict"])
        return obj
