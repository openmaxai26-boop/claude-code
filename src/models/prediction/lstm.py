"""
Bidirectional LSTM Predictor with Bahdanau-style Additive Attention
and MC-Dropout uncertainty estimation.

Output heads:
  direction_prob   -> sigmoid (P(up move))
  expected_return  -> linear
  predicted_vol    -> softplus (always positive)
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


# -----------------------------------------------------------------------
# Bahdanau additive attention
# -----------------------------------------------------------------------

class BahdanauAttention(nn.Module):
    """
    Additive (Bahdanau) attention over a sequence of hidden states.

    score(h_t) = v^T tanh(W_1 h_t + b)
    alpha       = softmax(score)
    context     = sum_t alpha_t * h_t
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        hidden_states : (batch, seq_len, hidden_dim)

        Returns
        -------
        context  : (batch, hidden_dim)
        alpha    : (batch, seq_len)   attention weights
        """
        energy = torch.tanh(self.W(hidden_states))      # (B, T, H)
        scores = self.v(energy).squeeze(-1)              # (B, T)
        alpha = torch.softmax(scores, dim=-1)            # (B, T)
        context = (alpha.unsqueeze(-1) * hidden_states).sum(dim=1)  # (B, H)
        return context, alpha


# -----------------------------------------------------------------------
# LSTM Predictor
# -----------------------------------------------------------------------

class LSTMPredictor(nn.Module):
    """
    Bidirectional LSTM with attention and three output heads.

    Architecture
    ------------
    Input (B, T, input_dim)
    -> BiLSTM(hidden_size=256, num_layers=3, dropout=0.3)
    -> BahdanauAttention over time steps
    -> context vector (B, 512)  [bi-directional: 2 * hidden_size]
    -> direction_head  : FC -> sigmoid
    -> magnitude_head  : FC -> linear
    -> volatility_head : FC -> softplus
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        attn_dim = hidden_size * 2  # bidirectional
        self.attention = BahdanauAttention(attn_dim)

        # Dropout for MC-Dropout inference
        self.mc_dropout = nn.Dropout(p=dropout)

        # Output heads
        self.direction_head = nn.Sequential(
            nn.Linear(attn_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.magnitude_head = nn.Sequential(
            nn.Linear(attn_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.volatility_head = nn.Sequential(
            nn.Linear(attn_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self._device)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        x : (batch, seq_len, input_dim)

        Returns
        -------
        dict with:
            direction_prob   (B, 1)
            expected_return  (B, 1)
            predicted_vol    (B, 1)
            attention_weights (B, T)
        """
        lstm_out, _ = self.lstm(x)              # (B, T, 2*H)
        lstm_out = self.mc_dropout(lstm_out)

        context, attn_weights = self.attention(lstm_out)  # (B, 2H), (B, T)
        context = self.mc_dropout(context)

        direction_prob = self.direction_head(context)
        expected_return = self.magnitude_head(context)
        predicted_vol = self.volatility_head(context)

        return {
            "direction_prob": direction_prob,
            "expected_return": expected_return,
            "predicted_vol": predicted_vol,
            "attention_weights": attn_weights,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.tensor(x, dtype=torch.float32).to(self._device)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int = 50,
        lr: float = 1e-3,
        patience: int = 10,
        max_norm: float = 1.0,
    ) -> Dict[str, list]:
        """
        Train with multi-task loss:
            total = BCE(direction) + Huber(return) + MSE(vol)

        Optimizer  : Adam
        Scheduler  : ReduceLROnPlateau (val loss)
        Clipping   : gradient max norm = 1.0

        Returns
        -------
        history dict
        """
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5, min_lr=1e-6
        )
        bce = nn.BCELoss()
        huber = nn.HuberLoss()
        mse = nn.MSELoss()

        history: Dict[str, list] = {"train_loss": [], "val_loss": []}
        best_val = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(1, epochs + 1):
            self.train()
            train_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                x_batch, dir_batch, ret_batch, vol_batch = [
                    b.to(self._device) for b in batch
                ]
                optimizer.zero_grad()
                out = self(x_batch)

                loss = (
                    bce(out["direction_prob"], dir_batch)
                    + huber(out["expected_return"], ret_batch)
                    + mse(out["predicted_vol"], vol_batch)
                )
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), max_norm)
                optimizer.step()

                train_loss += loss.item()
                n_batches += 1

            train_loss /= max(n_batches, 1)

            # Validation
            val_loss = self._eval_loss(val_loader, bce, huber, mse)
            scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_state is not None:
            self.load_state_dict(best_state)

        return history

    def _eval_loss(
        self,
        loader: torch.utils.data.DataLoader,
        bce: nn.Module,
        huber: nn.Module,
        mse: nn.Module,
    ) -> float:
        self.eval()
        total = 0.0
        n = 0
        with torch.no_grad():
            for batch in loader:
                x_batch, dir_batch, ret_batch, vol_batch = [
                    b.to(self._device) for b in batch
                ]
                out = self(x_batch)
                loss = (
                    bce(out["direction_prob"], dir_batch)
                    + huber(out["expected_return"], ret_batch)
                    + mse(out["predicted_vol"], vol_batch)
                )
                total += loss.item()
                n += 1
        return total / max(n, 1)

    # ------------------------------------------------------------------
    # Prediction with uncertainty
    # ------------------------------------------------------------------
    def predict(
        self,
        x: np.ndarray,
        n_samples: int = 50,
    ) -> Dict[str, np.ndarray]:
        """
        Point predictions plus MC-Dropout uncertainty estimates.

        Parameters
        ----------
        x : np.ndarray (batch, seq_len, input_dim)

        Returns
        -------
        dict with mean/std for each output head
        """
        return self.compute_uncertainty(x, n_samples=n_samples)

    def compute_uncertainty(
        self,
        x: np.ndarray,
        n_samples: int = 50,
    ) -> Dict[str, np.ndarray]:
        """
        MC-Dropout: run forward pass n_samples times with dropout active.

        Returns
        -------
        dict:
            direction_prob_mean, direction_prob_std
            expected_return_mean, expected_return_std
            predicted_vol_mean, predicted_vol_std
        """
        # Enable dropout during inference
        self.train()
        x_tensor = self._to_tensor(x)

        dir_samples = []
        ret_samples = []
        vol_samples = []

        with torch.no_grad():
            for _ in range(n_samples):
                out = self(x_tensor)
                dir_samples.append(out["direction_prob"].cpu().numpy())
                ret_samples.append(out["expected_return"].cpu().numpy())
                vol_samples.append(out["predicted_vol"].cpu().numpy())

        self.eval()

        dir_arr = np.stack(dir_samples, axis=0)   # (S, B, 1)
        ret_arr = np.stack(ret_samples, axis=0)
        vol_arr = np.stack(vol_samples, axis=0)

        return {
            "direction_prob_mean": dir_arr.mean(axis=0).squeeze(-1),
            "direction_prob_std": dir_arr.std(axis=0).squeeze(-1),
            "expected_return_mean": ret_arr.mean(axis=0).squeeze(-1),
            "expected_return_std": ret_arr.std(axis=0).squeeze(-1),
            "predicted_vol_mean": vol_arr.mean(axis=0).squeeze(-1),
            "predicted_vol_std": vol_arr.std(axis=0).squeeze(-1),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "input_dim": self.input_dim,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout_p,
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "LSTMPredictor":
        ckpt = torch.load(path, map_location="cpu")
        obj = cls(
            input_dim=ckpt["input_dim"],
            hidden_size=ckpt["hidden_size"],
            num_layers=ckpt["num_layers"],
            dropout=ckpt["dropout"],
        )
        obj.load_state_dict(ckpt["state_dict"])
        return obj
