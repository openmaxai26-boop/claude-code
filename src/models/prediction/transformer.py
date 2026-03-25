"""
Transformer Predictor for financial time-series forecasting.

Architecture:
  - Learnable positional embeddings
  - 6-layer Transformer Encoder (8 heads, d_model=256, ffn=1024)
  - Causal (upper-triangular) masking
  - Attention-weighted temporal pooling
  - Three output heads: direction, magnitude, volatility
  - MC-Dropout uncertainty estimation
  - Warmup + inverse-sqrt learning rate schedule
"""

from __future__ import annotations

import math
import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# -----------------------------------------------------------------------
# LR scheduler with warmup then inverse-sqrt decay
# -----------------------------------------------------------------------

class WarmupInvSqrtScheduler(optim.lr_scheduler._LRScheduler):
    """
    lr(step) = d_model^{-0.5} * min(step^{-0.5}, step * warmup^{-1.5})
    Follows Vaswani et al. (2017) "Attention Is All You Need".
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        d_model: int,
        warmup_steps: int = 4000,
        last_epoch: int = -1,
    ) -> None:
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self._step_count_custom = 0
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        self._step_count_custom += 1
        step = max(self._step_count_custom, 1)
        scale = self.d_model ** -0.5 * min(
            step ** -0.5, step * self.warmup_steps ** -1.5
        )
        return [scale for _ in self.base_lrs]


# -----------------------------------------------------------------------
# Attention-weighted temporal pooling
# -----------------------------------------------------------------------

class TemporalAttentionPooling(nn.Module):
    """
    Compute a scalar weight per time step and return a weighted sum.

    score(t) = w^T tanh(U h_t)
    alpha    = softmax(score)
    output   = sum_t alpha_t * h_t
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.U = nn.Linear(d_model, d_model)
        self.w = nn.Linear(d_model, 1, bias=False)

    def forward(self, enc_out: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        enc_out : (B, T, d_model)

        Returns
        -------
        pooled  : (B, d_model)
        """
        energy = torch.tanh(self.U(enc_out))   # (B, T, D)
        scores = self.w(energy).squeeze(-1)     # (B, T)
        alpha = torch.softmax(scores, dim=-1)   # (B, T)
        pooled = (alpha.unsqueeze(-1) * enc_out).sum(dim=1)  # (B, D)
        return pooled


# -----------------------------------------------------------------------
# Main model
# -----------------------------------------------------------------------

class TransformerPredictor(nn.Module):
    """
    Causal Transformer Encoder for sequence-to-scalar prediction.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_p = dropout
        self.max_seq_len = max_seq_len

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Learnable positional embeddings
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model),
        )

        # MC-Dropout layer
        self.mc_dropout = nn.Dropout(p=dropout)

        # Temporal pooling
        self.pooling = TemporalAttentionPooling(d_model)

        # Output heads
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.magnitude_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.volatility_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self._device)
        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ------------------------------------------------------------------
    # Causal mask
    # ------------------------------------------------------------------
    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular boolean mask: True = masked (ignored)."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, T, input_dim)

        Returns
        -------
        dict: direction_prob (B,1), expected_return (B,1), predicted_vol (B,1)
        """
        B, T, _ = x.shape

        # Input projection
        x_proj = self.input_proj(x)  # (B, T, D)

        # Add positional embeddings
        positions = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        positions = positions.clamp(max=self.max_seq_len - 1)
        x_proj = x_proj + self.pos_embedding(positions)             # (B, T, D)

        # Causal mask
        causal_mask = self._causal_mask(T, x.device)

        # Transformer encoding
        enc_out = self.transformer_encoder(
            x_proj, mask=causal_mask
        )  # (B, T, D)
        enc_out = self.mc_dropout(enc_out)

        # Temporal attention pooling
        pooled = self.pooling(enc_out)  # (B, D)
        pooled = self.mc_dropout(pooled)

        return {
            "direction_prob": self.direction_head(pooled),
            "expected_return": self.magnitude_head(pooled),
            "predicted_vol": self.volatility_head(pooled),
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
        lr: float = 1.0,
        warmup_steps: int = 4000,
        patience: int = 10,
        max_norm: float = 1.0,
    ) -> Dict[str, list]:
        """
        Train with warmup + inverse-sqrt LR schedule.

        Loss = BCE(direction) + Huber(return) + MSE(vol)
        """
        optimizer = optim.Adam(
            self.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-5
        )
        scheduler = WarmupInvSqrtScheduler(
            optimizer, d_model=self.d_model, warmup_steps=warmup_steps
        )

        bce = nn.BCELoss()
        huber = nn.HuberLoss()
        mse = nn.MSELoss()

        history: Dict[str, list] = {"train_loss": [], "val_loss": []}
        best_val = float("inf")
        patience_counter = 0
        best_state = None
        global_step = 0

        for epoch in range(1, epochs + 1):
            self.train()
            epoch_loss = 0.0
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
                scheduler.step()
                epoch_loss += loss.item()
                n_batches += 1
                global_step += 1

            epoch_loss /= max(n_batches, 1)
            val_loss = self._eval_loss(val_loader, bce, huber, mse)

            history["train_loss"].append(epoch_loss)
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
    # Prediction with MC-Dropout uncertainty
    # ------------------------------------------------------------------
    def predict(
        self,
        x: np.ndarray,
        n_samples: int = 50,
    ) -> Dict[str, np.ndarray]:
        return self.compute_uncertainty(x, n_samples=n_samples)

    def compute_uncertainty(
        self,
        x: np.ndarray,
        n_samples: int = 50,
    ) -> Dict[str, np.ndarray]:
        """MC-Dropout: enable dropout during inference."""
        self.train()
        x_tensor = self._to_tensor(x)

        dir_s, ret_s, vol_s = [], [], []
        with torch.no_grad():
            for _ in range(n_samples):
                out = self(x_tensor)
                dir_s.append(out["direction_prob"].cpu().numpy())
                ret_s.append(out["expected_return"].cpu().numpy())
                vol_s.append(out["predicted_vol"].cpu().numpy())

        self.eval()

        return {
            "direction_prob_mean": np.stack(dir_s).mean(axis=0).squeeze(-1),
            "direction_prob_std": np.stack(dir_s).std(axis=0).squeeze(-1),
            "expected_return_mean": np.stack(ret_s).mean(axis=0).squeeze(-1),
            "expected_return_std": np.stack(ret_s).std(axis=0).squeeze(-1),
            "predicted_vol_mean": np.stack(vol_s).mean(axis=0).squeeze(-1),
            "predicted_vol_std": np.stack(vol_s).std(axis=0).squeeze(-1),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "input_dim": self.input_dim,
                "d_model": self.d_model,
                "nhead": self.nhead,
                "num_encoder_layers": self.num_encoder_layers,
                "dim_feedforward": self.dim_feedforward,
                "dropout": self.dropout_p,
                "max_seq_len": self.max_seq_len,
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "TransformerPredictor":
        ckpt = torch.load(path, map_location="cpu")
        obj = cls(
            input_dim=ckpt["input_dim"],
            d_model=ckpt["d_model"],
            nhead=ckpt["nhead"],
            num_encoder_layers=ckpt["num_encoder_layers"],
            dim_feedforward=ckpt["dim_feedforward"],
            dropout=ckpt["dropout"],
            max_seq_len=ckpt["max_seq_len"],
        )
        obj.load_state_dict(ckpt["state_dict"])
        return obj
