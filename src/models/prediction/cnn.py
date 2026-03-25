"""
Multi-Scale 1D CNN Predictor for financial time-series forecasting.

Architecture:
  - Three parallel branches: kernel sizes 3, 7, 21
  - Each branch: 3 residual Conv1D blocks with BatchNorm + GELU
  - Concatenate -> global average pooling -> FC(192->256) -> output heads
  - OneCycleLR training schedule
  - MC-Dropout uncertainty estimation
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR


# -----------------------------------------------------------------------
# Residual Conv Block
# -----------------------------------------------------------------------

class ResidualConv1DBlock(nn.Module):
    """
    Single residual block for 1D convolution.

    x -> Conv1D -> BN -> GELU -> Conv1D -> BN -> + skip -> GELU
    Skip: 1x1 conv if channel mismatch, else identity.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        pad = kernel_size // 2  # "same" padding

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(p=dropout)
        self.act_out = nn.GELU()

        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T)"""
        residual = self.skip(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        return self.act_out(out + residual)


# -----------------------------------------------------------------------
# Multi-Scale Branch
# -----------------------------------------------------------------------

class MultiScaleBranch(nn.Module):
    """
    Three stacked ResidualConv1D blocks at a fixed kernel size.
    """

    def __init__(
        self,
        input_dim: int,
        n_filters: int = 64,
        kernel_size: int = 3,
        n_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_ch = input_dim
        for _ in range(n_layers):
            layers.append(
                ResidualConv1DBlock(in_ch, n_filters, kernel_size, dropout)
            )
            in_ch = n_filters
        self.blocks = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, input_dim, T) -> (B, n_filters, T)"""
        return self.blocks(x)


# -----------------------------------------------------------------------
# CNN Predictor
# -----------------------------------------------------------------------

class CNNPredictor(nn.Module):
    """
    Multi-scale 1D CNN with residual connections for market prediction.

    Input  : (B, T, input_dim)   [batch-first, as from DataLoader]
    Output : direction_prob, expected_return, predicted_vol
    """

    def __init__(
        self,
        input_dim: int,
        n_filters: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_filters = n_filters
        self.dropout_p = dropout

        # Three branches with different receptive fields
        self.branch3 = MultiScaleBranch(
            input_dim, n_filters, kernel_size=3, n_layers=3, dropout=dropout
        )
        self.branch7 = MultiScaleBranch(
            input_dim, n_filters, kernel_size=7, n_layers=3, dropout=dropout
        )
        self.branch21 = MultiScaleBranch(
            input_dim, n_filters, kernel_size=21, n_layers=3, dropout=dropout
        )

        concat_dim = n_filters * 3  # 192

        # Fusion FC
        self.fusion = nn.Sequential(
            nn.Linear(concat_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )

        # MC-Dropout
        self.mc_dropout = nn.Dropout(p=dropout)

        # Output heads
        self.direction_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.magnitude_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        self.volatility_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self._device)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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
        dict: direction_prob, expected_return, predicted_vol
        """
        # Transpose to (B, C, T) for Conv1d
        x_t = x.permute(0, 2, 1)  # (B, input_dim, T)

        f3 = self.branch3(x_t)   # (B, n_filters, T)
        f7 = self.branch7(x_t)
        f21 = self.branch21(x_t)

        # Global average pooling over time dimension
        g3 = f3.mean(dim=-1)    # (B, n_filters)
        g7 = f7.mean(dim=-1)
        g21 = f21.mean(dim=-1)

        concat = torch.cat([g3, g7, g21], dim=-1)  # (B, 192)
        concat = self.mc_dropout(concat)

        fused = self.fusion(concat)  # (B, 256)
        fused = self.mc_dropout(fused)

        return {
            "direction_prob": self.direction_head(fused),
            "expected_return": self.magnitude_head(fused),
            "predicted_vol": self.volatility_head(fused),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.tensor(x, dtype=torch.float32).to(self._device)

    # ------------------------------------------------------------------
    # Training with OneCycleLR
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
        Train with OneCycleLR scheduler, multi-task loss, and early stopping.
        """
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * epochs

        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy="cos",
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
    # Prediction with uncertainty
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
        """MC-Dropout uncertainty estimation."""
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
                "n_filters": self.n_filters,
                "dropout": self.dropout_p,
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "CNNPredictor":
        ckpt = torch.load(path, map_location="cpu")
        obj = cls(
            input_dim=ckpt["input_dim"],
            n_filters=ckpt["n_filters"],
            dropout=ckpt["dropout"],
        )
        obj.load_state_dict(ckpt["state_dict"])
        return obj
