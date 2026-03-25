"""
Ensemble Predictor
Combines LSTM + Transformer + CNN predictions with:
  - Regime-adaptive weighting
  - Scipy-optimised weight fitting on validation Sharpe
  - Platt scaling calibration
  - Final signal generation with confidence and risk scoring
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .lstm import LSTMPredictor
from .transformer import TransformerPredictor
from .cnn import CNNPredictor


# -----------------------------------------------------------------------
# Regime-specific base weights
# -----------------------------------------------------------------------

# Each row: [lstm_w, transformer_w, cnn_w] (sum to 1)
_REGIME_BASE_WEIGHTS: Dict[str, List[float]] = {
    "BULL": [0.40, 0.35, 0.25],      # LSTM excels at trends
    "BEAR": [0.35, 0.35, 0.30],      # balanced, slight CNN boost
    "RANGE": [0.25, 0.35, 0.40],     # Transformer + CNN for mean-reversion patterns
    "HIGH_VOL": [0.20, 0.30, 0.50],  # CNN best at fast pattern recognition
    "UNKNOWN": [1 / 3, 1 / 3, 1 / 3],
}


# -----------------------------------------------------------------------
# Ensemble
# -----------------------------------------------------------------------

class EnsemblePredictor:
    """
    Meta-learner that combines LSTM, Transformer and CNN predictions.

    Weight optimisation: scipy L-BFGS-B minimises negative Sharpe on
    the validation set by finding the best convex combination of models.

    Platt scaling: isotonic or logistic regression maps raw direction
    probabilities to calibrated probabilities.
    """

    def __init__(
        self,
        lstm: LSTMPredictor,
        transformer: TransformerPredictor,
        cnn: CNNPredictor,
        weights: Optional[np.ndarray] = None,
    ) -> None:
        self.lstm = lstm
        self.transformer = transformer
        self.cnn = cnn

        # Base (equal) weights; updated by fit_weights
        if weights is not None:
            weights = np.asarray(weights, dtype=float)
            self.weights: np.ndarray = weights / weights.sum()
        else:
            self.weights = np.array([1 / 3, 1 / 3, 1 / 3])

        # Platt scaling models (one per regime or global)
        self._calibrator: Optional[LogisticRegression] = None
        self._calibrator_scaler: Optional[StandardScaler] = None
        self._calibrated: bool = False

    # ------------------------------------------------------------------
    # Individual model predictions (returns raw numpy dicts)
    # ------------------------------------------------------------------
    def _collect_predictions(
        self,
        x: np.ndarray,
        n_mc: int = 50,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        return {
            "lstm": self.lstm.predict(x, n_samples=n_mc),
            "transformer": self.transformer.predict(x, n_samples=n_mc),
            "cnn": self.cnn.predict(x, n_samples=n_mc),
        }

    # ------------------------------------------------------------------
    # Weighted combination
    # ------------------------------------------------------------------
    def _weighted_combine(
        self,
        preds: Dict[str, Dict[str, np.ndarray]],
        weights: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        keys = ["direction_prob_mean", "expected_return_mean", "predicted_vol_mean",
                "direction_prob_std", "expected_return_std", "predicted_vol_std"]

        combined: Dict[str, np.ndarray] = {}
        model_keys = ["lstm", "transformer", "cnn"]

        for key in keys:
            stacked = np.stack(
                [preds[m][key] for m in model_keys], axis=-1
            )  # (..., 3)
            combined[key] = (stacked * weights).sum(axis=-1)

        return combined

    # ------------------------------------------------------------------
    # Fit weights via Sharpe optimisation
    # ------------------------------------------------------------------
    def fit_weights(
        self,
        val_loader: torch.utils.data.DataLoader,
        n_mc: int = 20,
    ) -> np.ndarray:
        """
        Find ensemble weights that maximise Sharpe ratio on the validation set.

        Optimisation: L-BFGS-B on simplex (weights >= 0, sum = 1) using a
        negative Sharpe objective computed from direction-probability-signed
        return predictions.

        Returns
        -------
        Optimised weights array of shape (3,).
        """
        # Collect all validation batch predictions
        all_preds: Dict[str, List[np.ndarray]] = {
            "lstm": [], "transformer": [], "cnn": []
        }
        all_actual_returns: List[np.ndarray] = []

        device = next(self.lstm.parameters()).device

        for batch in val_loader:
            x_batch, dir_batch, ret_batch, vol_batch = [b.numpy() for b in batch]

            preds = self._collect_predictions(x_batch, n_mc=n_mc)
            for m in ["lstm", "transformer", "cnn"]:
                all_preds[m].append(preds[m]["direction_prob_mean"])

            all_actual_returns.append(ret_batch.squeeze(-1))

        # Concatenate
        model_dir: Dict[str, np.ndarray] = {
            m: np.concatenate(v) for m, v in all_preds.items()
        }
        actual_ret = np.concatenate(all_actual_returns)

        lstm_dir = model_dir["lstm"]
        trans_dir = model_dir["transformer"]
        cnn_dir = model_dir["cnn"]

        def neg_sharpe(w: np.ndarray) -> float:
            w = np.abs(w)
            w = w / (w.sum() + 1e-12)
            combined_dir = w[0] * lstm_dir + w[1] * trans_dir + w[2] * cnn_dir
            # Signal: buy if dir > 0.5, sell otherwise
            signal = np.where(combined_dir > 0.5, 1.0, -1.0)
            strategy_ret = signal * actual_ret
            mean_r = strategy_ret.mean()
            std_r = strategy_ret.std()
            if std_r < 1e-10:
                return 0.0
            sharpe = mean_r / std_r * np.sqrt(252)
            return -sharpe

        best_val = np.inf
        best_w = self.weights.copy()

        # Multiple random restarts
        rng = np.random.default_rng(42)
        for _ in range(10):
            x0 = rng.dirichlet([1, 1, 1])
            res = minimize(
                neg_sharpe,
                x0=x0,
                method="L-BFGS-B",
                bounds=[(0.0, 1.0)] * 3,
                options={"maxiter": 500},
            )
            if res.fun < best_val:
                best_val = res.fun
                best_w = np.abs(res.x)
                best_w /= best_w.sum() + 1e-12

        self.weights = best_w
        return self.weights

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(
        self,
        x: np.ndarray,
        regime: Optional[str] = None,
        n_mc: int = 50,
    ) -> Dict[str, np.ndarray]:
        """
        Weighted average of model predictions, optionally regime-adjusted.

        Parameters
        ----------
        x      : (B, T, input_dim) input array
        regime : one of 'BULL', 'BEAR', 'RANGE', 'HIGH_VOL', or None
        n_mc   : MC-Dropout samples

        Returns
        -------
        Combined prediction dict.
        """
        if regime is not None and regime in _REGIME_BASE_WEIGHTS:
            base = np.array(_REGIME_BASE_WEIGHTS[regime], dtype=float)
            # Blend 50/50 with learned weights
            w = 0.5 * base + 0.5 * self.weights
            w /= w.sum()
        else:
            w = self.weights

        preds = self._collect_predictions(x, n_mc=n_mc)
        return self._weighted_combine(preds, w)

    # ------------------------------------------------------------------
    # Final signal generation
    # ------------------------------------------------------------------
    def compute_final_signal(
        self,
        predictions_dict: Dict[str, np.ndarray],
        attention_weights: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> Dict:
        """
        Derive a structured trading signal from ensemble predictions.

        Parameters
        ----------
        predictions_dict : output of predict()
        attention_weights : (T,) array of temporal attention weights (optional)
        feature_names     : list of feature column names (optional)

        Returns
        -------
        dict with keys:
            signal          : 'BUY' | 'SELL' | 'HOLD'
            probability     : float 0-100
            confidence_score: float 0-1
            risk_level      : 'LOW' | 'MEDIUM' | 'HIGH'
            factors         : list of str describing top contributing factors
        """
        dir_mean = float(np.mean(predictions_dict["direction_prob_mean"]))
        dir_std = float(np.mean(predictions_dict["direction_prob_std"]))
        ret_mean = float(np.mean(predictions_dict["expected_return_mean"]))
        vol_mean = float(np.mean(predictions_dict["predicted_vol_mean"]))
        ret_std = float(np.mean(predictions_dict["expected_return_std"]))

        # Calibrate if available
        if self._calibrated and self._calibrator is not None:
            raw = np.array([[dir_mean, dir_std, ret_mean, vol_mean]])
            scaled = self._calibrator_scaler.transform(raw)
            dir_mean = float(self._calibrator.predict_proba(scaled)[0, 1])

        # Signal
        HOLD_ZONE = 0.07  # within 0.5 +/- this -> HOLD
        if dir_mean > 0.5 + HOLD_ZONE:
            signal = "BUY"
        elif dir_mean < 0.5 - HOLD_ZONE:
            signal = "SELL"
        else:
            signal = "HOLD"

        probability = round(dir_mean * 100, 2)

        # Confidence: penalise high epistemic uncertainty and low |dir - 0.5|
        conviction = abs(dir_mean - 0.5) * 2  # 0-1
        uncertainty_penalty = min(dir_std * 10, 1.0)
        confidence_score = round(float(conviction * (1 - uncertainty_penalty)), 4)
        confidence_score = max(0.0, min(1.0, confidence_score))

        # Risk level: based on predicted volatility and model disagreement
        annualized_vol = vol_mean  # already annualized from training targets
        disagreement = dir_std

        if annualized_vol > 0.35 or disagreement > 0.15:
            risk_level = "HIGH"
        elif annualized_vol > 0.20 or disagreement > 0.08:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # Top factors from attention weights
        factors: List[str] = []
        if attention_weights is not None and feature_names is not None:
            if len(attention_weights) == len(feature_names):
                top_idx = np.argsort(attention_weights)[::-1][:5]
                factors = [feature_names[i] for i in top_idx]
        elif attention_weights is not None:
            top_idx = np.argsort(attention_weights)[::-1][:5]
            factors = [f"feature_{i}" for i in top_idx]

        if not factors:
            factors = [
                f"direction_prob={dir_mean:.3f}",
                f"expected_return={ret_mean:.4f}",
                f"predicted_vol={vol_mean:.3f}",
                f"uncertainty={dir_std:.3f}",
            ]

        return {
            "signal": signal,
            "probability": probability,
            "confidence_score": confidence_score,
            "risk_level": risk_level,
            "factors": factors,
            "raw": {
                "direction_prob": dir_mean,
                "direction_std": dir_std,
                "expected_return": ret_mean,
                "return_std": ret_std,
                "predicted_vol": vol_mean,
            },
        }

    # ------------------------------------------------------------------
    # Platt scaling calibration
    # ------------------------------------------------------------------
    def calibrate(
        self,
        val_loader: torch.utils.data.DataLoader,
        n_mc: int = 20,
    ) -> None:
        """
        Fit a logistic regression (Platt scaling) on raw ensemble probabilities
        and auxiliary features to produce calibrated direction probabilities.
        """
        raw_features: List[np.ndarray] = []
        true_directions: List[np.ndarray] = []

        for batch in val_loader:
            x_batch, dir_batch, ret_batch, vol_batch = [b.numpy() for b in batch]
            preds = self.predict(x_batch, n_mc=n_mc)

            dir_m = preds["direction_prob_mean"]
            dir_s = preds["direction_prob_std"]
            ret_m = preds["expected_return_mean"]
            vol_m = preds["predicted_vol_mean"]

            feats = np.stack([dir_m, dir_s, ret_m, vol_m], axis=-1)
            raw_features.append(feats)
            true_directions.append((dir_batch.squeeze(-1) > 0.5).astype(int))

        X_cal = np.concatenate(raw_features, axis=0)
        y_cal = np.concatenate(true_directions, axis=0)

        scaler = StandardScaler()
        X_cal_scaled = scaler.fit_transform(X_cal)

        lr = LogisticRegression(C=1.0, max_iter=1000)
        lr.fit(X_cal_scaled, y_cal)

        self._calibrator = lr
        self._calibrator_scaler = scaler
        self._calibrated = True

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """
        Save all sub-models and ensemble metadata to a directory.

        Parameters
        ----------
        path : directory path (created if absent)
        """
        os.makedirs(path, exist_ok=True)
        self.lstm.save(os.path.join(path, "lstm.pt"))
        self.transformer.save(os.path.join(path, "transformer.pt"))
        self.cnn.save(os.path.join(path, "cnn.pt"))

        import joblib
        meta = {
            "weights": self.weights,
            "calibrator": self._calibrator,
            "calibrator_scaler": self._calibrator_scaler,
            "calibrated": self._calibrated,
        }
        joblib.dump(meta, os.path.join(path, "ensemble_meta.joblib"))

    @classmethod
    def load(cls, path: str) -> "EnsemblePredictor":
        """
        Load ensemble from a previously saved directory.

        Parameters
        ----------
        path : directory containing lstm.pt, transformer.pt, cnn.pt,
               ensemble_meta.joblib
        """
        import joblib

        lstm = LSTMPredictor.load(os.path.join(path, "lstm.pt"))
        transformer = TransformerPredictor.load(os.path.join(path, "transformer.pt"))
        cnn = CNNPredictor.load(os.path.join(path, "cnn.pt"))

        meta = joblib.load(os.path.join(path, "ensemble_meta.joblib"))

        obj = cls(lstm=lstm, transformer=transformer, cnn=cnn, weights=meta["weights"])
        obj._calibrator = meta["calibrator"]
        obj._calibrator_scaler = meta["calibrator_scaler"]
        obj._calibrated = meta["calibrated"]
        return obj
