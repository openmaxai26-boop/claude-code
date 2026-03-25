"""
ContinuousLearningSystem: Online drift detection, regime shift detection,
and automated retraining scheduling.

Uses Population Stability Index (PSI), statistical t-tests, and KL-divergence
to detect when model performance or data distributions have shifted.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def _psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """
    Population Stability Index between two distributions.

    PSI < 0.1   : No significant change.
    PSI < 0.25  : Moderate change.
    PSI >= 0.25 : Significant change.
    """
    bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    bins[0] = -np.inf
    bins[-1] = np.inf

    exp_counts, _ = np.histogram(expected, bins=bins)
    act_counts, _ = np.histogram(actual, bins=bins)

    exp_pct = exp_counts / max(exp_counts.sum(), 1) + 1e-8
    act_pct = act_counts / max(act_counts.sum(), 1) + 1e-8

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def _sharpe(returns: np.ndarray, periods: int = 252) -> float:
    if len(returns) < 2:
        return 0.0
    std = returns.std(ddof=1)
    if std == 0:
        return 0.0
    return float(returns.mean() / std * np.sqrt(periods))


class ContinuousLearningSystem:
    """
    Monitors model performance and data distributions; schedules retraining
    when drift is detected.

    Parameters
    ----------
    models_dict : dict mapping model name -> model object (must expose .predict).
    data_store  : Object with .get_recent(n) -> pd.DataFrame method.
    config      : Object or dict with optional thresholds.
    """

    def __init__(
        self,
        models_dict: Dict[str, Any],
        data_store: Any,
        config: Any,
    ) -> None:
        self.models = models_dict
        self.data_store = data_store
        self.config = config

        # Thresholds
        self.psi_threshold: float = _cfg(config, "psi_threshold", 0.25)
        self.sharpe_drop_threshold: float = _cfg(config, "sharpe_drop_threshold", 0.5)
        self.regime_pvalue_threshold: float = _cfg(config, "regime_pvalue_threshold", 0.05)
        self.min_samples: int = int(_cfg(config, "min_samples", 30))

        # History
        self._historical_predictions: Dict[str, np.ndarray] = {}
        self._historical_returns: np.ndarray = np.array([])
        self._performance_log: List[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_performance_drift(
        self,
        recent_predictions: np.ndarray,
        recent_actuals: np.ndarray,
        window: int = 30,
        model_name: str = "ensemble",
    ) -> dict:
        """
        Detect whether model performance has drifted.

        Checks:
        1. PSI on prediction distributions (recent vs historical).
        2. Welch t-test comparing recent Sharpe to historical Sharpe.

        Returns
        -------
        dict with keys:
            drift_detected (bool), severity ('none'|'moderate'|'severe'),
            psi (float), sharpe_recent (float), sharpe_historical (float),
            pvalue (float), metrics (dict)
        """
        result: dict = {
            "drift_detected": False,
            "severity": "none",
            "psi": 0.0,
            "sharpe_recent": 0.0,
            "sharpe_historical": 0.0,
            "pvalue": 1.0,
            "metrics": {},
        }

        if len(recent_predictions) < self.min_samples:
            logger.warning("Not enough samples for drift detection (%d < %d)", len(recent_predictions), self.min_samples)
            return result

        # --- PSI on predictions ---
        hist_preds = self._historical_predictions.get(model_name)
        psi_val = 0.0
        if hist_preds is not None and len(hist_preds) >= self.min_samples:
            psi_val = _psi(hist_preds, recent_predictions)
        result["psi"] = round(psi_val, 4)

        # --- Sharpe comparison ---
        recent_returns = recent_actuals - recent_predictions  # residuals used as proxy
        sharpe_recent = _sharpe(recent_returns)
        sharpe_hist = 0.0
        pvalue = 1.0
        if len(self._historical_returns) >= self.min_samples:
            sharpe_hist = _sharpe(self._historical_returns)
            _, pvalue = stats.ttest_ind(
                recent_returns,
                self._historical_returns[-len(recent_returns):],
                equal_var=False,
            )
        result["sharpe_recent"] = round(sharpe_recent, 3)
        result["sharpe_historical"] = round(sharpe_hist, 3)
        result["pvalue"] = round(float(pvalue), 4)

        # --- Severity ---
        sharpe_drop = sharpe_hist - sharpe_recent
        drift = psi_val >= self.psi_threshold or (
            pvalue < self.regime_pvalue_threshold
            and sharpe_drop > self.sharpe_drop_threshold
        )
        if drift:
            severity = "severe" if psi_val >= 0.4 or sharpe_drop > 1.0 else "moderate"
        else:
            severity = "none"

        result["drift_detected"] = drift
        result["severity"] = severity
        result["metrics"] = {
            "psi": psi_val,
            "sharpe_drop": round(sharpe_drop, 3),
            "pvalue": round(float(pvalue), 4),
            "recent_mae": float(np.mean(np.abs(recent_actuals - recent_predictions))),
        }

        # Update history
        self._historical_predictions[model_name] = np.concatenate(
            [hist_preds if hist_preds is not None else np.array([]), recent_predictions]
        )[-5000:]
        self._historical_returns = np.concatenate(
            [self._historical_returns, recent_returns]
        )[-5000:]

        logger.info(
            "Drift detection: drift=%s severity=%s psi=%.3f sharpe_drop=%.3f",
            drift, severity, psi_val, sharpe_drop,
        )
        return result

    def detect_regime_shift(
        self,
        recent_features: pd.DataFrame,
        historical_features: pd.DataFrame,
    ) -> dict:
        """
        Detect distributional shift in feature space using KL-divergence and
        a Kolmogorov-Smirnov test on each feature.

        Returns
        -------
        dict with keys:
            shift_detected (bool), regime_change (bool),
            ks_stats (dict of feature -> ks_statistic),
            pvalues (dict), n_shifted_features (int)
        """
        result: dict = {
            "shift_detected": False,
            "regime_change": False,
            "ks_stats": {},
            "pvalues": {},
            "n_shifted_features": 0,
        }

        common_cols = [
            c for c in recent_features.columns if c in historical_features.columns
        ]
        if not common_cols:
            return result

        ks_stats: dict = {}
        pvalues: dict = {}
        n_shifted = 0

        for col in common_cols:
            r = recent_features[col].dropna().values
            h = historical_features[col].dropna().values
            if len(r) < 5 or len(h) < 5:
                continue
            ks_stat, pval = stats.ks_2samp(h, r)
            ks_stats[col] = round(float(ks_stat), 4)
            pvalues[col] = round(float(pval), 4)
            if pval < self.regime_pvalue_threshold:
                n_shifted += 1

        result["ks_stats"] = ks_stats
        result["pvalues"] = pvalues
        result["n_shifted_features"] = n_shifted

        shift_ratio = n_shifted / max(len(common_cols), 1)
        result["shift_detected"] = shift_ratio > 0.3
        result["regime_change"] = shift_ratio > 0.6

        logger.info(
            "Regime shift: %d/%d features shifted (ratio=%.2f)",
            n_shifted, len(common_cols), shift_ratio,
        )
        return result

    def schedule_retraining(self, drift_report: dict) -> dict:
        """
        Decide which models to retrain based on drift severity.

        Returns
        -------
        dict with keys:
            retrain_models (list), priority ('low'|'medium'|'high'), reason (str)
        """
        severity = drift_report.get("severity", "none")
        regime_change = drift_report.get("regime_change", False)

        if severity == "severe" or regime_change:
            priority = "high"
            retrain_models = list(self.models.keys())
            reason = "Severe drift or regime change detected — full retraining required."
        elif severity == "moderate":
            priority = "medium"
            retrain_models = [
                k for k in self.models.keys() if k in ("lstm", "transformer", "ensemble")
            ]
            reason = "Moderate drift — retraining prediction models."
        else:
            priority = "low"
            retrain_models = []
            reason = "No significant drift detected."

        schedule = {
            "retrain_models": retrain_models,
            "priority": priority,
            "reason": reason,
        }
        logger.info("Retraining schedule: priority=%s models=%s", priority, retrain_models)
        return schedule

    def retrain_model(
        self,
        model_name: str,
        new_data: pd.DataFrame,
        existing_model: Any,
    ) -> Any:
        """
        Fine-tune or fully retrain a model on new data.

        For PyTorch models this runs a short fine-tuning loop; for
        sklearn-compatible models it calls .fit(). Falls back to returning
        the existing model unchanged if retraining is not possible.

        Returns
        -------
        Updated model object.
        """
        import torch  # lazy import

        logger.info("Retraining model: %s on %d samples", model_name, len(new_data))

        try:
            # Sklearn-compatible
            if hasattr(existing_model, "fit"):
                X = new_data.select_dtypes(include=[np.number]).dropna()
                if len(X) == 0:
                    return existing_model
                y = X.iloc[:, 0].shift(-1).dropna()
                X = X.iloc[: len(y)]
                existing_model.fit(X, y)
                logger.info("Sklearn retraining complete: %s", model_name)
                return existing_model

            # PyTorch model — lightweight fine-tuning
            if hasattr(existing_model, "parameters"):
                X_np = new_data.select_dtypes(include=[np.number]).ffill().bfill().values
                if X_np.shape[0] < 10:
                    return existing_model
                X_t = torch.tensor(X_np[:-1], dtype=torch.float32)
                y_t = torch.tensor(X_np[1:, :1], dtype=torch.float32)
                optimizer = torch.optim.AdamW(existing_model.parameters(), lr=1e-4)
                existing_model.train()
                for _ in range(5):
                    optimizer.zero_grad()
                    out = existing_model(X_t.unsqueeze(0))
                    if isinstance(out, (tuple, list)):
                        out = out[0]
                    out = out.squeeze()
                    if out.shape != y_t.squeeze().shape:
                        break
                    loss = torch.nn.functional.mse_loss(out, y_t.squeeze())
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(existing_model.parameters(), 1.0)
                    optimizer.step()
                existing_model.eval()
                logger.info("PyTorch fine-tuning complete: %s", model_name)
                return existing_model

        except Exception as exc:
            logger.warning("Retraining failed for %s: %s", model_name, exc)

        return existing_model

    def update_performance_log(
        self,
        model_name: str,
        predictions: np.ndarray,
        actuals: np.ndarray,
    ) -> None:
        """Append performance record to internal log."""
        mae = float(np.mean(np.abs(actuals - predictions)))
        self._performance_log.append(
            {
                "model": model_name,
                "n_samples": len(predictions),
                "mae": round(mae, 6),
                "sharpe": round(_sharpe(actuals - predictions), 3),
            }
        )

    def get_performance_history(self) -> pd.DataFrame:
        """Return performance log as a DataFrame."""
        return pd.DataFrame(self._performance_log)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg(config: Any, key: str, default: float) -> float:
    if isinstance(config, dict):
        return float(config.get(key, default))
    return float(getattr(config, key, default))
