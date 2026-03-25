"""
Market Regime Hidden Markov Model
Uses hmmlearn GaussianHMM to identify four market regimes:
BULL, BEAR, RANGE, HIGH_VOL.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# Colour palette for each regime
REGIME_COLORS = {
    "BULL": "#2ecc71",
    "BEAR": "#e74c3c",
    "RANGE": "#3498db",
    "HIGH_VOL": "#f39c12",
}


class MarketRegimeHMM:
    """
    Gaussian Hidden Markov Model for unsupervised market regime detection.

    Regimes (n_states = 4):
        0 : BULL     – trending up, low volatility
        1 : BEAR     – trending down, elevated volatility
        2 : RANGE    – low trend, low volatility
        3 : HIGH_VOL – high volatility, unclear direction
    """

    # Feature column names expected from the input DataFrame
    FEATURE_COLS = [
        "return_1d",
        "vol_21d",
        "volume_change",
        "rsi",
        "market_corr",
    ]

    def __init__(
        self,
        n_states: int = 4,
        n_iter: int = 1000,
        covariance_type: str = "full",
        random_state: int = 42,
    ) -> None:
        self.n_states = n_states
        self.n_iter = n_iter
        self.covariance_type = covariance_type
        self.random_state = random_state

        self.model: Optional[GaussianHMM] = None
        self.transition_matrix_: Optional[np.ndarray] = None
        self.state_means_: Optional[np.ndarray] = None
        self._state_to_label: Dict[int, str] = {}
        self._fitted = False

    # ------------------------------------------------------------------
    # Feature preparation
    # ------------------------------------------------------------------
    def _prepare_features(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, pd.Index]:
        """
        Extract and standardise the feature matrix from a wide DataFrame.

        Missing columns are filled with 0 (neutral).
        """
        df = features_df.copy()

        available = [c for c in self.FEATURE_COLS if c in df.columns]

        # Build feature matrix, filling any missing columns with 0
        X_parts: List[pd.Series] = []
        for col in self.FEATURE_COLS:
            if col in df.columns:
                X_parts.append(df[col])
            else:
                X_parts.append(pd.Series(0.0, index=df.index, name=col))

        X_df = pd.concat(X_parts, axis=1).fillna(0.0)
        X_df = X_df.replace([np.inf, -np.inf], 0.0)

        return X_df.values.astype(float), X_df.index

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, features_df: pd.DataFrame) -> "MarketRegimeHMM":
        """
        Fit the Gaussian HMM on the feature matrix.

        Parameters
        ----------
        features_df : pd.DataFrame
            Wide DataFrame; see FEATURE_COLS for expected columns.

        Returns
        -------
        self
        """
        X, _ = self._prepare_features(features_df)

        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        self.model.fit(X)

        self.transition_matrix_ = self.model.transmat_.copy()
        self.state_means_ = self.model.means_.copy()

        # Assign semantic labels to states based on mean features
        self._assign_labels()
        self._fitted = True

        return self

    # ------------------------------------------------------------------
    # Label assignment
    # ------------------------------------------------------------------
    def _assign_labels(self) -> None:
        """
        Map integer HMM states to semantic labels by inspecting feature means.

        Heuristic:
        - Highest mean return   -> BULL
        - Lowest mean return    -> BEAR
        - Highest mean vol      -> HIGH_VOL (among remaining)
        - Remaining             -> RANGE
        """
        n = self.n_states
        # Feature indices: return=0, vol=1
        means = self.state_means_  # (n_states, n_features)

        return_means = means[:, 0]
        vol_means = means[:, 1] if means.shape[1] > 1 else np.zeros(n)

        ranked_by_return = np.argsort(return_means)  # ascending
        bull_state = int(ranked_by_return[-1])
        bear_state = int(ranked_by_return[0])

        remaining = [i for i in range(n) if i not in (bull_state, bear_state)]
        if len(remaining) == 0:
            # n_states == 2 edge-case
            self._state_to_label = {bull_state: "BULL", bear_state: "BEAR"}
            return

        high_vol_state = int(remaining[np.argmax(vol_means[remaining])])
        range_states = [i for i in remaining if i != high_vol_state]

        label_map: Dict[int, str] = {
            bull_state: "BULL",
            bear_state: "BEAR",
            high_vol_state: "HIGH_VOL",
        }
        for s in range_states:
            label_map[s] = "RANGE"

        self._state_to_label = label_map

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict_regime(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Decode the most likely regime sequence and return state probabilities.

        Returns
        -------
        pd.DataFrame with columns:
            regime_id, regime_label, prob_BULL, prob_BEAR, prob_RANGE, prob_HIGH_VOL
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        X, idx = self._prepare_features(features_df)

        state_seq = self.model.predict(X)
        posteriors = self.model.predict_proba(X)  # (T, n_states)

        regime_labels = [self._state_to_label.get(s, "UNKNOWN") for s in state_seq]

        out = pd.DataFrame({"regime_id": state_seq, "regime_label": regime_labels}, index=idx)

        # Add per-state probabilities with semantic names
        label_cols: Dict[str, np.ndarray] = {
            "BULL": np.zeros(len(X)),
            "BEAR": np.zeros(len(X)),
            "RANGE": np.zeros(len(X)),
            "HIGH_VOL": np.zeros(len(X)),
        }
        for state_id, label in self._state_to_label.items():
            if state_id < posteriors.shape[1]:
                label_cols[label] += posteriors[:, state_id]

        for label, probs in label_cols.items():
            out[f"prob_{label}"] = probs

        return out

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------
    def decode_regime(self, state_id: int) -> str:
        """Return the semantic label for a given HMM state integer."""
        return self._state_to_label.get(state_id, "UNKNOWN")

    # ------------------------------------------------------------------
    # Transition Probabilities
    # ------------------------------------------------------------------
    def compute_transition_probs(self) -> Dict[str, Dict[str, float]]:
        """
        Return transition probabilities as a nested dict.

        Returns
        -------
        dict: { 'BULL': {'BULL': 0.95, 'BEAR': 0.02, ...}, ... }
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted.")

        result: Dict[str, Dict[str, float]] = {}
        for from_state in range(self.n_states):
            from_label = self._state_to_label.get(from_state, f"STATE_{from_state}")
            result[from_label] = {}
            for to_state in range(self.n_states):
                to_label = self._state_to_label.get(to_state, f"STATE_{to_state}")
                result[from_label][to_label] = float(
                    self.transition_matrix_[from_state, to_state]
                )
        return result

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    def plot_regimes(
        self,
        price_df: pd.DataFrame,
        regime_series: pd.DataFrame,
    ):
        """
        Plot price series with regime background shading.

        Parameters
        ----------
        price_df : pd.DataFrame with a 'close' column.
        regime_series : pd.DataFrame as returned by predict_regime().

        Returns
        -------
        matplotlib.figure.Figure
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for plotting.")

        price_df = price_df.copy()
        price_df.columns = [c.lower() for c in price_df.columns]
        close = price_df["close"]

        fig, ax = plt.subplots(figsize=(16, 6))

        ax.plot(close.index, close.values, color="black", linewidth=1.0, zorder=3)

        # Shade background per regime
        labels = regime_series["regime_label"]
        prev_label = None
        start_idx = None

        for i, (ts, label) in enumerate(labels.items()):
            if label != prev_label:
                if prev_label is not None and start_idx is not None:
                    color = REGIME_COLORS.get(prev_label, "#cccccc")
                    ax.axvspan(start_idx, ts, alpha=0.25, color=color, zorder=1)
                start_idx = ts
                prev_label = label

        # Close final span
        if prev_label is not None and start_idx is not None:
            color = REGIME_COLORS.get(prev_label, "#cccccc")
            ax.axvspan(start_idx, close.index[-1], alpha=0.25, color=color, zorder=1)

        patches = [
            mpatches.Patch(color=v, label=k, alpha=0.6)
            for k, v in REGIME_COLORS.items()
        ]
        ax.legend(handles=patches, loc="upper left", framealpha=0.8)
        ax.set_title("Market Regimes (HMM)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Serialise model and metadata to disk using joblib."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "model": self.model,
            "transition_matrix": self.transition_matrix_,
            "state_means": self.state_means_,
            "state_to_label": self._state_to_label,
            "n_states": self.n_states,
            "n_iter": self.n_iter,
            "covariance_type": self.covariance_type,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str) -> "MarketRegimeHMM":
        """Load a saved MarketRegimeHMM from disk."""
        payload = joblib.load(path)
        obj = cls(
            n_states=payload["n_states"],
            n_iter=payload["n_iter"],
            covariance_type=payload["covariance_type"],
        )
        obj.model = payload["model"]
        obj.transition_matrix_ = payload["transition_matrix"]
        obj.state_means_ = payload["state_means"]
        obj._state_to_label = payload["state_to_label"]
        obj._fitted = True
        return obj
