"""
SignalGenerator: Produce standardised trading signals from ensemble model outputs.

Output format is fixed so downstream consumers can parse it reliably.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Risk-level multipliers used in ranking
_RISK_MULTIPLIER = {"LOW": 1.0, "MEDIUM": 0.75, "HIGH": 0.5}


@dataclass
class SignalOutput:
    asset: str
    timeframe: str
    signal: str          # 'BUY' | 'SELL' | 'HOLD'
    probability: float   # 0-100
    confidence_score: float  # 0-1
    risk_level: str      # 'LOW' | 'MEDIUM' | 'HIGH'
    factors: List[str]
    predicted_return: float
    predicted_volatility: float
    regime: str
    position_size_pct: float
    stop_loss_price: float
    take_profit_price: float
    historical_accuracy: Any   # 'UNKNOWN' or float
    timestamp: str

    def to_dict(self) -> dict:
        return {
            "asset": self.asset,
            "timeframe": self.timeframe,
            "signal": self.signal,
            "probability": round(self.probability, 2),
            "confidence_score": round(self.confidence_score, 4),
            "risk_level": self.risk_level,
            "factors": self.factors,
            "predicted_return": round(self.predicted_return, 6),
            "predicted_volatility": round(self.predicted_volatility, 6),
            "regime": self.regime,
            "position_size_pct": round(self.position_size_pct, 4),
            "stop_loss_price": round(self.stop_loss_price, 4),
            "take_profit_price": round(self.take_profit_price, 4),
            "historical_accuracy": self.historical_accuracy,
            "timestamp": self.timestamp,
        }


class SignalGenerator:
    """
    Wraps ensemble model output into standardised SignalOutput objects.

    Parameters
    ----------
    ensemble       : Model with .predict(features) -> (pred_return, pred_vol, proba).
    risk_manager   : Optional RiskManager to apply position-size limits.
    config         : Object or dict with optional parameters.
    """

    def __init__(
        self,
        ensemble: Any,
        risk_manager: Optional[Any] = None,
        config: Optional[Any] = None,
    ) -> None:
        self.ensemble = ensemble
        self.risk_manager = risk_manager
        self.config = config or {}

        self.stop_loss_pct: float = _cfg(config, "stop_loss_pct", 0.05)
        self.take_profit_ratio: float = _cfg(config, "take_profit_ratio", 2.0)
        self.max_position_pct: float = _cfg(config, "max_position_size", 0.10)
        self._accuracy_history: Dict[str, List[float]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_signal(
        self,
        symbol: str,
        timeframe: str,
        features: pd.DataFrame,
        regime: str,
        current_price: float = 1.0,
    ) -> SignalOutput:
        """
        Generate a single signal for one asset.

        Parameters
        ----------
        symbol       : Ticker (e.g. 'AAPL').
        timeframe    : Timeframe string (e.g. '1d').
        features     : Feature DataFrame (latest row used).
        regime       : Regime label string.
        current_price: Current market price (used for stop-loss/take-profit).

        Returns
        -------
        SignalOutput
        """
        pred_return, pred_vol, proba, top_factors = self._run_ensemble(features)

        signal_str, probability = self._classify_signal(pred_return, proba)
        risk_level = self._assess_risk(pred_vol, regime)
        confidence = self._compute_confidence(pred_vol, top_factors)
        position_size = self._compute_position_size(confidence, risk_level, pred_vol)
        stop_loss = current_price * (1 - self.stop_loss_pct)
        take_profit = current_price * (
            1 + abs(pred_return) * self.take_profit_ratio
        )
        hist_acc = self._get_historical_accuracy(symbol)
        ts = datetime.now(tz=timezone.utc).isoformat()

        return SignalOutput(
            asset=symbol,
            timeframe=timeframe,
            signal=signal_str,
            probability=probability,
            confidence_score=confidence,
            risk_level=risk_level,
            factors=top_factors,
            predicted_return=pred_return,
            predicted_volatility=pred_vol,
            regime=regime,
            position_size_pct=position_size,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
            historical_accuracy=hist_acc,
            timestamp=ts,
        )

    def format_output(self, signal: SignalOutput) -> dict:
        """Return signal as the mandatory output dict."""
        return signal.to_dict()

    def generate_portfolio_signals(
        self,
        symbols: List[str],
        timeframe: str,
        features_dict: Dict[str, pd.DataFrame],
        regime_dict: Dict[str, str],
        prices_dict: Optional[Dict[str, float]] = None,
    ) -> List[SignalOutput]:
        """
        Generate signals for every symbol in the portfolio.

        Parameters
        ----------
        symbols       : List of tickers.
        timeframe     : Common timeframe string.
        features_dict : Mapping symbol -> features DataFrame.
        regime_dict   : Mapping symbol -> regime label.
        prices_dict   : Optional mapping symbol -> current price.

        Returns
        -------
        List of SignalOutput (one per symbol where features are available).
        """
        signals = []
        for sym in symbols:
            if sym not in features_dict:
                logger.warning("No features for %s — skipping", sym)
                continue
            price = (prices_dict or {}).get(sym, 1.0)
            regime = regime_dict.get(sym, "unknown")
            try:
                sig = self.generate_signal(
                    sym, timeframe, features_dict[sym], regime, price
                )
                signals.append(sig)
            except Exception as exc:
                logger.error("Signal generation failed for %s: %s", sym, exc)
        return signals

    def filter_signals(
        self,
        signals: List[SignalOutput],
        min_probability: float = 55.0,
        min_confidence: float = 0.4,
    ) -> List[SignalOutput]:
        """Keep only signals that clear minimum thresholds."""
        return [
            s for s in signals
            if s.probability >= min_probability and s.confidence_score >= min_confidence
        ]

    def rank_signals(self, signals: List[SignalOutput]) -> List[SignalOutput]:
        """
        Sort signals by expected_return * confidence / risk_multiplier (descending).
        """
        def _score(s: SignalOutput) -> float:
            rm = _RISK_MULTIPLIER.get(s.risk_level, 0.5)
            return abs(s.predicted_return) * s.confidence_score * rm

        return sorted(signals, key=_score, reverse=True)

    def update_accuracy(self, symbol: str, was_correct: bool) -> None:
        """Record whether a previous signal was correct."""
        self._accuracy_history.setdefault(symbol, []).append(float(was_correct))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_ensemble(
        self, features: pd.DataFrame
    ) -> tuple[float, float, float, List[str]]:
        """
        Run ensemble model and extract predictions.

        Returns (pred_return, pred_vol, proba, top_factors).
        """
        try:
            row = features.select_dtypes(include=[np.number]).ffill().bfill().iloc[[-1]]
            result = self.ensemble.predict(row)

            if isinstance(result, dict):
                pred_return = float(result.get("predicted_return", 0.0))
                pred_vol = float(result.get("predicted_volatility", 0.02))
                proba = float(result.get("probability", 50.0))
                factors = result.get("factors", [])
            elif isinstance(result, (list, tuple)) and len(result) >= 3:
                pred_return = float(result[0])
                pred_vol = float(result[1])
                proba = float(result[2])
                factors = list(result[3]) if len(result) > 3 else []
            else:
                pred_return = float(np.squeeze(result)) if result is not None else 0.0
                pred_vol = 0.02
                proba = 50.0 + pred_return * 500  # heuristic
                factors = []

            # Top-5 feature importances if model exposes them
            if not factors and hasattr(self.ensemble, "feature_importances_"):
                fi = self.ensemble.feature_importances_
                top_idx = np.argsort(fi)[::-1][:5]
                factors = [features.columns[i] for i in top_idx if i < len(features.columns)]

            return pred_return, max(pred_vol, 1e-6), np.clip(proba, 0, 100), factors[:5]

        except Exception as exc:
            logger.warning("Ensemble prediction failed: %s", exc)
            return 0.0, 0.02, 50.0, []

    @staticmethod
    def _classify_signal(pred_return: float, proba: float) -> tuple[str, float]:
        """Map predicted return to BUY/SELL/HOLD and calibrated probability."""
        if pred_return > 0.005 and proba >= 55:
            return "BUY", float(proba)
        elif pred_return < -0.005 and proba <= 45:
            return "SELL", float(100 - proba)
        else:
            return "HOLD", float(50 + abs(pred_return) * 1000)

    @staticmethod
    def _assess_risk(pred_vol: float, regime: str) -> str:
        """Classify risk level from volatility and regime."""
        if pred_vol < 0.01 and "bull" in regime.lower():
            return "LOW"
        elif pred_vol > 0.03 or "bear" in regime.lower() or "crisis" in regime.lower():
            return "HIGH"
        return "MEDIUM"

    @staticmethod
    def _compute_confidence(pred_vol: float, factors: List[str]) -> float:
        """Confidence decays with volatility and improves with more factors."""
        base = max(0.0, 1.0 - pred_vol * 20)
        factor_bonus = min(len(factors) * 0.04, 0.2)
        return float(np.clip(base + factor_bonus, 0.0, 1.0))

    def _compute_position_size(
        self,
        confidence: float,
        risk_level: str,
        pred_vol: float,
    ) -> float:
        """Kelly-inspired position sizing capped by max_position_pct."""
        rm = _RISK_MULTIPLIER.get(risk_level, 0.5)
        raw = confidence * rm * (1.0 / max(pred_vol * 100, 1))
        capped = min(raw * self.max_position_pct, self.max_position_pct)
        if self.risk_manager is not None:
            try:
                approved = self.risk_manager.approve_position_size(capped)
                capped = approved if isinstance(approved, float) else capped
            except Exception:
                pass
        return float(np.clip(capped, 0.0, self.max_position_pct))

    def _get_historical_accuracy(self, symbol: str) -> Any:
        history = self._accuracy_history.get(symbol, [])
        if not history:
            return "UNKNOWN"
        return round(float(np.mean(history)), 4)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg(config: Any, key: str, default: float) -> float:
    if config is None:
        return default
    if isinstance(config, dict):
        return float(config.get(key, default))
    return float(getattr(config, key, default))
