"""
RiskManager: Institutional-grade portfolio and position risk management.

Provides hard limits, soft warnings, position sizing (Kelly + volatility parity),
drawdown monitoring, correlation limits, stop-loss triggers, and emergency halt.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Severity levels (ordered)
_SEVERITY_ORDER = {"OK": 0, "WARNING": 1, "CRITICAL": 2, "HALT": 3}


class RiskManager:
    """
    Central risk management component.

    All public methods return a tuple (passed: bool, details: dict) or a
    plain dict with an 'approved' key so callers can act without unpacking
    complex structures.

    Parameters
    ----------
    config : dict or object with attributes
        Must expose the following keys / attributes (defaults shown):
        - max_drawdown       : 0.15  (15%)
        - max_position       : 0.10  (10% of NAV)
        - max_portfolio_vol  : 0.20  (20% annualised)
        - stop_loss          : 0.05  (5% from entry)
        - correlation_limit  : 0.70
        - max_sector_exposure: 0.30  (30% in any single sector)
        - kelly_fraction     : 0.25  (quarter-Kelly scaling)
        - var_confidence     : 0.95
    """

    def __init__(self, config: Any) -> None:
        def _get(key: str, default: float) -> float:
            if isinstance(config, dict):
                return float(config.get(key, default))
            return float(getattr(config, key, default))

        self.max_drawdown: float = _get("max_drawdown", 0.15)
        self.max_position: float = _get("max_position", 0.10)
        self.max_portfolio_vol: float = _get("max_portfolio_vol", 0.20)
        self.stop_loss: float = _get("stop_loss", 0.05)
        self.correlation_limit: float = _get("correlation_limit", 0.70)
        self.max_sector_exposure: float = _get("max_sector_exposure", 0.30)
        self.kelly_fraction: float = _get("kelly_fraction", 0.25)
        self.var_confidence: float = _get("var_confidence", 0.95)

        # Internal state
        self._halted: bool = False
        self._halt_reason: str = ""
        self._trade_history: List[Dict[str, float]] = []  # for Kelly estimation

        logger.info(
            "RiskManager initialised | max_dd=%.0f%% | max_pos=%.0f%% | "
            "max_vol=%.0f%% | stop=%.0f%%",
            self.max_drawdown * 100,
            self.max_position * 100,
            self.max_portfolio_vol * 100,
            self.stop_loss * 100,
        )

    # ------------------------------------------------------------------
    # Position size check
    # ------------------------------------------------------------------

    def check_position_size(
        self,
        symbol: str,
        size: float,
        portfolio_value: float,
        current_positions: Dict[str, float],
    ) -> Tuple[bool, str]:
        """
        Verify that a proposed new position size does not breach limits.

        Parameters
        ----------
        symbol : str
            Asset identifier.
        size : float
            Proposed position value in dollars (can be negative for short).
        portfolio_value : float
            Current total NAV.
        current_positions : dict
            Mapping {symbol: dollar_value} of existing positions.

        Returns
        -------
        (passed, reason) where reason is a human-readable string.
        """
        if self._halted:
            return False, f"System halted: {self._halt_reason}"

        if portfolio_value <= 0:
            return False, "Portfolio value must be positive."

        position_pct = abs(size) / portfolio_value

        if position_pct > self.max_position:
            return (
                False,
                f"{symbol}: position {position_pct:.2%} exceeds max "
                f"{self.max_position:.2%} of NAV.",
            )

        # Gross leverage check
        all_positions = dict(current_positions)
        all_positions[symbol] = size
        gross_exposure = sum(abs(v) for v in all_positions.values())
        gross_leverage = gross_exposure / portfolio_value
        if gross_leverage > 2.0:
            return (
                False,
                f"Adding {symbol} would push gross leverage to "
                f"{gross_leverage:.2f}x (limit 2.0x).",
            )

        return True, "OK"

    # ------------------------------------------------------------------
    # Drawdown check
    # ------------------------------------------------------------------

    def check_drawdown(
        self,
        current_equity: float,
        peak_equity: float,
    ) -> Tuple[bool, str]:
        """
        Assess drawdown severity and whether trading should continue.

        Returns
        -------
        (can_trade: bool, severity: str)
            severity is one of 'OK', 'WARNING', 'CRITICAL', 'HALT'
        """
        if peak_equity <= 0:
            return True, "OK"

        drawdown = (peak_equity - current_equity) / peak_equity

        if drawdown < self.max_drawdown * 0.50:
            return True, "OK"
        elif drawdown < self.max_drawdown * 0.80:
            logger.warning("DRAWDOWN WARNING: %.2f%%", drawdown * 100)
            return True, "WARNING"
        elif drawdown < self.max_drawdown:
            logger.error("DRAWDOWN CRITICAL: %.2f%% (limit %.2f%%)",
                         drawdown * 100, self.max_drawdown * 100)
            return True, "CRITICAL"
        else:
            logger.critical("DRAWDOWN HALT: %.2f%% exceeds limit %.2f%%",
                            drawdown * 100, self.max_drawdown * 100)
            self._halted = True
            self._halt_reason = (
                f"Max drawdown breached: {drawdown:.2%} > {self.max_drawdown:.2%}"
            )
            return False, "HALT"

    # ------------------------------------------------------------------
    # Volatility / portfolio risk check
    # ------------------------------------------------------------------

    def check_volatility_exposure(
        self,
        positions: Dict[str, float],
        vol_forecasts: Dict[str, float],
        correlation_matrix: np.ndarray,
        asset_order: Optional[List[str]] = None,
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Estimate ex-ante portfolio volatility and check against limit.

        Parameters
        ----------
        positions : dict {symbol: weight (fraction of NAV)}
        vol_forecasts : dict {symbol: annualised daily vol}
        correlation_matrix : np.ndarray (n x n)
        asset_order : list of symbol names matching matrix rows/cols
            If None, sorted(positions.keys()) is used.

        Returns
        -------
        (within_limits: bool, details: dict)
            details contains 'portfolio_vol', 'limit', 'excess_vol'
        """
        symbols = asset_order or sorted(positions.keys())
        n = len(symbols)
        if n == 0:
            return True, {"portfolio_vol": 0.0, "limit": self.max_portfolio_vol}

        w = np.array([positions.get(s, 0.0) for s in symbols])
        vols = np.array([vol_forecasts.get(s, 0.20) for s in symbols])

        if correlation_matrix.shape != (n, n):
            # Build identity covariance as fallback
            corr = np.eye(n)
        else:
            corr = np.array(correlation_matrix)

        # Covariance = diag(vol) @ corr @ diag(vol)
        cov = np.outer(vols, vols) * corr
        portfolio_var = float(w @ cov @ w)
        portfolio_vol = float(np.sqrt(max(portfolio_var, 0.0)))

        within_limits = portfolio_vol <= self.max_portfolio_vol
        details = {
            "portfolio_vol": portfolio_vol,
            "limit": self.max_portfolio_vol,
            "excess_vol": max(0.0, portfolio_vol - self.max_portfolio_vol),
        }

        if not within_limits:
            logger.warning(
                "Portfolio vol %.2f%% exceeds limit %.2f%%",
                portfolio_vol * 100, self.max_portfolio_vol * 100,
            )
        return within_limits, details

    # ------------------------------------------------------------------
    # Correlation check
    # ------------------------------------------------------------------

    def check_correlation_limits(
        self,
        new_symbol: str,
        new_position_weight: float,
        current_positions: Dict[str, float],
        correlation_matrix: np.ndarray,
        asset_order: List[str],
    ) -> Tuple[bool, List[str]]:
        """
        Check whether adding a new position would breach correlation limits
        with any existing position.

        Returns
        -------
        (passed: bool, violations: list of (symbol, correlation) strings)
        """
        if new_symbol not in asset_order:
            return True, []

        new_idx = asset_order.index(new_symbol)
        violations: List[str] = []

        for sym, weight in current_positions.items():
            if sym == new_symbol or abs(weight) < 1e-6:
                continue
            if sym not in asset_order:
                continue
            sym_idx = asset_order.index(sym)
            try:
                corr = float(correlation_matrix[new_idx, sym_idx])
            except (IndexError, TypeError):
                continue

            if abs(corr) > self.correlation_limit:
                violations.append(
                    f"{new_symbol}<->{sym}: corr={corr:.3f} > limit {self.correlation_limit:.2f}"
                )

        passed = len(violations) == 0
        return passed, violations

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def compute_position_size(
        self,
        signal_strength: float,
        volatility: float,
        portfolio_value: float,
        method: str = "volatility_parity",
        target_vol: float = 0.15,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None,
    ) -> float:
        """
        Compute the recommended dollar position size.

        Methods
        -------
        'volatility_parity':
            size = (target_vol / asset_vol) * portfolio_value
        'kelly':
            f* = (p*b - q) / b  where b = avg_win / avg_loss
            Applied at kelly_fraction (quarter-Kelly by default)
        'combined':
            Returns min(kelly * kelly_fraction, volatility_parity)

        Parameters
        ----------
        signal_strength : float
            Normalised signal in [0, 1]; scales the final size.
        volatility : float
            Forecasted daily volatility of the asset (annualised).
        portfolio_value : float
            Current NAV.
        target_vol : float
            Desired annualised contribution per position (volatility parity).
        win_rate, avg_win, avg_loss : float or None
            Required for Kelly calculation.

        Returns
        -------
        float : dollar position size (unsigned).
        """
        if portfolio_value <= 0:
            return 0.0

        vol = max(volatility, 1e-4)

        # --- Volatility parity ---
        vol_parity_size = (target_vol / vol) * portfolio_value
        vol_parity_size = min(vol_parity_size, self.max_position * portfolio_value)

        if method == "volatility_parity":
            return float(abs(signal_strength) * vol_parity_size)

        # --- Kelly ---
        kelly_size = 0.0
        if win_rate is not None and avg_win is not None and avg_loss is not None:
            p = float(np.clip(win_rate, 1e-6, 1.0 - 1e-6))
            q = 1.0 - p
            b = abs(avg_win) / (abs(avg_loss) + 1e-12)
            kelly_f = (p * b - q) / (b + 1e-12)
            kelly_f = max(0.0, kelly_f)  # no negative Kelly
            fractional_kelly = kelly_f * self.kelly_fraction
            kelly_size = fractional_kelly * portfolio_value
            kelly_size = min(kelly_size, self.max_position * portfolio_value)

        if method == "kelly":
            return float(abs(signal_strength) * kelly_size)

        # --- Combined: take the minimum ---
        if kelly_size > 0:
            combined = min(kelly_size, vol_parity_size)
        else:
            combined = vol_parity_size

        return float(abs(signal_strength) * combined)

    # ------------------------------------------------------------------
    # Stop-loss
    # ------------------------------------------------------------------

    def apply_stop_loss(
        self,
        position_size: float,
        entry_price: float,
        current_price: float,
        is_long: bool = True,
    ) -> bool:
        """
        Determine whether a hard stop-loss should trigger.

        Parameters
        ----------
        position_size : float
            Absolute position size (dollars).  Ignored in this check.
        entry_price : float
            Price at which the position was entered.
        current_price : float
            Current mark price.
        is_long : bool
            True for long positions, False for short.

        Returns
        -------
        bool : True means the stop is triggered (close the position).
        """
        if entry_price <= 0:
            return False

        if is_long:
            loss_pct = (entry_price - current_price) / entry_price
        else:
            loss_pct = (current_price - entry_price) / entry_price

        triggered = loss_pct >= self.stop_loss
        if triggered:
            logger.warning(
                "STOP-LOSS triggered: entry=%.4f, current=%.4f, loss=%.2f%%",
                entry_price, current_price, loss_pct * 100,
            )
        return triggered

    # ------------------------------------------------------------------
    # Emergency halt
    # ------------------------------------------------------------------

    def emergency_halt(self, reason: str) -> Dict[str, Any]:
        """
        Trigger an emergency halt, flagging the system to close all positions.

        Returns
        -------
        dict with keys: action, reason, timestamp, previous_halt_state
        """
        prev_state = self._halted
        self._halted = True
        self._halt_reason = reason

        payload: Dict[str, Any] = {
            "action": "CLOSE_ALL",
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "previous_halt_state": prev_state,
        }
        logger.critical("EMERGENCY HALT: %s", reason)
        return payload

    def reset_halt(self) -> None:
        """Manually reset the halt flag (use with caution)."""
        self._halted = False
        self._halt_reason = ""
        logger.info("RiskManager halt reset.")

    @property
    def is_halted(self) -> bool:
        return self._halted

    # ------------------------------------------------------------------
    # Risk report
    # ------------------------------------------------------------------

    def generate_risk_report(
        self,
        positions: Dict[str, float],
        portfolio_value: float,
        equity_curve: pd.Series,
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive risk report.

        Parameters
        ----------
        positions : dict {symbol: dollar_value}
        portfolio_value : float
        equity_curve : pd.Series with DatetimeIndex

        Returns
        -------
        dict with keys: timestamp, portfolio_value, positions_summary,
                        max_drawdown, current_drawdown, gross_leverage,
                        net_leverage, concentration, var_95, volatility_30d,
                        halted, warnings
        """
        warnings: List[str] = []

        # Drawdown
        if len(equity_curve) > 1:
            peak = equity_curve.cummax()
            drawdowns = (equity_curve - peak) / (peak + 1e-12)
            max_dd = float(drawdowns.min())
            current_dd = float(drawdowns.iloc[-1])
        else:
            max_dd = 0.0
            current_dd = 0.0

        if abs(current_dd) > self.max_drawdown * 0.80:
            warnings.append(
                f"Drawdown {current_dd:.2%} approaching limit {self.max_drawdown:.2%}"
            )

        # Leverage
        total_long = sum(v for v in positions.values() if v > 0)
        total_short = sum(abs(v) for v in positions.values() if v < 0)
        gross_leverage = (total_long + total_short) / (portfolio_value + 1e-12)
        net_leverage = (total_long - total_short) / (portfolio_value + 1e-12)

        if gross_leverage > 1.80:
            warnings.append(f"Gross leverage {gross_leverage:.2f}x approaching 2.0x limit")

        # Concentration
        if portfolio_value > 0:
            weights = {s: abs(v) / portfolio_value for s, v in positions.items()}
            max_position_weight = max(weights.values()) if weights else 0.0
            if max_position_weight > self.max_position:
                warnings.append(
                    f"Max single position {max_position_weight:.2%} > limit {self.max_position:.2%}"
                )
        else:
            weights = {}
            max_position_weight = 0.0

        # Historical VaR (95%)
        if len(equity_curve) > 30:
            returns = equity_curve.pct_change().dropna()
            var_95 = float(np.percentile(returns, (1 - self.var_confidence) * 100))
            vol_30d = float(returns.tail(30).std() * np.sqrt(252))
        else:
            var_95 = 0.0
            vol_30d = 0.0

        if vol_30d > self.max_portfolio_vol:
            warnings.append(
                f"Realised 30-day vol {vol_30d:.2%} > limit {self.max_portfolio_vol:.2%}"
            )

        report: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "portfolio_value": portfolio_value,
            "positions_summary": {
                "count": len(positions),
                "long_count": sum(1 for v in positions.values() if v > 0),
                "short_count": sum(1 for v in positions.values() if v < 0),
                "weights": weights,
                "max_position_weight": max_position_weight,
            },
            "max_drawdown": max_dd,
            "current_drawdown": current_dd,
            "gross_leverage": gross_leverage,
            "net_leverage": net_leverage,
            "var_95": var_95,
            "volatility_30d": vol_30d,
            "halted": self._halted,
            "halt_reason": self._halt_reason,
            "warnings": warnings,
            "risk_score": self._compute_risk_score(
                current_dd, gross_leverage, vol_30d, max_position_weight
            ),
        }
        return report

    def _compute_risk_score(
        self,
        drawdown: float,
        leverage: float,
        vol: float,
        max_weight: float,
    ) -> float:
        """
        Composite risk score in [0, 1] where 1 = maximum risk.
        Equally weighted average of four normalised sub-scores.
        """
        dd_score = min(abs(drawdown) / (self.max_drawdown + 1e-12), 1.0)
        lev_score = min(leverage / 2.0, 1.0)
        vol_score = min(vol / (self.max_portfolio_vol + 1e-12), 1.0)
        conc_score = min(max_weight / (self.max_position + 1e-12), 1.0)
        return float((dd_score + lev_score + vol_score + conc_score) / 4.0)

    # ------------------------------------------------------------------
    # Portfolio validation
    # ------------------------------------------------------------------

    def validate_portfolio(
        self,
        proposed_positions: Dict[str, float],
        current_portfolio: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Validate a proposed set of positions against all risk rules.

        Parameters
        ----------
        proposed_positions : dict {symbol: target_weight (fraction of NAV)}
        current_portfolio : dict with keys:
            - 'value': float (current NAV)
            - 'positions': dict {symbol: current_weight}
            - 'equity_curve': pd.Series (optional)
            - 'peak_equity': float (optional)

        Returns
        -------
        dict with keys:
            approved_positions : dict of approved {symbol: weight}
            rejected : list of dicts {symbol, weight, reason}
            warnings : list of warning strings
            risk_summary : dict
        """
        if self._halted:
            return {
                "approved_positions": {},
                "rejected": [
                    {"symbol": s, "weight": w, "reason": f"System halted: {self._halt_reason}"}
                    for s, w in proposed_positions.items()
                ],
                "warnings": [f"System halted: {self._halt_reason}"],
                "risk_summary": {"halted": True},
            }

        portfolio_value = float(current_portfolio.get("value", 1_000_000.0))
        current_positions_w = current_portfolio.get("positions", {})
        current_positions_usd = {
            s: w * portfolio_value for s, w in current_positions_w.items()
        }

        # Drawdown check
        peak_equity = float(current_portfolio.get("peak_equity", portfolio_value))
        dd_ok, dd_severity = self.check_drawdown(portfolio_value, peak_equity)

        approved: Dict[str, float] = {}
        rejected: List[Dict[str, Any]] = []
        warnings: List[str] = []

        if dd_severity == "WARNING":
            warnings.append(
                f"Drawdown WARNING – new position sizes reduced by 50%."
            )
        elif dd_severity == "CRITICAL":
            warnings.append(
                "Drawdown CRITICAL – only existing positions may be reduced."
            )

        for symbol, target_weight in proposed_positions.items():
            target_usd = target_weight * portfolio_value

            if dd_severity == "HALT":
                rejected.append({
                    "symbol": symbol,
                    "weight": target_weight,
                    "reason": "System in HALT state – all new trades blocked.",
                })
                continue

            if dd_severity == "CRITICAL":
                # In critical DD state, only allow reducing existing positions
                current_w = current_positions_w.get(symbol, 0.0)
                if abs(target_weight) > abs(current_w):
                    rejected.append({
                        "symbol": symbol,
                        "weight": target_weight,
                        "reason": "CRITICAL drawdown – can only reduce, not increase positions.",
                    })
                    continue

            if dd_severity == "WARNING":
                # Scale down proposed position
                target_usd *= 0.50
                target_weight *= 0.50

            pos_ok, pos_reason = self.check_position_size(
                symbol, target_usd, portfolio_value, current_positions_usd
            )
            if not pos_ok:
                rejected.append({
                    "symbol": symbol,
                    "weight": target_weight,
                    "reason": pos_reason,
                })
                continue

            approved[symbol] = target_weight

        risk_summary = {
            "drawdown_severity": dd_severity,
            "total_proposed": len(proposed_positions),
            "total_approved": len(approved),
            "total_rejected": len(rejected),
            "gross_leverage": sum(abs(w) for w in approved.values()),
        }

        return {
            "approved_positions": approved,
            "rejected": rejected,
            "warnings": warnings,
            "risk_summary": risk_summary,
        }

    # ------------------------------------------------------------------
    # Trade history helpers (for Kelly estimation)
    # ------------------------------------------------------------------

    def record_trade(
        self, symbol: str, pnl_pct: float, side: str = "long"
    ) -> None:
        """Record a completed trade for Kelly criterion estimation."""
        self._trade_history.append({
            "symbol": symbol,
            "pnl_pct": pnl_pct,
            "side": side,
        })
        # Keep last 500 trades
        if len(self._trade_history) > 500:
            self._trade_history = self._trade_history[-500:]

    def estimate_kelly_params(
        self, symbol: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Estimate win_rate, avg_win, avg_loss from trade history.

        Parameters
        ----------
        symbol : str or None
            If provided, filter to trades for that symbol only.

        Returns
        -------
        dict with keys: win_rate, avg_win, avg_loss, n_trades
        """
        history = self._trade_history
        if symbol:
            history = [t for t in history if t["symbol"] == symbol]

        if not history:
            return {"win_rate": 0.5, "avg_win": 0.02, "avg_loss": 0.01, "n_trades": 0}

        pnls = [t["pnl_pct"] for t in history]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        win_rate = len(wins) / len(pnls)
        avg_win = float(np.mean(wins)) if wins else 0.02
        avg_loss = float(abs(np.mean(losses))) if losses else 0.01

        return {
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "n_trades": len(pnls),
        }
