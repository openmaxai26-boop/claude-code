"""
PortfolioOptimizer: Mean-variance, risk-parity, max-Sharpe and Black-Litterman
optimisation using cvxpy + scipy.

DynamicRebalancer: Threshold- and schedule-based portfolio rebalancing.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.optimize as sco
import cvxpy as cp

logger = logging.getLogger(__name__)

_BPS = 1e-4  # one basis point


class PortfolioOptimizer:
    """
    Multi-method portfolio optimiser.

    All methods accept expected_returns as a (n,) array and cov_matrix as
    an (n, n) positive-semi-definite matrix.  Weights are always returned as
    a (n,) numpy array.

    Parameters
    ----------
    n_assets : int
        Number of assets.
    risk_aversion : float
        Lambda in the mean-variance objective.  Higher = more risk averse.
    max_position : float
        Upper bound on any single weight.
    min_position : float
        Lower bound on any single weight (negative = allow shorts).
    max_turnover : float
        Maximum allowable one-way turnover per rebalance.
    """

    def __init__(
        self,
        n_assets: int,
        risk_aversion: float = 2.0,
        max_position: float = 0.10,
        min_position: float = -0.05,
        max_turnover: float = 0.20,
    ) -> None:
        self.n_assets = n_assets
        self.risk_aversion = risk_aversion
        self.max_position = max_position
        self.min_position = min_position
        self.max_turnover = max_turnover

        logger.info(
            "PortfolioOptimizer | n=%d | ra=%.1f | max_pos=%.2f | "
            "min_pos=%.2f | max_turn=%.2f",
            n_assets, risk_aversion, max_position, min_position, max_turnover,
        )

    # ------------------------------------------------------------------
    # Mean-Variance (MVO)
    # ------------------------------------------------------------------

    def optimize_mean_variance(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        prev_weights: Optional[np.ndarray] = None,
        constraints: Optional[List[Any]] = None,
    ) -> np.ndarray:
        """
        Solve the classic Markowitz MVO problem with cvxpy.

        Objective:
            Maximise  w' mu  -  (risk_aversion/2) * w' Sigma w

        Constraints:
            sum(w) = 1
            min_position <= w_i <= max_position
            |w - w_prev| <= max_turnover  (if prev_weights provided)

        Returns
        -------
        np.ndarray shape (n_assets,)
        """
        mu = np.array(expected_returns, dtype=float)
        Sigma = np.array(cov_matrix, dtype=float)
        n = self.n_assets

        # Regularise covariance for numerical stability
        Sigma += np.eye(n) * 1e-8

        w = cp.Variable(n)
        ret_term = mu @ w
        risk_term = cp.quad_form(w, cp.psd_wrap(Sigma))
        objective = cp.Maximize(ret_term - (self.risk_aversion / 2.0) * risk_term)

        cons = [
            cp.sum(w) == 1.0,
            w >= self.min_position,
            w <= self.max_position,
        ]

        if prev_weights is not None:
            w_prev = np.array(prev_weights, dtype=float)
            cons.append(cp.norm(w - w_prev, 1) <= 2.0 * self.max_turnover)

        if constraints:
            cons.extend(constraints)

        prob = cp.Problem(objective, cons)
        try:
            prob.solve(solver=cp.CLARABEL, warm_start=True)
        except cp.SolverError:
            prob.solve(solver=cp.SCS, warm_start=True)

        if w.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
            logger.warning(
                "MVO solver status: %s – returning equal weights.", prob.status
            )
            return np.ones(n) / n

        result = np.array(w.value)
        result = np.clip(result, self.min_position, self.max_position)
        # Re-normalise to sum=1
        total = result.sum()
        if abs(total) > 1e-8:
            result = result / total
        return result

    # ------------------------------------------------------------------
    # Risk Parity
    # ------------------------------------------------------------------

    def optimize_risk_parity(
        self,
        cov_matrix: np.ndarray,
        initial_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Equal Risk Contribution (ERC) portfolio via scipy minimisation.

        Minimises the sum of squared pairwise differences of marginal risk
        contributions:
            sum_i sum_j (RC_i - RC_j)^2
        where RC_i = w_i * (Sigma @ w)_i

        Returns
        -------
        np.ndarray shape (n_assets,)
        """
        Sigma = np.array(cov_matrix, dtype=float) + np.eye(self.n_assets) * 1e-8
        n = self.n_assets

        def _risk_contributions(weights: np.ndarray) -> np.ndarray:
            port_var = weights @ Sigma @ weights
            marginal = Sigma @ weights
            rc = weights * marginal / (np.sqrt(port_var) + 1e-12)
            return rc

        def _objective(weights: np.ndarray) -> float:
            rc = _risk_contributions(weights)
            # Sum of squared pairwise differences
            total = 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    total += (rc[i] - rc[j]) ** 2
            return total

        # Gradient
        def _objective_grad(weights: np.ndarray) -> np.ndarray:
            eps = 1e-7
            grad = np.zeros(n)
            base = _objective(weights)
            for k in range(n):
                w_up = weights.copy()
                w_up[k] += eps
                grad[k] = (_objective(w_up) - base) / eps
            return grad

        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(max(0.0, self.min_position), self.max_position)] * n

        if initial_weights is not None:
            w0 = np.array(initial_weights)
        else:
            w0 = np.ones(n) / n

        result = sco.minimize(
            _objective,
            w0,
            method="SLSQP",
            jac=_objective_grad,
            bounds=bounds,
            constraints=cons,
            options={"ftol": 1e-12, "maxiter": 1000, "disp": False},
        )

        if not result.success:
            logger.warning("Risk parity solver: %s – fallback to equal weight.", result.message)
            return np.ones(n) / n

        weights = np.clip(result.x, 0.0, self.max_position)
        total = weights.sum()
        return weights / total if total > 1e-8 else np.ones(n) / n

    # ------------------------------------------------------------------
    # Maximum Sharpe
    # ------------------------------------------------------------------

    def optimize_max_sharpe(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_free_rate: float = 0.05,
    ) -> np.ndarray:
        """
        Maximum Sharpe Ratio portfolio via the Markowitz two-fund separation.

        Uses the tangency portfolio approach: solve for y = w / (1' w) with
        the change-of-variables  y = Sigma^{-1} (mu - rf * 1)  then normalise.

        Falls back to scipy if the direct formula is ill-conditioned.

        Returns
        -------
        np.ndarray shape (n_assets,)
        """
        mu = np.array(expected_returns, dtype=float)
        Sigma = np.array(cov_matrix, dtype=float) + np.eye(self.n_assets) * 1e-8
        n = self.n_assets
        rf_daily = risk_free_rate / 252.0
        excess = mu - rf_daily

        # Direct tangency formula
        try:
            Sigma_inv = np.linalg.inv(Sigma)
            y = Sigma_inv @ excess
            if y.sum() > 1e-8:
                w_tang = y / y.sum()
                w_tang = np.clip(w_tang, self.min_position, self.max_position)
                total = w_tang.sum()
                if abs(total) > 1e-8:
                    return w_tang / total
        except np.linalg.LinAlgError:
            pass

        # Fallback: scipy
        def _neg_sharpe(w: np.ndarray) -> float:
            port_ret = float(w @ mu)
            port_vol = float(np.sqrt(w @ Sigma @ w))
            if port_vol < 1e-12:
                return 0.0
            return -((port_ret - rf_daily) / port_vol)

        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(self.min_position, self.max_position)] * n
        w0 = np.ones(n) / n

        res = sco.minimize(
            _neg_sharpe, w0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"ftol": 1e-12, "maxiter": 1000},
        )

        if not res.success:
            logger.warning("Max-Sharpe solver failed – returning MVO weights.")
            return self.optimize_mean_variance(expected_returns, cov_matrix)

        w = np.clip(res.x, self.min_position, self.max_position)
        total = w.sum()
        return w / total if abs(total) > 1e-8 else np.ones(n) / n

    # ------------------------------------------------------------------
    # Efficient frontier
    # ------------------------------------------------------------------

    def compute_efficient_frontier(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        n_points: int = 50,
        risk_free_rate: float = 0.05,
    ) -> List[Dict[str, Any]]:
        """
        Trace the efficient frontier.

        Returns
        -------
        list of dicts, each with keys:
            'target_return', 'portfolio_return', 'portfolio_vol',
            'sharpe', 'weights'
        """
        mu = np.array(expected_returns, dtype=float)
        Sigma = np.array(cov_matrix, dtype=float) + np.eye(self.n_assets) * 1e-8
        n = self.n_assets

        min_ret = float(np.min(mu))
        max_ret = float(np.max(mu))
        target_returns = np.linspace(min_ret * 0.90, max_ret * 1.10, n_points)

        frontier: List[Dict[str, Any]] = []

        for target in target_returns:
            w = cp.Variable(n)
            port_var = cp.quad_form(w, cp.psd_wrap(Sigma))
            cons = [
                cp.sum(w) == 1.0,
                w >= self.min_position,
                w <= self.max_position,
                mu @ w >= target,
            ]
            prob = cp.Problem(cp.Minimize(port_var), cons)
            try:
                prob.solve(solver=cp.CLARABEL, warm_start=True)
            except cp.SolverError:
                prob.solve(solver=cp.SCS, warm_start=True)

            if w.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
                continue

            weights = np.clip(np.array(w.value), self.min_position, self.max_position)
            port_ret = float(weights @ mu)
            port_vol = float(np.sqrt(max(weights @ Sigma @ weights, 0.0)))
            rf_daily = risk_free_rate / 252.0
            sharpe = (port_ret - rf_daily) / (port_vol + 1e-12) * np.sqrt(252)

            frontier.append({
                "target_return": float(target),
                "portfolio_return": port_ret * 252,  # annualised
                "portfolio_vol": port_vol * np.sqrt(252),
                "sharpe": float(sharpe),
                "weights": weights.tolist(),
            })

        return frontier

    # ------------------------------------------------------------------
    # Portfolio metrics
    # ------------------------------------------------------------------

    def compute_portfolio_metrics(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_free_rate: float = 0.05,
    ) -> Dict[str, float]:
        """
        Compute standard portfolio statistics.

        Returns
        -------
        dict with: annualised_return, annualised_vol, sharpe, diversification_ratio,
                   effective_n, max_weight, min_weight
        """
        w = np.array(weights, dtype=float)
        mu = np.array(expected_returns, dtype=float)
        Sigma = np.array(cov_matrix, dtype=float) + np.eye(len(w)) * 1e-8

        port_ret_daily = float(w @ mu)
        port_var = float(w @ Sigma @ w)
        port_vol_daily = float(np.sqrt(max(port_var, 0.0)))

        ann_ret = port_ret_daily * 252
        ann_vol = port_vol_daily * np.sqrt(252)
        rf_daily = risk_free_rate / 252.0
        sharpe = (port_ret_daily - rf_daily) / (port_vol_daily + 1e-12) * np.sqrt(252)

        # Diversification ratio: weighted sum of individual vols / portfolio vol
        individual_vols = np.sqrt(np.diag(Sigma))
        weighted_vol_sum = float(w @ individual_vols)
        diversification_ratio = weighted_vol_sum / (port_vol_daily + 1e-12)

        # Effective N (Herfindahl-based)
        herfindahl = float(np.sum(w ** 2))
        effective_n = 1.0 / (herfindahl + 1e-12)

        return {
            "annualised_return": ann_ret,
            "annualised_vol": ann_vol,
            "sharpe": float(sharpe),
            "diversification_ratio": float(diversification_ratio),
            "effective_n": float(effective_n),
            "max_weight": float(w.max()),
            "min_weight": float(w.min()),
            "portfolio_var": port_var,
        }

    # ------------------------------------------------------------------
    # Rebalance
    # ------------------------------------------------------------------

    def rebalance(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        portfolio_value: float,
        transaction_cost_bps: float = 5.0,
    ) -> Dict[str, Any]:
        """
        Compute trades required to move from current to target weights.

        Returns
        -------
        dict with keys:
            trades           : np.ndarray (n,) delta weights
            trade_values     : np.ndarray (n,) dollar trades
            total_turnover   : float (one-way)
            estimated_cost   : float (dollar cost)
            new_weights      : np.ndarray
        """
        curr = np.array(current_weights, dtype=float)
        tgt = np.array(target_weights, dtype=float)

        delta = tgt - curr
        trade_values = delta * portfolio_value
        one_way_turnover = float(np.sum(np.abs(delta)) / 2.0)
        cost = float(np.sum(np.abs(trade_values)) * transaction_cost_bps * _BPS)

        # New weights after paying costs
        new_weights = tgt.copy()
        cost_weight = cost / (portfolio_value + 1e-12)
        # Reduce positions proportionally to cover cost
        new_weights *= (1.0 - cost_weight)

        return {
            "trades": delta,
            "trade_values": trade_values,
            "total_turnover": one_way_turnover,
            "estimated_cost": cost,
            "new_weights": new_weights,
        }

    # ------------------------------------------------------------------
    # Black-Litterman
    # ------------------------------------------------------------------

    def black_litterman(
        self,
        market_weights: np.ndarray,
        cov_matrix: np.ndarray,
        views_matrix: np.ndarray,
        view_returns: np.ndarray,
        view_uncertainty: Optional[np.ndarray] = None,
        tau: float = 0.05,
        risk_aversion: Optional[float] = None,
    ) -> np.ndarray:
        """
        Black-Litterman posterior expected returns.

        Parameters
        ----------
        market_weights : np.ndarray (n,)
            Market-cap weights (or any prior portfolio weights).
        cov_matrix : np.ndarray (n, n)
        views_matrix : np.ndarray (k, n)
            Each row encodes one view as a portfolio (P matrix).
        view_returns : np.ndarray (k,)
            Expected return for each view (Q vector).
        view_uncertainty : np.ndarray (k, k) or None
            Diagonal uncertainty matrix (Omega).  If None, inferred from
            tau * P @ Sigma @ P'.
        tau : float
            Uncertainty scaling parameter (typical range 0.01-0.10).
        risk_aversion : float or None
            Override the instance risk_aversion.

        Returns
        -------
        np.ndarray (n,) : posterior expected returns (daily, raw scale).
        """
        ra = risk_aversion if risk_aversion is not None else self.risk_aversion
        w_mkt = np.array(market_weights, dtype=float)
        Sigma = np.array(cov_matrix, dtype=float) + np.eye(self.n_assets) * 1e-8
        P = np.array(views_matrix, dtype=float)
        Q = np.array(view_returns, dtype=float)

        # Implied equilibrium returns: pi = ra * Sigma @ w_mkt
        pi = ra * Sigma @ w_mkt

        # Prior covariance scaled by tau
        tau_Sigma = tau * Sigma

        # View uncertainty (Omega)
        if view_uncertainty is not None:
            Omega = np.array(view_uncertainty, dtype=float)
        else:
            Omega = np.diag(np.diag(tau * P @ Sigma @ P.T))

        # Black-Litterman posterior formula
        # mu_BL = [(tau*Sigma)^{-1} + P'Omega^{-1}P]^{-1} * [(tau*Sigma)^{-1}*pi + P'Omega^{-1}Q]
        tau_Sigma_inv = np.linalg.inv(tau_Sigma)
        Omega_inv = np.linalg.inv(Omega + np.eye(len(Q)) * 1e-12)

        M1 = tau_Sigma_inv + P.T @ Omega_inv @ P
        M2 = tau_Sigma_inv @ pi + P.T @ Omega_inv @ Q

        try:
            posterior_returns = np.linalg.solve(M1, M2)
        except np.linalg.LinAlgError:
            logger.warning("Black-Litterman: singular matrix – returning equilibrium returns.")
            posterior_returns = pi.copy()

        return posterior_returns


# ---------------------------------------------------------------------------
# Dynamic Rebalancer
# ---------------------------------------------------------------------------


class DynamicRebalancer:
    """
    Decides when to rebalance and computes the required trades.

    Triggers rebalancing on two criteria (either suffices):
    1. Drift: any weight has drifted more than ``rebalance_threshold`` from target.
    2. Calendar: enough time has elapsed since last rebalance (``rebalance_freq``).

    Parameters
    ----------
    optimizer : PortfolioOptimizer
    rebalance_threshold : float
        Maximum tolerated absolute weight drift before triggering rebalance.
    rebalance_freq : str
        One of 'daily', 'weekly', 'monthly', 'quarterly'.
    """

    _FREQ_DAYS: Dict[str, int] = {
        "daily": 1,
        "weekly": 5,
        "monthly": 21,
        "quarterly": 63,
        "semi-annual": 126,
        "annual": 252,
    }

    def __init__(
        self,
        optimizer: PortfolioOptimizer,
        rebalance_threshold: float = 0.05,
        rebalance_freq: str = "monthly",
    ) -> None:
        self.optimizer = optimizer
        self.rebalance_threshold = rebalance_threshold
        self.rebalance_freq = rebalance_freq.lower()
        self._last_rebalance: Optional[datetime] = None

        if self.rebalance_freq not in self._FREQ_DAYS:
            raise ValueError(
                f"rebalance_freq must be one of {list(self._FREQ_DAYS)}, "
                f"got '{rebalance_freq}'"
            )

    # ------------------------------------------------------------------

    def should_rebalance(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        current_dt: Optional[datetime] = None,
    ) -> Tuple[bool, str]:
        """
        Determine whether a rebalance is warranted.

        Returns
        -------
        (should_rebalance: bool, reason: str)
        """
        curr = np.array(current_weights)
        tgt = np.array(target_weights)
        max_drift = float(np.max(np.abs(curr - tgt)))

        # Drift check
        if max_drift >= self.rebalance_threshold:
            return True, f"Drift {max_drift:.3f} >= threshold {self.rebalance_threshold:.3f}"

        # Calendar check
        now = current_dt or datetime.now(timezone.utc)
        if self._last_rebalance is None:
            return True, "No prior rebalance – initial allocation"

        elapsed_days = (now - self._last_rebalance).days
        freq_days = self._FREQ_DAYS[self.rebalance_freq]
        if elapsed_days >= freq_days:
            return (
                True,
                f"Scheduled rebalance: {elapsed_days}d >= {freq_days}d ({self.rebalance_freq})",
            )

        return False, f"No rebalance: drift={max_drift:.3f}, elapsed={elapsed_days}d"

    # ------------------------------------------------------------------

    def compute_rebalance_trades(
        self,
        current_positions: Dict[str, float],
        target_weights: np.ndarray,
        prices: np.ndarray,
        symbols: List[str],
        portfolio_value: float,
    ) -> pd.DataFrame:
        """
        Compute the trades required to reach target_weights.

        Parameters
        ----------
        current_positions : dict {symbol: shares_held}
        target_weights : np.ndarray (n,)
        prices : np.ndarray (n,)
        symbols : list of str
        portfolio_value : float

        Returns
        -------
        pd.DataFrame with columns:
            symbol, current_shares, target_shares, delta_shares,
            current_weight, target_weight, delta_weight,
            trade_value, direction
        """
        n = len(symbols)
        rows: List[Dict[str, Any]] = []

        for i, sym in enumerate(symbols):
            price = float(prices[i]) if prices[i] > 0 else 1.0
            curr_shares = float(current_positions.get(sym, 0.0))
            curr_value = curr_shares * price
            curr_weight = curr_value / (portfolio_value + 1e-12)

            tgt_weight = float(target_weights[i])
            tgt_value = tgt_weight * portfolio_value
            tgt_shares = tgt_value / price

            delta_shares = tgt_shares - curr_shares
            trade_value = delta_shares * price

            rows.append({
                "symbol": sym,
                "current_shares": curr_shares,
                "target_shares": tgt_shares,
                "delta_shares": delta_shares,
                "current_weight": curr_weight,
                "target_weight": tgt_weight,
                "delta_weight": tgt_weight - curr_weight,
                "trade_value": trade_value,
                "direction": "BUY" if delta_shares > 0 else ("SELL" if delta_shares < 0 else "HOLD"),
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------

    def execute_rebalance(
        self,
        trades: pd.DataFrame,
        risk_manager: Any,
        portfolio_value: float,
        current_positions: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Apply risk checks and mark trades as executed or rejected.

        Parameters
        ----------
        trades : pd.DataFrame from compute_rebalance_trades
        risk_manager : RiskManager instance
        portfolio_value : float
        current_positions : dict {symbol: dollar_value} for risk checks

        Returns
        -------
        dict with keys:
            executed_trades : pd.DataFrame (subset that passed risk checks)
            rejected_trades : pd.DataFrame
            total_cost : float (estimated)
            turnover : float
        """
        if current_positions is None:
            current_positions = {}

        executed_rows: List[int] = []
        rejected_rows: List[int] = []
        reasons: Dict[int, str] = {}

        for idx, row in trades.iterrows():
            sym = row["symbol"]
            trade_val = abs(row["trade_value"])

            if trade_val < 1.0:  # sub-dollar trades: skip silently
                executed_rows.append(idx)
                continue

            ok, reason = risk_manager.check_position_size(
                sym,
                row["target_weight"] * portfolio_value,
                portfolio_value,
                current_positions,
            )
            if ok:
                executed_rows.append(idx)
                # Update running positions dict for subsequent checks
                current_positions[sym] = row["target_weight"] * portfolio_value
            else:
                rejected_rows.append(idx)
                reasons[idx] = reason

        executed_df = trades.loc[executed_rows].copy()
        rejected_df = trades.loc[rejected_rows].copy()
        if rejected_rows:
            rejected_df["rejection_reason"] = [reasons.get(i, "") for i in rejected_rows]

        total_cost = float(
            executed_df["trade_value"].abs().sum() * 5.0 * _BPS
        )  # 5 bps default
        turnover = float(executed_df["delta_weight"].abs().sum() / 2.0)

        # Record rebalance time
        self._last_rebalance = datetime.now(timezone.utc)

        return {
            "executed_trades": executed_df,
            "rejected_trades": rejected_df,
            "total_cost": total_cost,
            "turnover": turnover,
            "rebalance_timestamp": self._last_rebalance.isoformat(),
        }
