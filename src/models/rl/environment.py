"""
TradingEnvironment: A Gymnasium-compatible reinforcement-learning environment
for multi-asset portfolio management with realistic execution costs.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

logger = logging.getLogger(__name__)


class TradingEnvironment(gym.Env):
    """
    Multi-asset continuous-action trading environment.

    Observation space:
        - market_features: (n_features,) float32  -- price/vol/technical features
        - portfolio_state: (n_assets + 2,) float32 -- positions + cash_pct + unrealized_pnl_pct
        - regime: () int64                          -- discrete market regime label (0..n_regimes-1)

    Action space:
        Box(-1, 1, shape=(n_assets,)) -- target position weights as fraction of NAV
        Negative = short, positive = long.  |w| <= max_position_pct enforced at step time.

    Reward:
        risk_adjusted_pnl = daily_return / (rolling_vol_21d + 1e-8)
                          - 3 * max(0, drawdown - 0.05)
                          - 0.5 * mean(|delta_positions|)
    """

    metadata = {"render_modes": ["human", "ansi"]}

    # ------------------------------------------------------------------ init --

    def __init__(
        self,
        df: pd.DataFrame,
        n_assets: int,
        initial_capital: float = 1_000_000.0,
        transaction_cost_bps: float = 5.0,
        slippage_bps: float = 3.0,
        max_position_pct: float = 0.10,
        n_regimes: int = 4,
        lookback: int = 21,
        render_mode: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        df : pd.DataFrame
            Multi-level column DataFrame with at minimum columns:
            ('close', symbol), ('volume', symbol), ('feature_*', symbol_or_global)
            Index must be a DatetimeIndex sorted ascending.
        n_assets : int
            Number of tradeable assets (must match columns in df).
        initial_capital : float
            Starting NAV in dollars.
        transaction_cost_bps : float
            One-way transaction cost in basis points.
        slippage_bps : float
            Base market-impact / slippage in basis points.
        max_position_pct : float
            Maximum |weight| per single asset as fraction of NAV.
        n_regimes : int
            Number of discrete market regimes produced by the regime model.
        lookback : int
            Rolling window size (bars) used for volatility / reward computation.
        render_mode : str or None
            'human' prints to stdout; None suppresses output.
        """
        super().__init__()

        self.df = df.copy()
        self.n_assets = n_assets
        self.initial_capital = initial_capital
        self.tc_bps = transaction_cost_bps
        self.sl_bps = slippage_bps
        self.max_position_pct = max_position_pct
        self.n_regimes = n_regimes
        self.lookback = lookback
        self.render_mode = render_mode

        # Determine feature columns (everything that is not close/volume/regime)
        self._close_cols = [c for c in df.columns if str(c).startswith("close")]
        self._vol_cols = [c for c in df.columns if str(c).startswith("volume")]
        self._feature_cols = [
            c for c in df.columns
            if not str(c).startswith("close")
            and not str(c).startswith("volume")
            and not str(c).startswith("regime")
        ]
        self._regime_col = next(
            (c for c in df.columns if str(c).startswith("regime")), None
        )

        self.n_features = max(len(self._feature_cols), 1)

        # ---- Spaces ----
        self.observation_space = spaces.Dict(
            {
                "market_features": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.n_features,),
                    dtype=np.float32,
                ),
                "portfolio_state": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.n_assets + 2,),  # positions + cash_pct + unrealised_pnl_pct
                    dtype=np.float32,
                ),
                "regime": spaces.Discrete(self.n_regimes),
            }
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_assets,),
            dtype=np.float32,
        )

        # ---- Episode state (initialised by reset) ----
        self._current_step: int = 0
        self._max_steps: int = len(self.df) - 1
        self._portfolio_value: float = initial_capital
        self._cash: float = initial_capital
        self._positions: np.ndarray = np.zeros(n_assets, dtype=np.float64)  # shares held
        self._position_weights: np.ndarray = np.zeros(n_assets, dtype=np.float64)
        self._entry_prices: np.ndarray = np.zeros(n_assets, dtype=np.float64)
        self._peak_value: float = initial_capital
        self._daily_returns: deque = deque(maxlen=lookback)
        self._trade_count: int = 0
        self._total_cost: float = 0.0

    # ---------------------------------------------------------------- reset ---

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)

        # Allow custom start offset via options
        start_offset = 0
        if options and "start_step" in options:
            start_offset = int(options["start_step"])

        self._current_step = start_offset
        self._portfolio_value = self.initial_capital
        self._cash = self.initial_capital
        self._positions = np.zeros(self.n_assets, dtype=np.float64)
        self._position_weights = np.zeros(self.n_assets, dtype=np.float64)
        self._entry_prices = np.zeros(self.n_assets, dtype=np.float64)
        self._peak_value = self.initial_capital
        self._daily_returns = deque(maxlen=self.lookback)
        self._trade_count = 0
        self._total_cost = 0.0

        obs = self._get_observation()
        info = self._build_info()
        return obs, info

    # ----------------------------------------------------------------- step ---

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute one bar of simulation.

        Parameters
        ----------
        action : np.ndarray shape (n_assets,)
            Raw target weights in [-1, 1].  Clipped to max_position_pct.

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        assert self.action_space.contains(action.astype(np.float32)), \
            f"Action {action} out of action space bounds."

        # 1. Clip raw actions to maximum position size
        target_weights = np.clip(action, -self.max_position_pct, self.max_position_pct)

        # 2. Get current prices
        prices_before = self._get_prices(self._current_step)

        # 3. Execute trades → realised position weights after costs
        realised_weights, execution_cost = self._execute_trades(
            target_weights, prices_before
        )
        self._position_weights = realised_weights
        self._total_cost += execution_cost

        # 4. Advance time
        self._current_step += 1

        # 5. Mark-to-market at new prices
        prices_after = self._get_prices(self._current_step)
        pnl_raw = self._mark_to_market(prices_before, prices_after)

        prev_portfolio_value = self._portfolio_value
        self._portfolio_value = (
            self._cash
            + np.sum(self._positions * prices_after)
        )

        daily_return = (self._portfolio_value - prev_portfolio_value) / (
            prev_portfolio_value + 1e-12
        )
        self._daily_returns.append(daily_return)

        # 6. Update peak for drawdown tracking
        if self._portfolio_value > self._peak_value:
            self._peak_value = self._portfolio_value

        # 7. Compute reward
        reward = self._compute_reward(daily_return, self._portfolio_value)

        # 8. Episode termination
        terminated = self._portfolio_value <= self.initial_capital * 0.20  # 80% loss → stop
        truncated = self._current_step >= self._max_steps

        # 9. Observation & info
        obs = self._get_observation()
        info = self._build_info(
            daily_return=daily_return,
            execution_cost=execution_cost,
            pnl_raw=pnl_raw,
        )

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), terminated, truncated, info

    # -------------------------------------------------------- reward ---------

    def _compute_reward(
        self,
        daily_return: float,
        portfolio_value: float,
    ) -> float:
        """
        Composite risk-adjusted reward:
            sharpe_component  = daily_return / rolling_vol_21d
            drawdown_penalty  = -3 * max(0, current_drawdown - 0.05)
            turnover_penalty  = -0.5 * mean(|delta_positions|)
        """
        # Sharpe component
        if len(self._daily_returns) >= 2:
            rolling_vol = float(np.std(list(self._daily_returns), ddof=1))
        else:
            rolling_vol = 0.01  # bootstrap value

        sharpe_component = daily_return / (rolling_vol + 1e-8)

        # Drawdown penalty
        current_drawdown = (self._peak_value - portfolio_value) / (
            self._peak_value + 1e-12
        )
        drawdown_penalty = -3.0 * max(0.0, current_drawdown - 0.05)

        # Turnover penalty (stored from _execute_trades)
        turnover_penalty = -0.5 * float(np.mean(np.abs(self._delta_weights)))

        reward = sharpe_component + drawdown_penalty + turnover_penalty
        return float(np.clip(reward, -10.0, 10.0))

    # ---------------------------------------------------- observation --------

    def _get_observation(self) -> Dict[str, np.ndarray]:
        step = min(self._current_step, self._max_steps)
        row = self.df.iloc[step]

        # Market features
        if self._feature_cols:
            market_features = row[self._feature_cols].values.astype(np.float32)
            market_features = np.nan_to_num(market_features, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            market_features = np.zeros(self.n_features, dtype=np.float32)

        # Portfolio state: [weights (n_assets), cash_pct, unrealised_pnl_pct]
        cash_pct = self._cash / (self._portfolio_value + 1e-12)
        unrealised_pnl_pct = (self._portfolio_value - self.initial_capital) / (
            self.initial_capital + 1e-12
        )
        portfolio_state = np.concatenate(
            [
                self._position_weights.astype(np.float32),
                np.array([cash_pct, unrealised_pnl_pct], dtype=np.float32),
            ]
        )

        # Regime
        if self._regime_col is not None:
            regime_raw = row[self._regime_col]
            regime = int(regime_raw) if not np.isnan(float(regime_raw)) else 0
        else:
            regime = 0
        regime = max(0, min(regime, self.n_regimes - 1))

        return {
            "market_features": market_features.astype(np.float32),
            "portfolio_state": portfolio_state.astype(np.float32),
            "regime": np.int64(regime),
        }

    # --------------------------------------------------- trade execution -----

    def _execute_trades(
        self,
        target_weights: np.ndarray,
        prices: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Convert target portfolio weights into actual trades respecting costs.

        Parameters
        ----------
        target_weights : np.ndarray (n_assets,)
            Desired weight per asset as fraction of current NAV.
        prices : np.ndarray (n_assets,)
            Current ask/bid mid-prices.

        Returns
        -------
        realised_weights : np.ndarray
            Actual weights after clipping and cash constraint.
        total_cost_usd : float
            Dollar cost of all trades (transaction fees + slippage).
        """
        nav = self._portfolio_value
        target_values = target_weights * nav  # target $ position per asset

        # Current $ position values
        current_values = self._positions * prices

        delta_values = target_values - current_values
        self._delta_weights = delta_values / (nav + 1e-12)  # store for reward

        total_cost_usd = 0.0
        realised_weights = np.zeros(self.n_assets, dtype=np.float64)

        for i in range(self.n_assets):
            dv = delta_values[i]
            price = prices[i]

            if price <= 0 or abs(dv) < 1.0:  # ignore sub-dollar trades
                realised_weights[i] = current_values[i] / (nav + 1e-12)
                continue

            # Slippage: adverse price movement proportional to trade size
            trade_sign = np.sign(dv)
            slippage_price_impact = price * (self.sl_bps / 10_000.0)
            effective_price = price + trade_sign * slippage_price_impact

            # Transaction cost (dollar amount)
            gross_trade = abs(dv)
            tc_dollar = gross_trade * (self.tc_bps / 10_000.0)
            slippage_dollar = gross_trade * (self.sl_bps / 10_000.0)
            total_cost_usd += tc_dollar + slippage_dollar

            # Shares traded
            shares_delta = dv / (effective_price + 1e-12)
            new_shares = self._positions[i] + shares_delta

            # Update cash
            self._cash -= (shares_delta * effective_price + tc_dollar)
            self._positions[i] = new_shares

            # Track entry price (VWAP-style)
            if new_shares > 0:
                prev_value = current_values[i]
                added_value = shares_delta * effective_price if shares_delta > 0 else 0.0
                total_sh = abs(new_shares)
                if total_sh > 1e-12:
                    self._entry_prices[i] = (
                        prev_value + added_value
                    ) / total_sh
            else:
                self._entry_prices[i] = 0.0

            self._trade_count += 1

        # Recompute weights from actual positions
        for i in range(self.n_assets):
            realised_weights[i] = (self._positions[i] * prices[i]) / (nav + 1e-12)

        return realised_weights, total_cost_usd

    # -------------------------------------------------- mark-to-market -------

    def _mark_to_market(
        self, prices_before: np.ndarray, prices_after: np.ndarray
    ) -> float:
        """Compute unrealised PnL from price movement between two bars."""
        return float(np.sum(self._positions * (prices_after - prices_before)))

    # -------------------------------------------------------- helpers ---------

    def _get_prices(self, step: int) -> np.ndarray:
        """Extract close prices for all assets at given step."""
        step = min(step, self._max_steps)
        row = self.df.iloc[step]
        if self._close_cols:
            prices = row[self._close_cols[: self.n_assets]].values.astype(np.float64)
        else:
            prices = np.ones(self.n_assets, dtype=np.float64)
        prices = np.where(prices <= 0, 1.0, prices)
        return prices

    def _build_info(
        self,
        daily_return: float = 0.0,
        execution_cost: float = 0.0,
        pnl_raw: float = 0.0,
    ) -> Dict[str, Any]:
        current_drawdown = (self._peak_value - self._portfolio_value) / (
            self._peak_value + 1e-12
        )
        return {
            "portfolio_value": self._portfolio_value,
            "cash": self._cash,
            "positions": self._positions.copy(),
            "position_weights": self._position_weights.copy(),
            "daily_return": daily_return,
            "cumulative_return": (self._portfolio_value / self.initial_capital) - 1.0,
            "current_drawdown": current_drawdown,
            "peak_value": self._peak_value,
            "execution_cost": execution_cost,
            "total_cost": self._total_cost,
            "trade_count": self._trade_count,
            "step": self._current_step,
            "pnl_raw": pnl_raw,
        }

    # ----------------------------------------------------------- render ------

    def render(self, mode: str = "human") -> Optional[str]:
        """Print a formatted portfolio summary to stdout."""
        step = self._current_step
        date = self.df.index[min(step, self._max_steps)]
        pv = self._portfolio_value
        ret = (pv / self.initial_capital - 1.0) * 100.0
        dd = (self._peak_value - pv) / (self._peak_value + 1e-12) * 100.0

        summary = (
            f"\n{'='*60}\n"
            f" Step {step:>6} | Date: {date}\n"
            f" Portfolio NAV : ${pv:>14,.2f}\n"
            f" Cumulative Ret: {ret:>+8.2f}%\n"
            f" Current DD    : {dd:>8.2f}%\n"
            f" Cash          : ${self._cash:>14,.2f}\n"
            f" Trade Count   : {self._trade_count:>6}\n"
            f" Total Costs   : ${self._total_cost:>12,.2f}\n"
            f"{'='*60}"
        )

        if mode == "human":
            print(summary)
            return None
        return summary

    def close(self) -> None:
        pass
