"""
BacktestEngine: Realistic bar-by-bar simulation with no look-ahead bias.

Supports walk-forward validation and Monte Carlo simulation.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.backtesting.metrics import MetricsResult, PerformanceMetrics

logger = logging.getLogger(__name__)


def _safe_float(value: float, default: float = 0.0) -> float:
        """Convert NaN / Inf float values to a JSON-safe default."""
        try:
                    if value is None:
                                    return default
                                f = float(value)
                    if math.isnan(f) or math.isinf(f):
                                    return default
                                return f
except (TypeError, ValueError):
        return default


@dataclass
class ExecutionResult:
        symbol: str
        requested_size: float
        executed_size: float
        execution_price: float
        slippage_cost: float
        transaction_cost: float
        total_cost: float


@dataclass
class BacktestResults:
        equity_curve: pd.Series
        trade_log: pd.DataFrame
        positions_history: pd.DataFrame
        metrics: MetricsResult
        walk_forward_results: List["BacktestResults"] = field(default_factory=list)

    def summary(self) -> dict:
                m = self.metrics
                return {
                    "total_return_pct": _safe_float(round(_safe_float(m.total_return) * 100, 2)),
                    "annualized_return_pct": _safe_float(round(_safe_float(m.annualized_return) * 100, 2)),
                    "annualized_vol_pct": _safe_float(round(_safe_float(m.annualized_volatility) * 100, 2)),
                    "sharpe_ratio": _safe_float(round(_safe_float(m.sharpe_ratio), 3)),
                    "sortino_ratio": _safe_float(round(_safe_float(m.sortino_ratio), 3)),
                    "calmar_ratio": _safe_float(round(_safe_float(m.calmar_ratio), 3)),
                    "max_drawdown_pct": _safe_float(round(_safe_float(m.max_drawdown) * 100, 2)),
                    "max_drawdown_duration_bars": int(_safe_float(m.max_drawdown_duration)),
                    "win_rate_pct": _safe_float(round(_safe_float(m.win_rate) * 100, 2)),
                    "profit_factor": _safe_float(round(_safe_float(m.profit_factor), 3)),
                    "expectancy": _safe_float(round(_safe_float(m.expectancy), 2)),
                    "n_trades": m.n_trades,
                    "var_95_pct": _safe_float(round(_safe_float(m.value_at_risk_95) * 100, 3)),
                    "cvar_95_pct": _safe_float(round(_safe_float(m.conditional_var_95) * 100, 3)),
                    "ulcer_index": _safe_float(round(_safe_float(m.ulcer_index), 4)),
                }


class BacktestEngine:
        """
            Realistic backtesting engine.

                Parameters
                    ----------
                        initial_capital : Starting cash in USD.
                            transaction_cost_bps : One-way transaction cost in basis points.
                                slippage_bps : Base slippage in bps; actual = bps * sqrt(size / avg_volume).
                                    min_trade_usd : Minimum trade notional to execute.
                                        latency_ms : Simulated latency (informational, not currently modelled in time).
                                            """

    def __init__(
                self,
                initial_capital: float = 1_000_000,
                transaction_cost_bps: float = 5,
                slippage_bps: float = 3,
                min_trade_usd: float = 1_000,
                latency_ms: int = 50,
    ) -> None:
                self.initial_capital = initial_capital
                self.tc_bps = transaction_cost_bps / 10_000
                self.slippage_bps = slippage_bps / 10_000
                self.min_trade_usd = min_trade_usd
                self.latency_ms = latency_ms
                self._metrics = PerformanceMetrics()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
                self,
                signals_df: pd.DataFrame,
                prices_df: pd.DataFrame,
                risk_manager: Optional[Any] = None,
                portfolio_optimizer: Optional[Any] = None,
    ) -> BacktestResults:
                """
                        Run a bar-by-bar backtest.

                                Parameters
                                        ----------
                                                signals_df : DataFrame indexed by date; columns = assets;
                                                                     values = target weight in [-1, 1] (0 = flat).
                                                                             prices_df  : DataFrame indexed by date; columns = assets;
                                                                                                  OHLCV columns expected as MultiIndex (asset, field)
                                                                                                                       or simple close prices.
                                                                                                                               risk_manager : Optional RiskManager instance for limit checks.
                                                                                                                                       portfolio_optimizer: Ignored in this loop (weights come from signals_df).
                                                                                                                                       
                                                                                                                                               Returns
                                                                                                                                                       -------
                                                                                                                                                               BacktestResults
                                                                                                                                                                       """
                # Normalise prices to close only
                prices = self._extract_close(prices_df)
        volumes = self._extract_volume(prices_df)

        # Align dates
        common_dates = signals_df.index.intersection(prices.index)
        signals = signals_df.loc[common_dates]
        prices = prices.loc[common_dates]
        volumes = volumes.loc[common_dates] if volumes is not None else None

        cash = self.initial_capital
        positions: Dict[str, float] = {col: 0.0 for col in signals.columns}
        equity_history: Dict[pd.Timestamp, float] = {}
        position_history: List[dict] = []
        trade_log_rows: List[dict] = []

        for i, date in enumerate(common_dates):
                        row_prices = prices.loc[date]
                        row_signals = signals.loc[date]
                        row_volumes = volumes.loc[date] if volumes is not None else None

            # Portfolio value at open of bar
                        port_value = cash + sum(
                            positions.get(sym, 0.0) * float(row_prices.get(sym, 0.0))
                            for sym in positions
                        )

            # Target positions in USD notional
                        target_weights = row_signals.clip(-1, 1)
                        for sym in target_weights.index:
                                            target_w = float(target_weights[sym])
                                            price = float(row_prices.get(sym, np.nan))
                                            if np.isnan(price) or price <= 0:
                                                                    continue
                                                                target_notional = target_w * port_value
                                            current_notional = positions.get(sym, 0.0) * price
                                            delta_notional = target_notional - current_notional
                                            if abs(delta_notional) < self.min_trade_usd:
                                                                    continue
                                                                avg_vol = (
                                                float(row_volumes[sym])
                                                if row_volumes is not None and sym in row_volumes
                                                else 1e6
                                            )
                                            exec_result = self._simulate_execution(
                                                sym, price, avg_vol, delta_notional
                                            )
                                            shares_delta = exec_result.executed_size / exec_result.execution_price
                                            positions[sym] = positions.get(sym, 0.0) + shares_delta
                                            cash -= (
                                                exec_result.executed_size
                                                + exec_result.transaction_cost
                                                + exec_result.slippage_cost
                                            )
                                            if exec_result.executed_size != 0:
                                                                    trade_log_rows.append(
                                                                                                {
                                                                                                                                "date": date,
                                                                                                                                "symbol": sym,
                                                                                                                                "shares_delta": shares_delta,
                                                                                                                                "execution_price": exec_result.execution_price,
                                                                                                                                "notional": exec_result.executed_size,
                                                                                                                                "transaction_cost": exec_result.transaction_cost,
                                                                                                                                "slippage_cost": exec_result.slippage_cost,
                                                                                                                                "pnl": 0.0,  # filled at close
                                                                                                    }
                                                                    )

                                        # End-of-bar portfolio value
                                        eob_value = cash + sum(
                                                            positions.get(sym, 0.0) * float(row_prices.get(sym, 0.0))
                                                            for sym in positions
                                        )
            equity_history[date] = eob_value
            position_history.append(
                                {"date": date, "cash": cash, **{s: positions.get(s, 0.0) for s in signals.columns}}
            )

        equity_curve = pd.Series(equity_history)
        trade_log = self._build_trade_log(trade_log_rows, prices)
        positions_history = pd.DataFrame(position_history).set_index("date")
        metrics = self._metrics.compute_all(equity_curve, trade_log)
        return BacktestResults(
                        equity_curve=equity_curve,
                        trade_log=trade_log,
                        positions_history=positions_history,
                        metrics=metrics,
        )

    def walk_forward_validation(
                self,
                data_df: pd.DataFrame,
                n_splits: int = 5,
                train_window: int = 252,
                test_window: int = 63,
    ) -> List[BacktestResults]:
                """
                        Expanding / rolling walk-forward validation.
                                Returns one BacktestResults per out-of-sample window.
                                        """
        results = []
        total_len = len(data_df)
        start = train_window
        for split in range(n_splits):
                        test_start = start + split * test_window
            test_end = test_start + test_window
            if test_end > total_len:
                                break
                            test_data = data_df.iloc[test_start:test_end]
            close_cols = (
                                [c for c in test_data.columns if "close" in str(c).lower()]
                                or test_data.columns.tolist()
            )
            prices = test_data[close_cols[:1]].copy()
            prices.columns = ["ASSET"]
            signals = pd.DataFrame(
                                0.0, index=prices.index, columns=prices.columns
            )
            result = self.run(signals, prices)
            results.append(result)
            logger.info("Walk-forward split %d/%d complete", split + 1, n_splits)
        return results

    def monte_carlo_simulation(
                self,
                returns: pd.Series,
                n_simulations: int = 1_000,
                n_days: int = 252,
    ) -> pd.DataFrame:
                """
                        Bootstrap Monte Carlo over historical return distribution.
                                Returns a DataFrame of shape (n_days, n_simulations) containing
                                        simulated equity paths (starting from 1.0).
                                                """
        ret_arr = returns.dropna().values
        paths = np.ones((n_days + 1, n_simulations))
        rng = np.random.default_rng(42)
        for sim in range(n_simulations):
                        sampled = rng.choice(ret_arr, size=n_days, replace=True)
            paths[1:, sim] = np.cumprod(1 + sampled)
        return pd.DataFrame(paths, columns=[f"sim_{i}" for i in range(n_simulations)])

    def generate_report(self, results: BacktestResults) -> dict:
                """Return full performance report as a plain dict (JSON-safe)."""
        report = results.summary()
        report["avg_trade_pnl"] = _safe_float(round(_safe_float(results.metrics.avg_trade_pnl), 2))
        report["std_trade_pnl"] = _safe_float(round(_safe_float(results.metrics.std_trade_pnl), 2))
        report["beta"] = _safe_float(round(_safe_float(results.metrics.beta), 3))
        report["alpha_annualized"] = _safe_float(round(_safe_float(results.metrics.alpha), 4))
        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _simulate_execution(
                self,
                symbol: str,
                price: float,
                avg_volume: float,
                delta_notional: float,
    ) -> ExecutionResult:
                """Apply market impact model and transaction costs."""
        size = abs(delta_notional)
        direction = np.sign(delta_notional)
        # Market impact: slippage grows with sqrt of participation rate
        participation = size / max(avg_volume * price, 1)
        impact_pct = self.slippage_bps * np.sqrt(participation)
        slippage_cost = size * impact_pct
        # Execution price (adversarial)
        exec_price = price * (1 + direction * impact_pct)
        transaction_cost = size * self.tc_bps
        return ExecutionResult(
                        symbol=symbol,
                        requested_size=delta_notional,
                        executed_size=delta_notional,
                        execution_price=exec_price,
                        slippage_cost=slippage_cost,
                        transaction_cost=transaction_cost,
                        total_cost=slippage_cost + transaction_cost,
        )

    @staticmethod
    def _extract_close(df: pd.DataFrame) -> pd.DataFrame:
                """Return a DataFrame of close prices from various input formats."""
        if isinstance(df.columns, pd.MultiIndex):
                        # (symbol, field) multi-index
                        try:
                                            return df.xs("Close", axis=1, level=1)
except KeyError:
                return df.xs(df.columns.get_level_values(1)[0], axis=1, level=1)
        close_cols = [c for c in df.columns if "close" in str(c).lower()]
        if close_cols:
                        return df[close_cols].rename(columns=lambda c: c.replace("_close", "").replace("close_", ""))
        return df

    @staticmethod
    def _extract_volume(df: pd.DataFrame) -> Optional[pd.DataFrame]:
                """Return volume DataFrame or None."""
        if isinstance(df.columns, pd.MultiIndex):
                        try:
                                            return df.xs("Volume", axis=1, level=1)
except KeyError:
                return None
        vol_cols = [c for c in df.columns if "volume" in str(c).lower()]
        if vol_cols:
                        return df[vol_cols].rename(columns=lambda c: c.replace("_volume", "").replace("volume_", ""))
        return None

    @staticmethod
    def _build_trade_log(rows: List[dict], prices: pd.DataFrame) -> pd.DataFrame:
                """Convert raw trade rows to a proper trade log with PnL."""
        if not rows:
                        return pd.DataFrame(
                            columns=[
                                                    "date",
                                                    "symbol",
                                                    "shares_delta",
                                                    "execution_price",
                                                    "notional",
                                                    "transaction_cost",
                                                    "slippage_cost",
                                                    "pnl",
                                                    "return_pct",
                                                    "holding_days",
                            ]
        )
        df = pd.DataFrame(rows)
        # Simplified PnL: mark-to-market at last available price
        last_prices = prices.iloc[-1]
        df["pnl"] = df.apply(
                        lambda r: r["shares_delta"]
                        * (float(last_prices.get(r["symbol"], r["execution_price"])) - r["execution_price"])
                        - r["transaction_cost"]
                        - r["slippage_cost"],
                        axis=1,
        )
        df["return_pct"] = df["pnl"] / df["notional"].abs().replace(0, np.nan)
        df["holding_days"] = 0  # simplified
        return df
