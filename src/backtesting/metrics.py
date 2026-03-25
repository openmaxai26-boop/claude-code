"""
PerformanceMetrics: Compute all standard trading performance metrics from scratch.

No external finance libraries — all formulas implemented directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict


@dataclass
class MetricsResult:
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_pnl: float = 0.0
    std_trade_pnl: float = 0.0
    min_trade_pnl: float = 0.0
    max_trade_pnl: float = 0.0
    expectancy: float = 0.0
    value_at_risk_95: float = 0.0
    conditional_var_95: float = 0.0
    beta: float = 0.0
    alpha: float = 0.0
    information_ratio: float = 0.0
    ulcer_index: float = 0.0
    total_return: float = 0.0
    annualized_return: float = 0.0
    annualized_volatility: float = 0.0
    n_trades: int = 0
    extra: Dict = field(default_factory=dict)


class PerformanceMetrics:
    """All performance metrics computed from scratch."""

    def sharpe_ratio(
        self,
        returns: pd.Series,
        risk_free: float = 0.05,
        periods: int = 252,
    ) -> float:
        """Annualised Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        rf_daily = risk_free / periods
        excess = returns - rf_daily
        std = returns.std(ddof=1)
        if std == 0:
            return 0.0
        return float((excess.mean() / std) * np.sqrt(periods))

    def sortino_ratio(
        self,
        returns: pd.Series,
        risk_free: float = 0.05,
        periods: int = 252,
    ) -> float:
        """Sortino ratio using downside deviation."""
        if len(returns) < 2:
            return 0.0
        rf_daily = risk_free / periods
        excess = returns - rf_daily
        downside = returns[returns < 0]
        if len(downside) == 0:
            return float("inf")
        downside_std = np.sqrt((downside**2).mean())
        if downside_std == 0:
            return 0.0
        return float((excess.mean() / downside_std) * np.sqrt(periods))

    def calmar_ratio(self, returns: pd.Series, max_dd: float) -> float:
        """Calmar ratio = annualised return / |max drawdown|."""
        if max_dd == 0:
            return 0.0
        ann_ret = (1 + returns.mean()) ** 252 - 1
        return float(ann_ret / abs(max_dd))

    def max_drawdown(self, equity_curve: pd.Series) -> float:
        """Maximum peak-to-trough drawdown as a positive fraction."""
        if len(equity_curve) == 0:
            return 0.0
        roll_max = equity_curve.cummax()
        drawdown = (equity_curve - roll_max) / roll_max.replace(0, np.nan)
        return float(abs(drawdown.min()))

    def max_drawdown_duration(self, equity_curve: pd.Series) -> int:
        """Longest drawdown period in bars."""
        if len(equity_curve) == 0:
            return 0
        roll_max = equity_curve.cummax()
        in_dd = equity_curve < roll_max
        durations = []
        current = 0
        for flag in in_dd:
            if flag:
                current += 1
            else:
                if current:
                    durations.append(current)
                current = 0
        if current:
            durations.append(current)
        return int(max(durations)) if durations else 0

    def win_rate(self, trade_log: pd.DataFrame) -> float:
        """Fraction of trades with positive PnL."""
        if len(trade_log) == 0:
            return 0.0
        return float((trade_log["pnl"] > 0).mean())

    def profit_factor(self, trade_log: pd.DataFrame) -> float:
        """Gross profit / gross loss."""
        if len(trade_log) == 0:
            return 0.0
        gross_profit = trade_log.loc[trade_log["pnl"] > 0, "pnl"].sum()
        gross_loss = abs(trade_log.loc[trade_log["pnl"] < 0, "pnl"].sum())
        if gross_loss == 0:
            return float("inf")
        return float(gross_profit / gross_loss)

    def average_trade(self, trade_log: pd.DataFrame) -> dict:
        """Basic stats of trade PnL distribution."""
        if len(trade_log) == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        pnl = trade_log["pnl"]
        return {
            "mean": float(pnl.mean()),
            "std": float(pnl.std(ddof=1)) if len(pnl) > 1 else 0.0,
            "min": float(pnl.min()),
            "max": float(pnl.max()),
        }

    def expectancy(self, trade_log: pd.DataFrame) -> float:
        """win_rate * avg_win - loss_rate * avg_loss."""
        if len(trade_log) == 0:
            return 0.0
        wins = trade_log[trade_log["pnl"] > 0]["pnl"]
        losses = trade_log[trade_log["pnl"] < 0]["pnl"]
        wr = len(wins) / len(trade_log)
        avg_win = float(wins.mean()) if len(wins) else 0.0
        avg_loss = float(abs(losses.mean())) if len(losses) else 0.0
        return float(wr * avg_win - (1 - wr) * avg_loss)

    def value_at_risk(
        self, returns: pd.Series, confidence: float = 0.95
    ) -> float:
        """Historical simulation VaR (positive = loss)."""
        if len(returns) == 0:
            return 0.0
        return float(abs(np.percentile(returns, (1 - confidence) * 100)))

    def conditional_var(
        self, returns: pd.Series, confidence: float = 0.95
    ) -> float:
        """Expected shortfall beyond VaR (positive = loss)."""
        if len(returns) == 0:
            return 0.0
        var = np.percentile(returns, (1 - confidence) * 100)
        tail = returns[returns <= var]
        if len(tail) == 0:
            return float(abs(var))
        return float(abs(tail.mean()))

    def beta_alpha(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
        periods: int = 252,
    ) -> tuple[float, float]:
        """OLS beta and annualised alpha vs benchmark."""
        if len(returns) < 2 or len(benchmark_returns) < 2:
            return 0.0, 0.0
        aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
        if len(aligned) < 2:
            return 0.0, 0.0
        r = aligned.iloc[:, 0].values
        b = aligned.iloc[:, 1].values
        bvar = np.var(b, ddof=1)
        if bvar == 0:
            return 0.0, 0.0
        beta = float(np.cov(r, b, ddof=1)[0, 1] / bvar)
        alpha_daily = float(r.mean() - beta * b.mean())
        alpha_ann = float((1 + alpha_daily) ** periods - 1)
        return beta, alpha_ann

    def information_ratio(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
        periods: int = 252,
    ) -> float:
        """Annualised information ratio."""
        aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
        if len(aligned) < 2:
            return 0.0
        active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
        te = active.std(ddof=1)
        if te == 0:
            return 0.0
        return float((active.mean() / te) * np.sqrt(periods))

    def ulcer_index(self, equity_curve: pd.Series) -> float:
        """Ulcer Index = RMS of drawdowns."""
        if len(equity_curve) == 0:
            return 0.0
        roll_max = equity_curve.cummax()
        pct_dd = (equity_curve - roll_max) / roll_max.replace(0, np.nan).fillna(equity_curve)
        return float(np.sqrt((pct_dd**2).mean()))

    def compute_all(
        self,
        equity_curve: pd.Series,
        trade_log: pd.DataFrame,
        benchmark_returns: Optional[pd.Series] = None,
        periods: int = 252,
        risk_free: float = 0.05,
    ) -> MetricsResult:
        """Compute every metric and return a MetricsResult dataclass."""
        returns = equity_curve.pct_change().dropna()

        max_dd = self.max_drawdown(equity_curve)
        avg_trade = self.average_trade(trade_log)
        beta, alpha = (
            self.beta_alpha(returns, benchmark_returns, periods)
            if benchmark_returns is not None
            else (0.0, 0.0)
        )
        ir = (
            self.information_ratio(returns, benchmark_returns, periods)
            if benchmark_returns is not None
            else 0.0
        )

        total_ret = float(
            (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
            if len(equity_curve) >= 2
            else 0.0
        )
        n_days = len(returns)
        ann_ret = float((1 + total_ret) ** (periods / max(n_days, 1)) - 1)
        ann_vol = float(returns.std(ddof=1) * np.sqrt(periods)) if len(returns) > 1 else 0.0

        return MetricsResult(
            sharpe_ratio=self.sharpe_ratio(returns, risk_free, periods),
            sortino_ratio=self.sortino_ratio(returns, risk_free, periods),
            calmar_ratio=self.calmar_ratio(returns, max_dd),
            max_drawdown=max_dd,
            max_drawdown_duration=self.max_drawdown_duration(equity_curve),
            win_rate=self.win_rate(trade_log),
            profit_factor=self.profit_factor(trade_log),
            avg_trade_pnl=avg_trade["mean"],
            std_trade_pnl=avg_trade["std"],
            min_trade_pnl=avg_trade["min"],
            max_trade_pnl=avg_trade["max"],
            expectancy=self.expectancy(trade_log),
            value_at_risk_95=self.value_at_risk(returns, 0.95),
            conditional_var_95=self.conditional_var(returns, 0.95),
            beta=beta,
            alpha=alpha,
            information_ratio=ir,
            ulcer_index=self.ulcer_index(equity_curve),
            total_return=total_ret,
            annualized_return=ann_ret,
            annualized_volatility=ann_vol,
            n_trades=len(trade_log),
        )
