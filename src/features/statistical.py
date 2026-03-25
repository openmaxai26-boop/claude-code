"""
Statistical Feature Engine
Implements statistical / quantitative features for financial time series.
GARCH(1,1) is implemented via manual log-likelihood optimised with scipy.
Hurst exponent is computed via classical R/S analysis.
No arch package required.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class StatisticalFeatureEngine:
    """
    Computes a comprehensive set of statistical and risk features for
    financial return series, including volatility modelling, tail-risk
    metrics and market-structure statistics.
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Returns
    # ------------------------------------------------------------------
    def compute_returns(
        self,
        close: pd.Series,
        periods: List[int] = None,
    ) -> pd.DataFrame:
        """
        Log returns over multiple look-back windows.

        r(t, n) = ln(close_t / close_{t-n})
        """
        if periods is None:
            periods = [1, 5, 10, 21, 63]

        out: Dict[str, pd.Series] = {}
        for n in periods:
            out[f"return_{n}d"] = np.log(close / close.shift(n))

        return pd.DataFrame(out, index=close.index)

    # ------------------------------------------------------------------
    # Realized Volatility
    # ------------------------------------------------------------------
    def compute_volatility(
        self,
        returns: pd.Series,
        windows: List[int] = None,
    ) -> pd.DataFrame:
        """
        Realized (annualized) volatility over multiple rolling windows.

        vol(t, w) = std(r, window=w) * sqrt(252)
        """
        if windows is None:
            windows = [5, 10, 21, 63]

        out: Dict[str, pd.Series] = {}
        for w in windows:
            out[f"vol_{w}d"] = returns.rolling(w).std(ddof=1) * np.sqrt(252)

        return pd.DataFrame(out, index=returns.index)

    # ------------------------------------------------------------------
    # GARCH(1,1) volatility clustering
    # ------------------------------------------------------------------
    @staticmethod
    def _garch11_loglik(params: np.ndarray, returns: np.ndarray) -> float:
        """
        Negative log-likelihood for GARCH(1,1).

        sigma^2_t = omega + alpha * eps^2_{t-1} + beta * sigma^2_{t-1}
        """
        omega, alpha, beta = params

        # Parameter constraints enforced via penalty
        if omega <= 0 or alpha < 0 or beta < 0 or (alpha + beta) >= 1:
            return 1e10

        n = len(returns)
        sigma2 = np.empty(n)
        # Initialise with unconditional variance
        unc_var = omega / max(1 - alpha - beta, 1e-8)
        sigma2[0] = unc_var

        for t in range(1, n):
            sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]

        # Guard against non-positive variances
        sigma2 = np.maximum(sigma2, 1e-12)

        loglik = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + returns ** 2 / sigma2)
        return -loglik  # minimise negative log-likelihood

    def compute_volatility_clustering(
        self,
        returns: pd.Series,
    ) -> pd.DataFrame:
        """
        Fit GARCH(1,1) on the full return series and return:
        - garch_vol     : conditional annualized volatility at each time step
        - garch_persistence : alpha + beta (persistence coefficient)

        The optimisation uses L-BFGS-B with several restarts.
        """
        r = returns.dropna().values.astype(float)

        best_result = None
        best_nll = np.inf

        # Multiple starting points to avoid local minima
        starts = [
            [1e-6, 0.05, 0.90],
            [1e-5, 0.10, 0.85],
            [5e-6, 0.08, 0.88],
            [1e-4, 0.15, 0.80],
        ]

        for x0 in starts:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = minimize(
                    self._garch11_loglik,
                    x0=x0,
                    args=(r,),
                    method="L-BFGS-B",
                    bounds=[(1e-8, 1.0), (1e-8, 1.0), (1e-8, 1.0)],
                    options={"maxiter": 2000, "ftol": 1e-12},
                )
            if res.fun < best_nll:
                best_nll = res.fun
                best_result = res

        omega, alpha, beta = best_result.x
        persistence = float(alpha + beta)

        # Reconstruct sigma^2 path
        n = len(r)
        sigma2 = np.empty(n)
        unc_var = omega / max(1 - alpha - beta, 1e-8)
        sigma2[0] = unc_var
        for t in range(1, n):
            sigma2[t] = omega + alpha * r[t - 1] ** 2 + beta * sigma2[t - 1]

        garch_vol_series = pd.Series(
            np.sqrt(np.maximum(sigma2, 0)) * np.sqrt(252),
            index=returns.dropna().index,
        ).reindex(returns.index)

        persistence_series = pd.Series(persistence, index=returns.index)

        return pd.DataFrame(
            {
                "garch_vol": garch_vol_series,
                "garch_persistence": persistence_series,
            },
            index=returns.index,
        )

    # ------------------------------------------------------------------
    # Rolling Skewness & Kurtosis
    # ------------------------------------------------------------------
    def compute_skewness_kurtosis(
        self,
        returns: pd.Series,
        window: int = 21,
    ) -> pd.DataFrame:
        """
        Rolling skewness and excess kurtosis.

        skew     = E[(r-mu)^3] / sigma^3   (Fisher definition)
        kurtosis = E[(r-mu)^4] / sigma^4 - 3  (excess)
        """
        skew = returns.rolling(window).skew()
        kurt = returns.rolling(window).kurt()  # pandas returns excess kurtosis

        return pd.DataFrame(
            {"rolling_skew": skew, "rolling_kurtosis": kurt},
            index=returns.index,
        )

    # ------------------------------------------------------------------
    # Rolling Sharpe Ratio
    # ------------------------------------------------------------------
    def compute_sharpe(
        self,
        returns: pd.Series,
        risk_free: float = 0.05,
        window: int = 252,
    ) -> pd.DataFrame:
        """
        Rolling annualized Sharpe ratio.

        Sharpe = (mean_return_annualized - risk_free) / (vol_annualized)
        """
        daily_rf = risk_free / 252.0
        excess = returns - daily_rf

        mean_excess = excess.rolling(window).mean() * 252
        std_ann = returns.rolling(window).std(ddof=1) * np.sqrt(252)

        sharpe = mean_excess / std_ann.replace(0.0, np.nan)

        return pd.DataFrame({"rolling_sharpe": sharpe}, index=returns.index)

    # ------------------------------------------------------------------
    # Rolling Correlation Matrix
    # ------------------------------------------------------------------
    def compute_correlation_matrix(
        self,
        returns_df: pd.DataFrame,
        window: int = 63,
    ) -> pd.DataFrame:
        """
        Rolling pairwise correlation statistics.

        For each time t, compute the correlation matrix over [t-window, t]
        and return the mean pairwise off-diagonal correlation.
        """
        cols = returns_df.columns.tolist()
        n_assets = len(cols)

        if n_assets < 2:
            raise ValueError("Need at least 2 assets for correlation matrix.")

        mean_corr = pd.Series(index=returns_df.index, dtype=float)

        arr = returns_df.values  # (T, N)
        T = arr.shape[0]

        for t in range(window, T + 1):
            window_data = arr[t - window : t, :]
            # Drop assets with zero variance
            stds = window_data.std(axis=0)
            valid = stds > 0
            if valid.sum() < 2:
                mean_corr.iloc[t - 1] = np.nan
                continue
            sub = window_data[:, valid]
            corr = np.corrcoef(sub.T)
            n = corr.shape[0]
            mask = ~np.eye(n, dtype=bool)
            mean_corr.iloc[t - 1] = corr[mask].mean()

        return pd.DataFrame(
            {"mean_pairwise_corr": mean_corr},
            index=returns_df.index,
        )

    # ------------------------------------------------------------------
    # Beta / Alpha
    # ------------------------------------------------------------------
    def compute_beta(
        self,
        asset_returns: pd.Series,
        market_returns: pd.Series,
        window: int = 63,
    ) -> pd.DataFrame:
        """
        Rolling OLS beta and Jensen's alpha vs. market.

        beta  = cov(r_i, r_m) / var(r_m)
        alpha = mean(r_i) - beta * mean(r_m)   (daily, then annualized)
        """
        aligned = pd.concat(
            [asset_returns.rename("asset"), market_returns.rename("market")], axis=1
        ).dropna()

        beta_series = pd.Series(index=asset_returns.index, dtype=float)
        alpha_series = pd.Series(index=asset_returns.index, dtype=float)

        for t in range(window, len(aligned) + 1):
            chunk = aligned.iloc[t - window : t]
            r_a = chunk["asset"].values
            r_m = chunk["market"].values
            var_m = r_m.var(ddof=1)
            if var_m == 0:
                continue
            b = np.cov(r_a, r_m, ddof=1)[0, 1] / var_m
            a = (r_a.mean() - b * r_m.mean()) * 252  # annualized alpha
            idx = aligned.index[t - 1]
            beta_series[idx] = b
            alpha_series[idx] = a

        return pd.DataFrame(
            {"rolling_beta": beta_series, "rolling_alpha": alpha_series},
            index=asset_returns.index,
        )

    # ------------------------------------------------------------------
    # Hurst Exponent via R/S Analysis
    # ------------------------------------------------------------------
    def compute_hurst_exponent(
        self,
        price: pd.Series,
        window: int = 100,
    ) -> pd.DataFrame:
        """
        Hurst exponent using classical Rescaled Range (R/S) analysis.

        For a sub-series of length n:
          1. Compute mean-adjusted cumulative sum (Z).
          2. R = max(Z) - min(Z)
          3. S = std(sub-series)
          4. RS(n) = R / S
        Fit: ln(RS) = H * ln(n) + c  over multiple lags.

        H < 0.5  -> mean-reverting
        H = 0.5  -> random walk
        H > 0.5  -> trending
        """

        def _rs_single(x: np.ndarray) -> float:
            """R/S statistic for a 1D array."""
            n = len(x)
            mean = x.mean()
            z = np.cumsum(x - mean)
            r = z.max() - z.min()
            s = x.std(ddof=1)
            if s == 0:
                return np.nan
            return r / s

        def _hurst(x: np.ndarray) -> float:
            n = len(x)
            if n < 20:
                return np.nan
            lags = np.unique(
                np.floor(np.logspace(np.log10(10), np.log10(n), 20)).astype(int)
            )
            lags = lags[lags >= 10]
            rs_vals = []
            for lag in lags:
                rs_list = [
                    _rs_single(x[i : i + lag])
                    for i in range(0, n - lag + 1, lag)
                ]
                rs_list = [v for v in rs_list if not np.isnan(v) and v > 0]
                if rs_list:
                    rs_vals.append((lag, np.mean(rs_list)))
            if len(rs_vals) < 4:
                return np.nan
            lags_arr = np.log([v[0] for v in rs_vals])
            rs_arr = np.log([v[1] for v in rs_vals])
            # OLS slope
            A = np.column_stack([lags_arr, np.ones_like(lags_arr)])
            coef, *_ = np.linalg.lstsq(A, rs_arr, rcond=None)
            return float(coef[0])

        log_price = np.log(price.replace(0.0, np.nan)).dropna()
        hurst_vals = pd.Series(index=price.index, dtype=float)

        for t in range(window, len(log_price) + 1):
            chunk = log_price.iloc[t - window : t].values
            h = _hurst(chunk)
            hurst_vals[log_price.index[t - 1]] = h

        return pd.DataFrame({"hurst_exponent": hurst_vals}, index=price.index)

    # ------------------------------------------------------------------
    # Drawdown Series
    # ------------------------------------------------------------------
    def compute_drawdown_series(
        self,
        equity: pd.Series,
    ) -> pd.DataFrame:
        """
        Drawdown series, maximum drawdown, and duration.

        drawdown(t) = (equity(t) - running_max(t)) / running_max(t)
        max_drawdown = min(drawdown)
        duration = number of consecutive bars in drawdown (below previous peak)
        """
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max.replace(0.0, np.nan)

        max_dd = drawdown.min()  # scalar, most negative value

        # Duration: length of current drawdown streak
        in_dd = (drawdown < 0).astype(int)
        duration = in_dd.groupby((in_dd != in_dd.shift()).cumsum()).cumsum()

        max_dd_series = pd.Series(max_dd, index=equity.index)

        return pd.DataFrame(
            {
                "drawdown": drawdown,
                "max_drawdown": max_dd_series,
                "drawdown_duration": duration,
            },
            index=equity.index,
        )

    # ------------------------------------------------------------------
    # Tail Risk
    # ------------------------------------------------------------------
    def compute_tail_risk(
        self,
        returns: pd.Series,
    ) -> pd.DataFrame:
        """
        Value-at-Risk and Conditional Value-at-Risk via historical simulation.

        VaR(alpha)  = -quantile(returns, 1-alpha)
        CVaR(alpha) = -mean(returns[returns <= -VaR])
        """
        r = returns.dropna()

        results: Dict[str, float] = {}
        for confidence in [0.95, 0.99]:
            level = 1.0 - confidence
            var = float(-np.quantile(r, level))
            tail = r[r <= -var]
            cvar = float(-tail.mean()) if len(tail) > 0 else var
            pct = int(confidence * 100)
            results[f"var_{pct}"] = var
            results[f"cvar_{pct}"] = cvar

        out = pd.DataFrame(
            {k: pd.Series(v, index=returns.index) for k, v in results.items()},
            index=returns.index,
        )
        return out

    # ------------------------------------------------------------------
    # Master feature builder
    # ------------------------------------------------------------------
    def compute_all_features(
        self,
        df: pd.DataFrame,
        market_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute all statistical features.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain at least a 'close' column (and optionally 'volume').
        market_df : pd.DataFrame
            Market benchmark; must contain a 'close' column.

        Returns
        -------
        pd.DataFrame with all statistical features.
        """
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        market_df = market_df.copy()
        market_df.columns = [c.lower() for c in market_df.columns]

        close = df["close"]
        market_close = market_df["close"]

        returns_df_multi = self.compute_returns(close)
        r1d = returns_df_multi["return_1d"]

        market_returns_df = self.compute_returns(market_close)
        market_r1d = market_returns_df["return_1d"]

        frames = [
            returns_df_multi,
            self.compute_volatility(r1d),
            self.compute_volatility_clustering(r1d),
            self.compute_skewness_kurtosis(r1d),
            self.compute_sharpe(r1d),
            self.compute_beta(r1d, market_r1d),
            self.compute_hurst_exponent(close),
            self.compute_drawdown_series(close),
            self.compute_tail_risk(r1d),
        ]

        result = pd.concat(frames, axis=1)
        result = result.ffill(limit=1)
        result = result.dropna(how="all")

        return result
