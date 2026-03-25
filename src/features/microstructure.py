"""
Microstructure Feature Engine
Implements market-microstructure metrics from scratch using NumPy/Pandas.
Covers order imbalance, liquidity, VPIN, Kyle lambda, Amihud, Roll spread.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class MicrostructureFeatureEngine:
    """
    Computes market microstructure features that characterise liquidity,
    order flow and the mechanics of price discovery.
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Order Imbalance
    # ------------------------------------------------------------------
    def compute_order_imbalance(
        self,
        bids_vol: pd.Series,
        asks_vol: pd.Series,
    ) -> pd.DataFrame:
        """
        Order imbalance ratio (OIR) and signed order flow.

        OIR           = (bids_vol - asks_vol) / (bids_vol + asks_vol)
        signed_flow   = bids_vol - asks_vol   (raw net buying pressure)
        """
        total = (bids_vol + asks_vol).replace(0.0, np.nan)
        oir = (bids_vol - asks_vol) / total
        signed_flow = bids_vol - asks_vol

        return pd.DataFrame(
            {
                "order_imbalance": oir,
                "signed_order_flow": signed_flow,
            },
            index=bids_vol.index,
        )

    # ------------------------------------------------------------------
    # Bid-Ask Spread
    # ------------------------------------------------------------------
    def compute_bid_ask_spread(
        self,
        bid: pd.Series,
        ask: pd.Series,
    ) -> pd.DataFrame:
        """
        Bid-ask spread in basis points and relative spread.

        spread_bps      = (ask - bid) / mid * 10_000
        relative_spread = (ask - bid) / mid
        """
        mid = (bid + ask) / 2.0
        spread = ask - bid
        spread_bps = spread / mid.replace(0.0, np.nan) * 10_000.0
        relative_spread = spread / mid.replace(0.0, np.nan)

        return pd.DataFrame(
            {
                "spread_bps": spread_bps,
                "relative_spread": relative_spread,
            },
            index=bid.index,
        )

    # ------------------------------------------------------------------
    # Liquidity Pressure
    # ------------------------------------------------------------------
    def compute_liquidity_pressure(
        self,
        order_book_dict: Dict,
    ) -> pd.DataFrame:
        """
        Liquidity pressure from a snapshot order book dictionary.

        Expected format::
            {
                'bids': [(price, volume), ...],   # sorted best→worst
                'asks': [(price, volume), ...],   # sorted best→worst
                'timestamp': pd.Timestamp | None,
            }

        Returns a single-row DataFrame with:
        - weighted_bid_pressure  : volume-weighted mean bid deviation from mid
        - weighted_ask_pressure  : volume-weighted mean ask deviation from mid
        - depth_imbalance_5      : (bid_vol_top5 - ask_vol_top5) / total_top5
        - depth_imbalance_10     : same for top 10 levels
        - depth_imbalance_20     : same for top 20 levels
        """
        bids: List[tuple] = order_book_dict.get("bids", [])
        asks: List[tuple] = order_book_dict.get("asks", [])
        ts = order_book_dict.get("timestamp", None)

        def _depth_imbalance(bids_slice, asks_slice) -> float:
            bv = sum(v for _, v in bids_slice)
            av = sum(v for _, v in asks_slice)
            total = bv + av
            return (bv - av) / total if total > 0 else np.nan

        def _weighted_pressure(levels, mid, side: str) -> float:
            if not levels or mid == 0:
                return np.nan
            prices = np.array([p for p, _ in levels], dtype=float)
            vols = np.array([v for _, v in levels], dtype=float)
            total_vol = vols.sum()
            if total_vol == 0:
                return np.nan
            deviations = np.abs(prices - mid) / mid
            return float(np.average(deviations, weights=vols))

        best_bid = bids[0][0] if bids else np.nan
        best_ask = asks[0][0] if asks else np.nan
        mid = (best_bid + best_ask) / 2.0 if (bids and asks) else np.nan

        data = {
            "weighted_bid_pressure": _weighted_pressure(bids, mid, "bid"),
            "weighted_ask_pressure": _weighted_pressure(asks, mid, "ask"),
            "depth_imbalance_5": _depth_imbalance(bids[:5], asks[:5]),
            "depth_imbalance_10": _depth_imbalance(bids[:10], asks[:10]),
            "depth_imbalance_20": _depth_imbalance(bids[:20], asks[:20]),
        }

        idx = [ts] if ts is not None else [0]
        return pd.DataFrame(data, index=idx)

    # ------------------------------------------------------------------
    # Volume Profile
    # ------------------------------------------------------------------
    def compute_volume_profile(
        self,
        price: pd.Series,
        volume: pd.Series,
        n_bins: int = 20,
    ) -> pd.DataFrame:
        """
        Volume-at-price histogram with:
        - POC  : Point of Control (price bin with most volume)
        - VAH  : Value Area High (top of range containing 70 % of volume)
        - VAL  : Value Area Low
        """
        valid = ~(price.isna() | volume.isna())
        p = price[valid].values
        v = volume[valid].values

        if len(p) == 0:
            empty = pd.DataFrame(
                {"poc": [np.nan], "vah": [np.nan], "val": [np.nan]},
                index=[price.index[-1]] if len(price) else [0],
            )
            return empty

        bins = np.linspace(p.min(), p.max(), n_bins + 1)
        bin_indices = np.digitize(p, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        vol_at_price = np.zeros(n_bins)
        for i, vi in zip(bin_indices, v):
            vol_at_price[i] += vi

        poc_bin = int(np.argmax(vol_at_price))
        poc_price = float((bins[poc_bin] + bins[poc_bin + 1]) / 2.0)

        # Value area: accumulate from POC outward until >= 70 % total volume
        total_vol = vol_at_price.sum()
        target = 0.70 * total_vol
        accumulated = vol_at_price[poc_bin]
        lo, hi = poc_bin, poc_bin

        while accumulated < target:
            expand_lo = lo > 0
            expand_hi = hi < n_bins - 1

            if expand_lo and expand_hi:
                if vol_at_price[lo - 1] >= vol_at_price[hi + 1]:
                    lo -= 1
                    accumulated += vol_at_price[lo]
                else:
                    hi += 1
                    accumulated += vol_at_price[hi]
            elif expand_lo:
                lo -= 1
                accumulated += vol_at_price[lo]
            elif expand_hi:
                hi += 1
                accumulated += vol_at_price[hi]
            else:
                break

        val_price = float((bins[lo] + bins[lo + 1]) / 2.0)
        vah_price = float((bins[hi] + bins[hi + 1]) / 2.0)

        # Build histogram as a single row
        bin_centers = [(bins[i] + bins[i + 1]) / 2.0 for i in range(n_bins)]
        hist_data: Dict[str, float] = {
            f"vol_at_price_{i}": float(vol_at_price[i]) for i in range(n_bins)
        }
        hist_data.update({"poc": poc_price, "vah": vah_price, "val": val_price})

        idx = [price.index[-1]]
        return pd.DataFrame(hist_data, index=idx)

    # ------------------------------------------------------------------
    # VPIN
    # ------------------------------------------------------------------
    def compute_vpin(
        self,
        volume: pd.Series,
        buy_vol: pd.Series,
        sell_vol: pd.Series,
        bucket_size: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Volume-synchronized Probability of Informed Trading (Easley et al. 2012).

        VPIN_t = (1/n) * sum_{i=t-n+1}^{t} |V_buy_i - V_sell_i| / V_i

        We split the time series into equal-volume buckets. For each bucket
        the net order flow is |buy_vol - sell_vol| / bucket_volume.

        VPIN is the rolling mean over 50 buckets.
        """
        if bucket_size is None:
            bucket_size = float(volume.sum() / max(len(volume) // 50, 1))

        bucket_size = max(bucket_size, 1e-8)

        cum_vol = volume.cumsum().values
        buy = buy_vol.values
        sell = sell_vol.values
        vol = volume.values
        idx = volume.index

        bucket_imbalances: List[float] = []
        bucket_timestamps: List = []

        current_bucket_vol = 0.0
        current_buy = 0.0
        current_sell = 0.0

        for i in range(len(vol)):
            remaining = bucket_size - current_bucket_vol
            trade_vol = vol[i]

            if trade_vol == 0:
                continue

            while trade_vol >= remaining:
                frac = remaining / trade_vol
                current_buy += buy[i] * frac
                current_sell += sell[i] * frac
                trade_vol -= remaining

                bucket_vol_total = current_buy + current_sell
                imbalance = (
                    abs(current_buy - current_sell) / bucket_vol_total
                    if bucket_vol_total > 0
                    else 0.0
                )
                bucket_imbalances.append(imbalance)
                bucket_timestamps.append(idx[i])

                current_bucket_vol = 0.0
                current_buy = 0.0
                current_sell = 0.0
                remaining = bucket_size

            current_bucket_vol += trade_vol
            current_buy += buy[i] * (trade_vol / vol[i]) if vol[i] > 0 else 0.0
            current_sell += sell[i] * (trade_vol / vol[i]) if vol[i] > 0 else 0.0

        if not bucket_imbalances:
            return pd.DataFrame({"vpin": pd.Series(dtype=float)})

        n_window = min(50, len(bucket_imbalances))
        bucket_series = pd.Series(bucket_imbalances, index=bucket_timestamps)
        vpin = bucket_series.rolling(n_window).mean()

        # Resample back to original index (forward-fill)
        vpin_reindexed = vpin.reindex(vpin.index.union(idx)).ffill().reindex(idx)

        return pd.DataFrame({"vpin": vpin_reindexed}, index=idx)

    # ------------------------------------------------------------------
    # Kyle's Lambda
    # ------------------------------------------------------------------
    def compute_kyle_lambda(
        self,
        price_changes: pd.Series,
        order_flow: pd.Series,
    ) -> pd.DataFrame:
        """
        Kyle's lambda: price impact coefficient estimated via OLS.

        delta_p_t = lambda * OF_t + epsilon_t

        lambda = cov(delta_p, OF) / var(OF)

        A larger lambda implies lower liquidity / higher price impact.
        """
        aligned = pd.concat(
            [price_changes.rename("dp"), order_flow.rename("of")], axis=1
        ).dropna()

        dp = aligned["dp"].values
        of_ = aligned["of"].values

        var_of = np.var(of_, ddof=1)
        if var_of == 0:
            lam = np.nan
        else:
            lam = float(np.cov(dp, of_, ddof=1)[0, 1] / var_of)

        lam_series = pd.Series(lam, index=price_changes.index)

        return pd.DataFrame({"kyle_lambda": lam_series}, index=price_changes.index)

    # ------------------------------------------------------------------
    # Amihud Illiquidity
    # ------------------------------------------------------------------
    def compute_amihud_illiquidity(
        self,
        returns: pd.Series,
        volume: pd.Series,
        window: int = 21,
    ) -> pd.DataFrame:
        """
        Amihud (2002) illiquidity ratio.

        ILLIQ_t = (1/D) * sum_{d=t-D+1}^{t} |r_d| / (Volume_d)

        Larger values indicate less liquid markets.
        Dollar volume approximation: volume used directly.
        """
        abs_ret = returns.abs()
        dollar_vol = volume.replace(0.0, np.nan)
        illiq_daily = abs_ret / dollar_vol
        amihud = illiq_daily.rolling(window).mean() * 1e6  # scale for readability

        return pd.DataFrame({"amihud_illiquidity": amihud}, index=returns.index)

    # ------------------------------------------------------------------
    # Roll Spread
    # ------------------------------------------------------------------
    def compute_roll_spread(
        self,
        close: pd.Series,
    ) -> pd.DataFrame:
        """
        Roll (1984) implied spread from serial covariance of price changes.

        S = 2 * sqrt(-cov(delta_p_t, delta_p_{t-1}))

        If the serial covariance is positive (no bid-ask bounce) the implied
        spread is set to 0.
        """
        dp = close.diff()
        dp_lag = dp.shift(1)

        # Rolling serial covariance
        window = 21
        cov_series = pd.Series(index=close.index, dtype=float)

        dp_arr = dp.values
        dp_lag_arr = dp_lag.values

        for t in range(window, len(dp_arr) + 1):
            w_dp = dp_arr[t - window : t]
            w_lag = dp_lag_arr[t - window : t]
            mask = ~(np.isnan(w_dp) | np.isnan(w_lag))
            if mask.sum() < 4:
                continue
            c = np.cov(w_dp[mask], w_lag[mask], ddof=1)[0, 1]
            cov_series.iloc[t - 1] = c

        roll_spread = 2.0 * np.sqrt((-cov_series).clip(lower=0.0))

        return pd.DataFrame({"roll_spread": roll_spread}, index=close.index)

    # ------------------------------------------------------------------
    # Master feature builder
    # ------------------------------------------------------------------
    def compute_all_features(
        self,
        df: pd.DataFrame,
        order_book: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """
        Compute all microstructure features.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain: high, low, close, volume. Optionally:
            bid, ask, buy_vol, sell_vol, order_flow.
        order_book : dict, optional
            A single snapshot order book dict for liquidity pressure.

        Returns
        -------
        pd.DataFrame with all available microstructure features.
        """
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        frames: List[pd.DataFrame] = []

        # Bid-ask spread (if available)
        if "bid" in df.columns and "ask" in df.columns:
            frames.append(self.compute_bid_ask_spread(df["bid"], df["ask"]))

        # Order imbalance (if available)
        if "buy_vol" in df.columns and "sell_vol" in df.columns:
            frames.append(
                self.compute_order_imbalance(df["buy_vol"], df["sell_vol"])
            )

        # VPIN (if buy_vol / sell_vol available)
        if "buy_vol" in df.columns and "sell_vol" in df.columns:
            frames.append(
                self.compute_vpin(
                    df["volume"], df["buy_vol"], df["sell_vol"]
                )
            )

        # Kyle's lambda (if order_flow available)
        if "order_flow" in df.columns:
            frames.append(
                self.compute_kyle_lambda(df["close"].diff(), df["order_flow"])
            )

        # Amihud illiquidity
        returns = np.log(df["close"] / df["close"].shift(1))
        frames.append(self.compute_amihud_illiquidity(returns, df["volume"]))

        # Roll spread
        frames.append(self.compute_roll_spread(df["close"]))

        # Liquidity pressure from order book snapshot
        if order_book is not None:
            ob_ts = order_book.get("timestamp", df.index[-1])
            order_book["timestamp"] = ob_ts
            ob_df = self.compute_liquidity_pressure(order_book)
            # Broadcast scalar row to match full index
            for col in ob_df.columns:
                frames.append(
                    pd.DataFrame(
                        {col: pd.Series(ob_df[col].iloc[0], index=df.index)}
                    )
                )

        if not frames:
            return pd.DataFrame(index=df.index)

        result = pd.concat(frames, axis=1)
        result = result.ffill(limit=1)
        result = result.dropna(how="all")
        return result
