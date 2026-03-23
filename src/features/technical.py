"""
Technical Feature Engine
Implements all technical indicators from scratch using NumPy and Pandas.
No external TA libraries required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Optional


class TechnicalFeatureEngine:
    """
    Computes a comprehensive set of technical analysis features for financial time series.
    All indicators are implemented directly from their mathematical definitions.
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # RSI
    # ------------------------------------------------------------------
    def compute_rsi(
        self,
        close: pd.Series,
        period: int = 14,
    ) -> pd.DataFrame:
        """
        Relative Strength Index plus divergence and momentum signals.

        RSI = 100 - 100 / (1 + RS)
        RS  = avg_gain / avg_loss  (Wilder smoothing)

        Returns
        -------
        DataFrame with columns: rsi, rsi_divergence, rsi_momentum
        """
        delta = close.diff()

        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)

        # Wilder's smoothed moving average
        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))

        # RSI momentum: rate of change of RSI
        rsi_momentum = rsi.diff(5)

        # RSI divergence: price makes new high/low but RSI doesn't (rolling window = 5)
        price_high = close.rolling(5).max()
        rsi_high = rsi.rolling(5).max()
        price_low = close.rolling(5).min()
        rsi_low = rsi.rolling(5).min()

        bearish_div = ((close >= price_high) & (rsi < rsi_high)).astype(float)
        bullish_div = ((close <= price_low) & (rsi > rsi_low)).astype(float)
        rsi_divergence = bullish_div - bearish_div  # +1 bullish, -1 bearish, 0 none

        return pd.DataFrame(
            {
                "rsi": rsi,
                "rsi_divergence": rsi_divergence,
                "rsi_momentum": rsi_momentum,
            },
            index=close.index,
        )

    # ------------------------------------------------------------------
    # MACD
    # ------------------------------------------------------------------
    def compute_macd(
        self,
        close: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> pd.DataFrame:
        """
        Moving Average Convergence/Divergence.

        MACD line  = EMA(fast) - EMA(slow)
        Signal     = EMA(MACD, signal)
        Histogram  = MACD - Signal
        Crossover  = +1 when MACD crosses above Signal, -1 below, 0 otherwise
        """
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        # Crossover detection
        above = (macd_line > signal_line).astype(int)
        crossover = above.diff().fillna(0.0)  # +1 cross up, -1 cross down

        return pd.DataFrame(
            {
                "macd": macd_line,
                "macd_signal": signal_line,
                "macd_histogram": histogram,
                "macd_crossover": crossover,
            },
            index=close.index,
        )

    # ------------------------------------------------------------------
    # Bollinger Bands
    # ------------------------------------------------------------------
    def compute_bollinger_bands(
        self,
        close: pd.Series,
        period: int = 20,
        std: float = 2.0,
    ) -> pd.DataFrame:
        """
        Bollinger Bands: upper, lower, width, %B, squeeze indicator.

        %B      = (price - lower) / (upper - lower)
        Width   = (upper - lower) / middle
        Squeeze = width < 6-month minimum width (bool -> float)
        """
        middle = close.rolling(period).mean()
        rolling_std = close.rolling(period).std(ddof=1)

        upper = middle + std * rolling_std
        lower = middle - std * rolling_std

        width = (upper - lower) / middle.replace(0.0, np.nan)
        pct_b = (close - lower) / (upper - lower).replace(0.0, np.nan)

        # Squeeze: current width is below its 126-bar (6-month) minimum
        squeeze = (width < width.rolling(126).min()).astype(float)

        return pd.DataFrame(
            {
                "bb_upper": upper,
                "bb_lower": lower,
                "bb_width": width,
                "bb_pct_b": pct_b,
                "bb_squeeze": squeeze,
            },
            index=close.index,
        )

    # ------------------------------------------------------------------
    # VWAP
    # ------------------------------------------------------------------
    def compute_vwap(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
    ) -> pd.DataFrame:
        """
        Volume-Weighted Average Price (cumulative within session/full period).

        VWAP          = cumsum(typical_price * volume) / cumsum(volume)
        Deviation     = (close - VWAP) / VWAP
        VWAP slope    = first difference of VWAP (normalized by VWAP)
        """
        typical_price = (high + low + close) / 3.0
        tpv = typical_price * volume

        cum_tpv = tpv.cumsum()
        cum_vol = volume.cumsum()

        vwap = cum_tpv / cum_vol.replace(0.0, np.nan)
        deviation = (close - vwap) / vwap.replace(0.0, np.nan)
        vwap_slope = vwap.diff() / vwap.shift(1).replace(0.0, np.nan)

        return pd.DataFrame(
            {
                "vwap": vwap,
                "vwap_deviation": deviation,
                "vwap_slope": vwap_slope,
            },
            index=close.index,
        )

    # ------------------------------------------------------------------
    # ATR
    # ------------------------------------------------------------------
    def compute_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.DataFrame:
        """
        Average True Range and normalized ATR.

        TR  = max(high-low, |high-prev_close|, |low-prev_close|)
        ATR = Wilder EMA of TR
        NATR = ATR / close * 100
        """
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        natr = atr / close.replace(0.0, np.nan) * 100.0

        return pd.DataFrame(
            {"atr": atr, "natr": natr},
            index=close.index,
        )

    # ------------------------------------------------------------------
    # ADX
    # ------------------------------------------------------------------
    def compute_adx(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.DataFrame:
        """
        Average Directional Index (Wilder 1978).

        +DM = high - prev_high  if positive and > |low - prev_low| else 0
        -DM = prev_low - low    if positive and > |high - prev_high| else 0
        TR  = max(H-L, |H-PC|, |L-PC|)
        Smoothed with Wilder EMA(period)
        +DI = 100 * smooth(+DM) / smooth(TR)
        -DI = 100 * smooth(-DM) / smooth(TR)
        DX  = 100 * |+DI - -DI| / (+DI + -DI)
        ADX = Wilder EMA(DX, period)
        Trend strength: ADX > 25 -> trending, > 50 -> strong
        """
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_close = close.shift(1)

        up_move = high - prev_high
        down_move = prev_low - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        alpha = 1.0 / period
        plus_dm_s = (
            pd.Series(plus_dm, index=close.index)
            .ewm(alpha=alpha, min_periods=period, adjust=False)
            .mean()
        )
        minus_dm_s = (
            pd.Series(minus_dm, index=close.index)
            .ewm(alpha=alpha, min_periods=period, adjust=False)
            .mean()
        )
        tr_s = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

        plus_di = 100.0 * plus_dm_s / tr_s.replace(0.0, np.nan)
        minus_di = 100.0 * minus_dm_s / tr_s.replace(0.0, np.nan)

        di_sum = (plus_di + minus_di).replace(0.0, np.nan)
        dx = 100.0 * (plus_di - minus_di).abs() / di_sum
        adx = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

        trend_strength = pd.cut(
            adx,
            bins=[-np.inf, 20, 25, 50, np.inf],
            labels=[0, 1, 2, 3],
        ).astype(float)

        return pd.DataFrame(
            {
                "adx": adx,
                "plus_di": plus_di,
                "minus_di": minus_di,
                "adx_trend_strength": trend_strength,
            },
            index=close.index,
        )

    # ------------------------------------------------------------------
    # Ichimoku Cloud
    # ------------------------------------------------------------------
    def compute_ichimoku(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.DataFrame:
        """
        Ichimoku Kinko Hyo cloud system.

        Tenkan-sen  (9-period)   = (max_high + min_low) / 2
        Kijun-sen   (26-period)  = (max_high + min_low) / 2
        Senkou A    = (tenkan + kijun) / 2  shifted forward 26
        Senkou B    (52-period)  = (max_high + min_low) / 2  shifted forward 26
        Chikou span = close shifted back 26
        """
        def donchian_mid(h: pd.Series, l: pd.Series, n: int) -> pd.Series:
            return (h.rolling(n).max() + l.rolling(n).min()) / 2.0

        tenkan = donchian_mid(high, low, 9)
        kijun = donchian_mid(high, low, 26)
        senkou_a = ((tenkan + kijun) / 2.0).shift(26)
        senkou_b = donchian_mid(high, low, 52).shift(26)
        chikou = close.shift(-26)

        return pd.DataFrame(
            {
                "ichimoku_tenkan": tenkan,
                "ichimoku_kijun": kijun,
                "ichimoku_senkou_a": senkou_a,
                "ichimoku_senkou_b": senkou_b,
                "ichimoku_chikou": chikou,
            },
            index=close.index,
        )

    # ------------------------------------------------------------------
    # Stochastic Oscillator
    # ------------------------------------------------------------------
    def compute_stochastic(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k: int = 14,
        d: int = 3,
    ) -> pd.DataFrame:
        """
        Stochastic Oscillator.

        %K = 100 * (close - min_low_k) / (max_high_k - min_low_k)
        %D = SMA(%K, d)
        Overbought: %K > 80, Oversold: %K < 20
        """
        lowest_low = low.rolling(k).min()
        highest_high = high.rolling(k).max()

        denom = (highest_high - lowest_low).replace(0.0, np.nan)
        pct_k = 100.0 * (close - lowest_low) / denom
        pct_d = pct_k.rolling(d).mean()

        overbought = (pct_k > 80).astype(float)
        oversold = (pct_k < 20).astype(float)

        return pd.DataFrame(
            {
                "stoch_k": pct_k,
                "stoch_d": pct_d,
                "stoch_overbought": overbought,
                "stoch_oversold": oversold,
            },
            index=close.index,
        )

    # ------------------------------------------------------------------
    # OBV
    # ------------------------------------------------------------------
    def compute_obv(
        self,
        close: pd.Series,
        volume: pd.Series,
    ) -> pd.DataFrame:
        """
        On-Balance Volume and its normalized slope.

        OBV(t) = OBV(t-1) + volume if close > prev_close
                           - volume if close < prev_close
                           + 0      if close == prev_close
        """
        direction = np.sign(close.diff()).fillna(0.0)
        obv = (direction * volume).cumsum()
        obv_slope = obv.diff(5) / (obv.shift(5).abs().replace(0.0, np.nan))

        return pd.DataFrame(
            {"obv": obv, "obv_slope": obv_slope},
            index=close.index,
        )

    # ------------------------------------------------------------------
    # MFI
    # ------------------------------------------------------------------
    def compute_mfi(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int = 14,
    ) -> pd.DataFrame:
        """
        Money Flow Index.

        Typical Price = (H + L + C) / 3
        Raw Money Flow = Typical Price * Volume
        Money Flow Ratio = pos_MF / neg_MF  (over period)
        MFI = 100 - 100 / (1 + MFR)
        """
        tp = (high + low + close) / 3.0
        raw_mf = tp * volume

        tp_diff = tp.diff()
        pos_mf = raw_mf.where(tp_diff > 0, 0.0)
        neg_mf = raw_mf.where(tp_diff < 0, 0.0)

        pos_mf_sum = pos_mf.rolling(period).sum()
        neg_mf_sum = neg_mf.rolling(period).sum()

        mfr = pos_mf_sum / neg_mf_sum.replace(0.0, np.nan)
        mfi = 100.0 - (100.0 / (1.0 + mfr))

        return pd.DataFrame({"mfi": mfi}, index=close.index)

    # ------------------------------------------------------------------
    # Master feature builder
    # ------------------------------------------------------------------
    def compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all technical features for a OHLCV DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns: open, high, low, close, volume (case-insensitive).

        Returns
        -------
        pd.DataFrame
            Wide DataFrame with all technical features; NaN rows trimmed at front.
        """
        # Normalise column names
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")

        open_ = df["open"]
        high = df["high"]
        low = df["low"]
        close = df["close"]
        volume = df["volume"]

        frames = [
            self.compute_rsi(close),
            self.compute_macd(close),
            self.compute_bollinger_bands(close),
            self.compute_vwap(high, low, close, volume),
            self.compute_atr(high, low, close),
            self.compute_adx(high, low, close),
            self.compute_ichimoku(high, low, close),
            self.compute_stochastic(high, low, close),
            self.compute_obv(close, volume),
            self.compute_mfi(high, low, close, volume),
        ]

        result = pd.concat(frames, axis=1)

        # Forward-fill at most 1 bar (e.g. Ichimoku chikou near end of series)
        result = result.ffill(limit=1)

        # Drop rows where all values are NaN (warm-up period)
        result = result.dropna(how="all")

        return result
