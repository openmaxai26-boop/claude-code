"""
MarketDataCollector
===================
Fetches, validates, and normalises OHLCV and microstructure data from
yfinance (equities/ETFs) and ccxt (crypto order books).

All public methods return pandas DataFrames with a DatetimeIndex in UTC.
"""

from __future__ import annotations

import time
import warnings
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

# ccxt import is optional – only needed for order-book functionality
try:
    import ccxt
    _CCXT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CCXT_AVAILABLE = False
    logger.warning("ccxt not installed – order-book methods will be unavailable")

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RETRY_BASE_DELAY: float = 1.0   # seconds
_MAX_RETRIES: int = 5
_VOL_WINDOWS: List[int] = [5, 10, 21, 63]   # calendar days for realised vol


# ---------------------------------------------------------------------------
# Helper: retry with exponential backoff
# ---------------------------------------------------------------------------


def _retry(max_retries: int = _MAX_RETRIES, base_delay: float = _RETRY_BASE_DELAY):
    """Decorator: retry *func* with exponential backoff on any Exception."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    if attempt == max_retries:
                        logger.error(
                            f"[{func.__name__}] failed after {max_retries} attempts: {exc}"
                        )
                        raise
                    logger.warning(
                        f"[{func.__name__}] attempt {attempt}/{max_retries} failed "
                        f"({exc}); retrying in {delay:.1f}s"
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, 60.0)   # cap at 60 s

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# MarketDataCollector
# ---------------------------------------------------------------------------


class MarketDataCollector:
    """
    Institutional-grade market data collection layer.

    Parameters
    ----------
    ccxt_exchange_id : str
        Name of the CCXT exchange to use for crypto data (default: 'binance').
    ccxt_config : dict
        Extra kwargs passed to the CCXT exchange constructor (API keys, etc.).
    """

    def __init__(
        self,
        ccxt_exchange_id: str = "binance",
        ccxt_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._ccxt_exchange_id = ccxt_exchange_id
        self._ccxt_config = ccxt_config or {}
        self._exchange: Optional[Any] = None   # lazy-loaded
        logger.info(
            f"MarketDataCollector initialised (exchange={ccxt_exchange_id})"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_exchange(self) -> Any:
        """Return (and lazily initialise) the CCXT exchange instance."""
        if not _CCXT_AVAILABLE:
            raise ImportError("ccxt is required for order-book / crypto methods")
        if self._exchange is None:
            exchange_class = getattr(ccxt, self._ccxt_exchange_id)
            self._exchange = exchange_class(self._ccxt_config)
            logger.debug(f"CCXT exchange '{self._ccxt_exchange_id}' initialised")
        return self._exchange

    @staticmethod
    def _to_utc(dt: datetime) -> datetime:
        """Ensure *dt* is timezone-aware (UTC)."""
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    # ------------------------------------------------------------------
    # 1. OHLCV
    # ------------------------------------------------------------------

    @_retry()
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1d",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars for *symbol* using yfinance.

        Parameters
        ----------
        symbol    : Ticker string recognised by yfinance (e.g. 'AAPL', 'BTC-USD').
        timeframe : One of '1m','2m','5m','15m','30m','60m','90m','1h',
                    '1d','5d','1wk','1mo','3mo'.
        start     : Inclusive start datetime (UTC).  Defaults to 2 years ago.
        end       : Exclusive end datetime (UTC).  Defaults to now.

        Returns
        -------
        pd.DataFrame with columns [open, high, low, close, volume] and a
        UTC DatetimeIndex named 'datetime'.
        """
        if start is None:
            start = datetime.now(timezone.utc) - timedelta(days=730)
        if end is None:
            end = datetime.now(timezone.utc)

        start = self._to_utc(start)
        end = self._to_utc(end)

        logger.info(
            f"Fetching OHLCV | symbol={symbol} tf={timeframe} "
            f"start={start.date()} end={end.date()}"
        )

        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval=timeframe,
            auto_adjust=True,
            back_adjust=False,
        )

        if df.empty:
            logger.warning(f"No data returned for {symbol} [{timeframe}]")
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"]
            )

        # Standardise column names
        df.columns = df.columns.str.lower()
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "datetime"

        # Keep only canonical OHLCV columns (yfinance may return extras)
        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        df = df[[c for c in ohlcv_cols if c in df.columns]]

        df = self.normalize_ohlcv(df)
        df["symbol"] = symbol
        df["timeframe"] = timeframe

        logger.success(
            f"Fetched {len(df)} bars for {symbol} [{timeframe}]"
        )
        return df

    # ------------------------------------------------------------------
    # 2. Tick / microstructure simulation
    # ------------------------------------------------------------------

    def fetch_tick_data(
        self,
        symbol: str,
        date_: date,
        n_ticks_per_bar: int = 20,
    ) -> pd.DataFrame:
        """
        Simulate order-level tick data from minute OHLCV bars.

        The simulation uses a Brownian-bridge interpolation between the open and
        close of each minute bar, constrained to respect the bar's high/low.
        Bid-ask spread and trade sizes are generated with realistic
        microstructure noise (Roll model approximation).

        Parameters
        ----------
        symbol          : Ticker symbol.
        date_           : The calendar date for which to generate ticks.
        n_ticks_per_bar : Number of synthetic ticks per 1-minute bar.

        Returns
        -------
        pd.DataFrame with columns:
            timestamp, price, size, side ('buy'|'sell'), bid, ask, spread_bps
        """
        start_dt = datetime.combine(date_, datetime.min.time()).replace(
            tzinfo=timezone.utc
        )
        end_dt = start_dt + timedelta(days=1)

        logger.info(
            f"Generating synthetic tick data | symbol={symbol} date={date_}"
        )

        bars = self.fetch_ohlcv(symbol, "1m", start=start_dt, end=end_dt)
        if bars.empty:
            logger.warning(f"No minute bars available for {symbol} on {date_}")
            return pd.DataFrame()

        rng = np.random.default_rng(seed=int(date_.strftime("%Y%m%d")))
        ticks: List[Dict[str, Any]] = []

        for bar_ts, row in bars.iterrows():
            o, h, l, c, vol = (
                row["open"], row["high"], row["low"], row["close"], row["volume"]
            )

            if np.isnan(o) or np.isnan(c):
                continue

            # --- Brownian bridge from open to close ---
            t = np.linspace(0, 1, n_ticks_per_bar)
            # Intermediate values of the bridge at each tick step
            bridge = o + (c - o) * t + rng.normal(0, (h - l) / 6, n_ticks_per_bar)
            # Clip to bar high/low
            bridge = np.clip(bridge, l, h)
            bridge[0] = o
            bridge[-1] = c

            # Spread: Roll model c ≈ 2 * sqrt(-cov(r_t, r_{t-1})) ≈ 0.01–0.10%
            spread_frac = max(0.0001, rng.normal(0.0003, 0.0001))
            tick_interval_s = 60.0 / n_ticks_per_bar
            bar_start_ts = bar_ts.timestamp()

            for i, price in enumerate(bridge):
                half_spread = price * spread_frac / 2
                bid = price - half_spread
                ask = price + half_spread
                # Volume distributed roughly proportional to price movement
                size = max(1, int(vol / n_ticks_per_bar * rng.lognormal(0, 0.5)))
                side = "buy" if rng.random() > 0.5 else "sell"
                ts = pd.Timestamp(
                    bar_start_ts + i * tick_interval_s, unit="s", tz="UTC"
                )
                ticks.append(
                    {
                        "timestamp": ts,
                        "price": round(price, 4),
                        "size": size,
                        "side": side,
                        "bid": round(bid, 4),
                        "ask": round(ask, 4),
                        "spread_bps": round(spread_frac * 1e4, 2),
                    }
                )

        tick_df = pd.DataFrame(ticks).set_index("timestamp")
        logger.success(
            f"Generated {len(tick_df)} synthetic ticks for {symbol} on {date_}"
        )
        return tick_df

    # ------------------------------------------------------------------
    # 3. Order book (live – from CCXT)
    # ------------------------------------------------------------------

    @_retry()
    def fetch_order_book(
        self,
        symbol: str,
        depth: int = 20,
    ) -> Dict[str, Any]:
        """
        Fetch the current level-2 order book for a crypto symbol.

        Parameters
        ----------
        symbol : CCXT unified symbol, e.g. 'BTC/USDT'.
        depth  : Number of price levels on each side to return.

        Returns
        -------
        dict with keys:
            symbol      – str
            timestamp   – pd.Timestamp (UTC)
            bids        – np.ndarray shape (depth, 2) → [price, size]
            asks        – np.ndarray shape (depth, 2) → [price, size]
            mid_price   – float
            spread      – float (ask[0] - bid[0])
            spread_bps  – float
        """
        exchange = self._get_exchange()
        logger.info(f"Fetching order book | symbol={symbol} depth={depth}")

        raw = exchange.fetch_order_book(symbol, limit=depth)

        bids = np.array(raw["bids"][:depth], dtype=float)
        asks = np.array(raw["asks"][:depth], dtype=float)

        best_bid = bids[0, 0] if len(bids) else np.nan
        best_ask = asks[0, 0] if len(asks) else np.nan
        mid = (best_bid + best_ask) / 2 if not np.isnan(best_bid + best_ask) else np.nan
        spread = best_ask - best_bid if not np.isnan(best_bid + best_ask) else np.nan
        spread_bps = (spread / mid) * 1e4 if mid else np.nan

        result = {
            "symbol": symbol,
            "timestamp": pd.Timestamp(raw["timestamp"], unit="ms", tz="UTC"),
            "bids": bids,
            "asks": asks,
            "mid_price": mid,
            "spread": spread,
            "spread_bps": round(spread_bps, 2) if not np.isnan(spread_bps) else np.nan,
        }

        logger.success(
            f"Order book fetched | {symbol} mid={mid:.4f} spread={spread_bps:.2f}bps"
        )
        return result

    # ------------------------------------------------------------------
    # 4. Volatility surface
    # ------------------------------------------------------------------

    def fetch_volatility_surface(
        self,
        symbol: str,
        windows: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Compute the realised-volatility surface at multiple look-back windows.

        Parameters
        ----------
        symbol  : Ticker symbol.
        windows : List of look-back windows in trading days.
                  Defaults to [5, 10, 21, 63].

        Returns
        -------
        pd.DataFrame with a DatetimeIndex and one column per window:
            rv_5d, rv_10d, rv_21d, rv_63d  (annualised standard deviation)
        """
        if windows is None:
            windows = _VOL_WINDOWS

        logger.info(
            f"Computing volatility surface | symbol={symbol} windows={windows}"
        )

        # Fetch enough history to cover the longest window
        history_days = max(windows) * 2 + 30
        start = datetime.now(timezone.utc) - timedelta(days=history_days)
        df = self.fetch_ohlcv(symbol, "1d", start=start)

        if df.empty or "close" not in df.columns:
            logger.warning(f"Insufficient data for volatility surface: {symbol}")
            return pd.DataFrame()

        log_rets = np.log(df["close"] / df["close"].shift(1))

        vol_df = pd.DataFrame(index=df.index)
        trading_days_per_year = 252.0

        for w in windows:
            # Annualised realised vol using a rolling window
            col = f"rv_{w}d"
            vol_df[col] = log_rets.rolling(w).std() * np.sqrt(trading_days_per_year)

        # Additional derived metrics
        if "rv_21d" in vol_df and "rv_63d" in vol_df:
            vol_df["vol_ratio_short_long"] = vol_df["rv_21d"] / vol_df["rv_63d"]
        if "rv_5d" in vol_df and "rv_21d" in vol_df:
            vol_df["vol_momentum"] = vol_df["rv_5d"] - vol_df["rv_21d"]

        vol_df.dropna(how="all", inplace=True)
        vol_df["symbol"] = symbol

        logger.success(
            f"Volatility surface computed | {symbol} "
            f"rows={len(vol_df)} windows={windows}"
        )
        return vol_df

    # ------------------------------------------------------------------
    # 5. Normalise OHLCV (splits, dividends, gaps)
    # ------------------------------------------------------------------

    def normalize_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise a raw OHLCV DataFrame:
          - Drop duplicate index entries.
          - Sort by datetime ascending.
          - Cast numeric columns to float64.
          - Fill short intraday gaps (≤2 bars) by forward-filling price,
            setting volume to zero.
          - Remove bars where open == high == low == close == 0 (halted).
          - Enforce price consistency: low ≤ close ≤ high and low ≤ open ≤ high.

        Parameters
        ----------
        df : Raw OHLCV DataFrame (must have columns open/high/low/close/volume).

        Returns
        -------
        Cleaned pd.DataFrame.
        """
        if df.empty:
            return df

        df = df.copy()
        df = df[~df.index.duplicated(keep="last")]
        df.sort_index(inplace=True)

        price_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
        vol_cols = [c for c in ["volume"] if c in df.columns]

        for col in price_cols + vol_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Remove halted / zero bars
        if price_cols:
            zero_mask = (df[price_cols] == 0).all(axis=1)
            n_zero = zero_mask.sum()
            if n_zero:
                logger.debug(f"Removing {n_zero} all-zero bars")
            df = df[~zero_mask]

        # Forward-fill price gaps ≤ 2 consecutive missing bars
        df[price_cols] = df[price_cols].ffill(limit=2)

        # Zero-fill missing volume
        if vol_cols:
            df[vol_cols] = df[vol_cols].fillna(0)

        # Enforce OHLC consistency
        if all(c in df.columns for c in ["open", "high", "low", "close"]):
            df["high"] = df[["open", "high", "close"]].max(axis=1)
            df["low"] = df[["open", "low", "close"]].min(axis=1)

        # Drop remaining NaN rows
        df.dropna(subset=price_cols, inplace=True)

        return df

    # ------------------------------------------------------------------
    # 6. Data validation
    # ------------------------------------------------------------------

    def validate_data(
        self,
        df: pd.DataFrame,
        max_gap_bars: int = 5,
        zscore_threshold: float = 5.0,
    ) -> Tuple[bool, List[str]]:
        """
        Validate an OHLCV DataFrame for common data-quality issues.

        Checks performed:
          1. Non-empty with required columns.
          2. Monotonic datetime index.
          3. No negative prices or zero close.
          4. No excessive bar gaps (> max_gap_bars consecutive missing bars).
          5. No extreme log-return outliers (|z-score| > zscore_threshold).

        Parameters
        ----------
        df               : OHLCV DataFrame to validate.
        max_gap_bars     : Maximum allowed consecutive missing bars.
        zscore_threshold : Z-score magnitude beyond which a return is an outlier.

        Returns
        -------
        (is_valid : bool, issues : List[str])
            is_valid is True only if *all* checks pass.
            issues contains human-readable descriptions of any failures.
        """
        issues: List[str] = []

        if df is None or df.empty:
            issues.append("DataFrame is empty or None")
            return False, issues

        required_cols = {"open", "high", "low", "close"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
            return False, issues   # subsequent checks are meaningless

        # 2. Monotonic index
        if not df.index.is_monotonic_increasing:
            issues.append("DatetimeIndex is not monotonically increasing")

        # 3. Negative prices / zero close
        if (df["close"] <= 0).any():
            n = (df["close"] <= 0).sum()
            issues.append(f"{n} bars have non-positive close price")

        if (df["low"] < 0).any():
            n = (df["low"] < 0).sum()
            issues.append(f"{n} bars have negative low price")

        # 4. Gap detection (time-delta based on most common interval)
        if len(df) >= 3:
            deltas = df.index.to_series().diff().dropna()
            common_delta = deltas.mode().iloc[0]
            gap_mask = deltas > (common_delta * (max_gap_bars + 1))
            n_gaps = gap_mask.sum()
            if n_gaps:
                first_gap = df.index[gap_mask][0]
                issues.append(
                    f"{n_gaps} gap(s) detected exceeding {max_gap_bars} bars "
                    f"(first at {first_gap})"
                )

        # 5. Outlier detection via z-score of log returns
        log_rets = np.log(df["close"] / df["close"].shift(1)).dropna()
        if len(log_rets) >= 10:
            mu = log_rets.mean()
            sigma = log_rets.std()
            if sigma > 0:
                z_scores = (log_rets - mu) / sigma
                outliers = (z_scores.abs() > zscore_threshold).sum()
                if outliers:
                    issues.append(
                        f"{outliers} log-return outlier(s) with |z| > {zscore_threshold}"
                    )

        is_valid = len(issues) == 0
        if is_valid:
            logger.success(f"Data validation passed ({len(df)} bars)")
        else:
            logger.warning(
                f"Data validation found {len(issues)} issue(s): {issues}"
            )
        return is_valid, issues

    # ------------------------------------------------------------------
    # Convenience: bulk fetch
    # ------------------------------------------------------------------

    def fetch_multiple(
        self,
        symbols: List[str],
        timeframe: str = "1d",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV for a list of symbols.

        Returns
        -------
        dict mapping symbol → validated DataFrame (or empty DF on failure).
        """
        results: Dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                df = self.fetch_ohlcv(sym, timeframe, start, end)
                valid, issues = self.validate_data(df)
                if not valid:
                    logger.warning(f"{sym}: validation issues {issues}")
                results[sym] = df
            except Exception as exc:
                logger.error(f"Failed to fetch {sym}: {exc}")
                results[sym] = pd.DataFrame()
        return results
