"""
CryptoDataCollector
===================
Fetches crypto-specific data signals:
  - Funding rates (from CCXT perpetual swaps)
  - Open interest (from CCXT)
  - Liquidations (simulated from funding rate × OI changes)
  - Exchange flows (on-chain proxy, mock; real data via Glassnode API)
  - Dominance (from price × supply proxy)
  - Derived crypto features (fear/greed proxy, basis, funding premium)

NOTE ON API KEYS:
  - Glassnode on-chain flows require GLASSNODE_API_KEY in .env (UNKNOWN).
  - Exchange-level trade data (Binance, Bybit): no key needed for public
    market data endpoints, but rate limits apply.
"""

from __future__ import annotations

import hashlib
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

try:
    import ccxt
    _CCXT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CCXT_AVAILABLE = False
    logger.warning("ccxt not installed – CryptoDataCollector will have limited functionality")

try:
    import yfinance as yf
    _YFINANCE_AVAILABLE = True
except ImportError:
    _YFINANCE_AVAILABLE = False

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Approximate circulating supply (millions of coins) for dominance proxy
_CIRCULATING_SUPPLY: Dict[str, float] = {
    "BTC": 19.7e6,
    "ETH": 120.0e6,
    "BNB": 153.8e6,
    "SOL": 450.0e6,
    "XRP": 54_000.0e6,
}

# CCXT unified symbols for perpetual swaps (for funding rate / OI)
_PERP_SYMBOLS: Dict[str, str] = {
    "BTC": "BTC/USDT:USDT",
    "ETH": "ETH/USDT:USDT",
    "SOL": "SOL/USDT:USDT",
    "BNB": "BNB/USDT:USDT",
}

_GLASSNODE_BASE = "https://api.glassnode.com/v1/metrics"


# ---------------------------------------------------------------------------
# Helper: retry decorator
# ---------------------------------------------------------------------------


def _retry(max_retries: int = 4, base_delay: float = 1.0):
    def decorator(func):
        def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    if attempt == max_retries:
                        logger.error(f"[{func.__name__}] gave up: {exc}")
                        raise
                    logger.warning(f"[{func.__name__}] attempt {attempt} failed ({exc}); retry in {delay:.1f}s")
                    time.sleep(delay)
                    delay = min(delay * 2, 30.0)
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# CryptoDataCollector
# ---------------------------------------------------------------------------


class CryptoDataCollector:
    """
    Collects crypto-native data signals for use in portfolio models.

    Parameters
    ----------
    exchange_id : str
        CCXT exchange ID for public market data (default: 'binance').
    ccxt_config : dict
        Additional CCXT exchange constructor kwargs (API keys, etc.).
    glassnode_api_key : str, optional
        Glassnode API key for on-chain data.  Falls back to GLASSNODE_API_KEY env var.
    cache_dir : Path, optional
        Directory for parquet cache files.
    cache_ttl_hours : int
        Hours before a cache file is considered stale.
    """

    def __init__(
        self,
        exchange_id: str = "binance",
        ccxt_config: Optional[Dict[str, Any]] = None,
        glassnode_api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        cache_ttl_hours: int = 4,
    ) -> None:
        if not _CCXT_AVAILABLE:
            logger.warning("ccxt unavailable – install with: pip install ccxt")

        self._exchange_id = exchange_id
        self._ccxt_config = ccxt_config or {}
        self._glassnode_key = glassnode_api_key or os.getenv("GLASSNODE_API_KEY")
        self._cache_dir = cache_dir or Path(".cache") / "crypto"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_ttl = timedelta(hours=cache_ttl_hours)
        self._exchange: Optional[Any] = None   # lazy-loaded
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "AI-TradingSystem/1.0"})
        logger.info(f"CryptoDataCollector initialised (exchange={exchange_id})")

    # ------------------------------------------------------------------
    # Exchange initialisation
    # ------------------------------------------------------------------

    def _get_exchange(self) -> Any:
        if not _CCXT_AVAILABLE:
            raise ImportError("ccxt is required for this method")
        if self._exchange is None:
            cls = getattr(ccxt, self._exchange_id)
            self._exchange = cls({**self._ccxt_config, "enableRateLimit": True})
            logger.debug(f"CCXT exchange '{self._exchange_id}' initialised")
        return self._exchange

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.parquet"

    def _is_fresh(self, path: Path) -> bool:
        if not path.exists():
            return False
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        return (datetime.now(timezone.utc) - mtime) < self._cache_ttl

    # ------------------------------------------------------------------
    # 1. Funding rate
    # ------------------------------------------------------------------

    @_retry()
    def fetch_funding_rate(
        self,
        symbol: str,
        lookback_days: int = 90,
    ) -> pd.DataFrame:
        """
        Fetch historical perpetual-swap funding rates for *symbol*.

        Funding rates are typically charged every 8 hours on most exchanges.
        This method fetches the history and resamples to a daily aggregate.

        Parameters
        ----------
        symbol       : Base currency symbol, e.g. 'BTC', 'ETH'.
        lookback_days: Number of calendar days of history.

        Returns
        -------
        pd.DataFrame (DatetimeIndex=UTC) with columns:
            funding_rate_8h     – raw 8-hour funding rate
            funding_rate_annual – annualised (× 3 × 365)
            funding_daily_avg   – daily average of three 8h periods
            funding_cumsum      – cumulative sum (carry proxy)
        """
        cache_key = f"funding_{symbol}_{lookback_days}"
        cache_path = self._cache_path(cache_key)
        if self._is_fresh(cache_path):
            return pd.read_parquet(cache_path)

        perp_sym = _PERP_SYMBOLS.get(symbol, f"{symbol}/USDT:USDT")
        logger.info(f"Fetching funding rates | {perp_sym} lookback={lookback_days}d")

        exchange = self._get_exchange()
        since_ms = int(
            (datetime.now(timezone.utc) - timedelta(days=lookback_days)).timestamp() * 1000
        )

        raw: List[Dict] = []
        while True:
            batch = exchange.fetch_funding_rate_history(
                perp_sym, since=since_ms, limit=500
            )
            if not batch:
                break
            raw.extend(batch)
            last_ts = batch[-1]["timestamp"]
            if last_ts <= since_ms or len(batch) < 500:
                break
            since_ms = last_ts + 1
            time.sleep(0.25)

        if not raw:
            logger.warning(f"No funding rate history for {perp_sym}")
            return pd.DataFrame()

        df = pd.DataFrame(raw)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("datetime").sort_index()
        df["funding_rate_8h"] = pd.to_numeric(df["fundingRate"], errors="coerce")
        df["funding_rate_annual"] = df["funding_rate_8h"] * 3 * 365

        # Daily resample
        daily = df["funding_rate_8h"].resample("D").agg(
            funding_daily_avg="mean"
        )
        daily["funding_rate_8h_last"] = df["funding_rate_8h"].resample("D").last()
        daily["funding_rate_annual"] = daily["funding_daily_avg"] * 3 * 365
        daily["funding_cumsum"] = daily["funding_daily_avg"].cumsum()

        daily.to_parquet(cache_path)
        logger.success(f"Funding rates fetched: {len(daily)} days for {symbol}")
        return daily

    # ------------------------------------------------------------------
    # 2. Open interest
    # ------------------------------------------------------------------

    @_retry()
    def fetch_open_interest(
        self,
        symbol: str,
        lookback_days: int = 90,
    ) -> pd.DataFrame:
        """
        Fetch historical open interest (OI) for the perpetual swap of *symbol*.

        Parameters
        ----------
        symbol       : Base currency (e.g. 'BTC').
        lookback_days: Calendar days of history.

        Returns
        -------
        pd.DataFrame (DatetimeIndex=UTC daily) with columns:
            open_interest_usd   – notional open interest in USD
            oi_change_pct       – daily % change in OI
            oi_ma7              – 7-day moving average of OI
            oi_vs_ma            – OI relative to its 7-day MA (above/below)
        """
        cache_key = f"oi_{symbol}_{lookback_days}"
        cache_path = self._cache_path(cache_key)
        if self._is_fresh(cache_path):
            return pd.read_parquet(cache_path)

        perp_sym = _PERP_SYMBOLS.get(symbol, f"{symbol}/USDT:USDT")
        logger.info(f"Fetching open interest | {perp_sym}")

        exchange = self._get_exchange()
        since_ms = int(
            (datetime.now(timezone.utc) - timedelta(days=lookback_days)).timestamp() * 1000
        )

        raw: List[Dict] = []
        try:
            # Not all exchanges / symbols support fetchOpenInterestHistory
            batch = exchange.fetch_open_interest_history(
                perp_sym,
                timeframe="1d",
                since=since_ms,
                limit=lookback_days + 10,
            )
            raw.extend(batch)
        except (ccxt.NotSupported, AttributeError):
            logger.warning(
                f"fetchOpenInterestHistory not supported for {perp_sym} on "
                f"{self._exchange_id}; returning empty DataFrame"
            )
            return pd.DataFrame()

        if not raw:
            logger.warning(f"No open interest data returned for {perp_sym}")
            return pd.DataFrame()

        df = pd.DataFrame(raw)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("datetime").sort_index()

        # OI in USD = openInterestValue (contracts × contract size × price)
        if "openInterestValue" in df.columns:
            df["open_interest_usd"] = pd.to_numeric(df["openInterestValue"], errors="coerce")
        elif "openInterest" in df.columns:
            df["open_interest_usd"] = pd.to_numeric(df["openInterest"], errors="coerce")
        else:
            df["open_interest_usd"] = np.nan

        df["oi_change_pct"] = df["open_interest_usd"].pct_change() * 100
        df["oi_ma7"] = df["open_interest_usd"].rolling(7, min_periods=1).mean()
        df["oi_vs_ma"] = df["open_interest_usd"] / (df["oi_ma7"] + 1e-8) - 1

        result = df[["open_interest_usd", "oi_change_pct", "oi_ma7", "oi_vs_ma"]]
        result.to_parquet(cache_path)
        logger.success(f"Open interest fetched: {len(result)} rows for {symbol}")
        return result

    # ------------------------------------------------------------------
    # 3. Liquidations (simulated from funding rate + OI)
    # ------------------------------------------------------------------

    def fetch_liquidations(
        self,
        symbol: str,
        timeframe: str = "1d",
        lookback_days: int = 90,
    ) -> pd.DataFrame:
        """
        Estimate daily liquidation volumes for *symbol*.

        Direct liquidation data requires exchange-specific WebSocket feeds
        (e.g. Binance /ws/!forceOrder@arr) which are not reliably available
        via REST.  This method derives a realistic liquidation estimate using:

            estimated_liq ≈ |Δ OI| × liq_factor(funding_rate)

        where liq_factor increases when the funding rate is extreme (crowded
        positioning), following the intuition that large OI drawdowns during
        high-funding environments are primarily liquidation-driven.

        Formula:
            base_liq   = max(0, -ΔOI)  × (1 if price fell, else 0)   (long liqs)
                       + max(0, +ΔOI)  × (1 if price rose, else 0)   (short liqs)  [simplified]
            liq_factor = 1 + abs(funding_rate_8h) × 50   # empirical scaling
            liq_volume = base_liq × liq_factor

        Parameters
        ----------
        symbol       : Base currency (e.g. 'BTC').
        timeframe    : Aggregation period ('1d', '1h').
        lookback_days: Calendar days of history.

        Returns
        -------
        pd.DataFrame (DatetimeIndex=UTC) with columns:
            liq_long_usd        – estimated long liquidation volume in USD
            liq_short_usd       – estimated short liquidation volume in USD
            liq_total_usd       – total liquidation volume
            liq_imbalance       – (long - short) / total  ∈ [-1, +1]
        """
        logger.info(f"Estimating liquidations | symbol={symbol} tf={timeframe}")

        oi_df = self.fetch_open_interest(symbol, lookback_days=lookback_days)
        fr_df = self.fetch_funding_rate(symbol, lookback_days=lookback_days)

        if oi_df.empty or fr_df.empty:
            logger.warning(f"Cannot estimate liquidations for {symbol}: missing OI or funding rate")
            return pd.DataFrame()

        # Align to common daily index
        idx = oi_df.index.union(fr_df.index)
        oi = oi_df["open_interest_usd"].reindex(idx).ffill()
        fr = fr_df["funding_daily_avg"].reindex(idx).ffill().fillna(0)

        delta_oi = oi.diff()  # positive = OI grew, negative = OI fell

        # Scaling: extreme funding → crowded → higher liq rate when OI drops
        liq_factor = 1.0 + fr.abs() * 50.0

        # Long liquidations triggered when OI drops (leveraged longs blown out)
        liq_long = (-delta_oi.clip(upper=0)) * liq_factor
        # Short liquidations when OI rises fast during funding squeeze
        liq_short = (delta_oi.clip(lower=0)) * liq_factor * (fr > 0).astype(float)

        total = liq_long + liq_short
        imbalance = (liq_long - liq_short) / (total + 1e-8)

        result = pd.DataFrame(
            {
                "liq_long_usd": liq_long,
                "liq_short_usd": liq_short,
                "liq_total_usd": total,
                "liq_imbalance": imbalance,
            }
        ).dropna(how="all")

        logger.success(f"Liquidation estimates computed: {len(result)} rows for {symbol}")
        return result

    # ------------------------------------------------------------------
    # 4. Exchange flows (on-chain proxy)
    # ------------------------------------------------------------------

    def fetch_exchange_flows(
        self,
        symbol: str,
        lookback_days: int = 90,
    ) -> pd.DataFrame:
        """
        Fetch net exchange flows (inflow − outflow) for *symbol*.

        REAL IMPLEMENTATION NOTE:
          On-chain exchange flows require a blockchain data provider.
          Glassnode provides this via:
            GET https://api.glassnode.com/v1/metrics/transactions/transfers_volume_exchanges_net
            Parameters: a=BTC, i=24h, api_key=GLASSNODE_API_KEY (UNKNOWN – set in .env)

          CryptoQuant and IntoTheBlock also offer equivalent endpoints.

        This implementation:
          1. If GLASSNODE_API_KEY is set → calls the Glassnode REST API.
          2. Otherwise → returns a statistically realistic mock based on
             price and OI dynamics (negative flow = outflows dominate → bullish).

        Parameters
        ----------
        symbol       : Base currency (e.g. 'BTC', 'ETH').
        lookback_days: Calendar days of history.

        Returns
        -------
        pd.DataFrame (DatetimeIndex=UTC daily) with columns:
            inflow_usd     – estimated exchange inflows in USD
            outflow_usd    – estimated exchange outflows in USD
            netflow_usd    – inflow - outflow (negative = net accumulation)
            netflow_pct    – netflow as % of 30d avg volume
            flow_signal    – -1/0/+1  (outflow dominant / neutral / inflow dominant)
        """
        logger.info(f"Fetching exchange flows | symbol={symbol}")

        if self._glassnode_key:
            return self._fetch_glassnode_flows(symbol, lookback_days)
        else:
            logger.warning(
                f"GLASSNODE_API_KEY not set (UNKNOWN) – using mock exchange flows for {symbol}. "
                "Set GLASSNODE_API_KEY in .env for real on-chain data."
            )
            return self._mock_exchange_flows(symbol, lookback_days)

    @_retry()
    def _fetch_glassnode_flows(
        self, symbol: str, lookback_days: int
    ) -> pd.DataFrame:
        """Fetch exchange net flows from Glassnode API."""
        since_ts = int(
            (datetime.now(timezone.utc) - timedelta(days=lookback_days)).timestamp()
        )
        params = {
            "a": symbol.upper(),
            "i": "24h",
            "s": since_ts,
            "api_key": self._glassnode_key,
        }

        inflow_url = f"{_GLASSNODE_BASE}/transactions/transfers_volume_exchanges_net"
        resp = self._session.get(inflow_url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        dates = [pd.Timestamp(d["t"], unit="s", tz="UTC") for d in data]
        values = [d["v"] for d in data]

        netflow = pd.Series(values, index=dates, name="netflow_usd")
        df = netflow.to_frame()
        df["inflow_usd"] = df["netflow_usd"].clip(lower=0)
        df["outflow_usd"] = (-df["netflow_usd"]).clip(lower=0)
        df["netflow_pct"] = df["netflow_usd"] / (
            df["netflow_usd"].abs().rolling(30, min_periods=5).mean() + 1e-8
        )
        df["flow_signal"] = np.sign(df["netflow_usd"]).astype(int)
        return df

    def _mock_exchange_flows(
        self, symbol: str, lookback_days: int
    ) -> pd.DataFrame:
        """Generate mock exchange flow data with realistic statistical properties."""
        rng = np.random.default_rng(
            seed=int(hashlib.md5(symbol.encode()).hexdigest(), 16) % (2**32)
        )
        end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        dates = pd.date_range(
            end=end, periods=lookback_days, freq="D", tz="UTC"
        )

        # AR(1) flow process with fat tails
        phi = 0.7
        base_flow = 0.0
        flows = []
        for _ in range(lookback_days):
            base_flow = phi * base_flow + rng.standard_t(df=4) * 1e8
            flows.append(base_flow)

        netflow = pd.Series(flows, index=dates)
        avg_vol = netflow.abs().rolling(30, min_periods=5).mean().fillna(1e8)

        df = pd.DataFrame({"netflow_usd": netflow}, index=dates)
        df["inflow_usd"] = df["netflow_usd"].clip(lower=0)
        df["outflow_usd"] = (-df["netflow_usd"]).clip(lower=0)
        df["netflow_pct"] = df["netflow_usd"] / (avg_vol + 1e-8)
        df["flow_signal"] = np.sign(df["netflow_usd"]).astype(int)
        df.index.name = "datetime"
        return df

    # ------------------------------------------------------------------
    # 5. Dominance
    # ------------------------------------------------------------------

    def fetch_dominance(
        self,
        top_n: int = 5,
        lookback_days: int = 365,
    ) -> pd.DataFrame:
        """
        Estimate BTC and ETH market dominance using price × circulating supply.

        This is a proxy for CoinMarketCap's dominance metric; the actual
        total crypto market cap includes thousands of tokens which are not
        fetched here.  We use yfinance for the top-N asset prices.

        Parameters
        ----------
        top_n        : Number of assets to include in the market cap estimate.
        lookback_days: Calendar days of history.

        Returns
        -------
        pd.DataFrame (DatetimeIndex=UTC daily) with columns:
            btc_dominance       – BTC market cap / total top-N market cap
            eth_dominance       – ETH dominance
            altcoin_dominance   – 1 - btc_dom - eth_dom
            btc_mcap_usd        – BTC market cap (USD)
            total_mcap_usd      – total top-N market cap (USD)
        """
        assets = list(_CIRCULATING_SUPPLY.keys())[:top_n]
        logger.info(f"Computing dominance for {assets}")

        start = datetime.now(timezone.utc) - timedelta(days=lookback_days + 30)
        mcap_frames: Dict[str, pd.Series] = {}

        if not _YFINANCE_AVAILABLE:
            logger.warning("yfinance not available – cannot compute dominance")
            return pd.DataFrame()

        for asset in assets:
            yf_sym = f"{asset}-USD"
            try:
                ticker = yf.Ticker(yf_sym)
                hist = ticker.history(
                    start=start.strftime("%Y-%m-%d"),
                    interval="1d",
                    auto_adjust=True,
                )
                if hist.empty:
                    continue
                hist.index = pd.to_datetime(hist.index, utc=True)
                supply = _CIRCULATING_SUPPLY.get(asset, 1e6)
                mcap_frames[asset] = hist["Close"] * supply
            except Exception as exc:
                logger.warning(f"Could not fetch price for {asset}: {exc}")

        if not mcap_frames:
            logger.warning("No market cap data available for dominance calculation")
            return pd.DataFrame()

        mcap_df = pd.DataFrame(mcap_frames).sort_index().ffill(limit=5)
        total_mcap = mcap_df.sum(axis=1)

        result = pd.DataFrame(index=mcap_df.index)
        result["btc_mcap_usd"] = mcap_df.get("BTC", 0)
        result["eth_mcap_usd"] = mcap_df.get("ETH", 0)
        result["total_mcap_usd"] = total_mcap
        result["btc_dominance"] = result["btc_mcap_usd"] / (total_mcap + 1e-8)
        result["eth_dominance"] = result["eth_mcap_usd"] / (total_mcap + 1e-8)
        result["altcoin_dominance"] = 1 - result["btc_dominance"] - result["eth_dominance"]
        result.index.name = "datetime"
        result.dropna(how="all", inplace=True)

        logger.success(f"Dominance computed: {len(result)} rows")
        return result

    # ------------------------------------------------------------------
    # 6. Composite crypto features
    # ------------------------------------------------------------------

    def compute_crypto_features(
        self,
        symbol: str,
        ohlcv_df: Optional[pd.DataFrame] = None,
        lookback_days: int = 90,
    ) -> pd.DataFrame:
        """
        Compute a comprehensive set of crypto-native features.

        Features:
          - fear_greed_proxy      : composite of volatility, momentum, OI, funding
          - basis                 : spot − perp price (normalised by spot)
          - funding_premium       : annualised funding rate as risk-premium indicator
          - oi_price_ratio        : open interest / market cap (leverage proxy)
          - liq_pressure          : liq_total_usd / rolling avg OI (stress indicator)
          - netflow_z             : z-score of exchange net flow (selling pressure)
          - oi_momentum_5d        : 5-day change in OI
          - funding_momentum_5d   : 5-day change in average funding rate

        Parameters
        ----------
        symbol       : Base currency (e.g. 'BTC', 'ETH').
        ohlcv_df     : Optional OHLCV DataFrame for the spot market.
                       If None, fetched from yfinance.
        lookback_days: Calendar days of history.

        Returns
        -------
        pd.DataFrame of features with a UTC DatetimeIndex.
        """
        logger.info(f"Computing crypto features for {symbol}")

        # --- Fetch component data ---
        funding = self.fetch_funding_rate(symbol, lookback_days=lookback_days)
        oi = self.fetch_open_interest(symbol, lookback_days=lookback_days)
        liq = self.fetch_liquidations(symbol, lookback_days=lookback_days)
        flows = self.fetch_exchange_flows(symbol, lookback_days=lookback_days)

        # --- Fetch spot OHLCV if not provided ---
        if ohlcv_df is None and _YFINANCE_AVAILABLE:
            try:
                yf_sym = f"{symbol}-USD"
                start = datetime.now(timezone.utc) - timedelta(days=lookback_days + 30)
                ticker = yf.Ticker(yf_sym)
                hist = ticker.history(
                    start=start.strftime("%Y-%m-%d"),
                    interval="1d",
                    auto_adjust=True,
                )
                hist.index = pd.to_datetime(hist.index, utc=True)
                hist.columns = hist.columns.str.lower()
                ohlcv_df = hist
            except Exception as exc:
                logger.warning(f"Could not fetch OHLCV for {symbol}: {exc}")
                ohlcv_df = pd.DataFrame()

        # --- Build shared index ---
        all_indices = [df.index for df in [funding, oi, liq, flows] if not df.empty]
        if not all_indices:
            logger.warning(f"No component data available for {symbol} features")
            return pd.DataFrame()

        idx = all_indices[0]
        for other in all_indices[1:]:
            idx = idx.union(other)

        feats = pd.DataFrame(index=idx)
        feats.index.name = "datetime"

        # --- Funding premium ---
        if not funding.empty and "funding_rate_annual" in funding.columns:
            feats["funding_premium"] = funding["funding_rate_annual"].reindex(idx).ffill()
        else:
            feats["funding_premium"] = np.nan

        # --- OI features ---
        if not oi.empty and "open_interest_usd" in oi.columns:
            oi_reindexed = oi["open_interest_usd"].reindex(idx).ffill()
            feats["open_interest_usd"] = oi_reindexed
            feats["oi_momentum_5d"] = oi_reindexed.pct_change(5) * 100
        else:
            feats["open_interest_usd"] = np.nan
            feats["oi_momentum_5d"] = np.nan

        # --- Liquidation pressure ---
        if not liq.empty and "liq_total_usd" in liq.columns:
            liq_total = liq["liq_total_usd"].reindex(idx).ffill().fillna(0)
            feats["liq_total_usd"] = liq_total
            oi_ma = feats["open_interest_usd"].rolling(7, min_periods=1).mean()
            feats["liq_pressure"] = liq_total / (oi_ma + 1e-8)
        else:
            feats["liq_total_usd"] = np.nan
            feats["liq_pressure"] = np.nan

        # --- Exchange flow z-score ---
        if not flows.empty and "netflow_usd" in flows.columns:
            netflow = flows["netflow_usd"].reindex(idx).ffill().fillna(0)
            nf_mean = netflow.rolling(30, min_periods=5).mean()
            nf_std = netflow.rolling(30, min_periods=5).std()
            feats["netflow_z"] = (netflow - nf_mean) / (nf_std + 1e-8)
        else:
            feats["netflow_z"] = np.nan

        # --- Spot-market features ---
        if ohlcv_df is not None and not ohlcv_df.empty and "close" in ohlcv_df.columns:
            close = ohlcv_df["close"].reindex(idx).ffill()
            log_rets = np.log(close / close.shift(1))

            # Realised volatility (21-day annualised)
            rv21 = log_rets.rolling(21).std() * np.sqrt(252)
            feats["rv_21d"] = rv21

            # Price momentum
            feats["price_momentum_5d"] = close.pct_change(5) * 100
            feats["price_momentum_21d"] = close.pct_change(21) * 100

            # OI / Market cap (leverage proxy)
            if "open_interest_usd" in feats.columns:
                supply = _CIRCULATING_SUPPLY.get(symbol, 1e7)
                mcap = close * supply
                feats["oi_mcap_ratio"] = feats["open_interest_usd"] / (mcap + 1e-8)

            # Basis (spot price deviation from 7-day moving average as perp proxy)
            sma7 = close.rolling(7, min_periods=1).mean()
            feats["basis"] = (close - sma7) / (sma7 + 1e-8)

        else:
            feats["rv_21d"] = np.nan
            feats["price_momentum_5d"] = np.nan
            feats["price_momentum_21d"] = np.nan
            feats["basis"] = np.nan

        # --- Fear/greed composite proxy ---
        # Components (each normalised to [-1, +1]):
        #   +volatility (inverted: high vol → fear), +OI momentum, -netflow_z, +funding
        components = []

        if "rv_21d" in feats and feats["rv_21d"].notna().any():
            rv_z = (feats["rv_21d"] - feats["rv_21d"].rolling(90, min_periods=20).mean()) / (
                feats["rv_21d"].rolling(90, min_periods=20).std() + 1e-8
            )
            components.append(-rv_z.clip(-3, 3) / 3)   # high vol → fear → negative

        if "oi_momentum_5d" in feats and feats["oi_momentum_5d"].notna().any():
            oi_z = feats["oi_momentum_5d"].clip(-30, 30) / 30
            components.append(oi_z)                     # rising OI → greed

        if "netflow_z" in feats and feats["netflow_z"].notna().any():
            components.append(-feats["netflow_z"].clip(-3, 3) / 3)  # outflows → accumulation → greed

        if "funding_premium" in feats and feats["funding_premium"].notna().any():
            fp_z = feats["funding_premium"].clip(-1.5, 1.5) / 1.5
            components.append(fp_z)                     # high funding → greed

        if components:
            stacked = pd.concat(components, axis=1)
            feats["fear_greed_proxy"] = stacked.mean(axis=1)
        else:
            feats["fear_greed_proxy"] = np.nan

        # --- Funding momentum ---
        if "funding_premium" in feats:
            feats["funding_momentum_5d"] = feats["funding_premium"].diff(5)

        feats.dropna(how="all", inplace=True)
        logger.success(f"Crypto features computed: {feats.shape} for {symbol}")
        return feats
