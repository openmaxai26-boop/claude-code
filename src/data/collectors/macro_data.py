"""
MacroDataCollector
==================
Fetches macroeconomic time-series from FRED (Federal Reserve Economic Data)
and computes derived yield-curve features used as model inputs.

FRED CSV endpoint (no API key required for basic access):
  https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES_ID>

For higher rate-limits set FRED_API_KEY in your .env and the collector will
automatically switch to the FRED REST API.
"""

from __future__ import annotations

import io
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from loguru import logger

# ---------------------------------------------------------------------------
# FRED series identifiers
# ---------------------------------------------------------------------------

_YIELD_SERIES: Dict[str, str] = {
    "3m":  "DTB3",      # 3-Month Treasury Bill
    "2y":  "DGS2",      # 2-Year Treasury Constant Maturity
    "5y":  "DGS5",      # 5-Year Treasury Constant Maturity
    "10y": "DGS10",     # 10-Year Treasury Constant Maturity
    "30y": "DGS30",     # 30-Year Treasury Constant Maturity
}

_INFLATION_SERIES: Dict[str, str] = {
    "cpi_yoy":              "CPIAUCSL",     # CPI All Urban Consumers
    "pce_yoy":              "PCEPI",        # PCE Price Index
    "breakeven_5y":         "T5YIE",        # 5-Year Breakeven Inflation
    "breakeven_10y":        "T10YIE",       # 10-Year Breakeven Inflation
}

_ECONOMIC_SERIES: Dict[str, str] = {
    "pmi_manufacturing":    "MANEMP",       # Manufacturing Employment (proxy)
    "unemployment_rate":    "UNRATE",       # Civilian Unemployment Rate
    "gdp_growth":           "A191RL1Q225SBEA",  # Real GDP Growth Rate (quarterly)
    "retail_sales_mom":     "RSXFS",        # Advance Retail Sales
    "industrial_prod":      "INDPRO",       # Industrial Production Index
    "consumer_confidence":  "UMCSENT",      # U Michigan Consumer Sentiment
}

_POLICY_SERIES: Dict[str, str] = {
    "fed_funds_rate":       "FEDFUNDS",     # Effective Federal Funds Rate
    "fed_funds_target_upper": "DFEDTARU",  # Fed Funds Target Range Upper
    "fed_funds_target_lower": "DFEDTARL",  # Fed Funds Target Range Lower
}

_FRED_CSV_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="
_FRED_API_BASE = "https://api.stlouisfed.org/fred/series/observations"


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _retry(max_retries: int = 5, base_delay: float = 1.0):
    """Simple exponential-backoff retry decorator."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    if attempt == max_retries:
                        logger.error(f"[{func.__name__}] gave up after {max_retries} attempts: {exc}")
                        raise
                    logger.warning(f"[{func.__name__}] attempt {attempt} failed ({exc}); retry in {delay:.1f}s")
                    time.sleep(delay)
                    delay = min(delay * 2, 60.0)
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator


def _pct_change_yoy(series: pd.Series) -> pd.Series:
    """Compute year-over-year percentage change."""
    return series.pct_change(12) * 100  # assumes monthly frequency


# ---------------------------------------------------------------------------
# MacroDataCollector
# ---------------------------------------------------------------------------


class MacroDataCollector:
    """
    Downloads macroeconomic time-series from FRED and derives structured
    features for use in portfolio models.

    Parameters
    ----------
    fred_api_key : str, optional
        FRED API key.  If provided, uses the REST API (higher rate limit).
        Falls back to the public CSV endpoint if None.
    cache_dir : Path, optional
        Directory for parquet caching.  Defaults to `.cache/macro/`.
    cache_ttl_hours : int
        How many hours before the cached file is considered stale.
    """

    def __init__(
        self,
        fred_api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        cache_ttl_hours: int = 24,
    ) -> None:
        self._api_key = fred_api_key or os.getenv("FRED_API_KEY")
        self._cache_dir = cache_dir or Path(".cache") / "macro"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_ttl = timedelta(hours=cache_ttl_hours)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "AI-TradingSystem/1.0"})
        logger.info(
            f"MacroDataCollector initialised "
            f"(api_key={'set' if self._api_key else 'not set'}, "
            f"cache_dir={self._cache_dir})"
        )

    # ------------------------------------------------------------------
    # Internal: FRED fetch
    # ------------------------------------------------------------------

    def _cache_path(self, series_id: str) -> Path:
        return self._cache_dir / f"{series_id}.parquet"

    def _is_cache_fresh(self, path: Path) -> bool:
        if not path.exists():
            return False
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        return (datetime.now(timezone.utc) - mtime) < self._cache_ttl

    @_retry()
    def _fetch_fred_series(self, series_id: str) -> pd.Series:
        """
        Fetch a single FRED series.  Returns a pd.Series with a DatetimeIndex.
        Uses cache when fresh; writes through to parquet after every download.
        """
        cache_path = self._cache_path(series_id)
        if self._is_cache_fresh(cache_path):
            logger.debug(f"Cache hit: {series_id}")
            df = pd.read_parquet(cache_path)
            return df.iloc[:, 0]

        logger.info(f"Downloading FRED series: {series_id}")

        if self._api_key:
            # REST API path – supports more options and higher rate limits
            params = {
                "series_id": series_id,
                "api_key": self._api_key,
                "file_type": "json",
                "observation_start": "1990-01-01",
            }
            resp = self._session.get(_FRED_API_BASE, params=params, timeout=30)
            resp.raise_for_status()
            observations = resp.json().get("observations", [])
            dates = [o["date"] for o in observations]
            values = [np.nan if o["value"] == "." else float(o["value"]) for o in observations]
            series = pd.Series(values, index=pd.to_datetime(dates), name=series_id)
        else:
            # Public CSV path
            url = _FRED_CSV_BASE + series_id
            resp = self._session.get(url, timeout=30)
            resp.raise_for_status()
            series = pd.read_csv(
                io.StringIO(resp.text),
                index_col=0,
                parse_dates=True,
                na_values=[".", ""],
            ).iloc[:, 0]
            series.name = series_id

        series.index = pd.to_datetime(series.index, utc=True)
        series.index.name = "date"

        # Cache to parquet
        series.to_frame().to_parquet(cache_path)
        logger.debug(f"Cached {series_id} ({len(series)} observations)")
        return series

    def _fetch_multiple_series(
        self, series_dict: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Fetch several FRED series and join them into a single wide DataFrame,
        aligned on a daily index.
        """
        frames: Dict[str, pd.Series] = {}
        for name, series_id in series_dict.items():
            try:
                s = self._fetch_fred_series(series_id)
                s.name = name
                frames[name] = s
            except Exception as exc:
                logger.error(f"Failed to fetch {series_id} ({name}): {exc}")

        if not frames:
            return pd.DataFrame()

        df = pd.DataFrame(frames)
        df.index = pd.to_datetime(df.index, utc=True)
        df.sort_index(inplace=True)
        return df

    # ------------------------------------------------------------------
    # 1. Interest rates / yield curve
    # ------------------------------------------------------------------

    def fetch_interest_rates(self) -> pd.DataFrame:
        """
        Fetch US Treasury yield curve: 3m, 2y, 5y, 10y, 30y constant maturity.

        Returns
        -------
        pd.DataFrame with columns [3m, 2y, 5y, 10y, 30y] and a UTC DatetimeIndex.
        Values are in percent (e.g. 4.75 means 4.75 %).
        """
        logger.info("Fetching yield curve data from FRED")
        df = self._fetch_multiple_series(_YIELD_SERIES)

        if df.empty:
            logger.warning("Yield curve data unavailable")
            return df

        # Forward-fill weekends and holidays (up to 5 business days)
        df = df.resample("B").last().ffill(limit=5)
        df.dropna(how="all", inplace=True)

        logger.success(f"Yield curve fetched: {len(df)} rows, cols={list(df.columns)}")
        return df

    # ------------------------------------------------------------------
    # 2. Inflation
    # ------------------------------------------------------------------

    def fetch_inflation_data(self) -> pd.DataFrame:
        """
        Fetch inflation indicators: CPI, PCE, and 5y/10y breakeven inflation.

        Returns
        -------
        pd.DataFrame with columns for each series plus YoY change columns.
        """
        logger.info("Fetching inflation data from FRED")
        df = self._fetch_multiple_series(_INFLATION_SERIES)

        if df.empty:
            logger.warning("Inflation data unavailable")
            return df

        # Compute YoY percentage changes for index series
        for col in ("cpi_yoy", "pce_yoy"):
            if col in df.columns:
                df[f"{col}_chg"] = _pct_change_yoy(df[col])

        df = df.resample("B").last().ffill(limit=5)
        df.dropna(how="all", inplace=True)

        logger.success(f"Inflation data fetched: {len(df)} rows")
        return df

    # ------------------------------------------------------------------
    # 3. Economic indicators
    # ------------------------------------------------------------------

    def fetch_economic_indicators(self) -> pd.DataFrame:
        """
        Fetch broad economic indicators: PMI proxy, unemployment, GDP growth,
        retail sales, industrial production, and consumer confidence.

        Returns
        -------
        pd.DataFrame aligned to a business-day index (forward-filled up to 5 days).
        """
        logger.info("Fetching economic indicators from FRED")
        df = self._fetch_multiple_series(_ECONOMIC_SERIES)

        if df.empty:
            logger.warning("Economic indicators unavailable")
            return df

        # Compute MoM changes for flow series
        for col in ("retail_sales_mom", "industrial_prod"):
            if col in df.columns:
                df[f"{col}_mom"] = df[col].pct_change() * 100

        df = df.resample("B").last().ffill(limit=5)
        df.dropna(how="all", inplace=True)

        logger.success(f"Economic indicators fetched: {len(df)} rows")
        return df

    # ------------------------------------------------------------------
    # 4. Central bank policy
    # ------------------------------------------------------------------

    def fetch_central_bank_policy(self) -> Dict[str, object]:
        """
        Fetch current Fed policy rate and derive a simple forward-guidance score.

        The *forward_guidance_score* is a heuristic in [-1, +1]:
          +1  → rates likely rising   (fed funds rate is rising, below target upper)
          -1  → rates likely falling  (fed funds rate is falling, at target lower)
           0  → neutral / on hold

        Returns
        -------
        dict with keys:
            current_rate           – float (most recent effective fed-funds rate %)
            target_upper           – float
            target_lower           – float
            recent_change_bps      – float (change over last 30 days in bps)
            forward_guidance_score – float in [-1, +1]
            as_of_date             – str (ISO-8601)
            series_data            – pd.DataFrame (full history)
        """
        logger.info("Fetching central bank policy data from FRED")
        df = self._fetch_multiple_series(_POLICY_SERIES)

        result: Dict[str, object] = {
            "current_rate": np.nan,
            "target_upper": np.nan,
            "target_lower": np.nan,
            "recent_change_bps": np.nan,
            "forward_guidance_score": 0.0,
            "as_of_date": "",
            "series_data": df,
        }

        if df.empty:
            logger.warning("Central bank policy data unavailable")
            return result

        df_daily = df.resample("B").last().ffill(limit=5)

        current_rate = df_daily["fed_funds_rate"].dropna().iloc[-1] if "fed_funds_rate" in df_daily else np.nan
        target_upper = df_daily["fed_funds_target_upper"].dropna().iloc[-1] if "fed_funds_target_upper" in df_daily else np.nan
        target_lower = df_daily["fed_funds_target_lower"].dropna().iloc[-1] if "fed_funds_target_lower" in df_daily else np.nan

        # 30-day change in effective rate (in bps)
        if "fed_funds_rate" in df_daily:
            rate_series = df_daily["fed_funds_rate"].dropna()
            if len(rate_series) >= 30:
                recent_change_bps = (rate_series.iloc[-1] - rate_series.iloc[-30]) * 100
            else:
                recent_change_bps = np.nan
        else:
            recent_change_bps = np.nan

        # Forward guidance score heuristic
        score = 0.0
        if not (np.isnan(current_rate) or np.isnan(target_upper) or np.isnan(target_lower)):
            mid_target = (target_upper + target_lower) / 2
            gap = current_rate - mid_target   # negative → still hiking
            if not np.isnan(recent_change_bps):
                if recent_change_bps > 10:    # hiked recently
                    score = min(1.0, recent_change_bps / 100)
                elif recent_change_bps < -10:  # cut recently
                    score = max(-1.0, recent_change_bps / 100)

        as_of = df_daily.index[-1].isoformat() if not df_daily.empty else ""

        result.update(
            {
                "current_rate": float(current_rate) if not np.isnan(current_rate) else None,
                "target_upper": float(target_upper) if not np.isnan(target_upper) else None,
                "target_lower": float(target_lower) if not np.isnan(target_lower) else None,
                "recent_change_bps": float(recent_change_bps) if not np.isnan(recent_change_bps) else None,
                "forward_guidance_score": round(score, 4),
                "as_of_date": as_of,
            }
        )

        logger.success(
            f"Policy data: rate={result['current_rate']} "
            f"guidance={result['forward_guidance_score']}"
        )
        return result

    # ------------------------------------------------------------------
    # 5. Yield-curve features
    # ------------------------------------------------------------------

    def compute_yield_curve_features(
        self, rates_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Derive standard yield-curve shape features from a rates DataFrame.

        Features computed:
          - slope_2s10s      : 10y − 2y (spread in bps ×100)
          - slope_3m10y      : 10y − 3m
          - curvature        : 2·5y − 2y − 10y (butterfly)
          - level            : Average of 2y, 5y, 10y (parallel shift proxy)
          - inversion_flag   : 1 if 2s10s < 0, else 0
          - slope_momentum_5d: 5-day change in 2s10s
          - slope_momentum_21d: 21-day change in 2s10s
          - vol_3m_yield     : 21-day realised std of daily changes in 3m yield

        Parameters
        ----------
        rates_df : pd.DataFrame with columns [3m, 2y, 5y, 10y, 30y].
                   If None, fetched automatically.

        Returns
        -------
        pd.DataFrame of derived features with same DatetimeIndex as rates_df.
        """
        if rates_df is None:
            rates_df = self.fetch_interest_rates()

        if rates_df.empty:
            logger.warning("Cannot compute yield curve features: empty rates DataFrame")
            return pd.DataFrame()

        logger.info("Computing yield curve features")
        feats = pd.DataFrame(index=rates_df.index)

        # ----- Slope -----
        if "10y" in rates_df and "2y" in rates_df:
            feats["slope_2s10s"] = rates_df["10y"] - rates_df["2y"]
            feats["inversion_flag"] = (feats["slope_2s10s"] < 0).astype(int)
            feats["slope_momentum_5d"] = feats["slope_2s10s"].diff(5)
            feats["slope_momentum_21d"] = feats["slope_2s10s"].diff(21)
        else:
            feats["slope_2s10s"] = np.nan
            feats["inversion_flag"] = np.nan
            feats["slope_momentum_5d"] = np.nan
            feats["slope_momentum_21d"] = np.nan

        if "10y" in rates_df and "3m" in rates_df:
            feats["slope_3m10y"] = rates_df["10y"] - rates_df["3m"]
        else:
            feats["slope_3m10y"] = np.nan

        # ----- Curvature (butterfly) -----
        if all(c in rates_df for c in ("2y", "5y", "10y")):
            feats["curvature"] = (
                2 * rates_df["5y"] - rates_df["2y"] - rates_df["10y"]
            )
        else:
            feats["curvature"] = np.nan

        # ----- Level -----
        level_cols = [c for c in ("2y", "5y", "10y") if c in rates_df]
        if level_cols:
            feats["level"] = rates_df[level_cols].mean(axis=1)
        else:
            feats["level"] = np.nan

        # ----- Volatility of short rate -----
        if "3m" in rates_df:
            feats["vol_3m_yield"] = rates_df["3m"].diff().rolling(21).std()
        else:
            feats["vol_3m_yield"] = np.nan

        # ----- 30y - 10y term premium proxy -----
        if "30y" in rates_df and "10y" in rates_df:
            feats["term_premium_30y_10y"] = rates_df["30y"] - rates_df["10y"]
        else:
            feats["term_premium_30y_10y"] = np.nan

        feats.dropna(how="all", inplace=True)
        logger.success(f"Yield curve features computed: {feats.shape}")
        return feats

    # ------------------------------------------------------------------
    # Convenience: fetch all macro in one call
    # ------------------------------------------------------------------

    def fetch_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        """
        Convenience method: fetch rates, inflation, indicators, yield-curve
        features, and central-bank policy in a single call.

        Returns
        -------
        (rates_df, inflation_df, indicators_df, yield_features_df, policy_dict)
        """
        rates = self.fetch_interest_rates()
        inflation = self.fetch_inflation_data()
        indicators = self.fetch_economic_indicators()
        yc_features = self.compute_yield_curve_features(rates)
        policy = self.fetch_central_bank_policy()
        return rates, inflation, indicators, yc_features, policy
