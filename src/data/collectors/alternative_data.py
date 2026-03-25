"""
AlternativeDataCollector
========================
Collects and processes alternative data sources:
  - News sentiment via FinBERT (ProsusAI/finbert)
  - Social-media sentiment (mock structure; real APIs require keys)
  - ETF fund flows via yfinance volume proxy
  - Sentiment-based feature engineering

NOTE ON API KEYS:
  - News aggregation (e.g. NewsAPI, Refinitiv, Bloomberg): set NEWSAPI_KEY in .env
  - Social sentiment (StockTwits, Reddit, Twitter/X): requires OAuth tokens –
    placeholders marked with UNKNOWN in the relevant methods.
  - FinBERT inference can run locally (CPU or GPU) with no external key.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

# yfinance for ETF flow proxy
import yfinance as yf

# Optional: requests for news API calls
import requests

# HuggingFace Transformers for FinBERT
try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch
    _FINBERT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _FINBERT_AVAILABLE = False
    logger.warning("transformers/torch not installed – FinBERT sentiment unavailable")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FINBERT_MODEL_ID = "ProsusAI/finbert"
_NEWSAPI_BASE_URL = "https://newsapi.org/v2/everything"
_MAX_HEADLINE_TOKENS = 512
_BATCH_SIZE_INFERENCE = 32

# Known ETF AUM in USD billions (approximate, update regularly)
_ETF_AUM: Dict[str, float] = {
    "SPY": 500e9, "QQQ": 200e9, "IWM": 60e9,
    "GLD": 55e9,  "SLV": 10e9,  "USO": 3e9,
    "XLF": 40e9,  "XLK": 55e9,  "XLE": 35e9,
}


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
# FinBERT inference helper
# ---------------------------------------------------------------------------

class _FinBERTSentimentPipeline:
    """
    Wraps HuggingFace FinBERT for batch inference.
    Label mapping: positive → +1, neutral → 0, negative → -1.
    """

    _LABEL_MAP = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

    def __init__(self, model_id: str = _FINBERT_MODEL_ID, device: Optional[str] = None) -> None:
        if not _FINBERT_AVAILABLE:
            raise ImportError("transformers and torch are required for FinBERT sentiment")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading FinBERT from '{model_id}' on device={self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()
        logger.success("FinBERT loaded successfully")

    @torch.no_grad()
    def predict_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Run inference on a list of texts.

        Returns
        -------
        List of dicts: {label: str, score: float, sentiment: float}
        """
        results: List[Dict[str, float]] = []

        for i in range(0, len(texts), _BATCH_SIZE_INFERENCE):
            batch = texts[i : i + _BATCH_SIZE_INFERENCE]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=_MAX_HEADLINE_TOKENS,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            logits = self.model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

            for prob_row in probs:
                # id2label: {0: 'positive', 1: 'negative', 2: 'neutral'} for finbert
                label_id = int(prob_row.argmax())
                label = self.model.config.id2label[label_id].lower()
                score = float(prob_row.max())
                sentiment_value = self._LABEL_MAP.get(label, 0.0)
                results.append(
                    {
                        "label": label,
                        "score": score,
                        "sentiment": sentiment_value * score,  # confidence-weighted
                    }
                )

        return results


# ---------------------------------------------------------------------------
# AlternativeDataCollector
# ---------------------------------------------------------------------------


class AlternativeDataCollector:
    """
    Collects alternative data signals for equity and crypto assets.

    Parameters
    ----------
    newsapi_key : str, optional
        NewsAPI.org API key.  If not provided, tries NEWSAPI_KEY env var.
    cache_dir : Path, optional
        Directory for parquet caching of sentiment data.
    finbert_device : str, optional
        'cpu' or 'cuda'.  Auto-detected if None.
    """

    def __init__(
        self,
        newsapi_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        finbert_device: Optional[str] = None,
    ) -> None:
        self._newsapi_key = newsapi_key or os.getenv("NEWSAPI_KEY")
        self._cache_dir = cache_dir or Path(".cache") / "alternative"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._finbert_device = finbert_device
        self._finbert: Optional[_FinBERTSentimentPipeline] = None   # lazy
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "AI-TradingSystem/1.0"})
        logger.info("AlternativeDataCollector initialised")

    # ------------------------------------------------------------------
    # Lazy FinBERT loader
    # ------------------------------------------------------------------

    def _get_finbert(self) -> _FinBERTSentimentPipeline:
        if self._finbert is None:
            self._finbert = _FinBERTSentimentPipeline(device=self._finbert_device)
        return self._finbert

    # ------------------------------------------------------------------
    # Internal: cache helpers
    # ------------------------------------------------------------------

    def _cache_key(self, *args) -> str:
        raw = "_".join(str(a) for a in args)
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.parquet"

    def _save_cache(self, df: pd.DataFrame, key: str) -> None:
        df.to_parquet(self._cache_path(key))

    def _load_cache(self, key: str, ttl_hours: int = 6) -> Optional[pd.DataFrame]:
        path = self._cache_path(key)
        if not path.exists():
            return None
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        if (datetime.now(timezone.utc) - mtime) > timedelta(hours=ttl_hours):
            return None
        return pd.read_parquet(path)

    # ------------------------------------------------------------------
    # 1. News sentiment via FinBERT
    # ------------------------------------------------------------------

    @_retry()
    def fetch_news_sentiment(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        max_articles_per_day: int = 50,
    ) -> pd.DataFrame:
        """
        Fetch news headlines for *symbol* and score them with FinBERT.

        Data source: NewsAPI.org (requires NEWSAPI_KEY).
        If no API key is set, a realistic synthetic dataset is generated as a
        fallback so the downstream pipeline continues to function.

        Parameters
        ----------
        symbol               : Ticker or keyword (e.g. 'AAPL', 'Bitcoin').
        start                : Inclusive start datetime (UTC).
        end                  : Exclusive end datetime (UTC).
        max_articles_per_day : Cap on articles retrieved per day (API quota).

        Returns
        -------
        pd.DataFrame (DatetimeIndex=UTC day) with columns:
            sentiment_mean, sentiment_std, sentiment_sum,
            article_count, positive_pct, negative_pct, neutral_pct
        """
        cache_key = self._cache_key("news", symbol, start.date(), end.date())
        cached = self._load_cache(cache_key, ttl_hours=12)
        if cached is not None:
            logger.debug(f"News sentiment cache hit: {symbol}")
            return cached

        logger.info(f"Fetching news sentiment | symbol={symbol} {start.date()}→{end.date()}")

        if self._newsapi_key:
            raw_articles = self._fetch_newsapi_articles(symbol, start, end, max_articles_per_day)
        else:
            logger.warning(
                "NEWSAPI_KEY not set – generating synthetic sentiment data. "
                "Set NEWSAPI_KEY in .env to enable real news fetching."
            )
            raw_articles = self._generate_synthetic_articles(symbol, start, end)

        if not raw_articles:
            logger.warning(f"No articles found for {symbol}")
            return pd.DataFrame()

        # Group by date and score via FinBERT
        articles_by_date: Dict[str, List[str]] = {}
        for article in raw_articles:
            day_key = article["date"]
            articles_by_date.setdefault(day_key, []).append(article["text"])

        finbert = self._get_finbert()
        rows = []

        for day_str, texts in sorted(articles_by_date.items()):
            scores = finbert.predict_batch(texts)
            sentiments = [s["sentiment"] for s in scores]
            labels = [s["label"] for s in scores]

            n = len(sentiments)
            rows.append(
                {
                    "date": pd.Timestamp(day_str, tz="UTC"),
                    "sentiment_mean": float(np.mean(sentiments)),
                    "sentiment_std": float(np.std(sentiments)) if n > 1 else 0.0,
                    "sentiment_sum": float(np.sum(sentiments)),
                    "article_count": n,
                    "positive_pct": labels.count("positive") / n,
                    "negative_pct": labels.count("negative") / n,
                    "neutral_pct": labels.count("neutral") / n,
                }
            )

        df = pd.DataFrame(rows).set_index("date")
        df.index.name = "datetime"
        df.sort_index(inplace=True)

        self._save_cache(df, cache_key)
        logger.success(f"News sentiment computed for {symbol}: {len(df)} days")
        return df

    @_retry()
    def _fetch_newsapi_articles(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        max_per_day: int,
    ) -> List[Dict[str, str]]:
        """Call NewsAPI.org and return a list of {date, text} dicts."""
        articles: List[Dict[str, str]] = []
        current = start

        while current < end:
            day_end = min(current + timedelta(days=1), end)
            params = {
                "q": symbol,
                "from": current.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "to": day_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "language": "en",
                "sortBy": "relevancy",
                "pageSize": min(max_per_day, 100),
                "apiKey": self._newsapi_key,
            }
            resp = self._session.get(_NEWSAPI_BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            for art in data.get("articles", []):
                title = art.get("title") or ""
                desc = art.get("description") or ""
                text = (title + " " + desc).strip()
                if text:
                    articles.append(
                        {"date": current.strftime("%Y-%m-%d"), "text": text}
                    )

            current += timedelta(days=1)
            time.sleep(0.2)  # polite rate limit

        return articles

    def _generate_synthetic_articles(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> List[Dict[str, str]]:
        """
        Generate synthetic news headlines for testing purposes.
        Sentiment follows a mean-reverting random walk with fat tails.
        """
        rng = np.random.default_rng(
            seed=int(hashlib.md5(f"{symbol}{start}".encode()).hexdigest(), 16) % (2**32)
        )
        articles = []
        templates_positive = [
            f"{symbol} beats earnings estimates with strong revenue growth",
            f"Analysts upgrade {symbol} with higher price target",
            f"{symbol} announces share buyback program",
            f"{symbol} reports record quarterly profits",
        ]
        templates_negative = [
            f"{symbol} misses earnings expectations on weak guidance",
            f"Regulatory concerns weigh on {symbol} shares",
            f"{symbol} faces antitrust investigation",
            f"Analysts downgrade {symbol} on margin compression",
        ]
        templates_neutral = [
            f"{symbol} to report earnings next week",
            f"Investors watch {symbol} ahead of Fed decision",
            f"{symbol} maintains full-year guidance",
            f"{symbol} CEO speaks at industry conference",
        ]

        current = start
        while current < end:
            n_articles = rng.integers(3, 15)
            for _ in range(n_articles):
                label = rng.choice(["positive", "negative", "neutral"], p=[0.4, 0.3, 0.3])
                if label == "positive":
                    text = rng.choice(templates_positive)
                elif label == "negative":
                    text = rng.choice(templates_negative)
                else:
                    text = rng.choice(templates_neutral)
                articles.append({"date": current.strftime("%Y-%m-%d"), "text": text})
            current += timedelta(days=1)

        return articles

    # ------------------------------------------------------------------
    # 2. Social sentiment (mock structure)
    # ------------------------------------------------------------------

    def fetch_social_sentiment(
        self,
        symbol: str,
        lookback_days: int = 30,
    ) -> pd.DataFrame:
        """
        Fetch social-media sentiment for *symbol*.

        REAL IMPLEMENTATION NOTE:
          - StockTwits API: https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json
            Requires OAuth token: STOCKTWITS_TOKEN (UNKNOWN – set in .env)
          - Reddit API (PRAW): requires REDDIT_CLIENT_ID and REDDIT_SECRET (UNKNOWN)
          - Twitter/X API v2: requires TWITTER_BEARER_TOKEN (UNKNOWN)

        This implementation returns a realistic mock structure that mirrors the
        schema a production integration would produce, allowing downstream
        feature-engineering code to be fully tested without API keys.

        Returns
        -------
        pd.DataFrame (DatetimeIndex=UTC day) with columns:
            social_score       – composite sentiment score [-1, +1]
            social_volume      – estimated number of mentions
            viral_coefficient  – share / retweet ratio (virality proxy)
            bullish_pct        – fraction of bullish posts
            bearish_pct        – fraction of bearish posts
            net_sentiment      – bullish_pct - bearish_pct
        """
        logger.warning(
            f"fetch_social_sentiment: using MOCK data for {symbol}. "
            "Real data requires API keys: STOCKTWITS_TOKEN, REDDIT_CLIENT_ID, "
            "REDDIT_SECRET, TWITTER_BEARER_TOKEN (all UNKNOWN – set in .env)."
        )

        rng = np.random.default_rng(
            seed=int(hashlib.md5(symbol.encode()).hexdigest(), 16) % (2**32)
        )

        end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        dates = [end - timedelta(days=i) for i in range(lookback_days, 0, -1)]

        # Mean-reverting AR(1) sentiment process
        phi = 0.85
        sigma = 0.15
        score = 0.0
        rows = []
        for dt in dates:
            score = phi * score + rng.normal(0, sigma)
            score = np.clip(score, -1, 1)
            volume = int(rng.lognormal(8, 1.5))     # ~3k mentions/day baseline
            viral = float(np.clip(rng.exponential(0.05), 0, 0.5))
            bullish = float(np.clip(0.5 + score * 0.3 + rng.normal(0, 0.05), 0, 1))
            bearish = float(np.clip(0.5 - score * 0.3 + rng.normal(0, 0.05), 0, 1))
            total = bullish + bearish + 1e-8
            bullish /= total
            bearish /= total

            rows.append(
                {
                    "datetime": pd.Timestamp(dt, tz="UTC"),
                    "social_score": round(score, 4),
                    "social_volume": volume,
                    "viral_coefficient": round(viral, 4),
                    "bullish_pct": round(bullish, 4),
                    "bearish_pct": round(bearish, 4),
                    "net_sentiment": round(bullish - bearish, 4),
                }
            )

        df = pd.DataFrame(rows).set_index("datetime")
        logger.success(f"Mock social sentiment generated for {symbol}: {len(df)} days")
        return df

    # ------------------------------------------------------------------
    # 3. ETF flows (yfinance volume proxy)
    # ------------------------------------------------------------------

    def fetch_etf_flows(
        self,
        etf_symbols: Optional[List[str]] = None,
        lookback_days: int = 252,
    ) -> pd.DataFrame:
        """
        Estimate daily ETF fund flows using a volume × price proxy normalised
        by approximate AUM.

        Flow proxy formula:
            flow_pct = (volume × close − 30d_avg_volume × close_30d_avg) / AUM

        A positive value suggests net inflows; negative suggests outflows.
        This is an approximation – actual creation/redemption data requires
        provider-level data (e.g. iShares, State Street, Invesco APIs).

        Parameters
        ----------
        etf_symbols  : List of ETF tickers.  Defaults to the built-in set.
        lookback_days: Calendar days of history to fetch.

        Returns
        -------
        pd.DataFrame with columns [<sym>_flow_pct, <sym>_flow_usd, ...] and
        a UTC DatetimeIndex.
        """
        if etf_symbols is None:
            etf_symbols = list(_ETF_AUM.keys())

        logger.info(f"Fetching ETF flow proxy for: {etf_symbols}")

        start = datetime.now(timezone.utc) - timedelta(days=lookback_days + 60)
        frames: Dict[str, pd.DataFrame] = {}

        for sym in etf_symbols:
            try:
                ticker = yf.Ticker(sym)
                hist = ticker.history(
                    start=start.strftime("%Y-%m-%d"),
                    interval="1d",
                    auto_adjust=True,
                )
                if hist.empty:
                    continue

                hist.columns = hist.columns.str.lower()
                hist.index = pd.to_datetime(hist.index, utc=True)

                if "volume" not in hist or "close" not in hist:
                    continue

                dollar_volume = hist["volume"] * hist["close"]
                rolling_avg = dollar_volume.rolling(30).mean()
                aum = _ETF_AUM.get(sym, 10e9)

                flow_df = pd.DataFrame(
                    {
                        f"{sym}_flow_usd": dollar_volume - rolling_avg,
                        f"{sym}_flow_pct": (dollar_volume - rolling_avg) / aum,
                        f"{sym}_dollar_volume": dollar_volume,
                    },
                    index=hist.index,
                )
                frames[sym] = flow_df

            except Exception as exc:
                logger.error(f"ETF flow fetch failed for {sym}: {exc}")

        if not frames:
            logger.warning("No ETF flow data retrieved")
            return pd.DataFrame()

        result = pd.concat(frames.values(), axis=1).sort_index()
        result.dropna(how="all", inplace=True)

        logger.success(
            f"ETF flows computed: {result.shape[1]} columns, {len(result)} rows"
        )
        return result

    # ------------------------------------------------------------------
    # 4. Sentiment feature engineering
    # ------------------------------------------------------------------

    def compute_sentiment_features(
        self,
        sentiment_df: pd.DataFrame,
        price_series: Optional[pd.Series] = None,
        score_col: str = "sentiment_mean",
    ) -> pd.DataFrame:
        """
        Engineer predictive features from a daily sentiment DataFrame.

        Features computed:
          - sent_ma5, sent_ma21        : rolling mean (smoothed signal)
          - sent_momentum_5d           : 5-day change in sentiment
          - sent_momentum_21d          : 21-day change in sentiment
          - sent_vol_21d               : 21-day rolling standard deviation
          - sent_z_score               : z-score relative to 63-day window
          - sent_divergence            : sentiment direction vs price return
                                         (requires price_series)
          - sent_extreme_positive      : 1 if sent_z_score > 1.5
          - sent_extreme_negative      : 1 if sent_z_score < -1.5
          - sent_reversal_signal       : extreme reading + momentum reversal

        Parameters
        ----------
        sentiment_df : Daily DataFrame with at least *score_col* column.
        price_series : Daily close price for divergence computation.
        score_col    : Column name of the raw sentiment score to use.

        Returns
        -------
        pd.DataFrame of features with same DatetimeIndex as sentiment_df.
        """
        if sentiment_df.empty or score_col not in sentiment_df.columns:
            logger.warning("compute_sentiment_features: empty or missing score column")
            return pd.DataFrame()

        logger.info(f"Computing sentiment features (score_col={score_col})")

        score = sentiment_df[score_col].copy()
        feats = pd.DataFrame(index=sentiment_df.index)

        # --- Rolling means (smoothing) ---
        feats["sent_ma5"] = score.rolling(5, min_periods=1).mean()
        feats["sent_ma21"] = score.rolling(21, min_periods=5).mean()

        # --- Momentum ---
        feats["sent_momentum_5d"] = score.diff(5)
        feats["sent_momentum_21d"] = score.diff(21)

        # --- Volatility ---
        feats["sent_vol_21d"] = score.rolling(21, min_periods=5).std()

        # --- Z-score (cross-sectional normalisation) ---
        roll_mean = score.rolling(63, min_periods=21).mean()
        roll_std = score.rolling(63, min_periods=21).std()
        feats["sent_z_score"] = (score - roll_mean) / (roll_std + 1e-8)

        # --- Extreme readings ---
        feats["sent_extreme_positive"] = (feats["sent_z_score"] > 1.5).astype(int)
        feats["sent_extreme_negative"] = (feats["sent_z_score"] < -1.5).astype(int)

        # --- Reversal signal: extreme + momentum reversing ---
        feats["sent_reversal_signal"] = (
            (feats["sent_extreme_positive"] & (feats["sent_momentum_5d"] < 0)) |
            (feats["sent_extreme_negative"] & (feats["sent_momentum_5d"] > 0))
        ).astype(int)

        # --- Divergence from price (contrarian signal) ---
        if price_series is not None:
            price_aligned = price_series.reindex(sentiment_df.index, method="ffill")
            price_ret_5d = price_aligned.pct_change(5)
            sent_chg_5d = feats["sent_momentum_5d"]
            # Divergence: price rising but sentiment falling (or vice versa)
            feats["sent_price_divergence"] = np.where(
                (price_ret_5d > 0) & (sent_chg_5d < -0.05), -1,
                np.where(
                    (price_ret_5d < 0) & (sent_chg_5d > 0.05), 1, 0
                ),
            )

        # --- Carry-forward article count if available ---
        if "article_count" in sentiment_df.columns:
            feats["news_volume"] = sentiment_df["article_count"]
            feats["news_volume_ma5"] = feats["news_volume"].rolling(5, min_periods=1).mean()

        feats.dropna(how="all", inplace=True)
        logger.success(f"Sentiment features computed: {feats.shape}")
        return feats
