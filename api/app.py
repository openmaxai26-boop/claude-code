"""
API REST du système de trading.

Endpoints
---------
GET  /              → page d'accueil / statut
GET  /health        → health check (Railway l'utilise)
GET  /signals       → signaux actuels pour tous les symboles configurés
POST /webhook       → reçoit les alertes TradingView
GET  /backtest      → lance un backtest rapide et retourne les métriques
GET  /positions     → positions simulées en cours
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Trading System",
    description="Système de trading algorithmique propulsé par IA",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Config depuis variables d'environnement
# ---------------------------------------------------------------------------

SYMBOLS: List[str] = os.getenv("SYMBOLS", "AAPL,MSFT,GLD,BTC-USD").split(",")
TIMEFRAME: str = os.getenv("TIMEFRAME", "1d")
WEBHOOK_SECRET: str = os.getenv("WEBHOOK_SECRET", "")

# ---------------------------------------------------------------------------
# État interne simple (en mémoire)
# ---------------------------------------------------------------------------

_last_signals: List[dict] = []
_webhook_log: List[dict] = []
_positions: Dict[str, float] = {}  # symbol -> position size %


# ---------------------------------------------------------------------------
# Modèles Pydantic
# ---------------------------------------------------------------------------

class TradingViewAlert(BaseModel):
    """Format d'alerte TradingView (configuré dans le webhook TradingView)."""
    symbol: str
    action: str           # "buy" | "sell" | "close"
    price: Optional[float] = None
    timeframe: Optional[str] = None
    strategy: Optional[str] = None
    comment: Optional[str] = None


class SignalResponse(BaseModel):
    asset: str
    signal: str           # BUY | SELL | HOLD
    probability: float
    confidence_score: float
    risk_level: str
    predicted_return: float
    position_size_pct: float
    stop_loss_price: float
    take_profit_price: float
    regime: str
    timestamp: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "name": "AI Trading System",
        "version": "1.0.0",
        "status": "running",
        "symbols": SYMBOLS,
        "timeframe": TIMEFRAME,
        "timestamp": _now(),
        "endpoints": {
            "signals": "/signals",
            "webhook": "/webhook  [POST]",
            "backtest": "/backtest?symbol=AAPL&days=365",
            "positions": "/positions",
            "health": "/health",
        },
    }


@app.get("/health")
def health():
    """Railway vérifie cet endpoint pour savoir si l'app tourne."""
    return {"status": "ok", "timestamp": _now()}


@app.get("/signals")
def get_signals():
    """
    Génère des signaux pour tous les symboles configurés.
    Tente d'utiliser le vrai système ; retourne des signaux de démonstration si
    les dépendances lourdes (torch, yfinance…) ne sont pas disponibles.
    """
    signals = []

    for sym in SYMBOLS:
        try:
            signal = _generate_signal_real(sym)
        except Exception as exc:
            logger.warning("Fallback demo signal for %s: %s", sym, exc)
            signal = _demo_signal(sym)
        signals.append(signal)

    # Mémorise pour /positions
    global _last_signals
    _last_signals = signals

    return {
        "timestamp": _now(),
        "count": len(signals),
        "signals": signals,
    }


@app.post("/webhook")
async def tradingview_webhook(
    alert: TradingViewAlert,
    request: Request,
    x_webhook_secret: Optional[str] = Header(None),
):
    """
    Reçoit une alerte depuis TradingView.

    Comment configurer dans TradingView :
    1. Crée une alerte sur un indicateur ou une stratégie
    2. Dans "Webhook URL" → met l'URL de ton app Railway + /webhook
    3. Dans "Message" → colle ce JSON :
       {
         "symbol": "{{ticker}}",
         "action": "{{strategy.order.action}}",
         "price": {{close}},
         "timeframe": "{{interval}}",
         "strategy": "{{strategy.order.comment}}"
       }
    """
    # Vérification du secret (optionnel mais recommandé)
    if WEBHOOK_SECRET and x_webhook_secret != WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="Secret invalide")

    logger.info("Webhook reçu : %s %s @ %s", alert.action.upper(), alert.symbol, alert.price)

    record = {
        "received_at": _now(),
        "symbol": alert.symbol,
        "action": alert.action.upper(),
        "price": alert.price,
        "timeframe": alert.timeframe,
        "strategy": alert.strategy,
        "comment": alert.comment,
        "status": "received",
    }

    # Ici on pourrait envoyer l'ordre au broker (Alpaca, Binance…)
    # Pour l'instant on logue et on met à jour la position simulée
    _webhook_log.append(record)
    _update_position(alert.symbol, alert.action)

    return {
        "status": "ok",
        "message": f"Alerte traitée : {alert.action.upper()} {alert.symbol}",
        "record": record,
    }


@app.get("/webhook/log")
def webhook_log():
    """Historique des alertes reçues."""
    return {
        "count": len(_webhook_log),
        "alerts": _webhook_log[-50:],  # 50 dernières
    }


@app.get("/backtest")
def run_backtest(symbol: str = "AAPL", days: int = 365):
    """
    Lance un backtest rapide sur les `days` derniers jours.

    Exemple : GET /backtest?symbol=AAPL&days=252
    """
    try:
        return _run_backtest(symbol, days)
    except Exception as exc:
        logger.error("Backtest error: %s", exc)
        return _demo_backtest_result(symbol, days)


@app.get("/positions")
def get_positions():
    """Positions simulées en cours (issues des signaux et webhooks)."""
    return {
        "timestamp": _now(),
        "positions": _positions,
        "note": "Positions simulées — paper trading uniquement",
    }


# ---------------------------------------------------------------------------
# Logique interne
# ---------------------------------------------------------------------------

def _generate_signal_real(symbol: str) -> dict:
    """Tente d'utiliser le vrai pipeline (yfinance + backtesting)."""
    import yfinance as yf
    import numpy as np
    import sys
    sys.path.insert(0, ".")

    df = yf.download(symbol, period="60d", interval="1d", progress=False)
    if df.empty:
        raise ValueError(f"Pas de données pour {symbol}")

    close = df["Close"].squeeze()
    returns = close.pct_change().dropna()

    # Signal simple : momentum 20 jours
    momentum = float(returns.tail(20).mean())
    vol = float(returns.tail(20).std())
    pred_return = momentum
    pred_vol = vol
    proba = float(50 + momentum * 2000)
    proba = max(0, min(100, proba))

    if momentum > 0.001:
        signal = "BUY"
    elif momentum < -0.001:
        signal = "SELL"
    else:
        signal = "HOLD"

    price = float(close.iloc[-1])
    stop_loss = round(price * 0.95, 2)
    take_profit = round(price * (1 + abs(pred_return) * 2), 2)

    return {
        "asset": symbol,
        "signal": signal,
        "probability": round(proba, 1),
        "confidence_score": round(max(0, 1 - vol * 10), 2),
        "risk_level": "HIGH" if vol > 0.03 else ("LOW" if vol < 0.01 else "MEDIUM"),
        "predicted_return": round(pred_return, 6),
        "predicted_volatility": round(pred_vol, 6),
        "position_size_pct": round(min(0.10, (1 - vol * 10) * 0.10), 4),
        "stop_loss_price": stop_loss,
        "take_profit_price": take_profit,
        "current_price": price,
        "regime": "trending" if abs(momentum) > 0.001 else "ranging",
        "timestamp": _now(),
    }


def _demo_signal(symbol: str) -> dict:
    """Signal de démonstration quand les dépendances sont absentes."""
    import random
    import math
    seed = sum(ord(c) for c in symbol) + datetime.now().hour
    rng = random.Random(seed)
    pred_return = rng.uniform(-0.02, 0.02)
    pred_vol = rng.uniform(0.008, 0.035)
    proba = 50 + pred_return * 1000
    signal = "BUY" if pred_return > 0.005 else ("SELL" if pred_return < -0.005 else "HOLD")
    price = rng.uniform(100, 300)
    return {
        "asset": symbol,
        "signal": signal,
        "probability": round(max(0, min(100, proba)), 1),
        "confidence_score": round(max(0, 1 - pred_vol * 10), 2),
        "risk_level": "HIGH" if pred_vol > 0.025 else ("LOW" if pred_vol < 0.012 else "MEDIUM"),
        "predicted_return": round(pred_return, 6),
        "predicted_volatility": round(pred_vol, 6),
        "position_size_pct": 0.05,
        "stop_loss_price": round(price * 0.95, 2),
        "take_profit_price": round(price * 1.06, 2),
        "current_price": round(price, 2),
        "regime": "demo",
        "timestamp": _now(),
        "note": "⚠️ Signal de démonstration (installez yfinance pour les vrais signaux)",
    }


def _run_backtest(symbol: str, days: int) -> dict:
    import yfinance as yf
    import numpy as np
    import sys
    sys.path.insert(0, ".")
    from src.backtesting.engine import BacktestEngine
    from src.backtesting.metrics import PerformanceMetrics
    import pandas as pd

    period = f"{days}d" if days <= 730 else "5y"
    df = yf.download(symbol, period=period, interval="1d", progress=False)
    if df.empty:
        raise ValueError("Pas de données")

    close = df[["Close"]].copy()
    close.columns = [symbol]

    # Stratégie momentum simple
    returns = close.pct_change()
    signals = returns.shift(1).clip(-1, 1)
    signals.columns = [symbol]

    engine = BacktestEngine(initial_capital=100_000)
    results = engine.run(signals, close)
    report = engine.generate_report(results)
    report["symbol"] = symbol
    report["days_tested"] = len(close)
    report["strategy"] = "Momentum 1 jour"
    return report


def _demo_backtest_result(symbol: str, days: int) -> dict:
    return {
        "symbol": symbol,
        "days_tested": days,
        "strategy": "Momentum 1 jour",
        "total_return_pct": 12.4,
        "sharpe_ratio": 0.87,
        "max_drawdown_pct": -18.3,
        "win_rate_pct": 53.2,
        "note": "⚠️ Résultats de démonstration (installez yfinance pour le vrai backtest)",
    }


def _update_position(symbol: str, action: str) -> None:
    action = action.upper()
    if action == "BUY":
        _positions[symbol] = _positions.get(symbol, 0) + 0.05
    elif action == "SELL":
        _positions[symbol] = _positions.get(symbol, 0) - 0.05
    elif action == "CLOSE":
        _positions.pop(symbol, None)
    # Clamp entre -1 et 1
    if symbol in _positions:
        _positions[symbol] = max(-1.0, min(1.0, _positions[symbol]))


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()
