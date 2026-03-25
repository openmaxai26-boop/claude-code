"""
API REST du système de trading.

Endpoints
---------
GET  /              → page d'accueil / statut
GET  /health        → health check (Render l'utilise)
GET  /signals       → signaux actuels pour tous les symboles configurés
GET  /signals/history → historique des 50 derniers signaux
GET  /backtest      → lance un backtest rapide et retourne les métriques
GET  /positions     → positions en cours (paper trading)
GET  /dashboard     → tableau de bord HTML lisible dans le navigateur
POST /webhook       → reçoit des alertes externes (optionnel)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# ---------------------------------------------------------------------------
# Config depuis variables d'environnement
# ---------------------------------------------------------------------------

SYMBOLS: List[str] = os.getenv("SYMBOLS", "AAPL,MSFT,GLD,BTC-USD").split(",")
TIMEFRAME: str = os.getenv("TIMEFRAME", "1d")
WEBHOOK_SECRET: str = os.getenv("WEBHOOK_SECRET", "")
SCHEDULE_HOURS: int = int(os.getenv("SCHEDULE_HOURS", "1"))  # toutes les X heures

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
# État interne (en mémoire)
# ---------------------------------------------------------------------------

_last_signals: List[dict] = []
_signals_history: List[dict] = []   # 50 derniers runs
_positions: Dict[str, float] = {}   # symbol -> taille position %
_webhook_log: List[dict] = []
_scheduler_runs: List[str] = []     # timestamps des exécutions auto
_scheduler = None


# ---------------------------------------------------------------------------
# Scheduler automatique
# ---------------------------------------------------------------------------

def _run_signals_job():
    """Génère les signaux automatiquement (appelé par le scheduler)."""
    global _last_signals
    logger.info("⏰ Scheduler : génération automatique des signaux...")
    signals = []
    for sym in SYMBOLS:
        try:
            sig = _generate_signal_real(sym)
        except Exception as exc:
            logger.warning("Fallback demo signal pour %s : %s", sym, exc)
            sig = _demo_signal(sym)
        signals.append(sig)

    _last_signals = signals
    _signals_history.append({
        "run_at": _now(),
        "signals": signals,
    })
    # Garde les 50 derniers runs
    if len(_signals_history) > 50:
        _signals_history.pop(0)

    _scheduler_runs.append(_now())
    if len(_scheduler_runs) > 100:
        _scheduler_runs.pop(0)

    logger.info("✅ Signaux générés pour %d symboles", len(signals))


@app.on_event("startup")
def start_scheduler():
    """Lance le scheduler au démarrage de l'app."""
    global _scheduler
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        _scheduler = BackgroundScheduler()
        _scheduler.add_job(
            _run_signals_job,
            trigger="interval",
            hours=SCHEDULE_HOURS,
            id="signals_job",
            replace_existing=True,
        )
        _scheduler.start()
        logger.info("⏰ Scheduler démarré — signaux toutes les %dh", SCHEDULE_HOURS)
    except ImportError:
        logger.warning("APScheduler non installé — scheduler désactivé")

    # Génère les signaux immédiatement au démarrage
    _run_signals_job()


@app.on_event("shutdown")
def stop_scheduler():
    global _scheduler
    if _scheduler:
        _scheduler.shutdown()


# ---------------------------------------------------------------------------
# Modèles
# ---------------------------------------------------------------------------

class WebhookAlert(BaseModel):
    symbol: str
    action: str
    price: Optional[float] = None
    comment: Optional[str] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "name": "Système de trading IA",
        "version": "1.0.0",
        "status": "running",
        "symbols": SYMBOLS,
        "timeframe": TIMEFRAME,
        "scheduler": f"toutes les {SCHEDULE_HOURS}h",
        "dernier_run": _scheduler_runs[-1] if _scheduler_runs else "jamais",
        "timestamp": _now(),
        "endpoints": {
            "signaux": "/signals",
            "historique": "/signals/history",
            "backtest": "/backtest?symbol=AAPL&days=365",
            "positions": "/positions",
            "dashboard": "/dashboard",
            "health": "/health",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": _now()}


@app.get("/signals")
def get_signals(refresh: bool = False):
    """
    Retourne les derniers signaux calculés.
    Ajoute ?refresh=true pour forcer un recalcul immédiat.
    """
    global _last_signals
    if refresh or not _last_signals:
        _run_signals_job()
    return {
        "timestamp": _now(),
        "count": len(_last_signals),
        "signals": _last_signals,
    }


@app.get("/signals/history")
def get_signals_history():
    """Historique des 50 dernières générations de signaux."""
    return {
        "total_runs": len(_signals_history),
        "scheduler_runs": _scheduler_runs[-10:],
        "history": _signals_history[-10:],
    }


@app.get("/backtest")
def run_backtest(symbol: str = "AAPL", days: int = 365):
    """
    Backtest rapide sur les X derniers jours.
    Exemple : /backtest?symbol=AAPL&days=252
    """
    try:
        return _run_backtest(symbol, days)
    except Exception as exc:
        logger.error("Erreur backtest : %s", exc)
        return _demo_backtest_result(symbol, days)


@app.get("/positions")
def get_positions():
    return {
        "timestamp": _now(),
        "positions": _positions,
        "note": "Paper trading uniquement — aucun argent réel",
    }


@app.post("/webhook")
def webhook(alert: WebhookAlert, x_webhook_secret: Optional[str] = Header(None)):
    """Reçoit des alertes externes."""
    if WEBHOOK_SECRET and x_webhook_secret != WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="Secret invalide")
    record = {"received_at": _now(), **alert.dict()}
    _webhook_log.append(record)
    _update_position(alert.symbol, alert.action)
    return {"status": "ok", "record": record}


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    """Tableau de bord HTML — ouvrir dans le navigateur."""
    rows = ""
    for s in _last_signals:
        color = {"BUY": "#22c55e", "SELL": "#ef4444", "HOLD": "#f59e0b"}.get(s["signal"], "#6b7280")
        rows += f"""
        <tr>
          <td><b>{s['asset']}</b></td>
          <td style="color:{color};font-weight:bold">{s['signal']}</td>
          <td>{s['probability']}%</td>
          <td>{s['confidence_score']}</td>
          <td>{s['risk_level']}</td>
          <td>{s.get('current_price', '—')}</td>
          <td style="color:#ef4444">{s['stop_loss_price']}</td>
          <td style="color:#22c55e">{s['take_profit_price']}</td>
          <td>{s['regime']}</td>
        </tr>"""

    last_run = _scheduler_runs[-1] if _scheduler_runs else "Jamais"
    next_info = f"toutes les {SCHEDULE_HOURS}h"

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="refresh" content="60">
  <title>AI Trading System</title>
  <style>
    body {{ font-family: monospace; background: #0f172a; color: #e2e8f0; padding: 20px; }}
    h1 {{ color: #38bdf8; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
    th {{ background: #1e293b; padding: 10px; text-align: left; color: #94a3b8; }}
    td {{ padding: 10px; border-bottom: 1px solid #1e293b; }}
    .badge {{ background: #1e293b; padding: 4px 10px; border-radius: 20px; font-size: 12px; }}
    .info {{ color: #94a3b8; margin: 5px 0; font-size: 13px; }}
  </style>
</head>
<body>
  <h1>🤖 AI Trading System</h1>
  <p class="info">⏰ Dernier calcul : <b>{last_run}</b></p>
  <p class="info">🔄 Fréquence : <b>{next_info}</b> — Page actualisée toutes les 60s</p>
  <p class="info">📊 Symboles : <b>{', '.join(SYMBOLS)}</b></p>

  <table>
    <thead>
      <tr>
        <th>Action</th><th>Signal</th><th>Probabilité</th>
        <th>Confiance</th><th>Risque</th><th>Prix actuel</th>
        <th>Stop-loss</th><th>Take-profit</th><th>Régime</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>

  <br>
  <p class="info">💡 <a href="/signals" style="color:#38bdf8">/signals</a> (JSON) —
     <a href="/signals?refresh=true" style="color:#38bdf8">/signals?refresh=true</a> (forcer recalcul) —
     <a href="/backtest?symbol=AAPL&days=365" style="color:#38bdf8">/backtest</a></p>
</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# Logique interne
# ---------------------------------------------------------------------------

def _generate_signal_real(symbol: str) -> dict:
    import yfinance as yf
    import numpy as np

    df = yf.download(symbol, period="60d", interval="1d", progress=False)
    if df.empty:
        raise ValueError(f"Pas de données pour {symbol}")

    close = df["Close"].squeeze()
    returns = close.pct_change().dropna()
    momentum = float(returns.tail(20).mean())
    vol = float(returns.tail(20).std())
    proba = float(max(0, min(100, 50 + momentum * 2000)))

    if momentum > 0.001:
        signal = "BUY"
    elif momentum < -0.001:
        signal = "SELL"
    else:
        signal = "HOLD"

    price = float(close.iloc[-1])
    return {
        "asset": symbol,
        "signal": signal,
        "probability": round(proba, 1),
        "confidence_score": round(max(0, 1 - vol * 10), 2),
        "risk_level": "HIGH" if vol > 0.03 else ("LOW" if vol < 0.01 else "MEDIUM"),
        "predicted_return": round(momentum, 6),
        "predicted_volatility": round(vol, 6),
        "position_size_pct": round(min(0.10, (1 - vol * 10) * 0.10), 4),
        "stop_loss_price": round(price * 0.95, 2),
        "take_profit_price": round(price * (1 + abs(momentum) * 2), 2),
        "current_price": round(price, 2),
        "regime": "trending" if abs(momentum) > 0.001 else "ranging",
        "timestamp": _now(),
    }


def _demo_signal(symbol: str) -> dict:
    import random
    seed = sum(ord(c) for c in symbol) + datetime.now().hour
    rng = random.Random(seed)
    pred_return = rng.uniform(-0.02, 0.02)
    pred_vol = rng.uniform(0.008, 0.035)
    proba = 50 + pred_return * 1000
    signal = "BUY" if pred_return > 0.005 else ("SELL" if pred_return < -0.005 else "HOLD")
    price = rng.uniform(100, 300)
    return {
        "asset": symbol, "signal": signal,
        "probability": round(max(0, min(100, proba)), 1),
        "confidence_score": round(max(0, 1 - pred_vol * 10), 2),
        "risk_level": "MEDIUM", "predicted_return": round(pred_return, 6),
        "predicted_volatility": round(pred_vol, 6), "position_size_pct": 0.05,
        "stop_loss_price": round(price * 0.95, 2),
        "take_profit_price": round(price * 1.06, 2),
        "current_price": round(price, 2), "regime": "demo", "timestamp": _now(),
        "note": "⚠️ Démo",
    }


def _run_backtest(symbol: str, days: int) -> dict:
    import sys
    import pandas as pd
    import yfinance as yf
    sys.path.insert(0, ".")
    from src.backtesting.engine import BacktestEngine

    period = f"{days}d" if days <= 730 else "5y"
    df = yf.download(symbol, period=period, interval="1d", progress=False)
    if df.empty:
        raise ValueError("Pas de données")

    close = df[["Close"]].copy()
    close.columns = [symbol]
    signals = close.pct_change().shift(1).clip(-1, 1)
    signals.columns = [symbol]

    engine = BacktestEngine(initial_capital=100_000)
    results = engine.run(signals, close)
    report = engine.generate_report(results)
    report.update({"symbol": symbol, "days_tested": len(close), "strategy": "Momentum 1 jour"})
    return report


def _demo_backtest_result(symbol: str, days: int) -> dict:
    return {
        "symbol": symbol, "days_tested": days,
        "total_return_pct": 12.4, "sharpe_ratio": 0.87,
        "max_drawdown_pct": -18.3, "win_rate_pct": 53.2,
        "note": "⚠️ Résultats de démonstration",
    }


def _update_position(symbol: str, action: str) -> None:
    action = action.upper()
    if action in ("BUY", "LONG"):
        _positions[symbol] = min(1.0, _positions.get(symbol, 0) + 0.05)
    elif action in ("SELL", "SHORT"):
        _positions[symbol] = max(-1.0, _positions.get(symbol, 0) - 0.05)
    elif action == "CLOSE":
        _positions.pop(symbol, None)


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()
