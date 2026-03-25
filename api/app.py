"""
API REST du système de trading IA.

Endpoints
---------
GET  /              → statut
GET  /health        → health check Render
GET  /signals       → signaux BUY/SELL/HOLD en temps réel
GET  /signals/history → historique des runs
GET  /backtest      → backtest rapide
GET  /positions     → positions paper trading
POST /analyze       → analyse personnalisée (budget + objectif + stratégie)
GET  /dashboard     → tableau de bord HTML
POST /webhook       → alertes externes
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SYMBOLS: List[str] = os.getenv("SYMBOLS", "AAPL,MSFT,GLD,BTC-USD").split(",")
TIMEFRAME: str = os.getenv("TIMEFRAME", "1d")
WEBHOOK_SECRET: str = os.getenv("WEBHOOK_SECRET", "")
SCHEDULE_HOURS: int = int(os.getenv("SCHEDULE_HOURS", "1"))

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="AI Trading System", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---------------------------------------------------------------------------
# État interne
# ---------------------------------------------------------------------------

_last_signals: List[dict] = []
_signals_history: List[dict] = []
_positions: Dict[str, float] = {}
_webhook_log: List[dict] = []
_scheduler_runs: List[str] = []
_scheduler = None


# ---------------------------------------------------------------------------
# Stratégies disponibles
# ---------------------------------------------------------------------------

STRATEGIES = {
    "daily": {
        "label": "Trading quotidien",
        "lookback_days": 5,
        "stop_loss_pct": 0.02,       # -2%
        "take_profit_pct": 0.04,     # +4%
        "max_positions": 3,
        "assets": ["BTC-USD", "ETH-USD", "AAPL", "TSLA"],
        "description": "Positions ouvertes et fermées dans la journée ou la semaine. Risque élevé, gains potentiels rapides.",
    },
    "weekly": {
        "label": "Trading hebdomadaire",
        "lookback_days": 20,
        "stop_loss_pct": 0.05,       # -5%
        "take_profit_pct": 0.10,     # +10%
        "max_positions": 5,
        "assets": ["AAPL", "MSFT", "GOOGL", "BTC-USD", "GLD"],
        "description": "Positions tenues quelques jours à quelques semaines. Bon équilibre risque/rendement.",
    },
    "monthly": {
        "label": "Trading mensuel",
        "lookback_days": 60,
        "stop_loss_pct": 0.08,
        "take_profit_pct": 0.20,
        "max_positions": 6,
        "assets": ["AAPL", "MSFT", "GOOGL", "AMZN", "GLD", "BTC-USD"],
        "description": "Positions tenues de 2 semaines à 2 mois. Idéal pour travailleurs à temps plein.",
    },
    "yearly": {
        "label": "Investissement long terme",
        "lookback_days": 252,
        "stop_loss_pct": 0.15,
        "take_profit_pct": 0.50,
        "max_positions": 8,
        "assets": ["AAPL", "MSFT", "GOOGL", "AMZN", "BRK-B", "GLD", "BTC-USD", "SPY"],
        "description": "Positions tenues plusieurs mois à années. Le moins risqué sur le long terme.",
    },
}

OBJECTIVES = {
    "capital_preservation": {
        "label": "Préservation du capital",
        "risk_multiplier": 0.5,
        "description": "Priorité à ne pas perdre d'argent. Rendements faibles mais sûrs.",
    },
    "moderate_growth": {
        "label": "Croissance modérée",
        "risk_multiplier": 1.0,
        "description": "Équilibre entre sécurité et performance. Objectif ~10-20%/an.",
    },
    "aggressive_growth": {
        "label": "Croissance agressive",
        "risk_multiplier": 1.5,
        "description": "Maximiser les gains, accepter plus de volatilité.",
    },
    "income": {
        "label": "Revenus réguliers",
        "risk_multiplier": 0.8,
        "description": "Focus sur dividendes et actifs stables.",
    },
}


# ---------------------------------------------------------------------------
# Modèles Pydantic
# ---------------------------------------------------------------------------

class UserProfile(BaseModel):
    budget: float = Field(..., gt=0, description="Budget en USD")
    strategy: str = Field(..., description="daily | weekly | monthly | yearly")
    objective: str = Field(..., description="capital_preservation | moderate_growth | aggressive_growth | income")
    risk_tolerance: str = Field("medium", description="low | medium | high")
    currency: str = Field("USD", description="USD | EUR | CAD ...")

class WebhookAlert(BaseModel):
    symbol: str
    action: str
    price: Optional[float] = None
    comment: Optional[str] = None


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def _run_signals_job():
    global _last_signals
    logger.info("⏰ Génération automatique des signaux...")
    signals = []
    for sym in SYMBOLS:
        try:
            sig = _generate_signal(sym, lookback_days=20)
        except Exception as exc:
            logger.warning("Fallback démo pour %s : %s", sym, exc)
            sig = _demo_signal(sym)
        signals.append(sig)
    _last_signals = signals
    _signals_history.append({"run_at": _now(), "signals": signals})
    if len(_signals_history) > 50:
        _signals_history.pop(0)
    _scheduler_runs.append(_now())
    if len(_scheduler_runs) > 100:
        _scheduler_runs.pop(0)
    logger.info("✅ %d signaux générés", len(signals))


@app.on_event("startup")
def start_scheduler():
    global _scheduler
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        _scheduler = BackgroundScheduler()
        _scheduler.add_job(_run_signals_job, "interval", hours=SCHEDULE_HOURS, id="signals_job", replace_existing=True)
        _scheduler.start()
        logger.info("⏰ Scheduler démarré — toutes les %dh", SCHEDULE_HOURS)
    except ImportError:
        logger.warning("APScheduler absent — scheduler désactivé")
    _run_signals_job()


@app.on_event("shutdown")
def stop_scheduler():
    if _scheduler:
        _scheduler.shutdown()


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
        "dernier_run": _scheduler_runs[-1] if _scheduler_runs else "jamais",
        "timestamp": _now(),
        "endpoints": {
            "dashboard": "/dashboard",
            "signaux": "/signals",
            "analyse_personnalisée": "/analyze  [POST]",
            "backtest": "/backtest?symbol=AAPL&days=365",
            "documentation": "/docs",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": _now()}


@app.get("/signals")
def get_signals(refresh: bool = False):
    global _last_signals
    if refresh or not _last_signals:
        _run_signals_job()
    return {"timestamp": _now(), "count": len(_last_signals), "signals": _last_signals}


@app.get("/signals/history")
def get_signals_history():
    return {"total_runs": len(_signals_history), "history": _signals_history[-10:]}


@app.post("/analyze")
def analyze(profile: UserProfile):
    """
    Analyse personnalisée selon ton budget, objectif et stratégie.

    Exemple de body JSON :
    {
        "budget": 1000,
        "strategy": "weekly",
        "objective": "moderate_growth",
        "risk_tolerance": "medium",
        "currency": "USD"
    }
    """
    if profile.strategy not in STRATEGIES:
        raise HTTPException(400, f"Stratégie invalide. Choix : {list(STRATEGIES.keys())}")
    if profile.objective not in OBJECTIVES:
        raise HTTPException(400, f"Objectif invalide. Choix : {list(OBJECTIVES.keys())}")

    strat = STRATEGIES[profile.strategy]
    obj = OBJECTIVES[profile.objective]

    # Ajustement du risque selon le profil
    risk_mult = obj["risk_multiplier"]
    if profile.risk_tolerance == "low":
        risk_mult *= 0.6
    elif profile.risk_tolerance == "high":
        risk_mult *= 1.4

    # Taille max par position en $
    max_positions = strat["max_positions"]
    per_position_usd = (profile.budget / max_positions) * risk_mult
    per_position_usd = min(per_position_usd, profile.budget * 0.25)  # jamais > 25% du budget

    # Stop-loss adapté
    stop_loss_pct = strat["stop_loss_pct"]
    take_profit_pct = strat["take_profit_pct"]
    max_loss_usd = profile.budget * stop_loss_pct
    target_gain_usd = profile.budget * take_profit_pct

    # Génère les signaux pour les actifs de cette stratégie
    assets = strat["assets"]
    signals = []
    for sym in assets:
        try:
            sig = _generate_signal(sym, lookback_days=strat["lookback_days"])
            sig["position_usd"] = round(per_position_usd, 2)
            sig["stop_loss_usd"] = round(per_position_usd * stop_loss_pct, 2)
            sig["take_profit_usd"] = round(per_position_usd * take_profit_pct, 2)
            sig["shares_possible"] = round(per_position_usd / sig["current_price"], 4) if sig.get("current_price", 0) > 0 else "N/A"
            signals.append(sig)
        except Exception as exc:
            logger.warning("Pas de signal pour %s : %s", sym, exc)

    # Filtre : garde seulement les signaux actionnables
    actionable = [s for s in signals if s.get("signal") in ("BUY", "SELL")]
    actionable_sorted = sorted(actionable, key=lambda s: s.get("probability", 0), reverse=True)

    # Résumé du plan
    plan = {
        "résumé": {
            "budget_total": f"{profile.budget:,.0f} {profile.currency}",
            "stratégie": strat["label"],
            "objectif": obj["label"],
            "tolérance_risque": profile.risk_tolerance,
            "max_positions_simultanées": max_positions,
            "budget_par_position": f"{per_position_usd:,.0f} {profile.currency}",
            "perte_max_par_position": f"{per_position_usd * stop_loss_pct:,.0f} {profile.currency}  ({stop_loss_pct*100:.0f}%)",
            "gain_cible_par_position": f"{per_position_usd * take_profit_pct:,.0f} {profile.currency}  ({take_profit_pct*100:.0f}%)",
            "perte_max_totale_scénario_pire": f"{max_loss_usd * max_positions:,.0f} {profile.currency}",
            "gain_cible_total": f"{target_gain_usd * max_positions:,.0f} {profile.currency}",
        },
        "conseils": _generate_advice(profile, strat, obj),
        "signaux_actionnables": actionable_sorted[:max_positions],
        "tous_les_signaux": signals,
        "stratégie_détails": strat["description"],
        "objectif_détails": obj["description"],
    }
    return plan


@app.get("/positions")
def get_positions():
    return {"timestamp": _now(), "positions": _positions, "note": "Paper trading — aucun argent réel"}


@app.post("/webhook")
def webhook(alert: WebhookAlert, x_webhook_secret: Optional[str] = Header(None)):
    if WEBHOOK_SECRET and x_webhook_secret != WEBHOOK_SECRET:
        raise HTTPException(401, "Secret invalide")
    record = {"received_at": _now(), **alert.dict()}
    _webhook_log.append(record)
    _update_position(alert.symbol, alert.action)
    return {"status": "ok", "record": record}


@app.get("/backtest")
def run_backtest(symbol: str = "AAPL", days: int = 365):
    try:
        return _run_backtest(symbol, days)
    except Exception as exc:
        logger.error("Erreur backtest : %s", exc)
        return {"symbol": symbol, "days": days, "erreur": str(exc), "note": "Installez les dépendances complètes"}


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    rows = ""
    for s in _last_signals:
        color = {"BUY": "#22c55e", "SELL": "#ef4444", "HOLD": "#f59e0b"}.get(s.get("signal", ""), "#6b7280")
        emoji = {"BUY": "📈", "SELL": "📉", "HOLD": "⏸️"}.get(s.get("signal", ""), "")
        rows += f"""
        <tr>
          <td><b>{s.get('asset','')}</b></td>
          <td style="color:{color};font-weight:bold">{emoji} {s.get('signal','')}</td>
          <td>{s.get('probability','')}%</td>
          <td>{s.get('confidence_score','')}</td>
          <td>{s.get('risk_level','')}</td>
          <td><b>{s.get('current_price','—')}</b></td>
          <td style="color:#ef4444">{s.get('stop_loss_price','')}</td>
          <td style="color:#22c55e">{s.get('take_profit_price','')}</td>
          <td>{s.get('regime','')}</td>
        </tr>"""

    last_run = _scheduler_runs[-1] if _scheduler_runs else "Jamais"

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="refresh" content="60">
  <title>AI Trading System</title>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{font-family:monospace;background:#0f172a;color:#e2e8f0;padding:24px}}
    h1{{color:#38bdf8;margin-bottom:4px}}
    h2{{color:#94a3b8;font-size:14px;margin:20px 0 10px}}
    .info{{color:#94a3b8;font-size:13px;margin:4px 0}}
    table{{width:100%;border-collapse:collapse;margin-top:8px}}
    th{{background:#1e293b;padding:10px;text-align:left;color:#94a3b8;font-size:12px}}
    td{{padding:10px;border-bottom:1px solid #1e293b;font-size:13px}}
    .card{{background:#1e293b;border-radius:8px;padding:16px;margin:16px 0}}
    .form-row{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:12px}}
    .form-group{{display:flex;flex-direction:column;gap:4px;flex:1;min-width:140px}}
    label{{font-size:12px;color:#94a3b8}}
    select,input{{background:#0f172a;color:#e2e8f0;border:1px solid #334155;
      border-radius:4px;padding:8px;font-family:monospace;font-size:13px}}
    button{{background:#38bdf8;color:#0f172a;border:none;padding:10px 24px;
      border-radius:4px;font-weight:bold;cursor:pointer;font-size:14px}}
    button:hover{{background:#7dd3fc}}
    #result{{margin-top:16px;white-space:pre-wrap;font-size:12px;color:#a3e635;
      background:#0f172a;padding:12px;border-radius:4px;max-height:400px;overflow:auto}}
    a{{color:#38bdf8;text-decoration:none}}
  </style>
</head>
<body>
  <h1>🤖 AI Trading System</h1>
  <p class="info">⏰ Dernier calcul : <b>{last_run}</b> — 🔄 Refresh auto toutes les 60s</p>
  <p class="info">📊 Symboles suivis : <b>{', '.join(SYMBOLS)}</b></p>

  <h2>📡 SIGNAUX EN TEMPS RÉEL</h2>
  <table>
    <thead>
      <tr>
        <th>Actif</th><th>Signal</th><th>Probabilité</th><th>Confiance</th>
        <th>Risque</th><th>Prix actuel</th><th>Stop-loss</th><th>Take-profit</th><th>Régime</th>
      </tr>
    </thead>
    <tbody>{rows if rows else '<tr><td colspan="9" style="color:#94a3b8;padding:20px">Chargement en cours...</td></tr>'}</tbody>
  </table>

  <h2>🎯 ANALYSE PERSONNALISÉE</h2>
  <div class="card">
    <p class="info" style="margin-bottom:12px">Entre ton profil pour obtenir un plan adapté à ta situation.</p>
    <div class="form-row">
      <div class="form-group">
        <label>💰 Budget (USD)</label>
        <input type="number" id="budget" value="1000" min="10" step="100">
      </div>
      <div class="form-group">
        <label>📅 Stratégie</label>
        <select id="strategy">
          <option value="daily">Quotidien (day trading)</option>
          <option value="weekly" selected>Hebdomadaire</option>
          <option value="monthly">Mensuel</option>
          <option value="yearly">Long terme (1 an+)</option>
        </select>
      </div>
      <div class="form-group">
        <label>🎯 Objectif</label>
        <select id="objective">
          <option value="capital_preservation">Préserver mon capital</option>
          <option value="moderate_growth" selected>Croissance modérée</option>
          <option value="aggressive_growth">Croissance agressive</option>
          <option value="income">Revenus réguliers</option>
        </select>
      </div>
      <div class="form-group">
        <label>⚡ Tolérance au risque</label>
        <select id="risk">
          <option value="low">Faible — je dors bien</option>
          <option value="medium" selected>Moyenne</option>
          <option value="high">Élevée — je gère</option>
        </select>
      </div>
    </div>
    <button onclick="analyze()">🔍 Analyser mon profil</button>
    <div id="result" style="display:none"></div>
  </div>

  <p class="info">
    <a href="/signals?refresh=true">🔄 Forcer recalcul</a> &nbsp;|&nbsp;
    <a href="/backtest?symbol=AAPL&days=365">📊 Backtest AAPL</a> &nbsp;|&nbsp;
    <a href="/docs">📖 Documentation API</a>
  </p>

  <script>
  async function analyze() {{
    const btn = document.querySelector('button');
    const result = document.getElementById('result');
    btn.textContent = '⏳ Analyse en cours...';
    btn.disabled = true;
    result.style.display = 'block';
    result.textContent = 'Connexion aux marchés...';
    try {{
      const resp = await fetch('/analyze', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{
          budget: parseFloat(document.getElementById('budget').value),
          strategy: document.getElementById('strategy').value,
          objective: document.getElementById('objective').value,
          risk_tolerance: document.getElementById('risk').value,
          currency: 'USD'
        }})
      }});
      const data = await resp.json();
      // Affiche le résumé de façon lisible
      let txt = '=== MON PLAN DE TRADING ===\\n\\n';
      const r = data['résumé'];
      for (const [k, v] of Object.entries(r)) txt += `  ${{k.padEnd(35)}} ${{v}}\\n`;
      txt += '\\n=== CONSEILS PERSONNALISÉS ===\\n';
      (data['conseils'] || []).forEach(c => txt += `\\n  ✅ ${{c}}`);
      txt += '\\n\\n=== SIGNAUX ACTIONNABLES MAINTENANT ===\\n';
      (data['signaux_actionnables'] || []).forEach(s => {{
        txt += `\\n  ${{s.asset.padEnd(10)}} ${{s.signal.padEnd(5)}} | Prix: ${{s.current_price}} | Position: ${{s.position_usd}}$ | Stop: ${{s.stop_loss_usd}}$ | Target: ${{s.take_profit_usd}}$`;
      }});
      result.textContent = txt;
    }} catch(e) {{
      result.textContent = 'Erreur : ' + e.message;
    }}
    btn.textContent = '🔍 Analyser mon profil';
    btn.disabled = false;
  }}
  </script>
</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# Logique interne
# ---------------------------------------------------------------------------

def _generate_signal(symbol: str, lookback_days: int = 20) -> dict:
    import yfinance as yf
    import numpy as np

    period = f"{min(lookback_days * 3, 180)}d"
    df = yf.download(symbol, period=period, interval="1d", progress=False)
    if df.empty:
        raise ValueError(f"Pas de données pour {symbol}")

    close = df["Close"].squeeze()
    returns = close.pct_change().dropna()
    momentum = float(returns.tail(lookback_days).mean())
    vol = float(returns.tail(lookback_days).std())
    proba = float(max(0, min(100, 50 + momentum * 2000)))

    signal = "BUY" if momentum > 0.001 else ("SELL" if momentum < -0.001 else "HOLD")
    price = float(close.iloc[-1])

    return {
        "asset": symbol,
        "signal": signal,
        "probability": round(proba, 1),
        "confidence_score": round(max(0, 1 - vol * 10), 2),
        "risk_level": "HIGH" if vol > 0.03 else ("LOW" if vol < 0.01 else "MEDIUM"),
        "predicted_return": round(momentum, 6),
        "predicted_volatility": round(vol, 6),
        "stop_loss_price": round(price * 0.95, 2),
        "take_profit_price": round(price * (1 + abs(momentum) * 2), 2),
        "current_price": round(price, 2),
        "regime": "trending" if abs(momentum) > 0.001 else "ranging",
        "timestamp": _now(),
    }


def _demo_signal(symbol: str) -> dict:
    import random
    rng = random.Random(sum(ord(c) for c in symbol) + datetime.now().hour)
    r = rng.uniform(-0.02, 0.02)
    v = rng.uniform(0.008, 0.035)
    p = rng.uniform(100, 300)
    return {
        "asset": symbol, "signal": "BUY" if r > 0.005 else ("SELL" if r < -0.005 else "HOLD"),
        "probability": round(max(0, min(100, 50 + r * 1000)), 1),
        "confidence_score": round(max(0, 1 - v * 10), 2), "risk_level": "MEDIUM",
        "predicted_return": round(r, 6), "predicted_volatility": round(v, 6),
        "stop_loss_price": round(p * 0.95, 2), "take_profit_price": round(p * 1.06, 2),
        "current_price": round(p, 2), "regime": "demo", "timestamp": _now(),
    }


def _generate_advice(profile: UserProfile, strat: dict, obj: dict) -> List[str]:
    """Conseils personnalisés selon le profil."""
    advice = []

    if profile.budget < 500:
        advice.append("Petit budget : privilégie les cryptos (pas de minimum d'achat) et les ETFs fractionnés.")
    elif profile.budget < 2000:
        advice.append("Budget moyen : commence avec 2-3 positions maximum pour bien les suivre.")
    else:
        advice.append(f"Bon budget : tu peux diversifier sur {strat['max_positions']} positions simultanées.")

    if profile.strategy == "daily":
        advice.append("Day trading : suis les marchés en temps réel. Très chronophage et risqué pour débuter.")
        advice.append("Commence avec le paper trading (argent fictif) pendant au moins 3 mois avant l'argent réel.")
    elif profile.strategy == "weekly":
        advice.append("Trading hebdomadaire : vérifie tes positions 2-3 fois par semaine, pas besoin d'être rivé à l'écran.")
    elif profile.strategy == "monthly":
        advice.append("Trading mensuel : parfait si tu as un emploi. 30 min par semaine suffisent.")
    elif profile.strategy == "yearly":
        advice.append("Long terme : la stratégie la plus sûre historiquement. La patience est ta meilleure alliée.")

    if profile.objective == "capital_preservation":
        advice.append("Préservation : mets 50% en GLD (or) et SPY (ETF marché global) pour la stabilité.")
    elif profile.objective == "aggressive_growth":
        advice.append("Croissance agressive : accepte des baisses de -20% à -30% sans paniquer et vendre.")

    if profile.risk_tolerance == "low":
        advice.append("Risque faible : ne mets jamais plus de 5% de ton budget sur une seule position.")
    elif profile.risk_tolerance == "high":
        advice.append("Risque élevé : discipline absolue sur les stop-loss. Une perte sans limite peut tout effacer.")

    advice.append(f"Règle d'or : ne jamais investir plus que ce que tu peux te permettre de perdre entièrement.")
    return advice


def _run_backtest(symbol: str, days: int) -> dict:
    import sys
    import yfinance as yf
    sys.path.insert(0, ".")
    from src.backtesting.engine import BacktestEngine

    df = yf.download(symbol, period=f"{days}d" if days <= 730 else "5y", interval="1d", progress=False)
    if df.empty:
        raise ValueError("Pas de données")
    close = df[["Close"]].copy()
    close.columns = [symbol]
    signals = close.pct_change().shift(1).clip(-1, 1)
    signals.columns = [symbol]
    engine = BacktestEngine(initial_capital=100_000)
    results = engine.run(signals, close)
    report = engine.generate_report(results)
    report.update({"symbol": symbol, "days_tested": len(close)})
    return report


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
