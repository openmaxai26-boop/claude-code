"""
Institutional-Grade AI Trading System
Entry point for full pipeline execution.

Usage
-----
  python main.py --mode backtest --symbols AAPL MSFT GLD --timeframe 1d \
                 --start-date 2020-01-01 --end-date 2023-12-31

  python main.py --mode live --symbols AAPL BTC-USD GLD

  python main.py --mode train --symbols AAPL MSFT --start-date 2018-01-01 \
                 --end-date 2022-12-31
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("TradingSystem")


class TradingSystem:
    """
    Orchestrates the full AI trading pipeline.

    Components
    ----------
    - Data collection  : MarketDataCollector, CryptoDataCollector,
                         MacroDataCollector, AlternativeDataCollector
    - Feature engineering : TechnicalFeatureEngine, StatisticalFeatureEngine,
                            MicrostructureFeatureEngine
    - Regime detection : HMMRegimeDetector, RegimeClassifier
    - Prediction models: LSTMPredictor, TransformerPredictor, CNNPredictor,
                         EnsembleModel
    - RL agent         : TradingEnvironment, PPOTradingAgent
    - Risk management  : RiskManager
    - Portfolio        : PortfolioOptimizer
    - Backtesting      : BacktestEngine, PerformanceMetrics
    - Continuous learning : ContinuousLearningSystem
    - Output           : SignalGenerator
    """

    def __init__(self, config_path: str = ".env") -> None:
        self.config_path = config_path
        self.config: Any = None
        self.components: Dict[str, Any] = {}
        self._ready = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Initialise all components. Must be called before run_*."""
        logger.info("Initialising trading system …")

        # --- Config ---
        try:
            from config.settings import Settings
            self.config = Settings(_env_file=self.config_path)
        except Exception as exc:
            logger.warning("Could not load Settings (%s) — using defaults.", exc)
            self.config = _DefaultConfig()

        # --- Data collectors ---
        try:
            from src.data.collectors.market_data import MarketDataCollector
            self.components["market"] = MarketDataCollector(self.config)
        except Exception as exc:
            logger.warning("MarketDataCollector unavailable: %s", exc)

        try:
            from src.data.collectors.crypto_data import CryptoDataCollector
            self.components["crypto"] = CryptoDataCollector(self.config)
        except Exception as exc:
            logger.warning("CryptoDataCollector unavailable: %s", exc)

        try:
            from src.data.collectors.macro_data import MacroDataCollector
            self.components["macro"] = MacroDataCollector(self.config)
        except Exception as exc:
            logger.warning("MacroDataCollector unavailable: %s", exc)

        # --- Feature engines ---
        try:
            from src.features.technical import TechnicalFeatureEngine
            self.components["tech_features"] = TechnicalFeatureEngine()
        except Exception as exc:
            logger.warning("TechnicalFeatureEngine unavailable: %s", exc)

        try:
            from src.features.statistical import StatisticalFeatureEngine
            self.components["stat_features"] = StatisticalFeatureEngine()
        except Exception as exc:
            logger.warning("StatisticalFeatureEngine unavailable: %s", exc)

        try:
            from src.features.microstructure import MicrostructureFeatureEngine
            self.components["micro_features"] = MicrostructureFeatureEngine()
        except Exception as exc:
            logger.warning("MicrostructureFeatureEngine unavailable: %s", exc)

        # --- Regime detection ---
        try:
            from src.models.regime.hmm import HMMRegimeDetector
            self.components["hmm"] = HMMRegimeDetector(n_regimes=4)
        except Exception as exc:
            logger.warning("HMMRegimeDetector unavailable: %s", exc)

        try:
            from src.models.regime.classifier import RegimeClassifier
            self.components["regime_clf"] = RegimeClassifier(self.config)
        except Exception as exc:
            logger.warning("RegimeClassifier unavailable: %s", exc)

        # --- Prediction models ---
        try:
            from src.models.prediction.ensemble import EnsembleModel
            self.components["ensemble"] = EnsembleModel(self.config)
        except Exception as exc:
            logger.warning("EnsembleModel unavailable: %s", exc)

        # --- Risk manager ---
        try:
            from src.risk.manager import RiskManager
            self.components["risk"] = RiskManager(self.config)
        except Exception as exc:
            logger.warning("RiskManager unavailable: %s", exc)

        # --- Portfolio optimizer ---
        try:
            from src.portfolio.optimizer import PortfolioOptimizer
            self.components["portfolio"] = PortfolioOptimizer(self.config)
        except Exception as exc:
            logger.warning("PortfolioOptimizer unavailable: %s", exc)

        # --- Backtest engine ---
        try:
            from src.backtesting.engine import BacktestEngine
            self.components["backtest"] = BacktestEngine(
                initial_capital=getattr(self.config, "initial_capital", 1_000_000),
                transaction_cost_bps=getattr(self.config, "transaction_cost_bps", 5),
                slippage_bps=getattr(self.config, "slippage_bps", 3),
            )
        except Exception as exc:
            logger.warning("BacktestEngine unavailable: %s", exc)

        # --- Continuous learning ---
        try:
            from src.learning.continuous import ContinuousLearningSystem
            models_dict = {
                k: v for k, v in self.components.items()
                if k in ("ensemble", "hmm", "regime_clf")
            }
            self.components["learning"] = ContinuousLearningSystem(
                models_dict, data_store=None, config=self.config
            )
        except Exception as exc:
            logger.warning("ContinuousLearningSystem unavailable: %s", exc)

        # --- Signal generator ---
        try:
            from src.output.signals import SignalGenerator
            self.components["signals"] = SignalGenerator(
                ensemble=self.components.get("ensemble"),
                risk_manager=self.components.get("risk"),
                config=self.config,
            )
        except Exception as exc:
            logger.warning("SignalGenerator unavailable: %s", exc)

        self._ready = True
        logger.info("System initialised with %d components.", len(self.components))

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def run_data_pipeline(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Collect and merge market data for all symbols."""
        self._assert_ready()
        collector = self.components.get("market")
        if collector is None:
            logger.error("Market data collector not available.")
            return pd.DataFrame()

        frames = []
        for sym in symbols:
            try:
                df = collector.fetch(sym, start=start_date, end=end_date, interval=timeframe)
                df.columns = pd.MultiIndex.from_product([[sym], df.columns])
                frames.append(df)
                logger.info("Fetched %d rows for %s", len(df), sym)
            except Exception as exc:
                logger.warning("Failed to fetch %s: %s", sym, exc)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=1).sort_index()

    def run_training(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train all models on historical data. Returns trained models dict."""
        self._assert_ready()
        trained: Dict[str, Any] = {}

        ensemble = self.components.get("ensemble")
        if ensemble is None:
            logger.warning("No ensemble model to train.")
            return trained

        try:
            features = self._build_features(data)
            close = self._extract_close(data)
            returns = close.pct_change().dropna()

            # Train regime models first
            hmm = self.components.get("hmm")
            if hmm is not None:
                hmm.fit(returns)
                trained["hmm"] = hmm
                logger.info("HMM trained.")

            # Train ensemble
            X = features.dropna()
            if len(X) > 100:
                y = close.pct_change().shift(-1).dropna().reindex(X.index).dropna()
                X = X.reindex(y.index).dropna()
                ensemble.fit(X, y)
                trained["ensemble"] = ensemble
                logger.info("Ensemble trained on %d samples.", len(X))

        except Exception as exc:
            logger.error("Training failed: %s", exc)

        return trained

    def run_backtest(
        self,
        data: pd.DataFrame,
        models: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """Run full backtest on historical data."""
        self._assert_ready()
        engine = self.components.get("backtest")
        if engine is None:
            logger.error("BacktestEngine not available.")
            return None

        try:
            close = self._extract_close(data)
            features = self._build_features(data)
            ensemble = (models or {}).get("ensemble") or self.components.get("ensemble")

            # Generate signals bar by bar
            signals = pd.DataFrame(0.0, index=close.index, columns=close.columns)
            if ensemble is not None:
                for i in range(1, len(close)):
                    try:
                        feat_row = features.iloc[:i]
                        result = ensemble.predict(feat_row.iloc[[-1]])
                        pred = float(np.squeeze(result)) if not isinstance(result, dict) else float(result.get("predicted_return", 0))
                        signal_val = np.clip(pred * 20, -1, 1)  # scale to [-1, 1]
                        signals.iloc[i] = signal_val
                    except Exception:
                        pass

            results = engine.run(signals, close)
            return results

        except Exception as exc:
            logger.error("Backtest failed: %s", exc)
            return None

    def run_live_signals(
        self,
        symbols: List[str],
        timeframe: str,
    ) -> List[Any]:
        """Fetch latest data and generate live signals for all symbols."""
        self._assert_ready()
        sig_gen = self.components.get("signals")
        if sig_gen is None:
            logger.error("SignalGenerator not available.")
            return []

        signals_out = []
        for sym in symbols:
            try:
                collector = self.components.get("market")
                if collector is None:
                    continue
                df = collector.fetch(sym, period="30d", interval=timeframe)
                features = self._build_features_single(df)
                regime = self._detect_regime_single(df)
                price = float(df["Close"].iloc[-1]) if "Close" in df.columns else 1.0

                sig = sig_gen.generate_signal(sym, timeframe, features, regime, price)
                signals_out.append(sig)
            except Exception as exc:
                logger.warning("Live signal failed for %s: %s", sym, exc)

        return signals_out

    def run_full_pipeline(
        self,
        mode: str = "backtest",
        symbols: Optional[List[str]] = None,
        timeframe: str = "1d",
        start_date: str = "2020-01-01",
        end_date: str = "2023-12-31",
    ) -> Any:
        """
        Execute the complete pipeline.

        mode: 'backtest' | 'live' | 'train'
        """
        self._assert_ready()
        symbols = symbols or ["AAPL", "MSFT", "GOOGL", "GLD", "BTC-USD"]

        if mode == "live":
            logger.info("=== LIVE MODE ===")
            signals = self.run_live_signals(symbols, timeframe)
            self._print_signals(signals)
            return signals

        logger.info("=== %s MODE | %s | %s → %s ===", mode.upper(), symbols, start_date, end_date)
        data = self.run_data_pipeline(symbols, timeframe, start_date, end_date)

        if data.empty:
            logger.error("No data collected — aborting.")
            return None

        trained = self.run_training(data)

        if mode == "train":
            logger.info("Training complete. Models: %s", list(trained.keys()))
            return trained

        # Backtest
        results = self.run_backtest(data, trained)
        if results is not None:
            engine = self.components.get("backtest")
            report = engine.generate_report(results) if engine else {}
            self._print_report(report)
            return results

        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assert_ready(self) -> None:
        if not self._ready:
            raise RuntimeError("Call setup() before running the pipeline.")

    def _build_features(self, data: pd.DataFrame) -> pd.DataFrame:
        close = self._extract_close(data)
        tech = self.components.get("tech_features")
        stat = self.components.get("stat_features")
        frames = []
        if tech is not None:
            try:
                frames.append(tech.compute_all_features(close.rename(columns=lambda c: f"{c}_close")))
            except Exception as exc:
                logger.debug("TechFeatures failed: %s", exc)
        if stat is not None:
            try:
                rets = close.pct_change().dropna()
                frames.append(stat.compute_all_features(rets))
            except Exception as exc:
                logger.debug("StatFeatures failed: %s", exc)
        if not frames:
            return close.pct_change().fillna(0)
        return pd.concat(frames, axis=1).ffill().bfill()

    def _build_features_single(self, df: pd.DataFrame) -> pd.DataFrame:
        tech = self.components.get("tech_features")
        if tech is not None:
            try:
                return tech.compute_all_features(df)
            except Exception:
                pass
        return df.select_dtypes(include=[np.number]).ffill().bfill()

    def _detect_regime_single(self, df: pd.DataFrame) -> str:
        hmm = self.components.get("hmm")
        if hmm is None:
            return "unknown"
        try:
            close = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
            returns = close.pct_change().dropna().values.reshape(-1, 1)
            label = hmm.predict(returns)
            return str(label[-1])
        except Exception:
            return "unknown"

    @staticmethod
    def _extract_close(data: pd.DataFrame) -> pd.DataFrame:
        if isinstance(data.columns, pd.MultiIndex):
            try:
                return data.xs("Close", axis=1, level=1)
            except KeyError:
                pass
        close_cols = [c for c in data.columns if "close" in str(c).lower()]
        return data[close_cols] if close_cols else data

    @staticmethod
    def _print_signals(signals: list) -> None:
        ts = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"\n=== TRADING SIGNALS — {ts} ===\n")
        for s in signals:
            print(
                f"  {s.asset:<8} | {s.timeframe:<4} | {s.signal:<4} | "
                f"{s.probability:5.1f}% | Confidence: {s.confidence_score:.2f} | "
                f"Risk: {s.risk_level}"
            )
        print()

    @staticmethod
    def _print_report(report: dict) -> None:
        print("\n=== BACKTEST PERFORMANCE REPORT ===\n")
        for k, v in report.items():
            print(f"  {k:<32}: {v}")
        print()


# ---------------------------------------------------------------------------
# Default config (when settings.py / .env not present)
# ---------------------------------------------------------------------------

class _DefaultConfig:
    # Data
    symbols = ["AAPL", "MSFT", "GOOGL", "GLD", "BTC-USD"]
    timeframes = ["1d"]
    # Risk
    max_drawdown = 0.15
    max_position_size = 0.10
    max_portfolio_vol = 0.20
    stop_loss_pct = 0.05
    correlation_limit = 0.70
    # Backtesting
    initial_capital = 1_000_000
    transaction_cost_bps = 5
    slippage_bps = 3
    # Model
    stop_loss_pct = 0.05
    take_profit_ratio = 2.0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Institutional-Grade AI Trading System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["backtest", "live", "train"],
        default="backtest",
        help="Execution mode.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["AAPL", "MSFT", "GLD"],
        help="List of ticker symbols.",
    )
    parser.add_argument(
        "--timeframe",
        default="1d",
        help="Data timeframe (e.g. 1d, 1h, 15m).",
    )
    parser.add_argument(
        "--start-date",
        default="2020-01-01",
        help="Backtest/training start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        default="2023-12-31",
        help="Backtest/training end date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--config",
        default=".env",
        help="Path to .env configuration file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    system = TradingSystem(config_path=args.config)
    system.setup()

    results = system.run_full_pipeline(
        mode=args.mode,
        symbols=args.symbols,
        timeframe=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    sys.exit(0)
