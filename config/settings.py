"""
Pydantic-based settings for the institutional-grade AI trading system.
All values can be overridden via environment variables or a .env file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Sub-section dataclasses (pure-Python, serialisable)
# ---------------------------------------------------------------------------


@dataclass
class DataConfig:
    """Market & alternative-data collection settings."""

    # Equity symbols
    equity_symbols: List[str] = field(default_factory=lambda: [
        "SPY", "QQQ", "IWM", "GLD", "SLV", "USO",
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "GS",
    ])

    # Crypto symbols (CCXT unified format)
    crypto_symbols: List[str] = field(default_factory=lambda: [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    ])

    # Commodity proxies (ETFs)
    commodity_symbols: List[str] = field(default_factory=lambda: [
        "GLD", "SLV", "USO", "DBA", "DBB",
    ])

    # OHLCV timeframes accepted by yfinance / ccxt
    timeframes: List[str] = field(default_factory=lambda: [
        "1m", "5m", "15m", "1h", "4h", "1d",
    ])

    # Default lookback window in bars for feature engineering
    lookback_periods: List[int] = field(default_factory=lambda: [
        5, 10, 21, 63, 126, 252,
    ])

    # Data sources priority order
    data_sources: List[str] = field(default_factory=lambda: [
        "yfinance", "ccxt", "fred", "alternative",
    ])

    # Number of calendar days of history to fetch on initial load
    history_days: int = 1825  # ~5 years

    # Maximum allowed data gap before a bar is flagged (minutes)
    max_gap_minutes: int = 60

    # Z-score threshold for outlier detection
    outlier_zscore_threshold: float = 5.0

    # Forward-fill limit for macro data (trading days)
    macro_ffill_days: int = 5


@dataclass
class ModelConfig:
    """Neural-network architecture hyper-parameters."""

    # --- LSTM ---
    lstm_hidden_size: int = 256
    lstm_num_layers: int = 3
    lstm_dropout: float = 0.30
    lstm_bidirectional: bool = False

    # --- Transformer ---
    transformer_d_model: int = 256
    transformer_nhead: int = 8
    transformer_num_encoder_layers: int = 4
    transformer_dim_feedforward: int = 1024
    transformer_dropout: float = 0.10
    transformer_max_seq_len: int = 512

    # --- 1-D CNN ---
    cnn_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    cnn_kernel_sizes: List[int] = field(default_factory=lambda: [3, 5, 7])
    cnn_dropout: float = 0.20

    # --- Ensemble weights (must sum to 1.0) ---
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: {
        "lstm": 0.35,
        "transformer": 0.40,
        "cnn": 0.25,
    })

    # Feature dimensionality fed to all models
    input_features: int = 128
    output_classes: int = 3          # BUY / HOLD / SELL
    regression_output_size: int = 1  # price return

    # Activation
    activation: str = "gelu"

    # Normalisation inside models
    use_layer_norm: bool = True
    use_batch_norm: bool = False


@dataclass
class RiskConfig:
    """Portfolio and position-level risk constraints."""

    # Maximum peak-to-trough drawdown before forced de-risking
    max_drawdown_pct: float = 0.15

    # Maximum single-position weight in portfolio NAV
    max_position_size: float = 0.10

    # Maximum annualised portfolio volatility target
    max_portfolio_vol: float = 0.20

    # Hard stop-loss per position (from entry price)
    stop_loss_pct: float = 0.05

    # Trailing stop (from high-water-mark since entry)
    trailing_stop_pct: float = 0.07

    # Pairwise correlation limit before positions are reduced
    correlation_limit: float = 0.70

    # VaR confidence level
    var_confidence: float = 0.95

    # CVaR / Expected Shortfall confidence level
    cvar_confidence: float = 0.99

    # Maximum gross leverage
    max_gross_leverage: float = 2.00

    # Maximum net leverage
    max_net_leverage: float = 1.00

    # Sector concentration limit
    max_sector_weight: float = 0.30

    # Minimum cash buffer as fraction of NAV
    min_cash_buffer: float = 0.05

    # Kelly fraction scaling (fractional Kelly)
    kelly_fraction: float = 0.25


@dataclass
class RLConfig:
    """Reinforcement-learning (PPO via stable-baselines3) settings."""

    # PPO hyper-parameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    clip_range: float = 0.20
    n_steps: int = 2048
    batch_size: int = 256
    n_epochs: int = 10

    # Entropy coefficient for exploration
    ent_coef: float = 0.01

    # Value-function coefficient
    vf_coef: float = 0.50

    # Gradient clip norm
    max_grad_norm: float = 0.50

    # GAE lambda for advantage estimation
    gae_lambda: float = 0.95

    # Total environment time-steps for training
    total_timesteps: int = 5_000_000

    # Number of parallel environments
    n_envs: int = 4

    # Reward scaling factor
    reward_scale: float = 1.0

    # Observation normalisation
    normalize_observations: bool = True

    # Reward normalisation
    normalize_rewards: bool = True

    # Evaluation frequency (in steps)
    eval_freq: int = 10_000

    # Number of evaluation episodes
    n_eval_episodes: int = 10


@dataclass
class TrainingConfig:
    """Supervised-learning training loop settings."""

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    # test_ratio implicitly = 1 - train_ratio - val_ratio = 0.15

    epochs: int = 200
    batch_size: int = 512

    # Number of epochs without improvement before stopping
    early_stopping_patience: int = 20

    # Factor for learning-rate scheduler (ReduceLROnPlateau)
    lr_scheduler_factor: float = 0.50
    lr_scheduler_patience: int = 10
    lr_scheduler_min_lr: float = 1e-6

    # Initial learning rate for supervised models
    learning_rate: float = 1e-3

    # Weight decay (L2 regularisation)
    weight_decay: float = 1e-4

    # Gradient clipping max norm
    grad_clip_norm: float = 1.0

    # Mixed-precision training
    use_amp: bool = True

    # Sequence lookback length (bars)
    lookback: int = 60

    # Prediction horizon (bars ahead)
    horizon: int = 1

    # Number of DataLoader worker processes
    num_workers: int = 4

    # Random seed for reproducibility
    seed: int = 42

    # Optuna hyper-parameter search trials
    optuna_trials: int = 100


@dataclass
class BacktestingConfig:
    """Simulation and back-test engine settings."""

    initial_capital: float = 1_000_000.0

    # Transaction costs in basis points (1 bps = 0.01%)
    transaction_cost_bps: float = 5.0

    # Market-impact / slippage in basis points
    slippage_bps: float = 3.0

    # Minimum daily dollar volume for a position to be considered liquid
    min_liquidity_usd: float = 1_000_000.0

    # Borrow cost for short positions (annualised bps)
    short_borrow_cost_bps: float = 50.0

    # Margin interest rate (annualised %)
    margin_interest_rate: float = 0.05

    # Rebalance frequency: 'daily', 'weekly', 'monthly'
    rebalance_frequency: str = "daily"

    # Mark-to-market price: 'open', 'close', 'vwap'
    fill_price: str = "close"

    # Whether to account for dividends / corporate actions
    adjust_for_dividends: bool = True

    # Risk-free rate proxy (annualised) for Sharpe calculation
    risk_free_rate: float = 0.05

    # Benchmark symbol for alpha/beta calculation
    benchmark_symbol: str = "SPY"

    # Enable walk-forward optimisation
    walk_forward: bool = True

    # Walk-forward window size in trading days
    walk_forward_train_days: int = 504   # ~2 years
    walk_forward_test_days: int = 63    # ~1 quarter


@dataclass
class PathsConfig:
    """Filesystem layout for data, models, logs, and results."""

    # Base project root (resolved at runtime)
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def model_dir(self) -> Path:
        return self.project_root / "models"

    @property
    def log_dir(self) -> Path:
        return self.project_root / "logs"

    @property
    def results_dir(self) -> Path:
        return self.project_root / "results"

    @property
    def cache_dir(self) -> Path:
        return self.project_root / ".cache"

    def make_dirs(self) -> None:
        """Create all directories if they do not already exist."""
        for attr in ("data_dir", "raw_data_dir", "processed_data_dir",
                     "model_dir", "log_dir", "results_dir", "cache_dir"):
            getattr(self, attr).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Top-level Settings (pydantic-settings loads from env / .env file)
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """
    Unified settings class.  Values are loaded with this priority:
      1. Environment variables (prefixed per section, e.g. DATA__HISTORY_DAYS=3650)
      2. .env file in the project root
      3. Defaults defined in the dataclasses above
    """

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parents[1] / ".env"),
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- Application meta ----
    app_name: str = Field(default="AI-Trading-System", description="Application name")
    environment: str = Field(default="development", description="development | staging | production")
    debug: bool = Field(default=False, description="Enable verbose debug output")
    log_level: str = Field(default="INFO", description="Loguru log level")

    # ---- API keys (loaded from env / .env) ----
    fred_api_key: Optional[str] = Field(default=None, description="FRED API key (optional, raises rate limit without it)")
    alpaca_api_key: Optional[str] = Field(default=None, description="Alpaca Markets API key")
    alpaca_secret_key: Optional[str] = Field(default=None, description="Alpaca Markets secret")
    polygon_api_key: Optional[str] = Field(default=None, description="Polygon.io API key")
    glassnode_api_key: Optional[str] = Field(default=None, description="Glassnode on-chain data API key")
    newsapi_key: Optional[str] = Field(default=None, description="NewsAPI.org key for news sentiment")
    binance_api_key: Optional[str] = Field(default=None, description="Binance API key")
    binance_secret_key: Optional[str] = Field(default=None, description="Binance secret key")

    # ---- Database ----
    database_url: str = Field(
        default="sqlite:///trading.db",
        description="SQLAlchemy connection string; default SQLite, set to postgresql://... for prod",
    )

    # ---- Redis ----
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_password: Optional[str] = Field(default=None, description="Redis password")
    redis_db: int = Field(default=0, description="Redis database index")

    # ---- Sub-section configs (instantiated with defaults; override via env) ----
    # NOTE: Pydantic-settings v2 does not natively deserialise dataclasses from
    #       nested env vars, so we expose individual scalar overrides and build
    #       the dataclasses via properties.  Complex nested overrides should be
    #       done programmatically after calling get_settings().

    # Data overrides
    data_history_days: int = Field(default=1825)
    data_outlier_zscore_threshold: float = Field(default=5.0)
    data_macro_ffill_days: int = Field(default=5)

    # Model overrides
    model_lstm_hidden_size: int = Field(default=256)
    model_lstm_num_layers: int = Field(default=3)
    model_lstm_dropout: float = Field(default=0.30)
    model_transformer_d_model: int = Field(default=256)
    model_transformer_nhead: int = Field(default=8)

    # Risk overrides
    risk_max_drawdown_pct: float = Field(default=0.15)
    risk_max_position_size: float = Field(default=0.10)
    risk_max_portfolio_vol: float = Field(default=0.20)
    risk_stop_loss_pct: float = Field(default=0.05)
    risk_correlation_limit: float = Field(default=0.70)

    # RL overrides
    rl_learning_rate: float = Field(default=3e-4)
    rl_gamma: float = Field(default=0.99)
    rl_clip_range: float = Field(default=0.20)
    rl_n_steps: int = Field(default=2048)
    rl_batch_size: int = Field(default=256)

    # Training overrides
    train_ratio: float = Field(default=0.70)
    val_ratio: float = Field(default=0.15)
    train_epochs: int = Field(default=200)
    train_batch_size: int = Field(default=512)
    train_early_stopping_patience: int = Field(default=20)

    # Backtesting overrides
    backtest_initial_capital: float = Field(default=1_000_000.0)
    backtest_transaction_cost_bps: float = Field(default=5.0)
    backtest_slippage_bps: float = Field(default=3.0)
    backtest_min_liquidity_usd: float = Field(default=1_000_000.0)

    # -------------------------------------------------------------------------
    # Validators
    # -------------------------------------------------------------------------

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        allowed = {"development", "staging", "production"}
        if v.lower() not in allowed:
            raise ValueError(f"environment must be one of {allowed}, got '{v}'")
        return v.lower()

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return v.upper()

    @field_validator("train_ratio")
    @classmethod
    def validate_train_ratio(cls, v: float) -> float:
        if not (0.0 < v < 1.0):
            raise ValueError("train_ratio must be in (0, 1)")
        return v

    @field_validator("val_ratio")
    @classmethod
    def validate_val_ratio(cls, v: float) -> float:
        if not (0.0 < v < 1.0):
            raise ValueError("val_ratio must be in (0, 1)")
        return v

    # -------------------------------------------------------------------------
    # Convenience properties that return fully-populated dataclass instances
    # -------------------------------------------------------------------------

    @property
    def data(self) -> DataConfig:
        cfg = DataConfig()
        cfg.history_days = self.data_history_days
        cfg.outlier_zscore_threshold = self.data_outlier_zscore_threshold
        cfg.macro_ffill_days = self.data_macro_ffill_days
        return cfg

    @property
    def model(self) -> ModelConfig:
        cfg = ModelConfig()
        cfg.lstm_hidden_size = self.model_lstm_hidden_size
        cfg.lstm_num_layers = self.model_lstm_num_layers
        cfg.lstm_dropout = self.model_lstm_dropout
        cfg.transformer_d_model = self.model_transformer_d_model
        cfg.transformer_nhead = self.model_transformer_nhead
        return cfg

    @property
    def risk(self) -> RiskConfig:
        cfg = RiskConfig()
        cfg.max_drawdown_pct = self.risk_max_drawdown_pct
        cfg.max_position_size = self.risk_max_position_size
        cfg.max_portfolio_vol = self.risk_max_portfolio_vol
        cfg.stop_loss_pct = self.risk_stop_loss_pct
        cfg.correlation_limit = self.risk_correlation_limit
        return cfg

    @property
    def rl(self) -> RLConfig:
        cfg = RLConfig()
        cfg.learning_rate = self.rl_learning_rate
        cfg.gamma = self.rl_gamma
        cfg.clip_range = self.rl_clip_range
        cfg.n_steps = self.rl_n_steps
        cfg.batch_size = self.rl_batch_size
        return cfg

    @property
    def training(self) -> TrainingConfig:
        cfg = TrainingConfig()
        cfg.train_ratio = self.train_ratio
        cfg.val_ratio = self.val_ratio
        cfg.epochs = self.train_epochs
        cfg.batch_size = self.train_batch_size
        cfg.early_stopping_patience = self.train_early_stopping_patience
        return cfg

    @property
    def backtesting(self) -> BacktestingConfig:
        cfg = BacktestingConfig()
        cfg.initial_capital = self.backtest_initial_capital
        cfg.transaction_cost_bps = self.backtest_transaction_cost_bps
        cfg.slippage_bps = self.backtest_slippage_bps
        cfg.min_liquidity_usd = self.backtest_min_liquidity_usd
        return cfg

    @property
    def paths(self) -> PathsConfig:
        return PathsConfig()


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached global Settings instance."""
    settings = Settings()
    # Ensure filesystem layout exists
    settings.paths.make_dirs()
    return settings


# ---------------------------------------------------------------------------
# Module-level convenience instance
# ---------------------------------------------------------------------------

settings = get_settings()
