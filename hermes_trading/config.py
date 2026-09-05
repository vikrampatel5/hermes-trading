"""Hermes Trading Bot - Configuration."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

class Settings:
    """Configuration settings loaded from environment and defaults."""

    # Trading mode
    TRADING_MODE: str = os.getenv("TRADING_MODE", "paper")
    LIVE_TRADING_ENABLED: bool = os.getenv("LIVE_TRADING_ENABLED", "false").lower() == "true"

    # Exchange
    EXCHANGE_API_KEY: str = os.getenv("EXCHANGE_API_KEY", "")
    EXCHANGE_API_SECRET: str = os.getenv("EXCHANGE_API_SECRET", "")

    # API keys (optional)
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
    MACRO_API_KEY: str = os.getenv("MACRO_API_KEY", "")

    # Strategy parameters
    ASSET: str = os.getenv("ASSET", "BTC/USDT")
    MARKET: str = os.getenv("MARKET", "crypto")
    EXCHANGE: str = os.getenv("EXCHANGE", "delta_testnet")
    TIMEFRAME: str = os.getenv("TIMEFRAME", "5m")

    # Risk
    MAX_POSITION_SIZE_PCT: float = float(os.getenv("MAX_POSITION_SIZE_PCT", "5.0"))
    MAX_PORTFOLIO_EXPOSURE_PCT: float = float(os.getenv("MAX_PORTFOLIO_EXPOSURE_PCT", "10.0"))
    MAX_DRAWDOWN_PCT: float = float(os.getenv("MAX_DRAWDOWN_PCT", "8.0"))
    MAX_DAILY_LOSS_PCT: float = float(os.getenv("MAX_DAILY_LOSS_PCT", "3.0"))
    MAX_CONSECUTIVE_LOSSES: int = int(os.getenv("MAX_CONSECUTIVE_LOSSES", "3"))

    # Performance targets
    TARGET_RETURN_30D: float = float(os.getenv("TARGET_RETURN_30D", "0.20"))
    MIN_SHARPE: float = float(os.getenv("MIN_SHARPE", "1.2"))

    # Reflection
    REFLECTION_EVERY: int = int(os.getenv("REFLECTION_EVERY", "5"))

    # Operational
    WORKER_SLEEP_SECONDS: int = int(os.getenv("WORKER_SLEEP_SECONDS", "10"))
    HEARTBEAT_INTERVAL_SECONDS: int = int(os.getenv("HEARTBEAT_INTERVAL_SECONDS", "30"))

    def __post_init__(self):
        if self.TRADING_MODE not in ("paper", "live"):
            raise ValueError(f"Invalid TRADING_MODE: {self.TRADING_MODE}")
        if self.LIVE_TRADING_ENABLED and self.TRADING_MODE != "live":
            raise ValueError("LIVE_TRADING_ENABLED=true requires TRADING_MODE=live")

settings = Settings()