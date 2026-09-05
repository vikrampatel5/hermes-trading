"""Hermes Trading Bot - Configuration."""

import os
from pathlib import Path

# Load .env from project root safely
PROJECT_ROOT = Path(__file__).parent.parent
env_path = PROJECT_ROOT / ".env"

try:
    from dotenv import load_dotenv
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    # Manual .env loader fallback if python-dotenv is not installed
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k not in os.environ:
                        os.environ[k] = v

class Settings:
    """Configuration settings loaded from environment and defaults."""

    # Trading mode
    TRADING_MODE: str = os.getenv("HERMES_TRADING_MODE", os.getenv("TRADING_MODE", "paper"))
    LIVE_TRADING_ENABLED: bool = (
        os.getenv("HERMES_TRADING_I_ACCEPT_RISK", os.getenv("LIVE_TRADING_ENABLED", "false")).lower() == "true"
    )

    # Exchange
    EXCHANGE_API_KEY: str = os.getenv("EXCHANGE_API_KEY", "")
    EXCHANGE_API_SECRET: str = os.getenv("EXCHANGE_API_SECRET", "")

    # API keys (optional)
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
    MACRO_API_KEY: str = os.getenv("MACRO_API_KEY", "")

    # Strategy parameters
    ASSET: str = os.getenv("ASSET", "BTC/USDT")
    MARKET: str = os.getenv("MARKET", "crypto")
    EXCHANGE: str = os.getenv("EXCHANGE", "binance")
    TIMEFRAME: str = os.getenv("TIMEFRAME", "5m")

    # Risk
    MAX_POSITION_SIZE_PCT: float = float(os.getenv("MAX_POSITION_SIZE_PCT", "5.0"))
    MAX_PORTFOLIO_EXPOSURE_PCT: float = float(os.getenv("MAX_PORTFOLIO_EXPOSURE_PCT", "10.0"))
    MAX_DRAWDOWN_PCT: float = float(os.getenv("MAX_DRAWDOWN_PCT", "8.0"))
    MAX_DAILY_LOSS_PCT: float = float(os.getenv("MAX_DAILY_LOSS_PCT", "3.0"))
    MAX_CONSECUTIVE_LOSSES: int = int(os.getenv("MAX_CONSECUTIVE_LOSSES", "3"))

    # Performance targets
    TARGET_RETURN_30D: float = float(os.getenv("TARGET_RETURN_30D", "0.10"))
    MIN_SHARPE: float = float(os.getenv("MIN_SHARPE", "2.0"))

    # Reflection
    REFLECTION_EVERY: int = int(os.getenv("REFLECTION_EVERY", "3"))

    # Operational
    WORKER_SLEEP_SECONDS: int = int(os.getenv("WORKER_SLEEP_SECONDS", "10"))
    HEARTBEAT_INTERVAL_SECONDS: int = int(os.getenv("HEARTBEAT_INTERVAL_SECONDS", "30"))

settings = Settings()