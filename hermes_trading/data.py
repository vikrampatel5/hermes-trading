"""Hermes Trading Bot - Data Engine.

Handles historical data retrieval, current market data, validation,
and data quality checks for the trading system.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ccxt
import httpx
import numpy as np
import pandas as pd

# Project imports
from hermes_trading.config import settings


class DataEngine:
    """Historical and current market data engine."""

    def __init__(self):
        self.exchange = self._init_exchange()
        self.cache_dir = Path(__file__).parent.parent / "state" / "backtests"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _init_exchange(self) -> ccxt.Exchange:
        """Initialize the CCXT exchange connection."""
        exchange_id = settings.EXCHANGE
        kwargs = {
            "enableRateLimit": True,
        }
        # Handle Delta TestNet specially
        if exchange_id == "delta_testnet":
            exchange = ccxt.delta({
                "options": {
                    "defaultType": "spot",
                },
            })
            # Set testnet base URL for Delta
            exchange.base_url = "https://cdn-ind.testnet.deltaex.org"
        else:
            exchange = getattr(ccxt, exchange_id)(kwargs)
        return exchange

    def fetch_historical_data(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data from the exchange.

        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Candle timeframe (e.g., '5m', '1h', '1d')
            limit: Number of candles to fetch

        Returns:
            DataFrame with OHLCV data and normalized timestamps
        """
        try:
            raw = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            df = df.astype(float)
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to fetch historical data: {e}")

    def fetch_current_data(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """Fetch the latest candle for the symbol/timeframe.

        Args:
            symbol: Trading pair symbol
            timeframe: Candle timeframe

        Returns:
            Dict with latest candle data or None on failure
        """
        try:
            raw = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=1)
            if not raw:
                return None
            latest = raw[0]
            return {
                "timestamp": pd.to_datetime(latest[0], unit="ms"),
                "open": latest[1],
                "high": latest[2],
                "low": latest[3],
                "close": latest[4],
                "volume": latest[5],
            }
        except Exception as e:
            print(f"[data] Error fetching current data: {e}")
            return None

    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate data quality.

        Checks for:
        - Missing candles (gaps)
        - Duplicate timestamps
        - Malformed values (NaN, inf)
        - Sufficient data length

        Returns:
            (is_valid, issues) tuple
        """
        issues: List[str] = []

        if df.empty:
            issues.append("DataFrame is empty")
            return False, issues

        # Check for missing values
        if df.isnull().any().any():
            null_counts = df.isnull().sum()
            issues.append(f"Null values found: {null_counts.to_dict()}")

        # Check for inf values
        if np.isinf(df.select_dtypes(include=[np.floating]).values).any():
            issues.append("Infinite values found")

        # Check for duplicates
        if df.index.duplicated().any():
            dupes = df.index[df.index.duplicated()].tolist()
            issues.append(f"Duplicate timestamps: {dupes[:5]}")

        # Check for sufficient data (minimum 50 candles)
        if len(df) < 50:
            issues.append(f"Insufficient data: {len(df)} candles (minimum 50 required)")

        # Check for gaps (reasonable gap detection)
        if len(df) > 1:
            timestamps = df.index
            expected_interval = timestamps[1] - timestamps[0]
            for i in range(2, len(timestamps)):
                gap = timestamps[i] - timestamps[i - 1]
                if abs(gap - expected_interval) > expected_interval * 0.5:
                    issues.append(f"Potential gap at {timestamps[i]}: {gap} vs expected {expected_interval}")
                    break

        is_valid = len(issues) == 0
        return is_valid, issues

    def detect_missing_candles(
        self, df: pd.DataFrame, expected_interval_minutes: int
    ) -> List[pd.Timestamp]:
        """Detect gaps in candle data.

        Args:
            df: OHLCV DataFrame with timestamp index
            expected_interval_minutes: Expected candle duration in minutes

        Returns:
            List of timestamps where candles are missing
        """
        missing: List[pd.Timestamp] = []
        if len(df) < 2:
            return missing

        idx = df.index
        for i in range(1, len(idx)):
            gap = idx[i] - idx[i - 1]
            # Allow small timing variations
            tolerance = pd.Timedelta(minutes=expected_interval_minutes * 0.2)
            expected = pd.Timedelta(minutes=expected_interval_minutes)
            if abs(gap - expected) > tolerance:
                # Report the missing timestamp
                missing.append(idx[i - 1 + 1] - expected if i > 0 else idx[0] - expected)

        return missing

    def save_to_cache(self, df: pd.DataFrame, symbol: str, timeframe: str) -> None:
        """Save historical data to cache for offline use.

        Args:
            df: DataFrame to save
            symbol: Trading pair symbol
            timeframe: Candle timeframe
        """
        cache_file = self.cache_dir / f"{symbol.replace('/', '_')}_{timeframe}.json"
        df.to_json(cache_file, orient="records", date_format="iso")
        print(f"[data] Saved {len(df)} candles to cache: {cache_file}")

    def load_from_cache(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Load historical data from cache.

        Args:
            symbol: Trading pair symbol
            timeframe: Candle timeframe

        Returns:
            DataFrame if cached, None otherwise
        """
        cache_file = self.cache_dir / f"{symbol.replace('/', '_')}_{timeframe}.json"
        if cache_file.exists():
            df = pd.read_json(cache_file, orient="records", dates="iso")
            df.index = pd.to_datetime(df.index)
            df = df.astype(float)
            print(f"[data] Loaded {len(df)} candles from cache: {cache_file}")
            return df
        return None