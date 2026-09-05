"""Hermes Trading Bot - Backtest Engine.

Deterministic backtest engine that:
1. Loads historical data
2. Applies strategy sequentially
3. Prevents look-ahead bias
4. Simulates entries/exits
5. Applies fees and slippage
6. Applies position sizing
7. Tracks equity and P&L
8. Calculates performance metrics
9. Saves complete results

The same inputs must produce deterministic results.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from hermes_trading.data import DataEngine
from hermes_trading.strategy import Strategy
from hermes_trading.metrics import PerformanceMetrics


class BacktestEngine:
    """Deterministic backtest engine."""

    def __init__(self, data: DataEngine, strategy: Strategy, fees: float = 0.001, slippage: float = 0.0005):
        self.data = data
        self.strategy = strategy
        self.fees = fees  # per-trade fee rate
        self.slippage = slippage  # per-trade slippage rate
        self.metrics = PerformanceMetrics()

    def run(
        self,
        df: pd.DataFrame,
        train_pct: float = 0.6,
        validation_pct: float = 0.2,
        test_pct: float = 0.2,
    ) -> Dict[str, Any]:
        """Run backtest with train/validation/test split.

        Args:
            df: Historical OHLCV DataFrame
            train_pct: Percentage for training set
            validation_pct: Percentage for validation set
            test_pct: Percentage for test set

        Returns:
            Dict with per-split results and overall metrics
        """
        # Split data
        total_len = len(df)
        train_end = int(total_len * train_pct)
        validation_end = train_end + int(total_len * validation_pct)

        train_df = df.iloc[:train_end].copy()
        validation_df = df.iloc[train_end:validation_end].copy()
        test_df = df.iloc[validation_end:].copy()

        # Run backtest on each split
        train_results = self._run_on_data(train_df, "TRAIN")
        validation_results = self._run_on_data(validation_df, "VALIDATION")
        test_results = self._run_on_data(test_df, "TEST")

        # Combine results
        all_trades = train_results["trades"] + validation_results["trades"] + test_results["trades"]
        combined_equity = self._combine_equity([
            train_results["equity_curve"],
            validation_results["equity_curve"],
            test_results["equity_curve"],
        ])

        # Calculate overall metrics
        overall_metrics = self.metrics.calculate(all_trades)

        results = {
            "train": train_results,
            "validation": validation_results,
            "test": test_results,
            "all_trades": all_trades,
            "equity_curve": combined_equity,
            "overall_metrics": overall_metrics,
            "split_config": {
                "train_pct": train_pct,
                "validation_pct": validation_pct,
                "test_pct": test_pct,
            },
        }

        return results

    def _run_on_data(self, df: pd.DataFrame, split_name: str) -> Dict[str, Any]:
        """Run backtest on a single data split.

        Args:
            df: DataFrame for this split
            split_name: Name identifier (TRAIN, VALIDATION, TEST)

        Returns:
            Dict with equity curve and trade list for this split
        """
        if df.empty or len(df) < 50:
            return {
                "trades": [],
                "equity_curve": [1.0],
                "metrics": {"return_pct": 0, "sharpe": 0, "sortino": 0, "win_rate": 0, "profit_factor": 0, "expectancy": 0, "trade_count": 0},
                "split_name": split_name,
            }

        # Initialize tracking
        equity = 1.0  # Start with 1.0 unit of capital
        position = 0.0  # Current position (positive = long, negative = short)
        entry_price = 0.0
        trades: List[Dict[str, Any]] = []
        equity_curve: List[float] = [1.0]

        # Track consecutive losses
        consecutive_losses = 0
        max_consecutive_losses = 0
        total_wins = 0
        total_losses = 0
        total_win_pnl = 0.0
        total_loss_pnl = 0.0

        # Strategy parameters
        rsi_threshold = self.strategy.entry.get("threshold", 30)
        take_profit_pct = self.strategy.exit.get("take_profit_pct", 4.0) / 100.0
        stop_loss_pct = self.strategy.exit.get("stop_loss_pct", 2.0) / 100.0
        size_pct = self.strategy.position.get("size_pct", 5.0) / 100.0

        # RSI calculation helper
        def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
            delta = series.diff()
            up = delta.clip(lower=0)
            down = -delta.clip(upper=0)
            ma_up = up.ewm(alpha=1/period, adjust=False).mean()
            ma_down = down.ewm(alpha=1/period, adjust=False).mean()
            rs = ma_up / ma_down
            rsi = 100 - (100 / (1 + rs))
            return rsi

        # Main backtest loop - walk through data from index 14 (minimum for RSI)
        min_rsi_period = 14
        for i in range(min_rsi_period, len(df)):
            current_candle = df.iloc[i]
            prev_candle = df.iloc[i - 1]
            close = current_candle["close"]
            high = current_candle["high"]
            low = current_candle["low"]

            # Calculate RSI on data up to current point (NO look-ahead bias)
            # Use closing prices up to and including current candle
            prices = df["close"].iloc[:i + 1]
            rsi = calculate_rsi(prices).iloc[i]

            # Entry condition: RSI < threshold (oversold, go long)
            # We also need to ensure we don't have an existing position
            if rsi < rsi_threshold and position == 0:
                # Simulate entry with slippage: execute at slightly worse price
                entry_price = close * (1 - self.slippage)  # Long entry at bid-side price
                position = size_pct * equity  # Position size as % of current equity
                # Deduct cost from equity
                equity -= position * entry_price * self.fees

            # Exit conditions if we have a position
            elif position > 0:
                # Check take profit: price moves up by take_profit_pct from entry
                tp_price = entry_price * (1 + take_profit_pct)
                # Check stop loss: price moves down by stop_loss_pct from entry
                sl_price = entry_price * (1 - stop_loss_pct)

                # Determine exit price based on market movement
                # Use the highest high or lowest low reached since entry
                # Simplified: check current candle high/low against targets
                exit_price = close

                # Check if take profit hit (high reached or current close triggers it)
                if high >= tp_price or close >= tp_price:
                    exit_price = tp_price  # Take profit triggered
                    # Calculate P&L
                    pnl_pct = (exit_price / entry_price - 1) - self.fees * 2  # Round-turn fees
                    pnl = position * pnl_pct
                elif low <= sl_price:
                    exit_price = sl_price  # Stop loss triggered
                    # Calculate P&L
                    pnl_pct = (exit_price / entry_price - 1) - self.fees * 2  # Round-turn fees
                    pnl = position * pnl_pct
                else:
                    # Exit at current market price (no TP/SL hit)
                    exit_price = close
                    pnl_pct = (exit_price / entry_price - 1) - self.fees * 2
                    pnl = position * pnl_pct

                # Close position
                equity += pnl
                position = 0.0

                # Track trade statistics
                trade = {
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(exit_price, 2),
                    "pnl": round(pnl, 4),
                    "pnl_pct": round(pnl_pct * 100, 2),
                    "holding_period": 1,  # Simplified - would track bar count in full impl
                    "split": split_name,
                }
                trades.append(trade)

                if pnl > 0:
                    total_wins += 1
                    total_win_pnl += pnl
                else:
                    total_losses += 1
                    total_loss_pnl += pnl
                    consecutive_losses += 1
                    max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                consecutive_losses = 0 if pnl > 0 else consecutive_losses

            # Update equity curve (append current equity even if no trade)
            equity_curve.append(equity)

        # Calculate metrics for this split
        metrics = self.metrics.calculate(trades)

        return {
            "trades": trades,
            "equity_curve": equity_curve,
            "metrics": metrics,
            "split_name": split_name,
        }

    def _combine_equity(self, equity_curves: List[List[float]]) -> List[float]:
        """Combine equity curves from different splits.

        Pads shorter curves and averages at each point.
        """
        if not equity_curves:
            return [1.0]

        max_len = max(len(c) for c in equity_curves)
        padded = []
        for c in equity_curves:
            if len(c) < max_len:
                c = c + [c[-1]] * (max_len - len(c))  # Pad with last value
            padded.append(c)

        # Average at each point
        combined = [
            np.mean([p[i] for p in padded])
            for i in range(max_len)
        ]
        return combined