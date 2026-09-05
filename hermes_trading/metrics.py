"""Hermes Trading Bot - Performance Metrics.

Calculates comprehensive performance statistics from trade lists.

Metrics calculated:
- total return
- CAGR
- maximum drawdown
- Sharpe ratio
- Sortino ratio
- win rate
- profit factor
- expectancy
- trade count
- average trade
- largest win
- largest loss
- consecutive losses
- current strategy version
- current score
"""

import numpy as np
from typing import List, Dict, Any, Tuple


class PerformanceMetrics:
    """Calculate performance metrics from trade list."""

    @staticmethod
    def calculate(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate all performance metrics from a list of trades.

        Args:
            trades: List of trade dicts with 'pnl_pct' keys

        Returns:
            Dict with all calculated metrics
        """
        if not trades:
            return {
                "return_pct": 0.0,
                "cagr": 0.0,
                "max_drawdown": 0.0,
                "sharpe": 0.0,
                "sortino": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "expectancy": 0.0,
                "trade_count": 0,
                "average_trade": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
                "consecutive_losses": 0,
                "current_version": "unknown",
                "current_score": 0.0,
            }

        # Extract P&L percentages
        pnls = [t["pnl_pct"] for t in trades if "pnl_pct" in t]
        raw_pnls = [t.get("pnl", 0) for t in trades if "pnl" in t]

        trade_count = len(trades)
        winning_trades = [p for p in pnls if p > 0]
        losing_trades = [p for p in pnls if p < 0]

        # Win rate
        win_rate = len(winning_trades) / trade_count if trade_count > 0 else 0.0

        # Profit factor
        gross_profit = sum(w for w in winning_trades) if winning_trades else 0.0
        gross_loss = abs(sum(l for l in losing_trades)) if losing_trades else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0

        # Expectancy: expected value per trade
        expectancy = (win_rate * gross_profit / trade_count) - ((1 - win_rate) * gross_loss / trade_count) if trade_count > 0 else 0.0

        # Average trade
        average_trade = np.mean(raw_pnls) if raw_pnls else 0.0

        # Largest win and loss
        largest_win = max(winning_trades) if winning_trades else 0.0
        largest_loss = min(losing_trades) if losing_trades else 0.0

        # Consecutive losses
        consecutive_losses = PerformanceMetrics._max_consecutive_losing_streak(trades)

        # Total return (cumulative)
        total_return = sum(raw_pnls) / 100.0  # pnls are percentages, convert to decimal

        # CAGR - approximate assuming 1 year per 365 days of data
        # For now, use simple annualization
        days = len(trades) * 1  # rough estimate
        cagr = (1 + total_return) ** (365 / max(days, 1)) - 1 if days > 0 else 0.0

        # Maximum drawdown from equity curve (approximated from trade P&Ls)
        max_drawdown = PerformanceMetrics._calculate_max_drawdown(raw_pnls)

        # Sharpe ratio (risk-free rate assumed 0 for simplicity)
        sharpe = PerformanceMetrics._calculate_sharpe(raw_pnls)

        # Sortino ratio
        sortino = PerformanceMetrics._calculate_sortino(raw_pnls)

        return {
            "return_pct": round(total_return * 100, 2),
            "cagr": round(cagr, 4),
            "max_drawdown": round(max_drawdown, 4),
            "sharpe": round(sharpe, 4),
            "sortino": round(sortino, 4),
            "win_rate": round(win_rate, 4),
            "profit_factor": round(profit_factor, 4),
            "expectancy": round(expectancy, 4),
            "trade_count": trade_count,
            "average_trade": round(average_trade, 4),
            "largest_win": round(largest_win, 4),
            "largest_loss": round(largest_loss, 4),
            "consecutive_losses": consecutive_losses,
            "current_version": "unknown",
            "current_score": 0.0,
        }

    @staticmethod
    def _max_consecutive_losing_streak(trades: List[Dict[str, Any]]) -> int:
        """Find the maximum consecutive losing trades streak."""
        if not trades:
            return 0

        max_streak = 0
        current_streak = 0

        for trade in trades:
            pnl = trade.get("pnl", 0)
            if pnl < 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return max_streak

    @staticmethod
    def _calculate_sharpe(pnls: List[float], risk_free: float = 0.0) -> float:
        """Calculate Sharpe ratio.

        Sh = (Rp - Rf) / σR
        Where Rp is portfolio return, Rf is risk-free rate, σR is std dev of returns
        """
        if not pnls or len(pnls) < 2:
            return 0.0

        pnls_array = np.array(pnls, dtype=float)
        mean_return = np.mean(pnls_array)
        std_return = np.std(pnls_array, ddof=1)  # sample std dev

        if std_return == 0:
            return 0.0

        sharpe = (mean_return - risk_free) / std_return
        return sharpe

    @staticmethod
    def _calculate_sortino(pnls: List[float], target_return: float = 0.0) -> float:
        """Calculate Sortino ratio.

        So = (Rp - TARGET) / MAR
        Where MAR is the downside deviation
        """
        if not pnls or len(pnls) < 2:
            return 0.0

        pnls_array = np.array(pnls, dtype=float)
        mean_return = np.mean(pnls_array)

        # Downside deviation: only negative returns
        downside_returns = pnls_array[pnls_array < target_return]
        if len(downside_returns) == 0:
            # All returns above target, Sortino is effectively infinite
            return float('inf') if mean_return > target_return else 0.0

        downside_deviation = np.std(downside_returns, ddof=1)

        if downside_deviation == 0:
            return float('inf') if mean_return >= target_return else 0.0

        sortino = (mean_return - target_return) / downside_deviation
        return sortino

    @staticmethod
    def _calculate_max_drawdown(raw_pnls: List[float]) -> float:
        """Calculate maximum drawdown from raw P&L percentages.

        Approximates by tracking running peak-to-trough decline.
        """
        if not raw_pnls:
            return 0.0

        # Convert to equity curve assuming starting at 1.0
        equity = 1.0
        peak = 1.0
        max_dd = 0.0

        for pnl in raw_pnls:
            equity += pnl  # pnl is already a decimal change
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak
            max_dd = max(max_dd, drawdown)

        return max_dd