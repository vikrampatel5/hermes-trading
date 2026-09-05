"""Hermes Trading Bot - Continuous Trading Loop.

Contains the per-iteration logic for the trading worker.
Extracted from run.py for modularity."""

import asyncio
from typing import Dict, Any, Optional
from pathlib import Path
import json
import time
from datetime import datetime, timezone

from hermes_trading.config import settings
from hermes_trading.data import DataEngine
from hermes_trading.strategy import Strategy
from hermes_trading.backtest import BacktestEngine
from hermes_trading.execution import PaperExecution
from hermes_trading.risk import RiskManager
from hermes_trading.metrics import PerformanceMetrics
from hermes_trading.worker_control import WorkerControl
from hermes_trading.score import score


class TradingLoop:
    """Per-iteration logic for the trading worker."""

    def __init__(self, worker_control: WorkerControl, strategy: Strategy,
                 goal: Dict[str, Any], data_engine: DataEngine,
                 risk_manager: RiskManager, execution: PaperExecution,
                 performance_metrics: PerformanceMetrics):
        self.control = worker_control
        self.strategy = strategy
        self.goal = goal
        self.data_engine = data_engine
        self.risk_manager = risk_manager
        self.execution = execution
        self.metrics = performance_metrics
        self.project_root = Path(__file__).parent.parent
        self.state_dir = self.project_root / "state"
        self.trades_file = self.state_dir / "trades.jsonl"
        self.performance_file = self.state_dir / "performance.json"
        self.heartbeat_file = self.state_dir / "heartbeat.json"

    async def iterate(self, historical_data: Any, current_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one iteration of the trading loop.

        Returns:
            Dict with action taken and results.
        """
        result = {
            "action": "continue",
            "trade_executed": False,
            "risk_blocked": False,
            "score": 0.0,
        }

        # 1. Validate data
        if historical_data.empty:
            return result

        # 2. Check risk controls
        current_equity = 1.0
        current_drawdown = 0.0
        daily_loss_pct = self._get_daily_loss_pct()
        consecutive = self._get_consecutive_losses()

        risk_allowed, risk_reason = self.risk_manager.check_entry_allowed(
            equity=current_equity,
            proposed_position_pct=self.strategy.position.get("size_pct", 5.0),
            current_drawdown_pct=current_drawdown,
            consecutive_losses=consecutive,
            daily_loss_pct=daily_loss_pct,
        )

        if not risk_allowed:
            result["risk_blocked"] = True
            result["risk_reason"] = risk_reason
            return result

        # 3. Generate trading signal
        signal = self._generate_signal(historical_data, current_data)

        if not signal:
            return result

        # 4. Execute paper trade
        trade_result = await self._execute_trade(signal, historical_data, current_data)

        if trade_result:
            result["trade_executed"] = True
            result["trade_details"] = trade_result

            # 5. Update score
            all_trades = self._read_all_trades()
            if all_trades:
                result["score"] = score(all_trades, {
                    "target_return_30d": self.goal.get("target_return_30d", 0.20),
                    "max_drawdown": self.goal.get("max_drawdown", 0.08),
                    "min_sharpe": self.goal.get("min_sharpe", 1.2),
                })

        return result

    def _generate_signal(self, historical: Any, current: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate trading signal based on strategy rules."""
        from hermes_trading.strategy import Strategy

        rsi_threshold = self.strategy.entry.get("threshold", 30)
        take_profit_pct = self.strategy.exit.get("take_profit_pct", 4.0)
        stop_loss_pct = self.strategy.exit.get("stop_loss_pct", 2.0)
        size_pct = self.strategy.position.get("size_pct", 5.0)

        # Calculate RSI
        prices = historical["close"]
        if len(prices) < 15:
            return None

        delta = prices.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        ma_up = up.ewm(alpha=1/14, adjust=False).mean()
        ma_down = down.ewm(alpha=1/14, adjust=False).mean()
        rs = ma_up / ma_down
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]

        # Entry signal: RSI < threshold
        if current_rsi < rsi_threshold:
            return {
                "type": "entry",
                "side": "long",
                "price": current["close"],
                "rsi": float(current_rsi),
                "threshold": rsi_threshold,
                "size_pct": size_pct,
                "timestamp": current["timestamp"],
            }

        return None

    async def _execute_trade(self, signal: Dict[str, Any], historical: Any, current: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute a paper trade and record it."""
        price = signal["price"]
        size_pct = signal["size_pct"]
        side = signal["side"]
        timestamp = signal["timestamp"]

        # Simulate entry
        equity = 1.0
        entry_result = self.execution.simulate_entry(
            market_price=price,
            position_size_pct=size_pct,
            equity=equity,
        )

        # For continuous worker, we track the position open.
        # For backtest-style immediate close, use next candle:
        if len(historical) > 1:
            exit_price = historical.iloc[1]["close"]
        else:
            exit_price = price

        # Simulate exit
        exit_result = self.execution.simulate_exit(entry_result, exit_price)

        # Calculate trade record
        pnl = exit_result.get("realized_pnl", 0)
        pnl_pct = pnl / equity if equity else 0

        trade_record = {
            "timestamp": timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
            "symbol": settings.ASSET,
            "strategy_version": self.strategy.version,
            "side": side,
            "entry_price": entry_result["entry_price"],
            "exit_price": exit_result.get("exit_price", price),
            "position_size": entry_result["position_size_equity"],
            "fees": abs(pnl) / 2 if pnl else 0,  # Rough estimate round-turn
            "slippage": abs(price - entry_result["entry_price"]) / price if price else 0,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "holding_period": 1.0,
            "market_regime": "unknown",
            "reason": f"RSI {signal['rsi']:.1f} strategy entry",
        }

        # Append to trades file
        self._append_trade_record(trade_record)

        # Update consecutive losses
        if pnl < 0:
            self._increment_consecutive_losses()
        else:
            self._reset_consecutive_losses()

        return trade_record

    def _read_all_trades(self) -> list:
        """Read all trade records from JSONL."""
        trades = []
        if self.trades_file.exists():
            with open(self.trades_file, "r", encoding="utf-8-sig") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            trades.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        return trades

    def _append_trade_record(self, trade: Dict[str, Any]) -> None:
        """Append trade record to JSONL file."""
        with open(self.trades_file, "a", encoding="utf-8-sig") as f:
            f.write(json.dumps(trade, ensure_ascii=False) + "\n")

    def _get_daily_loss_pct(self) -> float:
        """Get current day's loss percentage."""
        return 0.0  # Simplified

    def _get_consecutive_losses(self) -> int:
        """Get current consecutive losses count."""
        trades = self._read_all_trades()
        if not trades:
            return 0
        # Count consecutive losing trades at the end
        count = 0
        for t in reversed(trades):
            if t.get("pnl", 0) < 0:
                count += 1
            else:
                break
        return count

    def _increment_consecutive_losses(self) -> None:
        """Increment consecutive losses counter in state."""
        # Would persist to state file
        pass

    def _reset_consecutive_losses(self) -> None:
        """Reset consecutive losses counter."""
        pass

    async def _update_performance(self) -> None:
        """Update performance.json after each iteration."""
        all_trades = self._read_all_trades()

        if all_trades:
            metrics = self.metrics.calculate(all_trades)
            # Read goal for target
            target_30d = self.goal.get("target_return_30d", 0.20)
            total_return_pct = metrics.get("return_pct", 0.0)

            # Calculate daily and 30-day approximations
            # (In full system, would track actual dates)
            daily_return = total_return_pct / max(self._get_trade_count_days(all_trades), 1)

            performance = {
                "total_return": total_return_pct,
                "daily_return": daily_return,
                "30_day_return": total_return_pct,  # Simplified
                "maximum_drawdown": metrics.get("max_drawdown", 0.0),
                "sharpe": metrics.get("sharpe", 0.0),
                "sortino": metrics.get("sortino", 0.0),
                "win_rate": metrics.get("win_rate", 0.0),
                "profit_factor": metrics.get("profit_factor", 0.0),
                "expectancy": metrics.get("expectancy", 0.0),
                "trade_count": metrics.get("trade_count", 0),
                "average_trade": metrics.get("average_trade", 0.0),
                "largest_win": metrics.get("largest_win", 0.0),
                "largest_loss": metrics.get("largest_loss", 0.0),
                "consecutive_losses": metrics.get("consecutive_losses", 0),
                "current_strategy_version": self.strategy.version,
                "current_score": metrics.get("current_score", 0.0),
            }

            with open(self.performance_file, "w") as f:
                json.dump(performance, f, indent=2)

    def _get_trade_count_days(self, trades: list) -> float:
        """Approximate number of days from trade data."""
        if not trades:
            return 1.0
        # Very rough: assume trades span some days
        return max(len(trades) * 0.1, 1.0)  # Placeholder