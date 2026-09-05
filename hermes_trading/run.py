"""Hermes Trading Bot - Continuous Trading Worker.

Runs continuously in paper mode, evaluating market data and executing
paper trades according to the current strategy.

The worker does NOT depend on Hermes remaining open. It runs as an
independent local process."""
import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_trading.config import settings
from hermes_trading.data import DataEngine
from hermes_trading.strategy import Strategy
from hermes_trading.backtest import BacktestEngine
from hermes_trading.execution import PaperExecution
from hermes_trading.risk import RiskManager
from hermes_trading.metrics import PerformanceMetrics
from hermes_trading.worker_control import WorkerControl


class TradingWorker:
    """Main trading worker that runs continuously."""

    def __init__(self):
        self.data_engine = DataEngine()
        self.risk_manager = RiskManager()
        self.execution = PaperExecution()
        self.control = WorkerControl()
        self.worker_start_time = datetime.now(timezone.utc)

        # State paths
        self.project_root = Path(__file__).parent.parent
        self.state_dir = self.project_root / "state"
        self.trades_file = self.state_dir / "trades.jsonl"
        self.performance_file = self.state_dir / "performance.json"
        self.heartbeat_file = self.state_dir / "heartbeat.json"
        self.strategy_file = self.state_dir / "strategy.yaml"
        self.goal_file = self.state_dir / "goal.yaml"

        # Load strategy and goal
        self.strategy = Strategy.load_from_file(self.strategy_file)
        self.goal = self._load_goal()

        # Performance tracking
        self.performance_metrics = PerformanceMetrics()
        self.closed_trades_count = 0
        self.daily_loss_tracker = 0.0
        self.consecutive_losses = 0
        self.last_daily_reset = datetime.now(timezone.utc).date()

    def _load_goal(self) -> Dict[str, Any]:
        """Load goal configuration from YAML."""
        if self.goal_file.exists():
            import yaml
            with open(self.goal_file, "r") as f:
                return yaml.safe_load(f) or {}
        return {
            "target_return_30d": 0.20,
            "max_drawdown": 0.08,
            "min_sharpe": 1.2,
            "reflection_every": 5,
        }

    async def run(self) -> None:
        """Main worker loop.

        Continuously:
        1. Fetch market data
        2. Validate data
        3. Load current strategy
        4. Evaluate strategy
        5. Evaluate risk controls
        6. Generate signal
        7. Execute paper trade if valid
        8. Update positions
        9. Evaluate exits
        10. Close positions
        11. Update performance
        12. Write trade record
        13. Update heartbeat
        14. Sleep
        15. Repeat
        """
        print("[worker] Booting hermes-trading worker...")

        while self.control.running:
            try:
                await self._iteration_cycle()
            except Exception as e:
                print(f"[worker] Error in iteration: {e}")
                # Log error but continue
                await self._write_error_log(e)

            # Sleep until next cycle
            await asyncio.sleep(settings.WORKER_SLEEP_SECONDS)

        print("[worker] Trading worker shutting down gracefully.")

    async def _iteration_cycle(self) -> None:
        """Execute one full iteration of the trading loop."""

        # 1. Fetch market data
        symbol = settings.ASSET
        timeframe = settings.TIMEFRAME
        raw_data = self.data_engine.fetch_historical_data(symbol, timeframe, limit=100)

        # 2. Validate data
        is_valid, issues = self.data_engine.validate_data(raw_data)
        if not is_valid:
            print(f"[worker] Data validation issues: {issues[:3]}")
            # Still proceed with available data, but log warnings

        # 3. Load current strategy (already loaded in __init__)
        # 4. Evaluate strategy - generate signal

        # Get the latest candle for signal generation
        current_data = self.data_engine.fetch_current_data(symbol, timeframe)
        if not current_data:
            return  # No data available this cycle

        # 5. Evaluate risk controls
        current_equity = 1.0  # Would be actual equity from state
        current_drawdown = 0.0  # Would track from performance
        daily_loss_pct = self._get_daily_loss_pct()
        consecutive = self.consecutive_losses

        # Also check against historical performance
        risk_allowed, risk_reason = self.risk_manager.check_entry_allowed(
            equity=current_equity,
            proposed_position_pct=self.strategy.position.get("size_pct", 5.0),
            current_drawdown_pct=current_drawdown,
            consecutive_losses=consecutive,
            daily_loss_pct=daily_loss_pct,
        )

        if not risk_allowed:
            # print(f"[worker] Risk blocked entry: {risk_reason}")
            pass  # Skip entry due to risk controls
        else:
            # 6. Generate signal
            signal = self._generate_signal(raw_data, current_data)

            # 7. Execute paper trade if valid signal
            if signal and risk_allowed:
                await self._execute_paper_trade(signal, raw_data, current_data)

        # 8. Update positions (check exits, close if needed)
        await self._check_exits(raw_data, current_data)

        # 9. Update performance tracking
        await self._update_performance()

        # 10. Write trade record (if any trade was closed this cycle)
        # Already handled in _execute_paper_trade

        # 11. Update heartbeat
        await self._update_heartbeat()

    def _generate_signal(self, historical: pd.DataFrame, current: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate a trading signal based on strategy rules.

        For the default RSI mean-reversion strategy:
        - Entry: RSI < threshold (oversold)
        - Exit: take_profit_pct or stop_loss_pct
        """
        from hermes_trading.strategy import Strategy

        # Get strategy parameters
        rsi_threshold = self.strategy.entry.get("threshold", 30)
        take_profit_pct = self.strategy.exit.get("take_profit_pct", 4.0)
        stop_loss_pct = self.strategy.exit.get("stop_loss_pct", 2.0)
        size_pct = self.strategy.position.get("size_pct", 5.0)

        # Calculate RSI
        if len(historical) < 15:
            return None

        prices = historical["close"]
        delta = prices.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        ma_up = up.ewm(alpha=1/14, adjust=False).mean()
        ma_down = down.ewm(alpha=1/14, adjust=False).mean()
        rs = ma_up / ma_down
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]

        # Check if we already have an open position (simplified)
        # In full implementation, track open positions

        # Entry signal: RSI < threshold and no existing position
        if current_rsi < rsi_threshold:
            # Generate entry signal
            price = current["close"]
            return {
                "type": "entry",
                "side": "long",
                "price": price,
                "rsi": float(current_rsi),
                "threshold": rsi_threshold,
                "size_pct": size_pct,
                "timestamp": current["timestamp"],
            }

        return None

    async def _execute_paper_trade(self, signal: Dict[str, Any], historical: pd.DataFrame, current: Dict[str, Any]) -> None:
        """Execute a paper trade simulation.

        Args:
            signal: Trade signal dict
            historical: Historical data
            current: Current market data
        """
        price = signal["price"]
        size_pct = signal["size_pct"]
        side = signal["side"]
        timestamp = signal["timestamp"]

        # Simulate entry
        equity = 1.0  # Would be actual equity
        entry_result = self.execution.simulate_entry(
            market_price=price,
            position_size_pct=size_pct,
            equity=equity,
        )

        # Record the trade - we'll close it in the same cycle for backtest-like behavior
        # In continuous mode, position would remain open until exit signal

        # For now, simulate immediate exit at next candle for backtesting
        # Get next price from historical data
        if len(historical) > 1:
            # Use next candle's close as exit price
            exit_price = historical.iloc[1]["close"] if len(historical) > 1 else price
        else:
            exit_price = price

        # Simulate exit
        exit_result = self.execution.simulate_exit(entry_result, exit_price)

        # Calculate trade details
        pnl_pct = exit_result.get("realized_pnl", 0) / equity if equity else 0
        holding_period_hours = 1.0  # Simplified

        # Create trade record
        trade_record = {
            "timestamp": timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
            "symbol": settings.ASSET,
            "strategy_version": self.strategy.version,
            "side": side,
            "entry_price": entry_result["entry_price"],
            "exit_price": exit_result.get("exit_price", price),
            "position_size": entry_result["position_size_equity"],
            "fees": abs(exit_result.get("realized_pnl", 0)) / 2 if exit_result.get("realized_pnl") else 0,  # Estimate
            "slippage": abs(price - entry_result["entry_price"]) / price if price else 0,
            "pnl": exit_result.get("realized_pnl", 0),
            "pnl_pct": pnl_pct,
            "holding_period": holding_period_hours,
            "market_regime": "unknown",  # Would classify from regime analysis
            "reason": f"RSI {signal['rsi']:.1f} crossed threshold {signal['threshold']}",
        }

        # Write trade record to JSONL
        self._append_trade_record(trade_record)

        # Update counters
        self.closed_trades_count += 1
        if trade_record["pnl"] < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        print(f"[worker] Paper trade executed: pnl={trade_record['pnl_pct']:.2f}%, reason={trade_record['reason'][:50]}")

    async def _check_exits(self, historical: pd.DataFrame, current: Dict[str, Any]) -> None:
        """Check and execute exit orders for open positions.

        In a full implementation, this would track open positions and
        check TP/SL levels against current market price.
        """
        # Simplified: if we have an open position, check exits
        # In continuous worker, positions remain open between cycles
        # This would check: price >= TP or price <= SL
        pass

    async def _update_performance(self) -> None:
        """Update performance.json with current statistics."""
        # Read existing trades
        trades = self._read_all_trades()

        if trades:
            # Calculate metrics from all trades
            pnls = [t.get("pnl", 0) for t in trades]
            total_return = sum(pnls) / 100.0  # Convert percentage to decimal

            metrics = self.performance_metrics.calculate(trades)

            # Add derived fields
            metrics["total_return"] = round(total_return * 100, 2)
            metrics["closed_trades"] = self.closed_trades_count

            # Write performance file
            with open(self.performance_file, "w") as f:
                json.dump(metrics, f, indent=2)
        else:
            # Write empty/default performance
            default_metrics = {
                "total_return": 0.0,
                "daily_return": 0.0,
                "30_day_return": 0.0,
                "maximum_drawdown": 0.0,
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
                "current_strategy_version": self.strategy.version,
                "current_score": 0.0,
            }
            with open(self.performance_file, "w") as f:
                json.dump(default_metrics, f, indent=2)

    def _read_all_trades(self) -> List[Dict[str, Any]]:
        """Read all trade records from JSONL file."""
        trades = []
        if self.trades_file.exists():
            with open(self.trades_file, "r", encoding="utf-8-sig") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            trade = json.loads(line)
                            trades.append(trade)
                        except json.JSONDecodeError:
                            continue
        return trades

    def _append_trade_record(self, trade: Dict[str, Any]) -> None:
        """Append a trade record to the JSONL file."""
        with open(self.trades_file, "a", encoding="utf-8-sig") as f:
            f.write(json.dumps(trade, ensure_ascii=False) + "\n")

    async def _update_heartbeat(self) -> None:
        """Update heartbeat.json with current worker state."""
        from hermes_trading.reflector_daemon import ReflectorDaemon

        # Get recent closed trades count
        trades = self._read_all_trades()
        closed_count = len(trades)

        heartbeat = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "worker_status": "running",
            "strategy_version": self.strategy.version,
            "last_trade": trades[-1]["timestamp"] if trades else None,
            "closed_trades": closed_count,
            "last_reflection": trades[-1].get("reflection_timestamp", None) if trades else None,
            "last_backtest": None,
            "current_score": 0.0,  # Would be calculated from current strategy
        }

        with open(self.heartbeat_file, "w") as f:
            json.dump(heartbeat, f, indent=2)

    def _get_daily_loss_pct(self) -> float:
        """Get the daily loss percentage tracker."""
        today = datetime.now(timezone.utc).date()
        if today != self.last_daily_reset:
            self.daily_loss_tracker = 0.0
            self.last_daily_reset = today
        return self.daily_loss_tracker

    async def _write_error_log(self, error: Exception) -> None:
        """Write error to logs file."""
        log_dir = self.state_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "worker_errors.log"
        with open(log_file, "a", encoding="utf-8") as f:
            ts = datetime.now(timezone.utc).isoformat()
            f.write(f"[{ts}] {type(error).__name__}: {str(error)}\n")

async def _health_handler(request):
    """Liveness probe. Touches heartbeat.json so Render's idle-detector sees activity.

    Render pauses free instances after ~15 min of no HTTP traffic. The trading
    loop already writes heartbeat.json every WORKER_SLEEP_SECONDS (default 10s),
    but on a paused/just-resumed instance that may not be enough. This handler
    is hit by an external cron (Render cron job, see render.yaml) once a minute
    to keep the instance warm.
    """
    from aiohttp import web
    project_root = Path(__file__).parent.parent
    heartbeat_file = project_root / "state" / "heartbeat.json"
    # Refresh the file's mtime + bump the last_keepalive field so the cron
    # verifier can confirm it actually reached the worker (not just the LB).
    try:
        if heartbeat_file.exists():
            with open(heartbeat_file, "r+") as f:
                hb = json.load(f) or {}
                hb["last_keepalive"] = datetime.now(timezone.utc).isoformat()
                hb["keepalive_source"] = "health_endpoint"
                f.seek(0)
                json.dump(hb, f, indent=2)
                f.truncate()
        else:
            with open(heartbeat_file, "w") as f:
                json.dump({
                    "last_keepalive": datetime.now(timezone.utc).isoformat(),
                    "keepalive_source": "health_endpoint",
                }, f, indent=2)
    except Exception:
        pass
    return web.json_response({
        "status": "ok",
        "service": "hermes-trading",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


async def _status_handler(request):
    from aiohttp import web
    project_root = Path(__file__).parent.parent
    heartbeat_file = project_root / "state" / "heartbeat.json"
    hb = {}
    if heartbeat_file.exists():
        try:
            with open(heartbeat_file, "r") as f:
                hb = json.load(f)
        except Exception:
            pass
    return web.json_response({
        "status": "running",
        "service": "hermes-trading-worker",
        "asset": settings.ASSET,
        "trading_mode": settings.TRADING_MODE,
        "heartbeat": hb
    })

async def async_main():
    import argparse
    from aiohttp import web

    parser = argparse.ArgumentParser(description="Hermes Trading Bot Worker")
    parser.add_argument("--asset", type=str, default=None, help="Asset ticker override")
    args, _ = parser.parse_known_args()

    if args.asset:
        settings.ASSET = args.asset

    print(f"[worker] Booting hermes-trading worker for {settings.ASSET}...")
    worker = TradingWorker()

    port = int(os.getenv("PORT", 10000))
    app = web.Application()
    app.router.add_get("/", _status_handler)
    app.router.add_get("/health", _health_handler)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    print(f"[worker] HTTP healthcheck & status server live on port {port}")

    # Run the worker trading loop
    await worker.run()

def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()