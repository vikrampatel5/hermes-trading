"""Hermes Trading Bot - Reflector Daemon.

This is a long-running local process that monitors the state directory
and triggers Hermes reflection cycles.

Its job is NOT to trade. Its job is to monitor and invoke reflection."""

import asyncio
import json
import os
import signal
import time
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_trading.reflect import Reflector


class ReflectorDaemon:
    """Long-running reflector daemon that monitors trades and triggers reflection."""

    def __init__(self):
        self.reflector = Reflector()
        self.running = True
        self.check_interval = 30  # seconds between checks
        self.project_root = Path(__file__).parent.parent
        self.state_dir = self.project_root / "state"
        self.trades_file = self.state_dir / "trades.jsonl"
        self.heartbeat_file = self.state_dir / "heartbeat.json"
        self.logs_dir = self.project_root / "state" / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.error_log = self.logs_dir / "errors.log"

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals."""
        print(f"[reflector_daemon] Received signal {signum}, shutting down...")
        self.running = False

    async def run(self) -> None:
        """Main daemon loop.

        Periodically checks for new closed trades and triggers reflection
        when the configured cadence is reached.
        """
        print("[reflector_daemon] Reflector daemon starting...")

        while self.running:
            try:
                await self._check_and_trigger_reflection()
            except Exception as e:
                print(f"[reflector_daemon] Error in check: {e}")
                self._log_error(e)

            # Sleep until next check
            await asyncio.sleep(self.check_interval)

        print("[reflector_daemon] Reflector daemon shutting down gracefully.")

    async def _check_and_trigger_reflection(self) -> None:
        """Check if reflection condition is met and trigger if so."""

        # Read trades file to count closed trades
        trade_count = 0
        if self.trades_file.exists():
            with open(self.trades_file, "r", encoding="utf-8-sig") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            trade = json.loads(line)
                            # Count only closed trades (those with exit_price)
                            if trade.get("exit_price", 0) != 0:
                                trade_count += 1
                        except json.JSONDecodeError:
                            continue

        # Get reflection cadence from goal
        goal = self.reflector.goal
        reflection_every = goal.get("reflection_every", 5) if goal else 5

        print(f"[reflector_daemon] Check: {trade_count} closed trades (need {reflection_every})")

        if trade_count >= reflection_every:
            print(f"[reflector_daemon] Reflection threshold reached! Triggering reflection cycle...")

            # Run reflection cycle
            result = self.reflector.run_reflection_cycle()

            # Update heartbeat
            await self._update_heartbeat(result)

            # Print result
            if result.get("action") in ("modified_strategy", "kept_current"):
                print(f"[reflector_daemon] Reflection complete: {result.get('action')}")
                if result.get("accepted"):
                    print(f"[reflector_daemon] New strategy version deployed")
                else:
                    print(f"[reflector_daemon] Current strategy retained")
            else:
                print(f"[reflector_daemon] Reflection skipped: {result.get('reason')}")
        else:
            # Not enough trades yet, just log periodically
            if trade_count > 0 and trade_count % 5 == 0:
                print(f"[reflector_daemon] Waiting for more trades: {trade_count}/{reflection_every}")

    def _update_heartbeat(self, result: Dict[str, Any]) -> None:
        """Update the heartbeat file with current state."""
        from hermes_trading.data import DataEngine
        import json

        # Count total trades
        total_trades = 0
        if self.trades_file.exists():
            with open(self.trades_file, "r", encoding="utf-8-sig") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            trade = json.loads(line)
                            total_trades += 1
                        except json.JSONDecodeError:
                            continue

        # Read current strategy version
        strategy_version = "unknown"
        strategy_file = self.state_dir / "strategy.yaml"
        if strategy_file.exists():
            import yaml
            try:
                with open(strategy_file, "r") as f:
                    config = yaml.safe_load(f)
                    strategy_version = config.get("version", "unknown")
            except Exception:
                pass

        # Read latest trade timestamp
        latest_trade = ""
        if self.trades_file.exists():
            with open(self.trades_file, "r", encoding="utf-8-sig") as f:
                trades = []
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            trade = json.loads(line)
                            trades.append(trade)
                        except json.JSONDecodeError:
                            continue
                if trades:
                    # Get newest by timestamp
                    latest_trade = max(trades, key=lambda t: t.get("timestamp", "")).get("timestamp", "")

        heartbeat = {
            "timestamp": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
            "worker_status": "running",
            "strategy_version": strategy_version,
            "last_trade": latest_trade,
            "closed_trades": total_trades,
            "last_reflection": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
            "last_backtest": None,
            "current_score": result.get("new_score", 0.0) if result else 0.0,
        }

        with open(self.heartbeat_file, "w") as f:
            json.dump(heartbeat, f, indent=2)

    def _log_error(self, error: Any) -> None:
        """Log an error to the error log file."""
        timestamp = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
        with open(self.error_log, "a") as f:
            f.write(f"{timestamp} - {error}\n")


def start_daemon() -> None:
    """Start the reflector daemon as an async service."""
    daemon = ReflectorDaemon()
    
    async def run_daemon():
        await daemon.run()
    
    # Run the daemon
    try:
        asyncio.run(run_daemon())
    except KeyboardInterrupt:
        print("[reflector_daemon] Interrupted by user")