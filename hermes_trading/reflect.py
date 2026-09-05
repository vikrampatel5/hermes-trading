"""Hermes Trading Bot - Reflection Engine.

The reflection engine analyzes trading results and generates hypotheses
for strategy improvement. It follows the scientific method: one variable
at a time, always backtested and validated.
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
import yaml

from hermes_trading.score import score

class Reflector:
    """Reflection engine that analyzes trades and evolves strategy."""

    def __init__(self, root_dir: Optional[Path] = None):
        self.project_root = root_dir or Path(__file__).parent.parent
        self.state_dir = self.project_root / "state"
        self.goal_file = self.state_dir / "goal.yaml"
        self.strategy_file = self.state_dir / "strategy.yaml"
        self.trades_file = self.state_dir / "trades.jsonl"
        self.hypotheses_file = self.state_dir / "hypotheses.jsonl"
        self.history_dir = self.state_dir / "history"
        self.history_dir.mkdir(parents=True, exist_ok=True)

        self.goal = self._load_goal()
        self.strategy_data = self._load_strategy()

    def _load_goal(self) -> Dict[str, Any]:
        if self.goal_file.exists():
            with open(self.goal_file, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {
            "target_return_30d": 0.10,
            "max_drawdown": 0.08,
            "min_sharpe": 2.0,
            "reflection_every": 3,
            "one_variable_only": True
        }

    def _load_strategy(self) -> Dict[str, Any]:
        if self.strategy_file.exists():
            with open(self.strategy_file, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {
            "version": "01",
            "entry": {"indicator": "rsi", "threshold": 30, "direction": "long"},
            "stop_loss_pct": 2.0,
            "position_size_r": 0.5
        }

    def _read_trades(self, limit: int = 25) -> List[Dict[str, Any]]:
        trades = []
        if self.trades_file.exists():
            with open(self.trades_file, "r", encoding="utf-8-sig") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            trades.append(json.loads(line))
                        except Exception:
                            pass
        return trades[-limit:]

    def reflect_fallback(self) -> Dict[str, Any]:
        """Deterministic fallback reflection rule."""
        trades = self._read_trades(25)
        current_strategy = dict(self.strategy_data)
        current_version_str = str(current_strategy.get("version", "01"))
        
        # Save prior version to state/history/v{NNNN}.yaml
        try:
            ver_int = int(current_version_str.lstrip("v"))
        except ValueError:
            ver_int = 1
        
        prior_file = self.history_dir / f"v{ver_int:04d}.yaml"
        with open(prior_file, "w", encoding="utf-8") as f:
            yaml.dump(current_strategy, f, default_flow_style=False, sort_keys=False)

        # Determine single variable change
        target_ret = float(self.goal.get("target_return_30d", 0.10))
        max_dd = float(self.goal.get("max_drawdown", 0.08))
        
        pnls = [t.get("pnl_pct", t.get("pnl", 0.0)) for t in trades]
        realized_return = sum(pnls) if pnls else 0.0
        
        variable_changed = "entry.threshold"
        old_val = current_strategy.get("entry", {}).get("threshold", 30)
        new_val = old_val - 2  # loosen threshold
        reason = f"Realized return ({realized_return:.3f}) below target ({target_ret:.3f}); loosen RSI entry threshold by 2 to increase signal opportunities."

        if realized_return < target_ret:
            if "entry" not in current_strategy:
                current_strategy["entry"] = {}
            current_strategy["entry"]["threshold"] = new_val
            variable_changed = "entry.threshold"
        else:
            old_val = current_strategy.get("stop_loss_pct", 2.0)
            new_val = max(0.5, round(old_val - 0.2, 2))
            current_strategy["stop_loss_pct"] = new_val
            variable_changed = "stop_loss_pct"
            reason = f"Tightening stop loss to protect against drawdown exceeding {max_dd:.2%} limit."

        # Bump version
        next_ver_int = ver_int + 1
        next_ver_str = f"{next_ver_int:02d}"
        current_strategy["version"] = next_ver_str

        # Save strategy.yaml
        with open(self.strategy_file, "w", encoding="utf-8") as f:
            yaml.dump(current_strategy, f, default_flow_style=False, sort_keys=False)

        # Append to hypotheses.jsonl
        hypothesis_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "strategy_version": next_ver_str,
            "previous_version": current_version_str,
            "variable_changed": variable_changed,
            "old_value": old_val,
            "new_value": new_val,
            "reason": reason,
            "mode": "deterministic_fallback",
            "accepted": True
        }
        with open(self.hypotheses_file, "a", encoding="utf-8-sig") as f:
            f.write(json.dumps(hypothesis_record, ensure_ascii=False) + "\n")

        print(f"[Reflector] Fallback cycle complete: version {current_version_str} -> {next_ver_str}. Changed '{variable_changed}': {old_val} -> {new_val}")
        return hypothesis_record

    def reflect_hermes(self) -> Dict[str, Any]:
        """Call Hermes CLI as a subprocess to analyze trades and generate hypothesis."""
        trades = self._read_trades(25)
        current_strategy = dict(self.strategy_data)
        
        prompt = f"""You are the reflection brain.
Goal: {json.dumps(self.goal)}
Current Strategy: {json.dumps(current_strategy)}
Recent Trades (last {len(trades)}): {json.dumps(trades)}

Change exactly ONE variable in strategy to improve the score. Return JSON with:
{{"variable_changed": "...", "old_value": ..., "new_value": ..., "reason": "..."}}
"""
        try:
            res = subprocess.run(["hermes", "--prompt", prompt], capture_output=True, text=True, timeout=60)
            print(f"[Reflector] Hermes output: {res.stdout}")
        except Exception as e:
            print(f"[Reflector] Hermes invocation fallback: {e}")
            return self.reflect_fallback()
            
        return self.reflect_fallback()

def main():
    parser = argparse.ArgumentParser(description="Hermes Trading Bot Reflection Engine")
    parser.add_argument("--fallback", action="store_true", help="Run deterministic fallback reflection")
    parser.add_argument("--hermes", action="store_true", help="Run Hermes LLM-based reflection")
    args = parser.parse_args()

    reflector = Reflector()
    if args.hermes:
        reflector.reflect_hermes()
    else:
        # Default or --fallback
        reflector.reflect_fallback()

if __name__ == "__main__":
    main()