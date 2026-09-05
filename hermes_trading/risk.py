"""Hermes Trading Bot - Risk Management.

Hard controls for position sizing, exposure, drawdown limits, and
other risk boundaries. When a hard risk limit is breached, new
entries are stopped."""
from typing import Dict, Any, List, Tuple
from pathlib import Path


class RiskManager:
    """Hard risk control manager.

    Enforces:
    - Maximum position size
    - Maximum portfolio exposure
    - Maximum drawdown
    - Maximum daily loss
    - Maximum consecutive losses
    - Stop loss / take profit enforcement
    """

    def __init__(self, settings=None):
        self.settings = settings or self._default_settings()

    def _default_settings(self) -> Dict[str, Any]:
        return {
            "max_position_size_pct": 5.0,       # Max % equity per position
            "max_portfolio_exposure_pct": 10.0,  # Max total open positions
            "max_drawdown_pct": 8.0,            # Max portfolio drawdown
            "max_daily_loss_pct": 3.0,          # Max daily loss
            "max_consecutive_losses": 3,        # Max consecutive losses before stop
        }

    def check_entry_allowed(
        self,
        equity: float,
        proposed_position_pct: float,
        current_drawdown_pct: float,
        consecutive_losses: int,
        daily_loss_pct: float,
    ) -> Tuple[bool, str]:
        """Check if a new entry is allowed under risk controls.

        Returns:
            (allowed, reason) tuple. If not allowed, reason explains why.
        """
        # Check maximum position size
        if proposed_position_pct > self.settings["max_position_size_pct"]:
            return False, f"Position size {proposed_position_pct}% exceeds max {self.settings['max_position_size_pct']}%"

        # Check maximum portfolio exposure (simplified - would track all open positions)
        if proposed_position_pct + self.settings["max_portfolio_exposure_pct"] > 100:
            return False, "Proposed position would exceed maximum portfolio exposure"

        # Check maximum drawdown
        if current_drawdown_pct >= self.settings["max_drawdown_pct"]:
            return False, f"Drawdown {current_drawdown_pct}% at or above maximum {self.settings['max_drawdown_pct']}%"

        # Check maximum daily loss
        if daily_loss_pct >= self.settings["max_daily_loss_pct"]:
            return False, f"Daily loss {daily_loss_pct}% at or above maximum {self.settings['max_daily_loss_pct']}%"

        # Check maximum consecutive losses
        if consecutive_losses >= self.settings["max_consecutive_losses"]:
            return False, f"Consecutive losses {consecutive_losses} at or above maximum {self.settings['max_consecutive_losses']}"

        return True, "Entry allowed"

    def check_exit_needed(
        self,
        entry_result: Dict[str, Any],
        current_price: float,
        original_stop_loss: float,
        original_take_profit: float,
    ) -> Dict[str, Any]:
        """Check if an exit is triggered by risk levels.

        Returns:
            Dict with exit reason and price information.
        """
        result = {
            "exit_needed": False,
            "exit_reason": "",
            "exit_price": None,
        }

        # Check trailing stop or original SL/TP
        # This is a simplified check - full implementation would track
        # price movement since entry
        price_change_pct = (current_price / entry_result.get("entry_price", current_price)) - 1

        # Stop loss check
        if price_change_pct <= -original_stop_loss / 100.0:
            result["exit_needed"] = True
            result["exit_reason"] = "stop_loss"
            result["exit_price"] = entry_result.get("entry_price", current_price) * (1 - original_stop_loss / 100.0)

        # Take profit check
        elif price_change_pct >= original_take_profit / 100.0:
            result["exit_needed"] = True
            result["exit_reason"] = "take_profit"
            result["exit_price"] = entry_result.get("entry_price", current_price) * (1 + original_take_profit / 100.0)

        return result