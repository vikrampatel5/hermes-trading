"""Hermes Trading Bot - Paper Execution.

Simulates order execution with fees and slippage.
No real orders are submitted.

Keep the execution interface separated from the strategy engine.
"""

from typing import Dict, Any, Optional


class PaperExecution:
    """Simulates paper trading execution.

    Handles:
    - Entry price simulation with slippage
    - Exit price simulation with slippage
    - Fee collection
    - Position size tracking
    - Realized and unrealized P&L
    """

    def __init__(self, fees: float = 0.001, slippage: float = 0.0005):
        self.fees = fees  # per-side fee rate
        self.slippage = slippage  # per-side slippage rate

    def simulate_entry(
        self,
        market_price: float,
        position_size_pct: float,
        equity: float,
    ) -> Dict[str, Any]:
        """Simulate entering a long position.

        Args:
            market_price: Current market price
            position_size_pct: Position size as percentage of equity
            equity: Current account equity

        Returns:
            Dict with entry simulation results
        """
        # Position size in equity terms
        position_size_equity = equity * (position_size_pct / 100.0)

        # Apply slippage: assume we buy at slightly above market
        entry_price = market_price * (1 + self.slippage)

        # Cost of position (including fees)
        position_cost = position_size_equity  # This is the notional exposure
        total_cost = position_cost * (1 + self.fees)  # Include entry fee

        # Remaining equity after entry
        remaining_equity = equity - total_cost

        return {
            "entry_price": entry_price,
            "position_size_equity": position_size_equity,
            "total_cost": total_cost,
            "remaining_equity": remaining_equity,
            "side": "long",
            "unrealized_pnl": 0.0,
        }

    def simulate_exit(
        self,
        entry_result: Dict[str, Any],
        exit_price: float,
    ) -> Dict[str, Any]:
        """Simulate exiting a position.

        Args:
            entry_result: Result from simulate_entry
            exit_price: Price at which we exit

        Returns:
            Dict with exit simulation results including realized P&L
        """
        position_size = entry_result["position_size_equity"]
        entry_price = entry_result["entry_price"]
        side = entry_result["side"]

        # Calculate realized P&L
        if side == "long":
            pnl = position_size * (exit_price / entry_price - 1 - 2 * self.fees)
            # Total fees round-turn: 2 * self.fees (entry + exit)
        else:
            # Short position - reversed logic
            pnl = position_size * (1 - exit_price / entry_price - 2 * self.fees)

        # Remaining equity after exit
        remaining_equity = entry_result["remaining_equity"] + abs(pnl) + entry_result["total_cost"]

        # Calculate slippage on exit
        exit_with_slippage = exit_price * (1 - self.slippage) if side == "long" else exit_price * (1 + self.slippage)

        # Recalculate P&L with exit slippage
        if side == "long":
            real_pnl = position_size * (exit_with_slippage / entry_price - 1 - 2 * self.fees)
        else:
            real_pnl = pnl  # Already accounted for

        return {
            "exit_price": exit_price,
            "realized_pnl": round(real_pnl, 4),
            "exit_with_slippage": round(exit_with_slippage, 4),
            "remaining_equity": round(remaining_equity, 4),
            "side": side,
        }

    def calculate_unrealized_pnl(
        self,
        entry_result: Dict[str, Any],
        current_price: float,
    ) -> float:
        """Calculate unrealized P&L for an open position.

        Args:
            entry_result: Result from simulate_entry
            current_price: Current market price

        Returns:
            Unrealized P&L amount
        """
        position_size = entry_result["position_size_equity"]
        entry_price = entry_result["entry_price"]

        if entry_result["side"] == "long":
            pnl = position_size * (current_price / entry_price - 1)
        else:
            pnl = 0.0  # Only long strategy for now

        # Subtract fees on unrealized portion
        pnl -= position_size * self.fees

        return pnl