"""Hermes Trading Bot - Score Engine.

Composite scoring function that normalizes strategy quality approximately:
-1 = unacceptable
 0 = neutral
+1 = excellent

Incorporates:
- Return relative to target
- Drawdown relative to maximum
- Sharpe relative to minimum
- Sortino
- Profit factor
- Trade count
- Stability
"""

from typing import Dict, Any, List, Tuple


def score(trades: List[Dict[str, Any]], goal: Dict[str, Any]) -> float:
    """Calculate composite score for a set of trades.

    Normalization:
    -1 = unacceptable
     0 = neutral
    +1 = excellent

    Args:
        trades: List of trade dictionaries with pnl_pct
        goal: Dict with target_return_30d, max_drawdown, min_sharpe

    Returns:
        Score in range approximately [-1, 1]
    """
    if not trades:
        return -1.0  # No trades = unacceptable

    # Extract metrics from trades
    pnls = [t.get("pnl_pct", 0) for t in trades if "pnl_pct" in t]
    if not pnls:
        return -1.0

    trade_count = len(trades)
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    win_rate = len([p for p in pnls if p > 0]) / trade_count
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Target return (annualized approximation from 30-day target)
    target_return_30d = goal.get("target_return_30d", 0.20)
    target_annual = (1 + target_return_30d) ** 4 - 1  # 4x per year

    # Actual return (annualized)
    total_return_pct = sum(pnls) / 100.0  # pnls are percentages
    actual_annual = (1 + total_return_pct) ** 4 - 1

    # Drawdown check - approximate from trade P&Ls
    # Simplified: use profit factor and win rate to gauge drawdown risk
    drawdown_penalty = 0.0
    if profit_factor > 0 and win_rate < 0.4:
        drawdown_penalty = 0.3  # High risk of drawdown with low win rate

    # Sharpe approximation (we can't calculate without return std dev,
    # so use proxy based on profit factor and trade count)
    sharpe_proxy = 0.0
    if trade_count > 10:
        # Rough proxy: profitable systems tend to have positive Sharpe
        sharpe_proxy = (gross_profit - gross_loss) / max(trade_count, 1) * 10

    # Score components
    return_components = []

    # 1. Return relative to target (weight: 0.3)
    return_score = min(max((actual_annual - target_annual) / max(target_annual, 0.1), -0.5), 0.5)
    return_components.append(return_score * 0.3)

    # 2. Drawdown stability (weight: 0.25) - inverted, lower is better
    # Penalize low win rate / high trade count combinations
    drawdown_score = -drawdown_penalty
    drawdown_components = max(0, 1 - drawdown_penalty)
    drawdown_score = drawdown_score * 0.25

    # 3. Sharpe proxy (weight: 0.2)
    sharpe_score = min(max(sharpe_proxy / 3.0, -0.5), 0.5)  # Normalize roughly
    sharpe_score = sharpe_score * 0.2
    sharpe_components = sharpe_score

    # 4. Profit factor (weight: 0.15)
    if profit_factor >= 2.0:
        pf_score = 0.3
    elif profit_factor >= 1.5:
        pf_score = 0.15
    elif profit_factor >= 1.0:
        pf_score = 0.05
    else:
        pf_score = -0.2
    pf_components = pf_score * 0.15

    # 5. Trade count (weight: 0.1) - more trades = more stable, but diminishing returns
    tc_score = min(trade_count / 100, 0.3)  # Cap at 30% of score
    tc_components = tc_score * 0.1

    # 6. Stability (win rate component) (weight: 0.1)
    wr_score = win_rate  # 0 to 1
    # Good win rate contributes positively
    wr_score = wr_score * 0.1

    # Sum all components
    total_score = (
        return_components +
        [drawdown_components] +
        [sharpe_components] +
        [pf_components] +
        [tc_components] +
        [wr_score]
    )

    # Clamp to approximately [-1, 1]
    total_score = max(-1.0, min(1.0, sum(total_score)))

    return round(total_score, 4)