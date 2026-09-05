"""Hermes Trading Bot - Multi-pair trading engine.

Trades a universe of Delta Exchange perpetual futures in parallel, applying
a half-max-leverage rule per pair (max 100x → use 50x, 50x → 25x, 20x → 10x).

Each pair has its own per-pair state file (open position, pending orders,
recent trades) so signals on one symbol never block another.
"""

import json
import time
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from hermes_trading.config import settings
from hermes_trading.adapters.delta import (
    fetch_ohlc_sync,
    get_positions_sync,
    place_order_sync,
)


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------

def load_universe() -> List[Dict[str, Any]]:
    """Load the trading universe from state/delta_universe.json.

    Falls back to a hard-coded list if the file is missing (first boot).
    """
    project_root = Path(__file__).parent.parent
    universe_file = project_root / "state" / "delta_universe.json"
    if universe_file.exists():
        with open(universe_file, "r") as f:
            data = json.load(f)
        return data.get("universe", [])
    return DEFAULT_UNIVERSE


def refresh_universe(api_key: str, api_secret: str, mode: str = "testnet") -> List[Dict[str, Any]]:
    """Re-fetch the universe from Delta and persist to state/delta_universe.json."""
    import hmac
    import hashlib

    base = "https://cdn-ind.testnet.deltaex.org" if mode == "testnet" else "https://api.india.delta.exchange"

    def sign(method, path, query="", body=""):
        ts = str(int(time.time()))
        msg = f"{method}{ts}{path}{query}{body}"
        sig = hmac.new(api_secret.encode(), msg.encode(), hashlib.sha256).hexdigest()
        return ts, sig

    # Products
    ts, sig = sign("GET", "/v2/products")
    headers = {"api-key": api_key, "signature": sig, "timestamp": ts, "User-Agent": "Universe/1.0"}
    products = httpx.get(f"{base}/v2/products", headers=headers, timeout=10.0).json().get("result", [])

    # Tickers
    ts, sig = sign("GET", "/v2/tickers")
    headers = {"api-key": api_key, "signature": sig, "timestamp": ts, "User-Agent": "Universe/1.0"}
    tickers = httpx.get(f"{base}/v2/tickers", headers=headers, timeout=10.0).json().get("result", [])
    ticker_map = {t.get("symbol"): t for t in tickers}

    universe = []
    for p in products:
        if p.get("contract_type") != "perpetual_futures" or p.get("state") != "live":
            continue
        sym = p.get("symbol")
        tk = ticker_map.get(sym, {})
        leverage_slider = p.get("ui_config", {}).get("leverage_slider_values", [])
        max_lev = max(leverage_slider) if leverage_slider else 10
        trading_lev = max_lev // 2 if max_lev >= 2 else 1
        universe.append({
            "product_id": p.get("id"),
            "symbol": sym,
            "underlying": p.get("underlying_asset", {}).get("symbol"),
            "mark_price": float(tk.get("close", 0) or 0),
            "volume_24h": float(tk.get("volume", 0) or 0),
            "max_leverage": max_lev,
            "trading_leverage": trading_lev,
            "initial_margin_pct": float(p.get("initial_margin", 0.5)),
            "max_leverage_notional": float(p.get("max_leverage_notional", 0)),
            "contract_value": float(p.get("contract_value", 1)),
            "tick_size": float(p.get("tick_size", 0.01)),
        })

    universe.sort(key=lambda x: x["volume_24h"], reverse=True)

    project_root = Path(__file__).parent.parent
    universe_file = project_root / "state" / "delta_universe.json"
    universe_file.parent.mkdir(parents=True, exist_ok=True)
    with open(universe_file, "w") as f:
        json.dump({"universe": universe, "total": len(universe), "last_updated": time.time()}, f, indent=2)

    return universe


# Hard-coded fallback (in case the JSON file is wiped)
DEFAULT_UNIVERSE = [
    {"product_id": 162764, "symbol": "1000SHIBUSD", "trading_leverage": 10, "max_leverage": 20, "tick_size": 0.000001, "contract_value": 1000, "max_leverage_notional": 10000},
    {"product_id": 101555, "symbol": "DOGEUSD", "trading_leverage": 50, "max_leverage": 100, "tick_size": 0.0001, "contract_value": 100, "max_leverage_notional": 10000},
    {"product_id": 101760, "symbol": "ADAUSD", "trading_leverage": 50, "max_leverage": 100, "tick_size": 0.0001, "contract_value": 1, "max_leverage_notional": 10000},
    {"product_id": 93723, "symbol": "XRPUSD", "trading_leverage": 50, "max_leverage": 100, "tick_size": 0.001, "contract_value": 1, "max_leverage_notional": 10000},
    {"product_id": 92572, "symbol": "SOLUSD", "trading_leverage": 50, "max_leverage": 100, "tick_size": 0.01, "contract_value": 1, "max_leverage_notional": 10000},
    {"product_id": 1699, "symbol": "ETHUSD", "trading_leverage": 50, "max_leverage": 100, "tick_size": 0.1, "contract_value": 0.01, "max_leverage_notional": 10000},
    {"product_id": 84, "symbol": "BTCUSD", "trading_leverage": 50, "max_leverage": 100, "tick_size": 0.1, "contract_value": 0.001, "max_leverage_notional": 10000},
]


# ---------------------------------------------------------------------------
# Leverage helper
# ---------------------------------------------------------------------------

def set_leverage(product_id: int, leverage: int, mode: str = "testnet",
                 api_key: str = None, api_secret: str = None) -> Dict[str, Any]:
    """
    Set the leverage for a product on Delta Exchange.

    Delta endpoint: POST /v2/products/{product_id}/orders/leverage
    Body: {"leverage": "<int as string>"}
    """
    import hmac
    import hashlib

    if not api_key or not api_secret:
        return {"status": "error", "error": "no credentials"}

    base = "https://cdn-ind.testnet.deltaex.org" if mode == "testnet" else "https://api.india.delta.exchange"

    path = f"/v2/products/{product_id}/orders/leverage"
    body = json.dumps({"leverage": str(leverage)})

    ts = str(int(time.time()))
    msg = f"POST{ts}{path}{body}"
    sig = hmac.new(api_secret.encode(), msg.encode(), hashlib.sha256).hexdigest()

    headers = {
        "api-key": api_key,
        "signature": sig,
        "timestamp": ts,
        "User-Agent": "Hermes-Trading/1.0",
        "Content-Type": "application/json",
    }

    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.post(f"{base}{path}", headers=headers, content=body)
            data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {"raw": resp.text}
            if resp.status_code == 200 and data.get("success"):
                return {"status": "ok", "leverage": leverage, "product_id": product_id}
            return {"status": "error", "error": data.get("error", data), "product_id": product_id, "leverage": leverage}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ---------------------------------------------------------------------------
# Per-pair state
# ---------------------------------------------------------------------------

@dataclass
class PairState:
    """Per-pair trading state. Persisted to disk so restarts are lossless."""
    symbol: str
    product_id: int
    trading_leverage: int
    position_size: int = 0            # contracts (positive=long, negative=short)
    entry_price: float = 0.0
    last_signal: str = "none"         # "long" / "short" / "none"
    last_signal_time: str = ""
    last_keepalive_signal: str = "none"
    recent_trades: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PairState":
        return cls(**d)


def load_pair_states(universe: List[Dict[str, Any]],
                     state_dir: Path) -> Dict[str, PairState]:
    states = {}
    for u in universe:
        sf = state_dir / f"pair_{u['symbol']}.json"
        if sf.exists():
            try:
                with open(sf, "r") as f:
                    states[u["symbol"]] = PairState.from_dict(json.load(f))
                continue
            except Exception:
                pass
        states[u["symbol"]] = PairState(
            symbol=u["symbol"],
            product_id=u["product_id"],
            trading_leverage=u.get("trading_leverage", 10),
        )
    return states


def save_pair_state(state_dir: Path, state: PairState) -> None:
    sf = state_dir / f"pair_{state.symbol}.json"
    with open(sf, "w") as f:
        json.dump(state.to_dict(), f, indent=2)


# ---------------------------------------------------------------------------
# Multi-pair engine
# ---------------------------------------------------------------------------

class MultiPairEngine:
    """
    Runs the same RSI-style mean-reversion strategy on every pair in the
    universe in parallel. Each pair is independent — a long BTC signal
    does not block a long ETH signal.

    Uses Delta Exchange testnet/live for both data and execution. Falls
    back to the existing Binance/Coingecko price adapter if Delta fails.
    """

    def __init__(self,
                 api_key: str,
                 api_secret: str,
                 mode: str = "testnet",
                 universe: Optional[List[Dict[str, Any]]] = None,
                 state_dir: Optional[Path] = None,
                 rsi_oversold: float = 30.0,
                 rsi_overbought: float = 70.0,
                 notional_per_trade_usd: float = 200.0,
                 take_profit_pct: float = 1.0,
                 stop_loss_pct: float = 0.5,
                 timeframe: str = "5m",
                 limit: int = 100):
        self.api_key = api_key
        self.api_secret = api_secret
        self.mode = mode
        self.state_dir = state_dir or (Path(__file__).parent.parent / "state")
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Refresh universe if requested
        if universe is None:
            universe = load_universe()
        self.universe = universe
        self.pair_states = load_pair_states(self.universe, self.state_dir)

        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.notional_per_trade_usd = notional_per_trade_usd
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.timeframe = timeframe
        self.limit = limit

        # Leverage-set tracking so we only POST to /leverage once per pair per session
        self._leverage_set: set = set()

    # ----- signal generation -----

    def _calculate_rsi(self, closes: List[float], period: int = 14) -> Optional[float]:
        if len(closes) < period + 1:
            return None
        deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        gains = [max(0, d) for d in deltas[-period:]]
        losses = [max(0, -d) for d in deltas[-period:]]
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _generate_signal(self, closes: List[float]) -> str:
        """Mean-reversion: oversold→long, overbought→short, else flat."""
        rsi = self._calculate_rsi(closes)
        if rsi is None:
            return "none"
        if rsi < self.rsi_oversold:
            return "long"
        if rsi > self.rsi_overbought:
            return "short"
        return "none"

    # ----- data -----

    def _fetch_candles(self, product_id: int, symbol: str) -> Optional[List[Dict[str, Any]]]:
        """Try Delta, fall back to Binance via the existing price adapter."""
        try:
            data = fetch_ohlc_sync(
                symbol=symbol,
                timeframe=self.timeframe,
                limit=self.limit,
                mode=self.mode,
                api_key=self.api_key,
                api_secret=self.api_secret,
            )
            if data.get("status") == "ok" and data.get("candles"):
                return data["candles"]
        except Exception:
            pass

        # Fallback: hit Binance public klines (no auth, works for major pairs)
        try:
            binance_sym = f"{symbol[:-3]}USDT" if symbol.endswith("USD") and not symbol.endswith("USDT") else symbol
            url = f"https://api.binance.com/api/v3/klines?symbol={binance_sym}&interval={self.timeframe}&limit={self.limit}"
            with httpx.Client(timeout=8.0) as client:
                resp = client.get(url)
                if resp.status_code == 200:
                    rows = resp.json()
                    return [
                        {"timestamp": r[0], "open": float(r[1]), "high": float(r[2]),
                         "low": float(r[3]), "close": float(r[4]), "volume": float(r[5])}
                        for r in rows
                    ]
        except Exception:
            pass
        return None

    # ----- execution -----

    def _set_leverage_if_needed(self, state: PairState) -> None:
        if state.symbol in self._leverage_set:
            return
        result = set_leverage(
            product_id=state.product_id,
            leverage=state.trading_leverage,
            mode=self.mode,
            api_key=self.api_key,
            api_secret=self.api_secret,
        )
        if result.get("status") == "ok":
            self._leverage_set.add(state.symbol)

    def _position_size_contracts(self, mark_price: float, contract_value: float) -> int:
        """Notional in USD / (mark * contract_value) = number of contracts."""
        if mark_price <= 0:
            return 0
        contract_notional = mark_price * contract_value
        contracts = self.notional_per_trade_usd / contract_notional
        return max(1, int(contracts))

    def _open_position(self, state: PairState, side: str, mark_price: float, universe_entry: Dict[str, Any]):
        contract_value = universe_entry.get("contract_value", 1)
        contracts = self._position_size_contracts(mark_price, contract_value)
        if contracts <= 0:
            return

        order = place_order_sync(
            side="buy" if side == "long" else "sell",
            quantity=contracts,
            order_type="market_order",
            symbol=state.symbol,
            mode=self.mode,
            api_key=self.api_key,
            api_secret=self.api_secret,
        )
        fill_price = float(order.get("average_fill_price") or mark_price)
        state.position_size = contracts if side == "long" else -contracts
        state.entry_price = fill_price
        state.last_signal = side
        state.last_signal_time = datetime.now(timezone.utc).isoformat()
        return order

    def _close_position(self, state: PairState, reason: str):
        if state.position_size == 0:
            return None
        side = "sell" if state.position_size > 0 else "buy"
        order = place_order_sync(
            side=side,
            quantity=abs(state.position_size),
            order_type="market_order",
            symbol=state.symbol,
            mode=self.mode,
            api_key=self.api_key,
            api_secret=self.api_secret,
            reduce_only=True,
        )
        fill_price = float(order.get("average_fill_price") or state.entry_price)
        pnl_pct = 0.0
        if state.entry_price > 0:
            if state.position_size > 0:
                pnl_pct = (fill_price - state.entry_price) / state.entry_price * 100
            else:
                pnl_pct = (state.entry_price - fill_price) / state.entry_price * 100
        trade = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": state.symbol,
            "side": "long" if state.position_size > 0 else "short",
            "entry_price": state.entry_price,
            "exit_price": fill_price,
            "contracts": abs(state.position_size),
            "leverage": state.trading_leverage,
            "pnl_pct": round(pnl_pct, 4),
            "reason": reason,
            "order_id": order.get("order_id"),
            "exchange": "delta_india",
            "mode": self.mode,
        }
        state.recent_trades.append(trade)
        state.recent_trades = state.recent_trades[-20:]
        state.position_size = 0
        state.entry_price = 0.0
        return trade

    def _check_exit(self, state: PairState, mark_price: float) -> Optional[str]:
        if state.position_size == 0 or state.entry_price == 0:
            return None
        if state.position_size > 0:
            pnl_pct = (mark_price - state.entry_price) / state.entry_price * 100
        else:
            pnl_pct = (state.entry_price - mark_price) / state.entry_price * 100
        if pnl_pct >= self.take_profit_pct:
            return "take_profit"
        if pnl_pct <= -self.stop_loss_pct:
            return "stop_loss"
        return None

    # ----- per-pair tick -----

    def tick_pair(self, symbol: str, universe_entry: Dict[str, Any]) -> Dict[str, Any]:
        state = self.pair_states[symbol]
        result = {"symbol": symbol, "action": "skipped"}

        # 1. Set leverage once
        self._set_leverage_if_needed(state)

        # 2. Fetch candles
        candles = self._fetch_candles(state.product_id, symbol)
        if not candles or len(candles) < 15:
            result["action"] = "no_data"
            return result

        closes = [c["close"] for c in candles]
        mark_price = closes[-1]

        # 3. Exit management first
        exit_reason = self._check_exit(state, mark_price)
        if exit_reason:
            trade = self._close_position(state, exit_reason)
            result["action"] = "exit"
            result["exit_reason"] = exit_reason
            result["trade"] = trade
            save_pair_state(self.state_dir, state)
            return result

        # 4. Entry signal
        signal = self._generate_signal(closes)
        state.last_keepalive_signal = signal  # tracked even if not acted on

        if signal != "none" and state.position_size == 0:
            try:
                order = self._open_position(state, signal, mark_price, universe_entry)
                result["action"] = f"open_{signal}"
                result["order"] = order
            except Exception as e:
                result["action"] = "order_failed"
                result["error"] = str(e)

        save_pair_state(self.state_dir, state)
        result["signal"] = signal
        result["mark_price"] = mark_price
        result["rsi"] = self._calculate_rsi(closes)
        return result

    def tick_all(self) -> List[Dict[str, Any]]:
        """Run one iteration across the entire universe in parallel (sequential here, but each pair is independent)."""
        results = []
        for u in self.universe:
            try:
                r = self.tick_pair(u["symbol"], u)
                results.append(r)
            except Exception as e:
                results.append({"symbol": u["symbol"], "action": "error", "error": str(e)})
        return results


# ---------------------------------------------------------------------------
# Run as a module
# ---------------------------------------------------------------------------

def main():
    api_key = settings.EXCHANGE_API_KEY
    api_secret = settings.EXCHANGE_API_SECRET
    if not api_key or not api_secret:
        print("[multi] No API credentials in .env — running in paper-only mode is fine, but live orders need keys.")
    mode = settings.DELTA_MODE
    print(f"[multi] Booting multi-pair engine, mode={mode}, universe_size={len(load_universe())}")
    engine = MultiPairEngine(
        api_key=api_key,
        api_secret=api_secret,
        mode=mode,
    )
    while True:
        results = engine.tick_all()
        trades = [r for r in results if r.get("action", "").startswith("open_") or r.get("action") == "exit"]
        if trades:
            for t in trades:
                print(f"[multi] {t['symbol']}: {t['action']} @ {t.get('mark_price', 'n/a')}")
        time.sleep(60)


if __name__ == "__main__":
    main()